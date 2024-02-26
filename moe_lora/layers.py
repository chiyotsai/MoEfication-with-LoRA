import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinearLayer(nn.Module):
    def __init__(self, original_weight, rank, useBias=True, alpha=1.0, merge=False):
        super(LoRALinearLayer, self).__init__()

        self.original_weight = nn.Parameter(original_weight)

        # Low-rank matrices A and B
        self.in_features, self.out_features = original_weight.shape
        self.rank = rank
        self.A = nn.Parameter(torch.empty(self.in_features, self.rank))
        self.B = nn.Parameter(torch.zeros(self.rank, self.out_features))
        nn.init.normal_(self.A)

        # Scaling factor
        self.scaling = alpha / self.rank
        
        # Bias layer
        self.useBias = useBias
        if self.useBias:
            self.bias = nn.Parameter(torch.zeros(self.out_features), requires_grad=True)
        else:
            self.bias = None

        # Weight merging flags
        self.merge_weights = merge 
        self.merged = False 

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lora_rank={self.rank}, bias={self.useBias})'

    def forward(self, x):
        # Check if the weights are merged or not
        if self.merged:
            adapted_weight = self.original_weight
        else:
            low_rank_update = self.A @ self.B * self.scaling
            adapted_weight = self.original_weight + low_rank_update

        if self.useBias:
            return F.linear(x, adapted_weight, self.bias)
        else:
            return F.linear(x, adapted_weight)

    
    def train(self, mode: bool = True):
        super(LoRALinearLayer, self).train(mode)
        if mode:
            # Training
            if self.merge_weights and self.merged:
                # Restore the original weights
                self.original_weight.data = self.original_weight_backup.clone()
                self.merged = False
        else:
            # Evaluation
            if self.merge_weights and not self.merged:
                # Backup original weights
                if not hasattr(self, 'original_weight_backup'):
                    self.original_weight_backup = self.original_weight.clone()
                # Merge the weights
                self.original_weight.data += ((self.A @ self.B) * self.scaling).data
                self.merged = True
