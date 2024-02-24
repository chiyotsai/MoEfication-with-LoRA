import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinearLayer(nn.Module):
    def __init__(self, original_weight, rank, dropout=0.1, alpha=1.0, merge=False):
        super(LoRALinearLayer, self).__init__()

        # Get the original weight and set it as a non-trainable parameter
        self.original_weight = original_weight.detach()
        self.original_weight.requires_grad = False

        # Low-rank matrices A and B
        in_features, out_features = original_weight.shape
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

        # Scaling factor
        self.scaling = alpha / rank
        
        # Bias layer (check if bias is present in the original layer)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=True)

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Weight merging flags
        self.merge_weights = merge 
        self.merged = False 

    def forward(self, x):
        x = self.dropout(x)
        
        # Check if the weights are merged or not
        if self.merged:
            adapted_weight = self.original_weight
        else:
            low_rank_update = self.A @ self.B * self.scaling
            adapted_weight = self.original_weight + low_rank_update
        
        return F.linear(x, adapted_weight.t(), self.bias)

    
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




