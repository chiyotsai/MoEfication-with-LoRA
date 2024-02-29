import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinearLayer(nn.Module):
    def __init__(self, original_weight, rank, useBias=True, dropout=0.1, alpha=1.0, merge=False):
        super(LoRALinearLayer, self).__init__()

        self.device = original_weight.device
        self.original_weight = nn.Parameter(original_weight).to(self.device)
        self.original_weight.requires_grad = False

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
            self.bias = nn.Parameter(torch.zeros(self.out_features).to(self.device), requires_grad=True)
        else:
            self.bias = None

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Weight merging flags
        self.merge_weights = merge 
        self.merged = False 

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, lora_rank={self.rank}, bias={self.useBias}, dropout={self.dropout.p})'

    def forward(self, x):
        x = self.dropout(x)
        
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
                if not hasattr(self, 'original_weight_backup'):
                    self.original_weight_backup = self.original_weight.clone()
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


class MoELoRALinearLayer(nn.Module):
    def __init__(self, original_weight, rank, num_experts, useBias=True, dropout=0.1, alpha=1.0, merge=False):
        super(MoELoRALinearLayer, self).__init__()

        self.device = original_weight.device
        self.original_weight = nn.Parameter(original_weight).to(self.device)
        self.original_weight.requires_grad = False

        # Construct LoRA tensors
        self.out_features, self.in_features = original_weight.shape
        self.rank = rank
        self.num_experts = num_experts
        self.A = nn.Parameter(torch.empty(self.num_experts, self.in_features, self.rank))
        self.B = nn.Parameter(torch.zeros(self.num_experts, self.rank, self.out_features))
        nn.init.normal_(self.A)

        # Scaling factor
        self.scaling = alpha / self.rank
        
        # Bias layer
        self.useBias = useBias
        if self.useBias:
            self.bias = nn.Parameter(torch.zeros(self.out_features).to(self.device), requires_grad=True)
        else:
            self.bias = None

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Weight merging flags
        self.merge_weights = merge 
        self.merged = False 

    def __repr__(self):
        return f'{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, A={self.A.shape}, B={self.B.shape} bias={self.useBias}, dropout={self.dropout.p})'

    def forward(self, x, labels):
        x = self.dropout(x)

        print('x: ', x.shape)
        print('original_weight: ', self.original_weight.shape)
        print('A: ', self.A.shape)
        print('B: ', self.B.shape)

        # Compute low-rank updates for all experts
        low_rank_updates = torch.einsum('eir,ero->eio', self.A, self.B) * self.scaling # Shape [num_experts, in_features, out_features]
        adapted_weights = self.original_weight.unsqueeze(0) + low_rank_updates

        # Select the adapted weight for each instance in the batch based on labels
        selected_adapted_weights = torch.zeros_like(self.original_weight).unsqueeze(0).expand(x.size(0), -1, -1)
        for i in range(x.size(0)):
            print(labels[i].shape)
            selected_adapted_weights[i] = adapted_weights[labels[i]]

        # Use the selected adapted weights to compute the output
        output = torch.bmm(x.unsqueeze(1), selected_adapted_weights).squeeze(1)

        if self.useBias:
            output += self.bias

        return output


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




