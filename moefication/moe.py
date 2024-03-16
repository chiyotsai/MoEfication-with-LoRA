import torch
import torch.nn.functional as F
import numpy as np

from fmoe.functions import MOEScatter, MOEGather
from fmoe.functions import prepare_forward
from fmoe.linear import MOELinear

from transformers.models.t5.modeling_t5 import T5DenseActDense
from copy import deepcopy


class GroundTruthMoEDenseActDense(torch.nn.Module):
    def __init__(self, original_layer, num_experts, top_k, moe_params_path):
        super().__init__()

        assert type(original_layer) == T5DenseActDense
        labels = torch.load(moe_params_path)

        patterns = [np.array(labels) == i for i in range(num_experts)]

        self.num_experts = num_experts
        self.patterns = torch.Tensor(np.array(patterns)).cuda()
        self.k = top_k
        self.mlp = torch.load(moe_params_path+'_input_compl').cuda()

        self.wi = deepcopy(original_layer.wi)
        self.wo = deepcopy(original_layer.wo)
        self.dropout = deepcopy(original_layer.dropout)
        self.act = deepcopy(original_layer.act)


    def forward(self, x):
        hidden_states = self.wi(x)
        hidden_states = self.act(hidden_states)

        with torch.no_grad():
            bsz, seq_len, hidden_size = hidden_states.shape
            hidden_states_relu = hidden_states.clone()
            hidden_states_relu = hidden_states_relu.view(-1, hidden_size)
            score = torch.matmul(hidden_states_relu, self.patterns.transpose(0, 1))
            labels = torch.topk(score, k=self.k, dim=-1)[1].view(bsz, seq_len, self.k)
            cur_mask = torch.nn.functional.embedding(labels, self.patterns).sum(-2)

        hidden_states[cur_mask == False] = 0  

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)

        return hidden_states


class NaiveMoEDenseActDense(torch.nn.Module):
    def __init__(self, original_layer, num_experts, top_k, moe_params_path):
        super().__init__()

        assert type(original_layer) == T5DenseActDense
        labels = torch.load(moe_params_path)

        patterns = [np.array(labels) == i for i in range(num_experts)]

        self.num_experts = num_experts
        self.patterns = torch.Tensor(np.array(patterns)).cuda()
        self.k = top_k
        self.mlp = torch.load(moe_params_path+'_input_compl').cuda()
        for params in self.mlp.parameters():
            params.requires_grad = False

        self.wi = deepcopy(original_layer.wi)
        self.wo = deepcopy(original_layer.wo)
        self.dropout = deepcopy(original_layer.dropout)
        self.act = deepcopy(original_layer.act)


    def forward(self, x):
        bsz, seq_len, hidden_size = x.shape

        hidden_states_mlp = x.view(-1, hidden_size)
        hidden_states_mlp = hidden_states_mlp / torch.norm(hidden_states_mlp, dim=-1).unsqueeze(-1)
        score = self.mlp(hidden_states_mlp)

        labels = torch.topk(score, k=self.k, dim=-1)[1].view(bsz, seq_len, self.k)
        cur_mask = torch.nn.functional.embedding(labels, self.patterns).sum(-2)

        hidden_states = self.wi(x)
        hidden_states = self.act(hidden_states)

        cur_mask = cur_mask.float()
        hidden_states = hidden_states * cur_mask

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)

        return hidden_states


def _fmoe_general_global_forward(inp, gate, expert_fn, num_expert, world_size, **kwargs):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, num_expert, world_size)
    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    x = MOEScatter.apply(
            inp,
            torch.div(pos, topk, rounding_mode='floor'),
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            world_size,
        )

    x = expert_fn(x, fwd_expert_count)

    out_batch_size = inp.shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    outp = MOEGather.apply(
            x,
            pos,
            local_expert_count,
            global_expert_count,
            out_batch_size,
            world_size,
        )
    return outp

class CustomGate(torch.nn.Module):
    def __init__(self, dim_in, expert_num, dtype, init_mean = 0.0, init_std = 0.02, length_scale=False):
        super().__init__()
        self.wg = torch.nn.Parameter(torch.empty((expert_num, dim_in), dtype=dtype))


    def forward(self, x):
        return F.linear(x, self.wg)

class CustomExpert(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden_size, expert_num, dtype, init_mean = 0.0, init_std = 0.02, dropout_p = 0.0, length_scale=False):
        super().__init__()

        self.expert_hidden_size = hidden_size // expert_num
        self.batched_fc1_w = torch.nn.Parameter(torch.empty((expert_num, self.expert_hidden_size, dim_in), dtype=dtype))
        self.batched_fc2_w = torch.nn.Parameter(torch.empty((expert_num, dim_out, self.expert_hidden_size), dtype=dtype))
        self.temp = torch.nn.Parameter(torch.empty((20 * self.expert_hidden_size, dim_in), dtype=dtype))

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        self.length_scale = length_scale

    def forward(self, x, fwd_expert_count):
        y = MOELinear.apply(x, fwd_expert_count, self.batched_fc1_w)
        y = F.relu(y)
        if self.dropout is not None:
            y = self.dropout(y)
        y = MOELinear.apply(y, fwd_expert_count, self.batched_fc2_w)
        return y


class MoEDenseActDense(torch.nn.Module):
    def __init__(self,
                 dim_in : int, 
                 dim_ff : int,
                 dim_out : int,
                 dtype = torch.float, 
                 int8 = False,
                 init_mean = 0.0, 
                 init_std = 0.02,
                 bias = False,
                 activate_fn = "gated_gelu",
                 length_scale : bool = False,
                 dropout_p = 0,
                 num_expert = 8,
                 top_k = 4,
        ):
        super().__init__()
        self.expert_num = num_expert
        self.gate = CustomGate(dim_in, self.expert_num, dtype, init_mean, init_std, length_scale=length_scale)
        self.experts = CustomExpert(dim_in, dim_out, dim_ff, self.expert_num, dtype, init_mean=init_mean, init_std=init_std, dropout_p=dropout_p, length_scale=length_scale)
        self.length_scale = length_scale
        self.k = top_k

    def forward(self, x):
        k = self.k
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size * seq_len, dim)

        gate = self.gate(x)
        _, gate_top_k_idx = torch.topk(
            gate, k=k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]

        fwd = _fmoe_general_global_forward(
            x, gate_top_k_idx, self.experts,
            self.expert_num, 1)
        moe_outp = fwd.view(-1, k, fwd.shape[-1])
        moe_outp = moe_outp.sum(dim=1).view(batch_size, seq_len, -1)

        return moe_outp
