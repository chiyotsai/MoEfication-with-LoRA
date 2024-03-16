import os
import types

import numpy as np
import torch
from transformers.models.t5.modeling_t5 import T5LayerFF, T5DenseActDense

from moefication import efficient_moe
from moe_lora import LoRALinearLayer, LoRADenseActDenseLayer
from utils.utils import LoadCheckpoint


class ModifierBase():
  """Base class for modifying the t5 model"""
  def __init__(self, ckpt_path=None, **kwargs):
    self.ckpt_path = ckpt_path

  
  def __call__(self, model):
    if self.ckpt_path:
      print(f'Loading Checkpoint {self.ckpt_path}')
      LoadCheckpoint(model, self.ckpt_path)


class MoEWithGTModifier(ModifierBase):
  "MoE-fy the t5 model."
  def __init__(self, moe_params_path, k, **kwargs):
    super().__init__(**kwargs)
    self.moe_params_path = moe_params_path
    self.k = k

  def __call__(self, model):
    super().__call__(model)

    def _forward(ffn_self, hidden_states):
        hidden_states = ffn_self.wi(hidden_states)
        hidden_states = ffn_self.act(hidden_states)

        if ffn_self.patterns is not None:
            # golden
            k = ffn_self.k
            bsz, seq_len, hidden_size = hidden_states.shape
            hidden_states_relu = hidden_states.clone()
            hidden_states_relu = hidden_states_relu.view(-1, hidden_size)
            score = torch.matmul(hidden_states_relu, ffn_self.patterns.transpose(0, 1))
            labels = torch.topk(score, k=k, dim=-1)[1].view(bsz, seq_len, k)
            cur_mask = torch.nn.functional.embedding(labels, ffn_self.patterns).sum(-2)
            hidden_states[cur_mask == False] = 0  
             
        hidden_states = ffn_self.dropout(hidden_states)
        hidden_states = ffn_self.wo(hidden_states)
        return hidden_states

    def modify_ffn(ffn, path):
        assert type(ffn) == T5DenseActDense
        labels = torch.load(path)
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(np.array(patterns)).cuda()
        ffn.k = self.k
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)   

    # encoder
    for layer_idx, layer in enumerate(model.encoder.block):
        ffn = layer.layer[1].DenseReluDense
        path = os.path.join(self.moe_params_path, 'encoder.block.{}.layer.1.DenseReluDense.wi.weight'.format(layer_idx))
        modify_ffn(ffn, path)

    #decoder
    for layer_idx, layer in enumerate(model.decoder.block):
        ffn = layer.layer[2].DenseReluDense
        path = os.path.join(self.moe_params_path, 'decoder.block.{}.layer.2.DenseReluDense.wi.weight'.format(layer_idx))
        modify_ffn(ffn, path)



class MoEModifier(ModifierBase):
  "MoE-fy the t5 model."
  def __init__(self, moe_params_path, k, **kwargs):
    super().__init__(**kwargs)
    self.moe_params_path = moe_params_path
    self.k = k

  def __call__(self, model):
    super().__call__(model)

    def _forward(ffn_self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states_mlp = hidden_states
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size)

        hidden_states_mlp = hidden_states_mlp / torch.norm(hidden_states_mlp, dim=-1).unsqueeze(-1)
        score = ffn_self.mlp(hidden_states_mlp)

        labels = torch.topk(score, k=self.k, dim=-1)[1].view(bsz, seq_len, self.k)
        cur_mask = torch.nn.functional.embedding(labels, ffn_self.patterns).sum(-2)

        hidden_states = ffn_self.wi(hidden_states)
        hidden_states = ffn_self.act(hidden_states)

        cur_mask = cur_mask.float()
        hidden_states = hidden_states * cur_mask

        hidden_states = ffn_self.dropout(hidden_states)
        hidden_states = ffn_self.wo(hidden_states)

        return hidden_states

    def modify_ffn(ffn, path):
        assert type(ffn) == T5DenseActDense
        labels = torch.load(path)
        cluster_num = max(labels)+1

        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(np.array(patterns)).cuda()
        ffn.k = self.k
        ffn.mlp = torch.load(path+'_input_compl').cuda()
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)

    # encoder
    for layer_idx, layer in enumerate(model.encoder.block):
        ffn = layer.layer[1].DenseReluDense
        layer.layer[1].DenseReluDense = efficient_moe.MoEDenseActDense(
          dim_in=ffn.wi.weight.shape[1], dim_ff=ffn.wi.weight.shape[0],
          dim_out=ffn.wo.weight.shape[0], dropout_p=0, num_expert=96, top_k=20)
        # path = os.path.join(self.moe_params_path, 'encoder.block.{}.layer.1.DenseReluDense.wi.weight'.format(layer_idx))
        # modify_ffn(ffn, path)

    #decoder
    for layer_idx, layer in enumerate(model.decoder.block):
        ffn = layer.layer[2].DenseReluDense
        ffn = layer.layer[1].DenseReluDense
        layer.layer[1].DenseReluDense = efficient_moe.MoEDenseActDense(
          dim_in=ffn.wi.weight.shape[1], dim_ff=ffn.wi.weight.shape[0],
          dim_out=ffn.wo.weight.shape[0], dropout_p=0, num_expert=96, top_k=20)
        #path = os.path.join(self.moe_params_path, 'decoder.block.{}.layer.2.DenseReluDense.wi.weight'.format(layer_idx))
        #modify_ffn(ffn, path)


def dfs_dense_relu_dense(model, leaf_callback, **kwargs):
  """Traverse through the model tree and call leaf_callback on DenseReluDense."""
  for _, module in model.named_children():
    if isinstance(module, T5LayerFF):
      dense_relu_dense = module.DenseReluDense
      module.DenseReluDense = leaf_callback(dense_relu_dense, **kwargs)
    else:
      # Recursively apply this function to child modules
      dfs_dense_relu_dense(module, leaf_callback, **kwargs)


class SFTBase():
  def __init__(self, modifier=None, **kwargs):
    self.modifier = modifier

  def __call__(self, model):
    if self.modifier:
      self.modifier(model)

    opt_params_patterns = [r'.*']
    return opt_params_patterns


class AddLoRA(SFTBase):
  def __init__(self, lora_rank, **kwargs):
    super().__init__(**kwargs)
    self.lora_rank = lora_rank

  def __call__(self, model):
    super().__call__(model)

    def _replace_linear_with_lora_linear(ffn, lora_rank, useBias=False):
        if hasattr(ffn, 'wi'):
            original_weight = ffn.wi.weight.data
            lora_layer = LoRALinearLayer(original_weight, lora_rank, useBias=useBias).cuda()
            ffn.wi = lora_layer

        if hasattr(ffn, 'wo'):
            original_weight = ffn.wo.weight.data
            lora_layer = LoRALinearLayer(original_weight, lora_rank, useBias=useBias).cuda()
            ffn.wo = lora_layer
        return ffn

    dfs_dense_relu_dense(model, _replace_linear_with_lora_linear, lora_rank=self.lora_rank, useBias=False)
    opt_params_patterns = [r'.*\.A', r'.*\.B']
    return opt_params_patterns


class AddDenseActDenseLoRA(SFTBase):
  def __init__(self, lora_rank, **kwargs):
    super().__init__(**kwargs)
    self.lora_rank = lora_rank

  def __call__(self, model):
    super().__call__(model)

    def _replace_with_lora(dense_act_dense, lora_rank):
      assert isinstance(dense_act_dense, T5DenseActDense)
      return LoRADenseActDenseLayer(dense_act_dense, lora_rank)

    dfs_dense_relu_dense(model, _replace_with_lora, lora_rank=self.lora_rank)

    opt_params_patterns = [r'.*\.A', r'.*\.B']
    return opt_params_patterns


def get_model_modifier(base_model_type,
                       tuning_type, **kwargs):
  # Convert the model to a MoE if needed
  if base_model_type == 'unmodified':
    modifier = ModifierBase(**kwargs)
  elif base_model_type == 'moe':
    modifier = MoEModifier(**kwargs)

  # Do either LoRA or Full Finetuning
  if tuning_type in ('full', 'fft'):
    return SFTBase(modifier=modifier, **kwargs)
  elif tuning_type == 'lora':
    return AddLoRA(modifier=modifier, **kwargs)
  elif tuning_type == 'dense_act_dense_lora':
    return AddDenseActDenseLoRA(modifier=modifier, **kwargs)
