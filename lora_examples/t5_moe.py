import torch
from datasets import load_dataset
from transformers import T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5LayerFF
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from moe_lora import LoRALinearLayer

def replace_linear_with_lora_linear(model, rank, useBias=False):
    for name, module in model.named_children():
        if isinstance(module, T5LayerFF):
            dense_relu_dense = module.DenseReluDense
            if hasattr(dense_relu_dense, 'wi'):
                original_weight = dense_relu_dense.wi.weight.data
                lora_layer = LoRALinearLayer(original_weight, rank, useBias=useBias)
                dense_relu_dense.wi = lora_layer
            
            if hasattr(dense_relu_dense, 'wo'):
                original_weight = dense_relu_dense.wo.weight.data
                lora_layer = LoRALinearLayer(original_weight, rank, useBias=useBias)
                dense_relu_dense.wo = lora_layer
        else:
            # Recursively apply this function to child modules
            replace_linear_with_lora_linear(module, rank, useBias=useBias)

def print_modules(model):
    for name, module in model.named_children():
        print(module)

def print_FF_layer(model, found=False):
    if found:
        return True
    for name, module in model.named_children():
        if isinstance(module, T5LayerFF):
            print(module)
            return True  
        else:
            found = print_FF_layer(module, found)
            if found:
                return True 
    return False 

if __name__ == "__main__":
    model = T5ForConditionalGeneration.from_pretrained('t5-base')
    print('Before replacing FF linear layers with LoRALinearLayer:')
    print_modules(model)
    print('')

    #replace_linear_with_lora_linear(model, rank=16, useBias=False)
    #print('After replacing FF linear layers with LoRALinearLayer:')
    #print_FF_layer(model)