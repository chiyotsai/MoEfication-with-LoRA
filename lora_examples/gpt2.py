from transformers import GPT2Tokenizer, GPT2LMHeadModel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from moe_lora import LoRALinearLayer

def replace_conv1d_with_lora(model, rank):
    for name, module in model.named_children():
        if module.__class__.__name__ == 'GPT2MLP'  and type(module.c_fc).__name__ == 'Conv1D':
            if hasattr(module, 'c_fc'):
                original_weight = module.c_fc.weight.data
                lora_layer = LoRALinearLayer(original_weight, rank)
                module.c_fc = lora_layer
            
            if hasattr(module, 'c_proj')  and type(module.c_proj).__name__ == 'Conv1D':
                original_weight = module.c_proj.weight.data
                lora_layer = LoRALinearLayer(original_weight, rank)
                module.c_proj = lora_layer
        else:
            # Recursively apply this function to child modules
            replace_conv1d_with_lora(module, rank)


def print_mlp_layer(model, found=False):
    if found:
        return True
    for name, module in model.named_children():
        if module.__class__.__name__ == 'GPT2MLP':
            print(module)
            return True  
        else:
            found = print_mlp_layer(module, found)
            if found:
                return True 
    return False 

def test_output(model, tokenizer, text):
    model.eval() 
    encoded_input = tokenizer(
        text, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_attention_mask=True
    )
    generated_text_ids = model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        max_length=100
    )
    generated_text = tokenizer.decode(generated_text_ids[0], skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    replace_conv1d_with_lora(model, rank=16)
    test_output(model, tokenizer, "When was Abraham Lincoln born?")

    # Clean up
    if hasattr(model, 'original_weight_backup'):
        del model.original_weight_backup  # Delete the backup weights
    del model  # Delete the model instance