import tqdm
import torch
import typing
import datasets
import transformers
from transformers.models.t5.modeling_t5 import T5LayerFF
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from moe_lora import LoRALinearLayer
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'

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

BATCH_SIZE = 64

POSITIVE_TOKEN = 1465
NEGATIVE_TOKEN = 2841

LORA_WEIGHT_NAMES = ['A', 'B']

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0

EVAL_STEPS= 100


def evaluate(model: transformers.PreTrainedModel, eval_set: datasets.Dataset, tokenizer: transformers.PreTrainedTokenizer):
  num_eval_steps = len(eval_set) // BATCH_SIZE
  eval_iter = eval_set.iter(batch_size=BATCH_SIZE, drop_last_batch=True)
  total_logits_loss = 0
  acc = 0
  with torch.no_grad():
    for batch in eval_iter:
      labels = torch.tensor(batch["label"]).to(device)
      sentences = [f"sst2 sentence: {sentence}" for sentence in batch["sentence"]]
      input_ids = tokenizer(text=sentences,
                            return_tensors="pt", padding=True).input_ids.to(device)
      dec_input_ids = tokenizer(text=["<extra_id_0>"] * BATCH_SIZE, return_tensors="pt").input_ids.to(device)[:, :1]

      output = model(input_ids=input_ids, labels=dec_input_ids)
      logits = output.logits.squeeze(dim=1).index_select(dim=1, index=torch.tensor([POSITIVE_TOKEN, NEGATIVE_TOKEN]).to(device))
      preds = logits[:,1] >= logits[:,0]

      logits_loss = F.cross_entropy(logits, labels)
      total_logits_loss += logits_loss
      acc += (preds != labels).double().mean()
    acc /= num_eval_steps
    logits_loss /= num_eval_steps

  return {'logits_loss' : logits_loss.item(), 'accuracy': acc.item()}


  
def main():
  sst2: datasets.Dataset = datasets.load_dataset('sst2')
  sst2_train : datasets.Dataset = sst2['train']
  sst2_eval : datasets.Dataset = sst2['validation']
  tokenizer = T5Tokenizer.from_pretrained('t5-base')

  model = T5ForConditionalGeneration.from_pretrained('t5-base')

  replace_linear_with_lora_linear(model, rank=16, useBias=False)
  model = model.to(device)

  lora_weights = []
  print('Searching for LoRA weights...')
  for name, param in model.named_parameters():
    if name.split('.')[-1] in LORA_WEIGHT_NAMES:
      print(f'  Adding {name} to list of LoRA weights to be optimized')
      lora_weights.append(param)

  metrics = evaluate(model, sst2_eval, tokenizer)
  print('Inital Model:')
  print(f"  Accuracy: {metrics['accuracy']}, logits_loss: {metrics['logits_loss']}")

  num_training_steps = len(sst2_train) // BATCH_SIZE

  optimizer = torch.optim.AdamW(lora_weights, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
  lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)
  
  train_iter = sst2_train.shuffle().iter(batch_size=BATCH_SIZE, drop_last_batch=True)
  progress_bar = tqdm.tqdm(train_iter, total=num_training_steps)

  for train_step, batch in enumerate(progress_bar):
    labels = torch.tensor(batch["label"]).to(device)
    sentences = [f"sst2 sentence: {sentence}" for sentence in batch["sentence"]]
    input_ids = tokenizer(text=sentences,
                          return_tensors="pt", padding=True).input_ids.to(device)
    dec_input_ids = tokenizer(text=["<extra_id_0>"] * BATCH_SIZE, return_tensors="pt").input_ids.to(device)[:, :1]

    output = model(input_ids=input_ids, labels=dec_input_ids)
    logits = output.logits.squeeze(dim=1).index_select(dim=1, index=torch.tensor([NEGATIVE_TOKEN, POSITIVE_TOKEN]).to(device))
    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    if train_step != 0 and train_step % EVAL_STEPS == 0:
      print("Evalutating Model:")
      metrics = evaluate(model, sst2_eval, tokenizer)
      print(f"  Accuracy: {metrics['accuracy']}, logits_loss: {metrics['logits_loss']}")

  if train_step % EVAL_STEPS != 0:
    print("Evalutating Model:")
    metrics = evaluate(model, sst2_eval, tokenizer)
    print(f"  Accuracy: {metrics['accuracy']}, logits_loss: {metrics['logits_loss']}")
    


if __name__ == "__main__":
  main()