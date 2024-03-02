import datetime
import os
import sys
from multiprocessing import cpu_count

import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import tqdm
import typing
import datasets
import transformers
from transformers.models.t5.modeling_t5 import T5LayerFF
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from moe_lora import LoRALinearLayer

T5_VARIANT = 't5-base'
EXP_NAME = "T5WithFFLoRA"
DATASET = 'race'
if DATASET == 'sst2':
  BATCH_SIZE = 64

  NUM_TRAIN_EPOCHS = 1
  VALIDATION_INTERVAL = 200

  MAX_SOURCE_LENGTH = 512

  LEARNING_RATE = 1e-5
elif DATASET == 'race':
  BATCH_SIZE = 4

  NUM_TRAIN_EPOCHS = 3
  VALIDATION_INTERVAL = 3000

  MAX_SOURCE_LENGTH = 1024

  LEARNING_RATE = 1e-4
else:
  raise ValueError(f'Unexpected DATASET {DATASET}')

LORA_WEIGHT_NAMES = ['A', 'B']


class LitT5WithFFLoRA(L.LightningModule):
  def __init__(self, t5_variant, output_classes, learning_rate,
               weight_decay=0, lora_rank=16, lr_scheduler='WarmupPoly',
               decay_steps=None):
    super().__init__()
    model = T5ForConditionalGeneration.from_pretrained(t5_variant)
    replace_linear_with_lora_linear(model, rank=lora_rank, useBias=False)
    self.model = model
    self.tokenizer = T5Tokenizer.from_pretrained(t5_variant, model_max_length=MAX_SOURCE_LENGTH)
    self.output_classes = output_classes
    num_classes = len(self.output_classes)
    self.output_class_tokens = self.tokenizer(output_classes, return_tensors='pt').input_ids
    assert(self.output_class_tokens.shape == (num_classes, 2))
    self.output_class_tokens = nn.Parameter(self.output_class_tokens[:, 0].squeeze(), requires_grad=False)
    self.loss_metric = torchmetrics.aggregation.MeanMetric()
    self.accuracy_metric = torchmetrics.classification.Accuracy(task='multiclass', num_classes=num_classes)

    self.lr_scheduler = lr_scheduler
    self.decay_steps = decay_steps
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay

    self.save_hyperparameters()

  def configure_optimizers(self):
    lora_weights = []
    print('Searching for LoRA weights...')
    for name, param in self.named_parameters():
      if name.split('.')[-1] in LORA_WEIGHT_NAMES:
        print(f' Adding {name} to list of LoRA weights to be optimized')
        lora_weights.append(param)

    optimizer = torch.optim.AdamW(
      lora_weights, lr=self.learning_rate, weight_decay=self.weight_decay)
    if self.lr_scheduler == 'Constant':
      lr_scheduler = transformers.get_constant_schedule(optimizer)
    elif self.lr_scheduler == 'WarmupPoly':
      lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=self.decay_steps,
        lr_end=self.learning_rate*0.1)
    elif self.lr_scheduler == 'WarmupCosineCyclic':
      lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=self.decay_steps)
    else:
      raise NotImplementedError(f'Unimplemented lr scheduler {self.lr_scheduler}')

    lr_scheduler_config = {
      'scheduler': lr_scheduler,
      'interval': 'step'
    }
    return {
      'optimizer': optimizer,
      'lr_scheduler': lr_scheduler_config
    }

  def forward(self, x):
    input_ids = self.tokenizer(
      text=[x['sentence']], return_tensors="pt").input_ids.to(self.device)
    dec_input_ids = torch.tensor([self.tokenizer.pad_token_id]).to(self.device)
    output = self.model(input_ids=input_ids, decoder_input_ids=dec_input_ids)
    return output

  def _load_one_batch(self, batch):
    labels = batch["label"].to(self.device)
    tokenized = self.tokenizer(
      text=batch["sentence"], return_tensors="pt", padding=True)
    input_ids = tokenized.input_ids.to(self.device)
    attention_mask = tokenized.attention_mask.to(self.device)
    dec_input_ids = torch.tensor(len(labels) * [[self.tokenizer.pad_token_id]]).to(self.device)
    return input_ids, attention_mask, dec_input_ids, labels
    
  def training_step(self, batch, batch_idx):
    input_ids, attention_mask, dec_input_ids, labels = self._load_one_batch(batch)

    output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=dec_input_ids)
    logits = output.logits.squeeze(dim=1).index_select(
      dim=1, index=self.output_class_tokens)
    loss = F.cross_entropy(logits, labels)
    
    self.loggers[0].log_metrics(
      {'loss': loss},
      self.global_step
    )

    return loss

  def validation_step(self, batch, batch_idx):
    input_ids, attention_mask, dec_input_ids, labels = self._load_one_batch(batch)

    output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=dec_input_ids)
    logits = output.logits.squeeze(dim=1).index_select(
      dim=1, index=self.output_class_tokens)
    preds = torch.argmax(logits, 1)

    logits_loss = F.cross_entropy(logits, labels)
    accuracy = (preds == labels).float().mean()

    # Setting logger to False 
    self.log('ckpt_metric', accuracy, batch_size=labels.size(0), logger=False)

    self.accuracy_metric(preds=logits, target=labels)
    self.loss_metric.update(logits_loss)

    return logits_loss

  def on_validation_epoch_end(self):
    accuracy = self.accuracy_metric.compute()
    loss = self.loss_metric.compute()
    self.loggers[1].log_metrics(
      {
        'loss': loss,
        'Metrics/accuracy': accuracy,
        'Metrics/logits': loss,
      },
      self.global_step
    )
    self.accuracy_metric.reset()
    self.loss_metric.reset()


def replace_linear_with_lora_linear(model, rank, useBias=False):
  for name, module in model.named_children():
    if isinstance(module, T5LayerFF):
      dense_relu_dense = module.DenseReluDense
      if hasattr(dense_relu_dense, 'wi'):
        original_weight = dense_relu_dense.wi.weight.data
        lora_layer = LoRALinearLayer(
          original_weight, rank, useBias=useBias)
        dense_relu_dense.wi = lora_layer

      if hasattr(dense_relu_dense, 'wo'):
        original_weight = dense_relu_dense.wo.weight.data
        lora_layer = LoRALinearLayer(
          original_weight, rank, useBias=useBias)
        dense_relu_dense.wo = lora_layer
    else:
      # Recursively apply this function to child modules
      replace_linear_with_lora_linear(module, rank, useBias=useBias)


def _process_sst2_sentence(example):
  example['sentence'] = f"sst2 sentence: {example['sentence']}"
  return example

def _process_race_sentence(example):
  abcd_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
  prompt = f'''question: {example['question']}
options: A: {example['options'][0]}, B: {example['options'][1]}, B: {example['options'][2]}, B: {example['options'][3]}
article: {example['article']}'''
  example['sentence'] = prompt
  example['label'] = abcd_mapping[example['answer']]
  return example


def load_dataset(dataset_name):
  if dataset_name == 'sst2':
    dataset = datasets.load_dataset('sst2')
    train, valid = dataset['train'], dataset['validation']

    train = train.map(_process_sst2_sentence)
    valid = valid.map(_process_sst2_sentence)
    output_classes = ('negative', 'positive')
  elif dataset_name == 'race':
    dataset = datasets.load_dataset('race', 'all')
    train, valid = dataset['train'], dataset['validation']

    train = train.map(_process_race_sentence)
    valid = valid.map(_process_race_sentence)
    output_classes = ('A', 'B', 'C', 'D')

  return train, valid, output_classes


def main():
  # Dataset
  train_set, valid_set, output_classes = load_dataset(DATASET)

  num_data_workers = max(cpu_count()//2, 1)
  train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=num_data_workers
  )
  valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_data_workers
  )

  # Model
  model = LitT5WithFFLoRA(t5_variant=T5_VARIANT, output_classes=output_classes,
                          learning_rate=LEARNING_RATE,
                          decay_steps=len(train_set)*NUM_TRAIN_EPOCHS//BATCH_SIZE)

  curr_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

  # saves top-K checkpoints based on validation accuracy
  checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    save_last=True, every_n_train_steps=VALIDATION_INTERVAL,
    save_top_k=3, monitor='ckpt_metric', mode='max',
    dirpath=os.path.join('lightning_logs', EXP_NAME, curr_time, 'checkpoints'),
    filename=EXP_NAME+'-{global_step}-{epoch:02d}-{step}-{ckpt_metric:.3f}'
  )

  # Log to train and valid sub-directory
  train_logger = L.pytorch.loggers.TensorBoardLogger(
    "lightning_logs", name=EXP_NAME, version=curr_time, sub_dir="train")
  valid_logger = L.pytorch.loggers.TensorBoardLogger(
    "lightning_logs", name=EXP_NAME, version=curr_time, sub_dir="valid")
  loggers = [train_logger, valid_logger]

  # Log learning rate
  lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
    logging_interval='step', log_momentum=True
  )

  # Train
  trainer = L.Trainer(default_root_dir=EXP_NAME, num_sanity_val_steps=100,
                      max_epochs=NUM_TRAIN_EPOCHS, val_check_interval=VALIDATION_INTERVAL,
                      logger=loggers, callbacks=[checkpoint_callback, lr_monitor])
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
  main()
