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
BATCH_SIZE = 64

NUM_TRAIN_EPOCHS = 2
VALIDATION_INTERVAL = 100

LEARNING_RATE = 1e-5

POSITIVE_TOKEN = 1465
NEGATIVE_TOKEN = 2841

LORA_WEIGHT_NAMES = ['A', 'B']


class LitT5WithFFLoRA(L.LightningModule):
  def __init__(self, t5_variant, learning_rate, weight_decay=0, lora_rank=16,
               lr_scheduler='WarmupPoly', decay_steps=None):
    super().__init__()
    model = T5ForConditionalGeneration.from_pretrained(t5_variant)
    replace_linear_with_lora_linear(model, rank=lora_rank, useBias=False)
    self.model = model
    self.tokenizer = T5Tokenizer.from_pretrained(t5_variant)
    self.output_class_tokens = nn.Parameter(torch.tensor([NEGATIVE_TOKEN, POSITIVE_TOKEN]), requires_grad=False)
    self.loss_metric = torchmetrics.aggregation.MeanMetric()
    self.accuracy_metric = torchmetrics.classification.Accuracy(task='binary')

    self.lr_scheduler = lr_scheduler
    self.decay_steps = decay_steps
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay

    self.save_hyperparameters()

  def forward(self, x):
    sentences = [f"sst2 sentence: {x}"]
    input_ids = self.tokenizer(
      text=sentences, return_tensors="pt", padding=True).input_ids
    dec_input_ids = self.tokenizer(
      text=["<extra_id_0>"], return_tensors="pt").input_ids[:, :1]
    output = self.model(input_ids=input_ids, labels=dec_input_ids)
    return output

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

  def _load_one_batch(self, batch):
    labels = batch["label"].to(self.device)
    sentences = [
      f"sst2 sentence: {sentence}" for sentence in batch["sentence"]]
    input_ids = self.tokenizer(
      text=sentences, return_tensors="pt", padding=True).input_ids.to(self.device)
    dec_input_ids = self.tokenizer(
      text=["<extra_id_0>"] * len(labels), return_tensors="pt").input_ids[:, :1].to(self.device)
    return input_ids, dec_input_ids, labels
    

  def training_step(self, batch, batch_idx):
    input_ids, dec_input_ids, labels = self._load_one_batch(batch)

    output = self.model(input_ids=input_ids, labels=dec_input_ids)
    logits = output.logits.squeeze(dim=1).index_select(
      dim=1, index=self.output_class_tokens)
    loss = F.cross_entropy(logits, labels)
    
    self.loggers[0].log_metrics(
      {'loss': loss},
      self.global_step
    )

    return loss

  def validation_step(self, batch, batch_idx):
    input_ids, dec_input_ids, labels = self._load_one_batch(batch)

    output = self.model(input_ids=input_ids, labels=dec_input_ids)
    logits = output.logits.squeeze(dim=1).index_select(
      dim=1, index=self.output_class_tokens)
    preds = logits[:, 1] >= logits[:, 0]

    logits_loss = F.cross_entropy(logits, labels)
    accuracy = (preds == labels).float().mean()

    # Setting logger to False 
    self.log('ckpt_metric', accuracy, batch_size=labels.size(0), logger=False)

    self.accuracy_metric(preds=preds, target=labels)
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


def main():
  # Dataset
  sst2: datasets.Dataset = datasets.load_dataset('sst2')
  sst2_train: datasets.Dataset = sst2['train']
  sst2_valid: datasets.Dataset = sst2['validation']

  num_data_workers = max(cpu_count()//2, 1)
  train_loader = torch.utils.data.DataLoader(
    sst2_train, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=num_data_workers
  )
  valid_loader = torch.utils.data.DataLoader(
    sst2_valid, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=num_data_workers
  )

  # Model
  model = LitT5WithFFLoRA(t5_variant=T5_VARIANT, learning_rate=LEARNING_RATE,
                          decay_steps=len(sst2_train)*NUM_TRAIN_EPOCHS//BATCH_SIZE)

  #curr_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
  curr_time = 'debug'

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
  trainer = L.Trainer(default_root_dir=EXP_NAME,
                      max_epochs=NUM_TRAIN_EPOCHS, val_check_interval=VALIDATION_INTERVAL,
                      logger=loggers, callbacks=[checkpoint_callback, lr_monitor])
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == "__main__":
  main()
