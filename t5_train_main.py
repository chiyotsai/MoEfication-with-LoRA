import argparse
import datetime
import os
import re

from multiprocessing import cpu_count

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
import tqdm
import utils

import model_modifier
from omegaconf import OmegaConf
from transformers import T5Tokenizer, T5ForConditionalGeneration


parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, default='configs/moe_t5_base_race_dad_lora_train.yaml', help='path to the config file for training')


class LitT5(L.LightningModule):
  def __init__(self, t5_variant, output_classes, learning_rate, model_modifier_config,
               weight_decay=0, lr_scheduler='Constant', max_source_length=512):
    super().__init__()
    model = T5ForConditionalGeneration.from_pretrained(t5_variant)
    # modifier will do the modifications for our experiments. e.g. moe-fy the model, adding lora, etc.
    modifier = model_modifier.get_model_modifier(**model_modifier_config)
    self.opt_params_patterns = modifier(model)
    self.model = model
    self.tokenizer = T5Tokenizer.from_pretrained(t5_variant, model_max_length=max_source_length)

    self.output_classes = output_classes
    self.output_class_tokens = utils.dataset_loader.get_output_class_tokens(self.tokenizer, output_classes)
    self.output_class_tokens = [class_tokens.to('cuda') for class_tokens in self.output_class_tokens]

    self.loss_metric = torchmetrics.aggregation.MeanMetric()
    self.accuracy_metric = torchmetrics.classification.Accuracy(task='multiclass', num_classes=len(self.output_classes))

    self.lr_scheduler = lr_scheduler
    self.learning_rate = learning_rate
    self.weight_decay = weight_decay

    self.save_hyperparameters()

  def configure_optimizers(self):
    lora_weights = []
    for name, param in self.named_parameters():
      for pattern in self.opt_params_patterns:
        if re.match(pattern, name):
          print(f' Adding {name} to list of weights to be optimized')
          lora_weights.append(param)

    optimizer = torch.optim.AdamW(
      lora_weights, lr=self.learning_rate, weight_decay=self.weight_decay)
    if self.lr_scheduler == 'Constant':
      lr_scheduler = transformers.get_constant_schedule(optimizer)
    elif self.lr_scheduler == 'ReduceLROnPlateau':
      lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min')
    else:
      raise NotImplementedError(f'Unimplemented lr scheduler {self.lr_scheduler}')

    lr_scheduler_config = {
      'scheduler': lr_scheduler,
      'interval': 'step',
      'monitor': 'Metrics/logits',
      'strict': False,
    }
    return {
      'optimizer': optimizer,
      'lr_scheduler': lr_scheduler_config
    }

  def forward(self, input_ids):
    dec_input_ids = torch.tensor([self.tokenizer.pad_token_id]).to(self.device)
    output = self.model(input_ids=input_ids, decoder_input_ids=dec_input_ids)
    return output

  def _load_one_batch(self, batch):
    labels = batch["label"].to(self.device)
    tokenized = self.tokenizer(
      text=batch["sentence"], return_tensors="pt", padding=True)
    input_ids = tokenized.input_ids.to(self.device)
    attention_mask = tokenized.attention_mask.to(self.device)
    # dec_input_ids = torch.tensor(len(labels) * [[self.tokenizer.pad_token_id]]).to(self.device)
    dec_input_ids = self.tokenizer(
      text=batch["label_str"], return_tensors="pt", padding=True).input_ids.to(self.device)
    return input_ids, attention_mask, dec_input_ids, labels

  def training_step(self, batch, batch_idx):
    input_ids, attention_mask, dec_input_ids, labels = self._load_one_batch(batch)

    output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=dec_input_ids)
    # logits = output.logits.squeeze(dim=1)[:, self.output_class_tokens]
    # loss = F.cross_entropy(logits, labels)
    loss = output.loss

    self.loggers[0].log_metrics(
      {'loss': loss},
      self.global_step
    )

    return loss

  def validation_step(self, batch, batch_idx):
    self.model.eval()
    input_ids, attention_mask, dec_input_ids, labels = self._load_one_batch(batch)

    # output = self.model(input_ids=input_ids, attention_mask=attention_mask,
    #                     labels=dec_input_ids)
    # logits = output.logits.squeeze(dim=1)[:, self.output_class_tokens]
    # preds = torch.argmax(logits, 1)

    # logits_loss = F.cross_entropy(logits, labels)
    # accuracy = (preds == labels).float().mean()

    output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=dec_input_ids)
    encoder_last_hidden_state = output.encoder_last_hidden_state
    loss = output.loss

    class_logits = []
    for class_token in self.output_class_tokens:
      class_token = class_token.repeat((input_ids.size(0), 1))
      output = self.model(encoder_outputs=(encoder_last_hidden_state,),
                          attention_mask=attention_mask,
                          labels=class_token)
      logits = output.logits
      # topk = logits.topk(k=40)[0][:, :, -1][:, :, None]
      # topk_mask = logits < topk
      # logits.data.masked_fill(topk_mask.bool(), -float('inf'))
      
      probs = torch.softmax(logits, dim=2)
      logits = torch.log(probs)
      logits = torch.gather(logits, dim=2, index=class_token[:, :, None]).squeeze(dim=2)
      class_logits.append(logits.sum(dim=1, keepdim=True))

    class_logits = torch.hstack(class_logits)
    preds = torch.argmax(class_logits, 1)
    accuracy = (preds == labels).float().mean()

    # Setting logger to False
    self.log('ckpt_metric', accuracy, batch_size=labels.size(0), logger=True)

    self.accuracy_metric(preds=class_logits, target=labels)
    self.loss_metric.update(loss)

    return loss

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

    
def SaveConfig(config, save_dir):
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
    OmegaConf.save(config=config, f=f)


def main():
  torch.set_float32_matmul_precision('medium')

  args = parser.parse_args()
  configs = OmegaConf.load(args.config)

  # Dataset
  train_set, valid_set, output_classes = \
    utils.dataset_loader.load_dataset(
      configs.data.dataset,
      # tokenizer=T5Tokenizer.from_pretrained(configs.model.t5_variant,
      #                                       model_max_length=configs.data.max_source_length)
    )

  # Model
  model = LitT5(t5_variant=configs.model.t5_variant, output_classes=output_classes,
                learning_rate=configs.model.learning_rate,
                lr_scheduler=configs.model.lr_scheduler,
                model_modifier_config=configs.model.model_modifier_config,
                max_source_length=configs.data.max_source_length)

  num_data_workers = max(cpu_count()//2, 1)
  train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=configs.data.batch_size, shuffle=True,
    num_workers=num_data_workers
  )
  valid_loader = torch.utils.data.DataLoader(
    valid_set, batch_size=configs.data.batch_size, shuffle=False,
    num_workers=num_data_workers
  )

  curr_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
  dir_name = os.path.join('lightning_logs', configs.exp_name, curr_time)
  SaveConfig(configs, dir_name)

  # saves top-K checkpoints based on validation accuracy
  checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
    save_last=True, every_n_train_steps=configs.trainer.val_check_interval,
    save_top_k=3, monitor='ckpt_metric', mode='max',
    dirpath=os.path.join('lightning_logs', configs.exp_name, curr_time, 'checkpoints'),
    filename=configs.exp_name+'-{global_step}-{epoch:02d}-{step}-{ckpt_metric:.3f}'
  )

  # Log to train and valid sub-directory
  train_logger = L.pytorch.loggers.TensorBoardLogger(
    "lightning_logs", name=configs.exp_name, version=curr_time, sub_dir="train")
  valid_logger = L.pytorch.loggers.TensorBoardLogger(
    "lightning_logs", name=configs.exp_name, version=curr_time, sub_dir="valid")
  loggers = [train_logger, valid_logger]

  # Log learning rate
  lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
    logging_interval='step', log_momentum=True
  )

  # Train
  trainer = L.Trainer(default_root_dir=configs.exp_name, num_sanity_val_steps=-1,
                      max_epochs=configs.trainer.max_epochs,
                      val_check_interval=configs.trainer.val_check_interval,
                      logger=loggers, callbacks=[checkpoint_callback, lr_monitor])
  trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


  print(checkpoint_callback.best_model_path)
    

if __name__ == "__main__":
  main()
