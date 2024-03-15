import argparse
from fvcore.nn import FlopCountAnalysis
from multiprocessing import cpu_count
import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torchmetrics
import transformers
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data.dataloader import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from utils import dataset_loader, utils
from omegaconf import OmegaConf
import time

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load the model rom')
parser.add_argument('--model_name', type=str, default='t5-base', help='model name in huggingface model hub')
parser.add_argument('--dataset', type=str, default='sst2', help='dataset used to for evaluation')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--validation_interval', type=int, default=1000, help='Validation Interval')

class T5Performance(L.LightningModule):
	def __init__(self, t5_variant, output_classes, learning_rate=1e-5, weight_decay=0,
							 lr_scheduler='WarmupPoly', decay_steps=None, model_max_length=512, dataset_name=None):
		super().__init__()
		model = T5ForConditionalGeneration.from_pretrained(t5_variant)
		self.t5_variant = t5_variant
		self.model = model
		self.tokenizer = T5Tokenizer.from_pretrained(t5_variant, model_max_length=model_max_length)

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

		self.validation_flops = []
		self.accuracy = []
		self.num_correct = 0
		self.inference_time = 0
		self.profiler = None

		self.dataset_name = dataset_name

		self.save_hyperparameters()
	
	def forward(self, input_ids, attention_mask=None):
		output = self.model(input_ids=input_ids, attention_mask=attention_mask)
		return output.logits

	def configure_optimizers(self):
		optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
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
		tokenized = self.tokenizer(
			text=batch["sentence"], return_tensors="pt", padding=True)
		input_ids = tokenized.input_ids.to(self.device)
		attention_mask = tokenized.attention_mask.to(self.device)
		dec_input_ids = torch.tensor(len(labels) * [[self.tokenizer.pad_token_id]]).to(self.device)
		
		return input_ids, attention_mask, dec_input_ids, labels

	def training_step(self, batch):
		input_ids, attention_mask, dec_input_ids, labels = self._load_one_batch(batch)
		output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
		loss = output.loss

		self.loggers[0].log_metrics(
			{'loss': loss},
			self.global_step
		)

		return loss

	def validation_step(self, batch):
		input_ids, attention_mask, dec_input_ids, labels = self._load_one_batch(batch)
		
		start_time = time.time()
		with record_function("model_inference"):
			outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec_input_ids)
		end_time = time.time()
		self.inference_time += end_time - start_time
		
		logits = outputs.logits.squeeze(dim=1)[:, self.output_class_tokens]
		preds = torch.argmax(logits, 1)
		self.num_correct += (preds == labels).sum().item()
		
		flops = FlopCountAnalysis(self.model, (input_ids, attention_mask, dec_input_ids)).total()
		self.validation_flops.append((flops, input_ids.size(0)))

		return {}

	def on_validation_start(self):
		self.profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)
		self.profiler.__enter__()
	
	def on_validation_epoch_end(self):
		self.profiler.__exit__(None, None, None)  # Stop the profiler

		total_examples = sum(batch_size for _, batch_size in self.validation_flops)
		accuracy = self.num_correct / total_examples
		total_flops = sum(flops for flops, _ in self.validation_flops)
		avg_flops_per_example = total_flops / total_examples
		self.log('accuracy', accuracy, prog_bar=True, logger=True)
		self.log('inference time', self.inference_time, prog_bar=True, logger=True)
		self.log('avg_flops_per_example', avg_flops_per_example, prog_bar=True, logger=True)

		profiler_summary = self.profiler.key_averages().table(sort_by="cuda_time_total")
		print(profiler_summary)

		# Save
		data_to_save = f"Accuracy: {accuracy}\n\nInference Time: {self.inference_time}\n\nAverage FLOPs per Example: {avg_flops_per_example}\n\nProfiler Summary:\n{profiler_summary}\n"
		file_path = f"analysis/{self.t5_variant}_{self.dataset_name}.txt"
		with open(file_path, "w") as file:
			file.write(data_to_save)
		
		# Reset
		self.validation_flops = []
		self.profiler = None

def main():
	args = parser.parse_args()
	print(f"Parsed arguments: batch_size={args.batch_size}, dataset={args.dataset}, checkpoint={args.checkpoint}")
	# configs = OmegaConf.load(args.config)

	model_name = args.model_name
	model_max_length = (2048 + 1024) // 2 if args.dataset == 'race' else 512
	tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=model_max_length)
	model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()
	if args.checkpoint:
		utils.LoadCheckpoint(model, args.checkpoint)

	# Dataset
	_, valid_set, output_classes = dataset_loader.load_dataset(args.dataset, tokenizer)
	# dataset = load_dataset(args.dataset)
	# valid_set = dataset['validation']
	# output_class_tokens = dataset_loader.get_output_class_tokens(tokenizer, output_classes)

	# Validation Data Loader
	num_data_workers = max(cpu_count()//2, 1)
	valid_loader = DataLoader(
		valid_set, 
		batch_size=args.batch_size, 
		shuffle=False,
    	num_workers=num_data_workers,
		# collate_fn=custom_collate_fn
  	)

	'''
	for batch in valid_loader:
		try:
			inputs, labels = batch['input_ids'], batch['labels']
		except Exception as e:
			print(f"Error processing batch: {e}")
			break
	'''

	# Model
	model = T5Performance(
		t5_variant=model_name, 
		output_classes=output_classes,
		model_max_length=model_max_length,
		dataset_name=args.dataset
	)

	# Trainer
	logger = L.pytorch.loggers.TensorBoardLogger("lightning_logs", name=args.checkpoint, version='performance', sub_dir="performance")
	trainer = L.Trainer(logger=logger, val_check_interval=args.validation_interval, max_epochs=1)
	trainer.validate(model=model, dataloaders=valid_loader)

if __name__ == "__main__":
	main()