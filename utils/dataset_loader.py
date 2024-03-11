"""Some utility functions to help with dataset loading"""

import datasets
import functools


def _tokenize_example(example, tokenizer):
  example['tokenized_input'] = tokenizer(
    text=example["sentence"], return_tensors="pt", padding=True)
  example['input_ids'] = example['tokenized_input'].input_ids
  example['attention_mask'] = example['tokenized_input'].attention_mask


def _process_sst2_sentence(example, output_classes, tokenizer=None):
  example['sentence'] = f"sst2 sentence: {example['sentence']}"
  example['label_str'] = output_classes[example['label']]

  if tokenizer:
    _tokenize_example(example, tokenizer)

  return example


def _process_mnli_sentence(example, output_classes, tokenizer=None):
  example['sentence'] = f"mnli hypothesis: {example['hypothesis']} premise: {example['premise']}"
  example['label_str'] = output_classes[example['label']]

  if tokenizer:
    _tokenize_example(example, tokenizer)

  return example


def _process_race_sentence(example, output_classes, tokenizer=None):
  str_to_idx_mapping = {s:idx for idx, s in enumerate(output_classes)}
  prompt = f'''question: {example['question']}
options: A: {example['options'][0]}, B: {example['options'][1]}, B: {example['options'][2]}, B: {example['options'][3]}
article: {example['article']}'''
  example['sentence'] = prompt
  example['label'] = str_to_idx_mapping[example['answer']]
  example['label_str'] = example['answer']

  if tokenizer:
    _tokenize_example(example, tokenizer)

  return example


def load_dataset(dataset_name, tokenizer=None):
  if dataset_name == 'sst2':
    dataset = datasets.load_dataset('sst2')
    if tokenizer:
      process_fn = functools.partial(_process_sst2_sentence, tokenizer=tokenizer)
    else:
      process_fn = _process_sst2_sentence

    train, valid = dataset['train'], dataset['validation']
    output_classes = ('negative', 'positive')
  elif dataset_name in ('multi_nli', 'mnli'):
    dataset = datasets.load_dataset('multi_nli')
    if tokenizer:
      process_fn = functools.partial(_process_mnli_sentence, tokenizer=tokenizer)
    else:
      process_fn = _process_mnli_sentence

    train, valid = dataset['train'], dataset['validation_matched']
    output_classes = ('entailment', 'neutral', 'contradiction')
  elif dataset_name == 'race':
    dataset = datasets.load_dataset('race', 'all')
    if tokenizer:
      process_fn = functools.partial(_process_race_sentence, tokenizer=tokenizer)
    else:
      process_fn = _process_race_sentence

    train, valid = dataset['train'], dataset['validation']
    output_classes = ('A', 'B', 'C', 'D')

  process_fn = functools.partial(process_fn, output_classes=output_classes)
  train = train.map(process_fn)
  valid = valid.map(process_fn)

  return train, valid, output_classes


def get_output_class_tokens(tokenizer, output_classes):
    output_class_tokens = [
      tokenizer(output_class, return_tensors='pt', padding=True).input_ids
      for output_class in output_classes
    ]
    return output_class_tokens