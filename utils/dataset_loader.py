"""Some utility functions to help with dataset loading"""

import datasets
import functools


def _tokenize_example(example, tokenizer):
  example['tokenized_input'] = tokenizer(
    text=example["sentence"], return_tensors="pt", padding=True)
  example['input_ids'] = example['tokenized_input'].input_ids
  example['attention_mask'] = example['tokenized_input'].attention_mask


def _process_sst2_sentence(example, tokenizer=None):
  example['sentence'] = f"sst2 sentence: {example['sentence']}"

  if tokenizer:
    _tokenize_example(example, tokenizer)

  return example


def _process_race_sentence(example, tokenizer=None):
  abcd_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
  prompt = f'''question: {example['question']}
options: A: {example['options'][0]}, B: {example['options'][1]}, B: {example['options'][2]}, B: {example['options'][3]}
article: {example['article']}'''
  example['sentence'] = prompt
  example['label'] = abcd_mapping[example['answer']]

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

    output_classes = ('negative', 'positive')
  elif dataset_name == 'race':
    dataset = datasets.load_dataset('race', 'all')
    if tokenizer:
      process_fn = functools.partial(_process_race_sentence, tokenizer=tokenizer)
    else:
      process_fn = _process_race_sentence

    output_classes = ('A', 'B', 'C', 'D')

  train, valid = dataset['train'], dataset['validation']
  train = train.map(process_fn)
  valid = valid.map(process_fn)

  return train, valid, output_classes


def get_output_class_tokens(tokenizer, output_classes):
    num_classes = len(output_classes)
    output_class_tokens = tokenizer(output_classes, return_tensors='pt').input_ids
    assert(output_class_tokens.shape == (num_classes, 2))
    output_class_tokens = output_class_tokens[:, 0].tolist()
    return output_class_tokens