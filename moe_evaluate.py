import argparse
import os
import torch
import tqdm
import types
import numpy as np

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5DenseActDense

import model_modifier
from utils import dataset_loader
from utils import utils

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='t5-base', help='model name in huggingface model hub')
parser.add_argument('--dataset', type=str, default='sst2', help='dataset used to for evaluation')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load the model rom')
parser.add_argument('--res_path', type=str, default='results/t5-base/base', help='path to store the results of moefication')
parser.add_argument('--k', type=int, default=20, help='Number of experts to activate')
parser.add_argument('--eval_type', type=str, choices=['base', 'moe_gt', 'moe_mlp'], help='Number of experts to activate')


def main():
    args = parser.parse_args()

    model_name = args.model_name
    if args.dataset == 'race':
        model_max_length = (2048 + 1024) // 2
    elif args.dataset in ['mnli', 'multi_nli']:
        model_max_length = 1024
    else:
        model_max_length = 512
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=model_max_length)

    model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()

    if args.eval_type == 'base':
        modifier = model_modifier.ModifierBase(
            ckpt_path=args.checkpoint)
    elif args.eval_type == 'moe_gt':
        params_path = os.path.join(args.res_path, 'param_split')
        modifier = model_modifier.MoEWithGTModifier(
            ckpt_path=args.checkpoint, k=args.k,
            moe_params_path=params_path)
    elif args.eval_type == 'moe_mlp':
        params_path = os.path.join(args.res_path, 'param_split')
        modifier = model_modifier.MoEModifier(
            ckpt_path=args.checkpoint, k=args.k,
            moe_params_path=params_path)
    else:
        raise NotImplementedError(f'Unexpected model type: {args.eval_type}')

    modifier(model)

    _, valid, output_classes = dataset_loader.load_dataset(args.dataset)
    output_class_tokens = dataset_loader.get_output_class_tokens(tokenizer, output_classes)
    output_class_tokens = [class_token.cuda() for class_token in output_class_tokens]

    num_correct = 0
    pbar = tqdm.tqdm(valid)
    model.eval()
    for step, instance in enumerate(pbar):
        input_ids = tokenizer(text=instance["sentence"], return_tensors="pt").input_ids.cuda()
        encoder_last_hidden_state = None
        class_logits = []
        for class_token in output_class_tokens:
            if encoder_last_hidden_state is None:
                output = model(input_ids=input_ids, labels=class_token)
                encoder_last_hidden_state = output.encoder_last_hidden_state
            else:
                output = model(encoder_outputs=(encoder_last_hidden_state,),
                               labels=class_token)
            
            probs = torch.softmax(output.logits, dim=2)
            logits = torch.log(probs)
            logits = torch.gather(logits, dim=2, index=class_token[:, :, None]).squeeze(dim=2)
            class_logits.append(logits.sum(dim=1, keepdim=True))

        class_logits = torch.hstack(class_logits)
        pred = torch.argmax(class_logits, 1)

        num_correct += (pred == instance['label']).item()

        pbar.set_description(f'Acc: {num_correct/(step + 1):.3f}')

    print(f'Acc: {num_correct/(step + 1):.3f}')


if __name__ == "__main__":
    main()
