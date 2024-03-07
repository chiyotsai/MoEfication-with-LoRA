import argparse
import functools
import os
import torch
import tqdm

from transformers import T5ForConditionalGeneration

import moefication
from utils import utils

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='t5-base', help='model name in huggingface model hub')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load the model rom')
parser.add_argument('--res_path', type=str, default='results/t5-base/base', help='path to store the results of moefication')
parser.add_argument('--num-expert', type=int, default=96, help='number of experts')
parser.add_argument('--templates', type=str, default='encoder.block.{}.layer.1.DenseReluDense.wi.weight,decoder.block.{}.layer.2.DenseReluDense.wi.weight', help='weight names of the first linear layer in each FFN (use comma to separate multiple templates)')


def main():
    args = parser.parse_args()
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
    
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    if args.checkpoint:
        utils.LoadCheckpoint(model, args.checkpoint)
        
    torch.save(model.state_dict(), os.path.join(args.res_path, 'model.pt'))

    config = moefication.utils.ModelConfig(os.path.join(args.res_path, 'model.pt'), args.res_path, split_num=args.num_expert)

    templates = args.templates.split(',')
    for template in templates:
        layer_pattern = template.format(r'(\d+)')
        GetLayerNumber = functools.partial(utils.GetLayerNumberWithPattern, pattern=layer_pattern)
        num_layers = max(map(GetLayerNumber, model.state_dict().keys())) + 1

        for i in tqdm.tqdm(range(num_layers)):
            split = moefication.utils.ParamSplit(config, template, i)
            split.split()
            split.save()


if __name__ == "__main__":
    main()
