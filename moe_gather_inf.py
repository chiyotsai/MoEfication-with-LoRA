import os
import types
import tqdm
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import numpy as np
from transformers.models.t5.modeling_t5 import T5DenseActDense

import argparse
import os
import torch
import tqdm
import types
import numpy as np

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5DenseActDense

from utils import dataset_loader
from utils import utils

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='t5-base', help='model name in huggingface model hub')
parser.add_argument('--dataset', type=str, default='sst2', help='dataset used to for evaluation')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load the model rom')
parser.add_argument('--res_path', type=str, default='results/t5-base/base', help='path to store the results of moefication')
parser.add_argument('--k', type=int, default=20, help='Number of experts to activate')
parser.add_argument('--num_info_samples', type=int, default=-1, help='Number of samples used to train the router. Set to 0 to use all training data.')
parser.add_argument('--shard_file_samples', type=int, default=512, help='Number of samples in each shard.')


def change_forward(model):
    """Modify the forward pass to track the model's hidden states."""

    def _forward(ffn_self, hidden_states):
        ffn_self.res.append(hidden_states.detach().cpu())

        hidden_states = ffn_self.wi(hidden_states)
        hidden_states = ffn_self.act(hidden_states)
             
        hidden_states = ffn_self.dropout(hidden_states)
        hidden_states = ffn_self.wo(hidden_states)
        return hidden_states

    def modify_ffn(ffn, res):
        assert type(ffn) == T5DenseActDense
        ffn.res = res
        ffn.forward = types.MethodType(_forward, ffn)   

    # encoder
    res = {}
    for layer_idx, layer in enumerate(model.encoder.block):
        ffn = layer.layer[1].DenseReluDense
        name = 'encoder.block.{}.layer.1.DenseReluDense.wi.weight'.format(layer_idx)
        res[name] = []
        modify_ffn(ffn, res[name]) 

    #decoder
    for layer_idx, layer in enumerate(model.decoder.block):
        ffn = layer.layer[2].DenseReluDense
        name = 'decoder.block.{}.layer.2.DenseReluDense.wi.weight'.format(layer_idx)
        res[name] = []
        modify_ffn(ffn, res[name])   
    
    return res


write_idx = 0


def write_and_clear_hidden_states(results, res_path):
    """Write the hidden states out and clear the dictionary.
    
    This is needed to avoid OOM errors.
    """
    global write_idx

    for k, v in results.items():
        v = [x.reshape(-1, x.shape[-1]) for x in v]
        if len(v) > 0:
            v = torch.cat(v, dim=0)
            save_dir = os.path.join(res_path, k + f'-shard-{write_idx}')
            torch.save(v, save_dir)

    for v in results.values():
        v.clear()
    write_idx += 1


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
    if args.checkpoint:
        utils.LoadCheckpoint(model, args.checkpoint)

    train, _, output_classes = dataset_loader.load_dataset(args.dataset, tokenizer=tokenizer)
    output_class_tokens = dataset_loader.get_output_class_tokens(tokenizer, output_classes)
    output_class_tokens = [class_token.cuda() for class_token in output_class_tokens]

    results = change_forward(model)

    if not os.path.isdir(args.res_path):
        os.makedirs(args.res_path)

    num_correct = 0
    pbar = tqdm.tqdm(train)
    for step, instance in enumerate(pbar):
        if args.num_info_samples != -1 and step > args.num_info_samples:
            break

        input_ids = torch.tensor(instance['input_ids']).cuda()
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

        if step != 0 and step % args.shard_file_samples == 0:
            write_and_clear_hidden_states(results, args.res_path)

        num_correct += (pred == instance['label']).item()
        pbar.set_description(f'Acc: {num_correct/(step + 1):.3f}')

    print(f'Acc: {num_correct/(step + 1):.3f}')
    write_and_clear_hidden_states(results, args.res_path)


if __name__ == "__main__":
    main()
