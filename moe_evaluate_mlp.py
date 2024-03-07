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


def change_forward(model, res_path, k=20):

    def _forward(ffn_self, hidden_states):
        
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states_mlp = hidden_states.clone().detach()
        hidden_states_mlp = hidden_states_mlp.view(-1, hidden_size)

        hidden_states_mlp = hidden_states_mlp / torch.norm(hidden_states_mlp, dim=-1).unsqueeze(-1)
        score = ffn_self.mlp(hidden_states_mlp)

        labels = torch.topk(score, k=k, dim=-1)[1].view(bsz, seq_len, k)
        cur_mask = torch.nn.functional.embedding(labels, ffn_self.patterns).sum(-2)
        
        hidden_states = ffn_self.wi(hidden_states)
        hidden_states = ffn_self.act(hidden_states)
        hidden_states[cur_mask == False] = 0  
        
        hidden_states = ffn_self.dropout(hidden_states)
        hidden_states = ffn_self.wo(hidden_states)
        return hidden_states

    def modify_ffn(ffn, path):
        assert type(ffn) == T5DenseActDense
        labels = torch.load(path)
        cluster_num = max(labels)+1
        patterns = []
        for i in range(cluster_num):
            patterns.append(np.array(labels) == i)
        ffn.patterns = torch.Tensor(patterns).cuda()
        ffn.k = k
        ffn.mlp = torch.load(path+'_input_compl').cuda()
        ffn.forward_old = ffn.forward
        ffn.forward = types.MethodType(_forward, ffn)   

    # encoder
    for layer_idx, layer in enumerate(model.encoder.block):
        ffn = layer.layer[1].DenseReluDense
        path = os.path.join(res_path, 'param_split', 'encoder.block.{}.layer.1.DenseReluDense.wi.weight'.format(layer_idx))
        modify_ffn(ffn, path) 

    #decoder
    for layer_idx, layer in enumerate(model.decoder.block):
        ffn = layer.layer[2].DenseReluDense
        path = os.path.join(res_path, 'param_split', 'decoder.block.{}.layer.2.DenseReluDense.wi.weight'.format(layer_idx))
        modify_ffn(ffn, path)    


def main():
    args = parser.parse_args()

    model_name = args.model_name
    model_max_length = (2048 + 1024) // 2 if args.dataset == 'race' else 512
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=model_max_length)
    model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()
    if args.checkpoint:
        utils.LoadCheckpoint(model, args.checkpoint)

    _, valid, output_classes = dataset_loader.load_dataset(args.dataset, tokenizer)
    output_class_tokens = dataset_loader.get_output_class_tokens(tokenizer, output_classes)

    change_forward(model, args.res_path, args.k)

    num_correct = 0
    pbar = tqdm.tqdm(valid)
    for step, instance in enumerate(pbar):
        input_ids = tokenizer(text=[instance['sentence']], return_tensors="pt").input_ids.cuda()
        dec_input_ids = tokenizer("<extra_id_0>", return_tensors="pt").input_ids.cuda()[:, :1]

        output = model(input_ids=input_ids, labels=dec_input_ids)
        logits = output.logits[:, 0, output_class_tokens]
        pred = torch.argmax(logits, 1)

        num_correct += (pred == instance['label']).item()

        pbar.set_description(f'Acc: {num_correct/(step + 1):.3f}')

    print(f'Acc: {num_correct/(step + 1):.3f}')


if __name__ == "__main__":
    main()