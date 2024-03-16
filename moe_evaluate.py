import argparse
import os
import time
import torch
import tqdm

from fvcore.nn import FlopCountAnalysis
from multiprocessing import cpu_count
from transformers import T5Tokenizer, T5ForConditionalGeneration

import model_modifier
from utils import dataset_loader

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='t5-base', help='model name in huggingface model hub')
parser.add_argument('--dataset', type=str, default='sst2', help='dataset used to for evaluation')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load the model rom')
parser.add_argument('--res_path', type=str, default='results/t5-base/base', help='path to store the results of moefication')
parser.add_argument('--k', type=int, default=20, help='Number of experts to activate')
parser.add_argument('--eval_type', type=str, choices=['base', 'moe_gt', 'moe_mlp'], help='Number of experts to activate')
parser.add_argument('--num_batches', type=int, default=None, help='Number of batches used for evaluation')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for eval')


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
    model.cuda()

    _, valid, output_classes = dataset_loader.load_dataset(args.dataset)
    output_class_tokens = dataset_loader.get_output_class_tokens(tokenizer, output_classes)
    output_class_tokens = [class_token.cuda() for class_token in output_class_tokens]

    num_correct = 0
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=args.batch_size, shuffle=False,
    num_workers=max(cpu_count()//2, 1))

    pbar = tqdm.tqdm(valid_loader, total=args.num_batches)
    model.eval()
    inference_time = 0
    flops = 0
    with torch.no_grad():
        for step, instance in enumerate(pbar):
            if step >= args.num_batches:
                break
            tokenized = tokenizer(text=instance["sentence"], return_tensors="pt", padding=True)
            input_ids = tokenized.input_ids.cuda()
            attention_mask = tokenized.attention_mask.cuda()
            encoder_last_hidden_state = None
            class_logits = []
            for class_token in output_class_tokens:
                class_token = class_token.repeat((input_ids.size(0), 1))
                if encoder_last_hidden_state is None:
                    start_time = time.time()
                    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=class_token)
                    end_time = time.time()
                    encoder_last_hidden_state = output.encoder_last_hidden_state
                else:
                    start_time = time.time()
                    output = model(encoder_outputs=(encoder_last_hidden_state,),
                                    attention_mask=attention_mask,
                                    labels=class_token)
                    end_time = time.time()
                inference_time += end_time - start_time

                if encoder_last_hidden_state is None:
                    analyzer = FlopCountAnalysis(model, (input_ids, attention_mask, None, None, None,
                                                        None, None, None, None, None,
                                                        None, class_token,))
                else:
                    analyzer = FlopCountAnalysis(model,
                                            (None, attention_mask, None, None, None,
                                            None, None, (encoder_last_hidden_state,), None, None,
                                            None, class_token,
                                            ))
                
                analyzer.tracer_warnings('none')
                flops += analyzer.total()
                probs = torch.softmax(output.logits, dim=2)
                logits = torch.log(probs)
                logits = torch.gather(logits, dim=2, index=class_token[:, :, None]).squeeze(dim=2)
                class_logits.append(logits.sum(dim=1, keepdim=True))

            class_logits = torch.hstack(class_logits)
            pred = torch.argmax(class_logits, 1)

            num_correct += (pred == instance['label'].cuda()).sum().item()

            pbar.set_description(f'Acc: {num_correct/((step + 1) * args.batch_size):.3f}')

    accuracy = num_correct / (args.batch_size * args.num_batches)
    inference_time = inference_time / (args.batch_size * args.num_batches)
    flops = flops / (args.batch_size * args.num_batches)
    print(f'Acc: {accuracy:.3f}')

    data_to_save = f"Accuracy: {accuracy}\n\nInference Time: {inference_time}\n\nAverage FLOPs per Example: {flops}\n\n"
    save_dir = 'analysis'
    file_path = f"{save_dir}/{args.model_name}_{args.dataset}_k{args.k}_{args.eval_type}.txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(file_path, "w") as file:
        file.write(data_to_save)


if __name__ == "__main__":
    main()
