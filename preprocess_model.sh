#!/bin/bash
set -e

MODEL_ARCHITECTURE='t5-base'
MODEL_CHECKPOINT='lightning_logs/T5BaseFinetuneRACE/2024-03-02-19:16:12/checkpoints/T5BaseFinetuneRACE-global_step=0-epoch=02-step=64000-ckpt_metric=0.715.ckpt'
DATASET="race"
RESULT_PATH="results/$MODEL_ARCHITECTURE/$DATASET"
TEMPLATES='encoder.block.{}.layer.1.DenseReluDense.wi.weight,decoder.block.{}.layer.2.DenseReluDense.wi.weight'
NUM_EXPERTS=96
K=20

echo "Clustering..."
cmd=(python moe_cluster_experts.py --model_name="$MODEL_ARCHITECTURE" --checkpoint="$MODEL_CHECKPOINT" \
  --res_path="$RESULT_PATH" --num-expert="$NUM_EXPERTS" --templates="$TEMPLATES")
echo "${cmd[*]}"
${cmd[@]}

echo "Evaluating Ground Truth..."
cmd=(python moe_evaluate_gt.py --model_name="$MODEL_ARCHITECTURE"  --checkpoint="$MODEL_CHECKPOINT" \
  --res_path="$RESULT_PATH" --k="$K" --dataset="$DATASET")
echo "${cmd[*]}"
${cmd[@]}

echo "Collecting hidden states..."
cmd=(python moe_gather_inf.py --model_name="$MODEL_ARCHITECTURE"  --checkpoint="$MODEL_CHECKPOINT" \
  --res_path="$RESULT_PATH" --dataset="$DATASET" --num_info_samples=2048 --shard_file_samples=512)
echo "${cmd[*]}"
${cmd[@]}

echo "Training router function..."
 cmd=(python moe_train_router.py --model_name="$MODEL_ARCHITECTURE"  --checkpoint="$MODEL_CHECKPOINT" \
   --res_path="$RESULT_PATH" --num-expert="$NUM_EXPERTS" --templates="$TEMPLATES")
echo "${cmd[*]}"
${cmd[@]}
 
echo "Evaluating MLP..."
cmd=(python moe_evaluate_mlp.py --model_name="$MODEL_ARCHITECTURE"  --checkpoint="$MODEL_CHECKPOINT" \
  --res_path="$RESULT_PATH" --k="$K" --dataset="$DATASET")
echo "${cmd[*]}"
${cmd[@]}