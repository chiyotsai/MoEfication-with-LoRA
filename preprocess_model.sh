#!/bin/bash
set -e

MODEL_ARCHITECTURE='t5-base'
MODEL_CHECKPOINT=''
TEMPLATES='encoder.block.{}.layer.1.DenseReluDense.wi.weight,decoder.block.{}.layer.2.DenseReluDense.wi.weight'
K=20

if [[ $MODEL_ARCHITECTURE == 't5-base' ]]; then
  NUM_EXPERTS=96
elif [[ $MODEL_ARCHITECTURE == 't5-large' ]]; then
  NUM_EXPERTS=128
else
  echo "Unknown architecutre $MODEL_ARCHITECTURE!"
fi

#K=1
#NUM_EXPERTS=8
#RESULT_PATH=results/"$MODEL_ARCHITECTURE"-"e$NUM_EXPERTS"-k"$K"/"$DATASET"

DATASET="sst2"
RESULT_PATH=results/"$MODEL_ARCHITECTURE"/"$DATASET"
NUM_EVAL_BATCHES=4

if [[ ! -d "$RESULT_PATH" ]]; then
  mkdir -p "$RESULT_PATH"
fi

if [[ $DATASET == 'sst2' ]]; then
  echo "Setting parameters for sst2..."
  # NUM_MLP_SAMPLES=10240
  # MLP_SAMPLE_SHARD_SIZE=1024
  NUM_MLP_SAMPLES=2560
  MLP_SAMPLE_SHARD_SIZE=512
  EVAL_BATCH_SIZE=128
elif [[ $DATASET == 'multi_nli' || $DATASET == 'mnli' ]]; then
  echo "Setting parameters for mnli..."
  NUM_MLP_SAMPLES=10240
  MLP_SAMPLE_SHARD_SIZE=1024
  EVAL_BATCH_SIZE=1024
elif [[ $DATASET == 'race' ]]; then
  echo "Setting parameters for race..."
  MODEL_CHECKPOINT='lightning_logs/T5BaseFinetuneRACE/2024-03-02-19:16:12/checkpoints/T5BaseFinetuneRACE-global_step=0-epoch=02-step=64000-ckpt_metric=0.715.ckpt'
  NUM_MLP_SAMPLES=2048
  MLP_SAMPLE_SHARD_SIZE=512
  EVAL_BATCH_SIZE=1024
else
  echo "Unknown dataset!"
  exit 1
fi

EVAL_ONLY=1

echo "Evaluating Baseline Model..."
cmd=(python moe_evaluate.py --model_name="$MODEL_ARCHITECTURE"  --checkpoint="$MODEL_CHECKPOINT" \
  --res_path="$RESULT_PATH" --num-expert="$NUM_EXPERTS" --k="$K" --dataset="$DATASET" --eval_type=base --num_batches="$NUM_EVAL_BATCHES" \
  --batch_size="$EVAL_BATCH_SIZE")
echo "${cmd[*]}"
${cmd[@]} | tee -a "$RESULT_PATH/eval_base.log"

if (( ! EVAL_ONLY )); then
  echo "Clustering..."
  cmd=(python moe_cluster_experts.py --model_name="$MODEL_ARCHITECTURE" --checkpoint="$MODEL_CHECKPOINT" \
    --res_path="$RESULT_PATH" --num-expert="$NUM_EXPERTS" --templates="$TEMPLATES")
  echo "${cmd[*]}"
  ${cmd[@]}
fi

echo "Evaluating MoE Ground Truth..."
cmd=(python moe_evaluate.py --model_name="$MODEL_ARCHITECTURE"  --checkpoint="$MODEL_CHECKPOINT" \
  --res_path="$RESULT_PATH" --num-expert="$NUM_EXPERTS" --k="$K" --dataset="$DATASET" --eval_type=moe_gt --num_batches="$NUM_EVAL_BATCHES" \
  --batch_size="$EVAL_BATCH_SIZE")
echo "${cmd[*]}"
${cmd[@]} | tee -a "$RESULT_PATH/eval_moe_gt.log"

if (( ! EVAL_ONLY )); then
  echo "Collecting hidden states..."
  cmd=(python moe_gather_inf.py --model_name="$MODEL_ARCHITECTURE"  --checkpoint="$MODEL_CHECKPOINT" \
    --res_path="$RESULT_PATH" --dataset="$DATASET" --num_info_samples="$NUM_MLP_SAMPLES" --shard_file_samples="$MLP_SAMPLE_SHARD_SIZE")
  echo "${cmd[*]}"
  ${cmd[@]}

  echo "Training router function..."
  cmd=(python moe_train_router.py --model_name="$MODEL_ARCHITECTURE"  --checkpoint="$MODEL_CHECKPOINT" \
    --res_path="$RESULT_PATH" --num-expert="$NUM_EXPERTS" --templates="$TEMPLATES")
  echo "${cmd[*]}"
  ${cmd[@]}
fi

echo "Evaluating MLP..."
cmd=(python moe_evaluate.py --model_name="$MODEL_ARCHITECTURE"  --checkpoint="$MODEL_CHECKPOINT" \
  --res_path="$RESULT_PATH" --num-expert="$NUM_EXPERTS" --k="$K" --dataset="$DATASET" --eval_type=moe_mlp --num_batches="$NUM_EVAL_BATCHES" \
  --batch_size="$EVAL_BATCH_SIZE")
echo "${cmd[*]}"
${cmd[@]} | tee -a "$RESULT_PATH/eval_moe_mlp.log"

if (( ! EVAL_ONLY )); then
  python t5_train_main.py "--config=configs/moe_t5_base_"$DATASET"_dad_lora_train.yaml"
fi