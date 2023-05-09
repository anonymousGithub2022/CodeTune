#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

task_name=code_x_glue_cc_code_to_code_trans
lang_id=0
tuner_name=PromptTuningTuner
model_id=0

lr=1e-2
bs=8
nt=10
train_num=256

seed=42
python main_simin.py \
        --task_name ${task_name} \
        --lang_id ${lang_id} \
        --tuner_name ${tuner_name} \
        --model_id ${model_id} \
        --train_num 256 \
        --epochs 10 \
        --lr ${lr} \
        --train_batch_size ${bs} \
        --prompt_num_token ${nt} \
        --output_dir results/${task_name}-${lang_id}/${train_num}/${tuner_name}-${model_id}-${lr}-${bs}-${nt}/${seed} \
        --seed ${seed} --do_eval
