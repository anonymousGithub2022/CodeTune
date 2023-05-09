#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

model_id=0

tuner_name=ConstantPromptE2ETuner
task_list=(code_x_glue_cc_code_to_code_trans code_x_glue_ct_code_to_text code_x_glue_tc_text_to_code)
nt_list=(0 1 2 3 4 5 6 7 8 9)
bs=8
train_num=256
EPOCH=30

python generate_outputs.py --model_id ${model_id}

#for nt in ${nt_list[*]}; do
#    for lang_id in 0; do
#        for lr in 3e-5; do
#            for seed in 43; do # 21 43 87
#                for task_name in ${task_list[*]}; do
#                    python main_yiming.py \
#                        --task_name ${task_name} \
#                        --lang_id ${lang_id} \
#                        --tuner_name ${tuner_name} \
#                        --model_id ${model_id} \
#                        --train_num ${train_num} \
#                        --epochs $EPOCH \
#                        --lr ${lr} \
#                        --train_batch_size ${bs} \
#                        --prompt_num_token ${nt} \
#                        --output_dir results/${task_name}-${lang_id}/${train_num}/${tuner_name}-${model_id}-${lr}-${bs}-${nt}/${seed} \
#                        --seed ${seed} \
#                        --do_eval
#                done;
#            done;
#        done;
#    done;
#done;
