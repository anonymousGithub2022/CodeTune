#!/bin/bash

#SBATCH -p p-A100

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

#module load cuda11.7/toolkit/11.7.0 slurm/slurm/21.08.8

export CUDA_VISIBLE_DEVICES=5

task_name=code_x_glue_tc_nl_code_search_adv
#code_x_glue_ct_code_to_text
#code_x_glue_tc_text_to_code
#code_x_glue_tc_nl_code_search_adv
#code_x_glue_cc_code_to_code_trans

tuner_name=PrefixTuningTuner
#PromptEncoderTuner
#PromptTuningTuner
#ModelTuner

bs=8
nt=10
train_num=256

for lang_id in 0; do
    for lr in 1e-2 2e-2 5e-2; do
        for model_id in 3; do
            for seed in 43; do # 21 43 87
                python main_yiming.py \
                        --task_name ${task_name} \
                        --lang_id ${lang_id} \
                        --tuner_name ${tuner_name} \
                        --model_id ${model_id} \
                        --train_num 256 \
                        --epochs 30 \
                        --lr ${lr} \
                        --train_batch_size ${bs} \
                        --prompt_num_token ${nt} \
                        --output_dir results/${task_name}-${lang_id}/${train_num}/${tuner_name}-${model_id}-${lr}-${bs}-${nt}/${seed} \
                        --seed ${seed} \
                        --do_eval
            done;
        done;
    done;
done;