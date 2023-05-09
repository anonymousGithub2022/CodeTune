import os.path

import torch
import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='code_x_glue_tc_text_to_code', type=str)
    parser.add_argument('--lang_id', default=0, type=int) #
    parser.add_argument('--tuner_name', default='CodeTuner', type=str)
    parser.add_argument('--model_id', default=1, type=int)
    parser.add_argument('--do_eval', action='store_true', default=True)

    parser.add_argument('--train_num', default=256, type=int)

    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--eps', default=1e-4, type=float)
    parser.add_argument('--prompt_num_token', default=10, type=int)
    parser.add_argument('--freeze_plm', default=True)

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='cm_tmp')

    args = parser.parse_args()
    return args

args = read_args()

task_name_list = [
    'code_x_glue_ct_code_to_text',
    'code_x_glue_tc_text_to_code',
    'code_x_glue_cc_code_to_code_trans',

]
args.lang_id = 0
args.train_num = 256
args.tuner_name = 'ConstantPromptE2ETuner'
args.lr = '3e-5'
args.bs = 8
args.seed = 43

pt_id = 10


for task_name in task_name_list:
    for model_id in range(1, 7):
        max_res = 0
        if model_id == 3:
            print(model_id, task_name, model_id, 0)
            continue
        for pt_id in range(10):
            args.nt = pt_id
            args.prompt_num_token = pt_id
            args.task_name = task_name
            args.model_id = model_id

            model_dir = f'results/{args.task_name}-{args.lang_id}/{args.train_num}/{args.tuner_name}-{args.model_id}-{args.lr}-{args.bs}-{args.nt}/{args.seed}'
            file_name = os.path.join(model_dir, 'val_bleu.txt')

            with open(file_name, 'r') as f:
                d = f.readlines()
                max_res = max(max_res, float(d[0].replace('\n', '')))
        print(model_id, task_name, model_id, max_res)