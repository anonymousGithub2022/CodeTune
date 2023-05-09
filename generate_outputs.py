import os.path
import argparse
import torch

from utils import common_prepare_everything
from utils import common_construct_data_loader
from utils import common_eval_model
from src.bleu import compute_blue_scores


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
args.tuner_name = 'ConstantPrompt'
args.lr = '1e-2'
args.bs = 8
args.seed = 43

try:
    results = torch.load(str(args.model_id) + '_constant.tar')
except:
    results = []


for model_id in range(1, 7):
    for task_name in task_name_list:
        max_res = 0
        for pt_id in range(10):
            args.nt = pt_id
            args.prompt_num_token = pt_id
            args.task_name = task_name
            args.model_id = model_id

            model_dir = f'results/{args.task_name}-{args.lang_id}/{args.train_num}/{args.tuner_name}-{args.model_id}-{args.lr}-{args.bs}-{args.nt}/{args.seed}'

            args.output_dir = model_dir

            tuner, train_set, val_set, test_set, save_dir = \
                common_prepare_everything(args, test_num=500)

            train_loader, val_loader, test_loader = \
                common_construct_data_loader(tuner, train_set, val_set, test_set, num_workers=1)
            tuner.tune(train_loader, val_loader)
            res = common_eval_model(tuner, test_loader, save_dir)
            # loss = tuner.compute_loss(val_loader)
            new_bleu_score_1 = compute_blue_scores(res[0], res[1], 'cm_tmp')
            new_bleu_score_2 = compute_blue_scores(res[0], res[2], 'cm_tmp')
            # print(new_bleu_score_1, new_bleu_score_2)
            results.append([task_name, args.model_id, pt_id, float(new_bleu_score_1), float(new_bleu_score_2)])
            print(task_name, model_id, new_bleu_score_1, new_bleu_score_2, 'successful')
            # torch.save(results, str(args.model_id) + '_constant.tar')
            print('--------------------------')
# for task_name in task_name_list:
#     for model_id in range(7):
#         save_dir = f'results/{task_name}-{lang_id}/{train_num}/{tuner_name}-{model_id}-{lr}-{bs}-{nt}/{seed}'
#         file_name = os.path.join(save_dir, 'val_bleu.txt')
#         with open(file_name, 'r') as f:
#             res = f.readlines()
#         print(save_dir, res )