import argparse
from utils import MODEL_NAME_LIST, TUNER_DICT
from utils import common_prepare_everything

tunner_name_list = [
    'ModelTuner',
    # 'PromptEncoderTuner',
    'ConstantPromptE2ETuner',
    # 'PromptEncoderE2ETuner',
]


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='code_x_glue_ct_code_to_text', type=str)
    parser.add_argument('--lang_id', default=0, type=int) #
    parser.add_argument('--tuner_name', default='PromptTuningTuner', type=str, choices=TUNER_DICT.keys())
    parser.add_argument('--model_id', default=4, type=int, choices=[_ for _ in range(len(MODEL_NAME_LIST))])
    parser.add_argument('--do_eval', action='store_true', default=True)

    parser.add_argument('--train_num', default=256, type=int)

    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--prompt_num_token', default=10, type=int)
    parser.add_argument('--freeze_plm', default=True)

    parser.add_argument('--output_dir', default='cm_tmp')

    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    return args


def common_compute_model_size(model):
    sum_p = 0
    for p in model.parameters():
        sum_p += p.numel()
    return sum_p

args = read_args()

for tunner_name in tunner_name_list:
    args.tuner_name = tunner_name
    for model_id in range(1, 7):
        args.model_id = model_id

        tuner, train_set, val_set, test_set, save_dir = \
            common_prepare_everything(args, test_num=100)
        if tunner_name in ['ModelTuner', 'ConstantPromptE2ETuner']:
            print(common_compute_model_size(tuner.model))
        else:
            tuner.model.print_trainable_parameters()
        print(tunner_name, model_id, '*******************************')
    print('---------------------------------------')