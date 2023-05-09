import torch
import argparse
import utils
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import default_data_collator
from torch.utils.data import DataLoader
import lightseq.inference as lsi
from transformers import pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

from peft_utils import preprocess_seq2seq, preprocess_clm
from utils import *

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='code_x_glue_tc_text_to_code', type=str)
    parser.add_argument('--lang_id', default=0, type=int) #
    parser.add_argument('--tuner_name', default='CodeTuner', type=str, choices=TUNER_DICT.keys())
    parser.add_argument('--model_id', default=3, type=int, choices=[_ for _ in range(len(MODEL_NAME_LIST))])
    parser.add_argument('--do_eval', action='store_true', default=True)

    parser.add_argument('--train_num', default=256, type=int)

    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--train_batch_size', default=8, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--eps', default=1e-4, type=float)
    parser.add_argument('--prompt_num_token', default=10, type=int)
    parser.add_argument('--freeze_plm', default=True)

    parser.add_argument('--output_dir', default='cm_tmp')

    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    return args


def main():
    from transformers import BertGenerationEncoder, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-350M-multi')
    print()
    encoder = BertGenerationEncoder.from_pretrained('microsoft/codebert-base')
    decoder = BertGenerationEncoder.from_pretrained('microsoft/codebert-base')
    print()


if __name__ == '__main__':
    main()