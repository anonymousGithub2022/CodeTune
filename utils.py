import os
import platform

import datasets
import torch
from transformers import default_data_collator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig
from transformers import pipeline
import numpy as np
import random
from nltk.translate.bleu_score import sentence_bleu
from loguru import logger
from src.data_module import *
from src.methods import ModelTuner
from src.methods import ConstantPrompt
from src.methods import PromptTuningTuner
from src.methods import PromptEncoderTuner
from src.methods import PrefixTuningTuner
from src.methods import CodeTuner
from src.methods import PrefixTuningE2ETuner
from src.methods import PromptEncoderE2ETuner
from src.methods import ConstantPromptE2ETuner
from peft_utils import preprocess_clm, preprocess_seq2seq
from peft import TaskType

TUNER_DICT = {
    # *********** Only select one from the following three ***********
    'CodeTuner': CodeTuner,                            # Not USE
    'PromptTuningTuner': PromptTuningTuner,            # Not USE
    'PromptEncoderTuner': PromptEncoderTuner,          # Fixed LM, Tune Prompt
    'ModelTuner': ModelTuner,                          # No Prompt, Tune Model
    'PrefixTuningTuner': PrefixTuningTuner,            # Not USE

    'PromptEncoderE2ETuner': PromptEncoderE2ETuner,    # tune both
    'PrefixTuningE2ETuner': PrefixTuningE2ETuner,      # Not USE

    'ConstantPrompt': ConstantPrompt,                  # constant prompt, no tunning
    'ConstantPromptE2ETuner': ConstantPromptE2ETuner,  # constant prompt, only tune model

    ## ***********  The following methods are not tested  ***********

    # 'FeatureTuner': FeatureTuner,
    # 'BlackBoxTuner': BlackBoxTuner,
}

DATA_CACHE_DIR = './Dataset/huggingface_codexglue'
if not os.path.isdir(DATA_CACHE_DIR):
    os.mkdir(DATA_CACHE_DIR)

PREPROCESSED_DATA_DIR = './Dataset/pre-process-codexglue'
if not os.path.isdir(PREPROCESSED_DATA_DIR):
    os.mkdir(PREPROCESSED_DATA_DIR)


TRAIN_DATA_SIZE_LIST = ['0', '4', '8', '16', '32', '64', '128', '256', '512', '1024', 'max']

'''
yiming's model_name_list

("roberta", "microsoft/codebert-base")

("t5", "Salesforce/codet5-small")
("t5", "Salesforce/codet5-base")
("t5", "Salesforce/codet5-large")

("gpt2", "Salesforce/codegen-350M-multi")
("gpt2", "Salesforce/codegen-350M-mono")
("gpt2", "Salesforce/codegen-350M-nl")

'''


MODEL_NAME_LIST = [
    ("roberta", "microsoft/codebert-base"),

    ("t5", "Salesforce/codet5-small"),
    ("t5", "Salesforce/codet5-base"),
    ("t5", "Salesforce/codet5-large"),

    ("gpt2", "Salesforce/codegen-350M-multi"),
    ("gpt2", "Salesforce/codegen-350M-mono"),
    ("gpt2", "Salesforce/codegen-350M-nl"),


    ## *********** We will not use the following model  *********

    # ("gpt2", "NinedayWang/PolyCoder-160M", LMTokenizerWrapper),    # PolyCodeer
    # ("gpt2", "NinedayWang/PolyCoder-0.4B", LMTokenizerWrapper),
    # Does not work
    # ("gpt2", 'EleutherAI/gpt-neo-125M', LMTokenizerWrapper),      # GPT-neo
    # ("gpt2", 'EleutherAI/gpt-neo-1.3b', LMTokenizerWrapper),
    # ("gpt2", 'codeparrot/codeparrot-small-multi', LMTokenizerWrapper),
    # ("gpt2", "codeparrot/codeparrot", LMTokenizerWrapper),        # Codeparrot
    # ("gpt2", 'codeparrot/codeparrot-small', LMTokenizerWrapper),
    # Too Large
    # ("gpt2", 'EleutherAI/gpt-j-6B', LMTokenizerWrapper),          # GPT-j
    # ("gpt2", 'EleutherAI/gpt-neox-20b', LMTokenizerWrapper),      # GPT-neox
    # (Salesforce/codegen-16B-multi )
]

TASK_ID_DICT = {
    "code_x_glue_ct_code_to_text": 0,
    "code_x_glue_tc_text_to_code": 1,
    "code_x_glue_tc_nl_code_search_adv": 2,
    # "code_x_glue_cc_clone_detection_big_clone_bench": 3,
    "code_x_glue_cc_code_to_code_trans": 4,

    # "code_x_glue_cc_code_completion_token": 5,          # Not Ready
    # "code_x_glue_cc_code_completion_line": 6,           # Not Ready
    # "code_x_glue_cc_clone_detection_poj104": 7,         # Not Ready

    # "code_x_glue_cc_defect_detection",               # Low accuracy
    # "code_x_glue_cc_cloze_testing_maxmin",           # No Ground truth
    # "code_x_glue_cc_cloze_testing_all",              # No Ground truth
    # "code_x_glue_cc_code_refinement",                # Not applicable
}

TASK_CONFIG_DICT = {
    "code_x_glue_ct_code_to_text": {
        'type': 'generation',
        'text_list': [
            'Generate comments for {src_lng} code: %s.',
            'Summarize {src_lng} code: %s.',
            'Generate comments for the following code snippet: %s.',
            'Summarize the following code snippet: %s.',

            '%s, Summarize the {src_lng} code: ',
            '%s, Generate comments for {src_lng} code: ',
            '%s, Summarize the code: ',
            '%s, Generate comments for code: ',

            'Given the {src_lng} code, %s , Generate comments',
            'Given the {src_lng} code, %s , Summarize',

            '%s',
        ]
    },
    "code_x_glue_tc_text_to_code": {
        'type': 'generation',
        'text_list': [
            'Generate {src_lng} code following the comments, %s.',
            'Generate code following the comments, %s.',
            'Synthesize {src_lng} code following the comments, %s.',
            'Synthesize code following the comments, %s.',

            '%s, Generate {src_lng} code: ',
            '%s, Generate code: ',
            '%s, Synthesize code: ',
            '%s, Synthesize {src_lng} code: ',

            'Given the instruction %s, Generate {src_lng} code: ',
            'Given the instruction %s, Synthesize {src_lng} code: ',

            '%s',
        ]
    },
    # "code_x_glue_tc_nl_code_search_adv": {
    #     'type': 'generation',
    #     'text': 'Generate code for '
    # },
    # 'code_x_glue_cc_clone_detection_big_clone_bench': {
    #     'type': 'classification'
    # },
    'code_x_glue_cc_code_to_code_trans': {
        'type': 'generation',
        'text_list': [
            'Translate code to {tgt_lng} code: %s ',
            'Translate to {tgt_lng}: %s ',
            'Transfer {src_lng} code to {tgt_lng} code: %s ',
            'Transfer {src_lng} to {tgt_lng}: %s ',

            '%s, Given the code, transfer to {tgt_lng}:',
            '%s, Given the code, translate to {tgt_lng}:',
            '%s, Given the {src_lng} code, transfer to {tgt_lng}:',
            '%s, Given the {src_lng} code, translate to {tgt_lng}:',

            'Given the code, %s, transfer to {tgt_lng} code:',
            'Given the {src_lng} code, %s, translate to {tgt_lng} code:',

            '%s',
        ]
    },
}


def common_get_root_dir():
    release_id = platform.release()
    if release_id == '5.14.0-1058-oem':    # weilab
        return '/disk/CM/Project/CodeTuneS'


def common_get_task_config(data_id, src_lang):
    task_config = {}
    if data_id == 0:
        templateText = 'Generate comment for %s ' % src_lang

    elif data_id == 1:
        templateText = 'Generate %s code for ' % src_lang

    elif data_id == 2:
        templateText = 'Generate %s code for ' % src_lang

    elif data_id == 3:
        templateText = 'Code A: {"placeholder":"text_a"}, Code B: {"placeholder":"text_b"}, Are these two code snippets a clone pair {"special": "<eos>"} {"mask"}'

    elif data_id == 4:
        assert src_lang in ['java', 'cs']
        tgt_lang = 'cs' if src_lang == 'java' else 'java'
        templateText = 'Translate following %s {"placeholder":"text_a"} to %s {"special": "<eos>"} {"mask"}' % (src_lang, tgt_lang)
    task_config['manu_text'] = templateText


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def common_split_data(ori_dataset, k_shot=None):
    if k_shot is None:
        # no split
        return ori_dataset, None
    all_indices = list(range(len(ori_dataset)))
    np.random.shuffle(all_indices)
    select_index = all_indices[:k_shot]
    rest_index = all_indices[k_shot:]

    if isinstance(ori_dataset, List):
        new_train_data = [ori_dataset[i] for i in select_index]
        new_dev_data = [ori_dataset[i] for i in rest_index]

    elif isinstance(ori_dataset, datasets.Dataset):
        new_train_data = ori_dataset.select(select_index)
        new_dev_data = ori_dataset.select(rest_index)
    else:
        raise NotImplementedError
    return new_train_data, new_dev_data


def common_load_dataset(task_id, lang_id):
    data_path = os.path.join(PREPROCESSED_DATA_DIR, 'task:%d_lang:%d_train.new_dataset' % (task_id, lang_id))
    train_set = datasets.load_from_disk(data_path)
    # data_path = os.path.join(PREPROCESSED_DATA_DIR, 'task:%d_lang:%d_test.new_dataset' % (task_id, lang_id))
    # test_set = datasets.load_from_disk(data_path)
    return train_set, None


def common_get_tune_config(args):
    tune_config = {
        'train_batch_size': args.train_batch_size,
        'test_batch_size': args.test_batch_size,
        'freeze_plm': args.freeze_plm,
        'epochs': args.epochs,
        'lr': args.lr,
        'eps': args.eps,
        'prompt_num_token': args.prompt_num_token,
        'output_dir': args.output_dir
    }
    return tune_config


def common_prepare_everything(args, test_num=1000):
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name
    task_id = TASK_ID_DICT[task_name]
    task_config = TASK_CONFIG_DICT[task_name]

    tune_config = common_get_tune_config(args)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    lang_id = args.lang_id
    assert lang_id < len(LANGUAGE_DICT[task_name]), f"Invalid language id for {task_name}"
    src_lang, tgt_lang = LANGUAGE_DICT[task_name][lang_id]
    if args.tuner_name in ['ConstantPrompt', 'ConstantPromptE2ETuner']:
        prompt = task_config['text_list'][args.prompt_num_token]
        prompt.replace('{src_lng}', src_lang)
        prompt.replace('{tgt_lng}', tgt_lang)
        task_config['text'] = prompt

    train_set, _ = common_load_dataset(task_id, lang_id)

    train_set, rest_set = common_split_data(train_set, k_shot=args.train_num)
    val_set, rest_set = common_split_data(rest_set, k_shot=args.train_num)
    test_set, _ = common_split_data(rest_set, k_shot=test_num)

    model_id = args.model_id
    model_type, model_path = MODEL_NAME_LIST[model_id]
    tuner_name = args.tuner_name
    tuner_class = TUNER_DICT[tuner_name]
    if model_type == 't5' and tuner_name in ['PromptTuningTuner', 'PromptEncoderTuner']:
        tuner_class = PrefixTuningTuner  # (Error) Only works for Seq2Seq

    print(model_type, model_path)

    tuner = tuner_class(
        model_type, model_path, tune_config, task_config, device
    )

    return tuner, train_set, val_set, test_set, tune_config['output_dir']


def common_prompt_adding(tunner, dataset):
    if isinstance(tunner, ConstantPrompt):
        def add_prompt(examples):
            examples['input_text_a'] = [tunner.text_template % d for d in examples['input_text_a']]
            return examples

        prompted_dataset = dataset.map(
            add_prompt,
            batched=True,
            num_proc=1,
            desc='Running constant prompt adding on dataset'
        )
        return prompted_dataset
    return dataset


def common_construct_data_loader(tuner, train_set, val_set, test_set, num_workers=8):
    if tuner.task_type == TaskType.SEQ_2_SEQ_LM:
        data_preprocess_func = preprocess_seq2seq
        test_batch = tuner.tune_config['test_batch_size']
    elif tuner.task_type == TaskType.CAUSAL_LM:
        data_preprocess_func = preprocess_clm
        test_batch = tuner.tune_config['test_batch_size']
    else:
        raise NotImplementedError

    train_set = common_prompt_adding(tuner, train_set)
    test_set = common_prompt_adding(tuner, test_set)
    val_set = common_prompt_adding(tuner, val_set)

    process_train_set = data_preprocess_func(
        train_set, tuner.tokenizer, is_compute_loss=True)
    process_val_set = data_preprocess_func(
        val_set, tuner.tokenizer, is_compute_loss=True)
    process_test_set = data_preprocess_func(
        test_set, tuner.tokenizer, is_compute_loss=False)

    train_loader = DataLoader(
        process_train_set, batch_size=tuner.tune_config['train_batch_size'], shuffle=True,
        collate_fn=default_data_collator, num_workers=num_workers)
    val_loader = DataLoader(
        process_val_set, batch_size=test_batch,
        shuffle=False, collate_fn=default_data_collator, num_workers=num_workers)
    test_loader = DataLoader(
        process_test_set, batch_size=test_batch, shuffle=False,
        collate_fn=default_data_collator)
    return train_loader, val_loader, test_loader


def common_eval_model(tuner, test_loader, save_dir):
    tuner.plm = tuner.plm.from_pretrained(save_dir)
    if not isinstance(tuner, ModelTuner) and not isinstance(tuner, ConstantPrompt):
        tuner.model = tuner.model.from_pretrained(tuner.plm, save_dir)
    else:
        tuner.model = tuner.plm  # no model for ModelTuner

    res = tuner.evaluate_generation(test_loader, max_length=1000)  # TODO
    return res

