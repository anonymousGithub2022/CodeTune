import os
import torch

from src.data_module.data_loader import LANGUAGE_DICT, DATA_MODULE_CLASS
from utils import DATA_CACHE_DIR, PREPROCESSED_DATA_DIR, TASK_ID_DICT

from datasets import Dataset

splits = ['train', 'test']

for (dataset_name) in TASK_ID_DICT:
    print(dataset_name)
    task_id = TASK_ID_DICT[dataset_name]

    for l_id, lang in enumerate(LANGUAGE_DICT[dataset_name]):
        train_save_path = os.path.join(PREPROCESSED_DATA_DIR, 'task:%d_lang:%d_train.new_dataset' % (task_id, l_id))
        test_save_path = os.path.join(PREPROCESSED_DATA_DIR, 'task:%d_lang:%d_test.new_dataset' % (task_id, l_id))

        data_module_class = DATA_MODULE_CLASS[dataset_name]
        data_bundle = data_module_class(
            DATA_CACHE_DIR, lang).my_load(splits)
        train_set = data_bundle['train']
        test_set = data_bundle['test']

        print(train_set[0])
        print(train_set[0])

        train_set.save_to_disk(train_save_path)
        test_set.save_to_disk(test_save_path)
        print(dataset_name, l_id, 'successful')
        print('----------------------------------------')

