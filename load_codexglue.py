'''
This script implements the basic data loader class for
codexglue benchmark for general purpose usage.
'''

import json
import os.path
# from torch.utils.data import Dataset
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from functools import partial


AVAILABLE_TASK_LIST = [
    ('Code-Text/code-to-text', 'go'),
    ('Code-Text/code-to-text', 'java'),
    ('Code-Text/code-to-text', 'javascript'),
    ('Code-Text/code-to-text', 'php'),
    ('Code-Text/code-to-text', 'python'),
    ('Code-Text/code-to-text', 'ruby'),

    ('Text-Code/text-to-code', None),
    #Text-Code/NL-code-search-WebQuery   # TODO

    ('Text-Code/NL-code-search-Adv', 'go'),
    ('Text-Code/NL-code-search-Adv', 'java'),
    ('Text-Code/NL-code-search-Adv', 'javascript'),
    ('Text-Code/NL-code-search-Adv', 'php'),
    ('Text-Code/NL-code-search-Adv', 'python'),
    ('Text-Code/NL-code-search-Adv', 'ruby'),

    # Code-Code/Clone-detection-BigCloneBench,      # TODO

    ('Code-Code/Clone-detection-POJ-104', None),


    # 'Code-Code/ClozeTesting-all',                # TODO
    # 'Code-Code/ClozeTesting-maxmin',              # TODO
    ('Code-Code/CodeCompletion-line', 'python'),
    ('Code-Code/CodeCompletion-line', 'java'),

    ('Code-Code/CodeCompletion-token', 'python'),
    ('Code-Code/CodeCompletion-token', 'java'),

    # 'Code-Code/Defect-detection',                   # TODO
    ('Code-Code/Method-Generation', 'go'),
    ('Code-Code/Method-Generation', 'java'),
    ('Code-Code/Method-Generation', 'javascript'),
    ('Code-Code/Method-Generation', 'php'),
    ('Code-Code/Method-Generation', 'python'),
    ('Code-Code/Method-Generation', 'ruby'),
    # 'Code-Code/code-refinement',                    #TODO

    ('Code-Code/code-to-code-trans', 'cs'),
    ('Code-Code/code-to-code-trans', 'java')
]


# class CodeXGLUEDataset():
#     def __getitem_code_to_text__(self, index):
#         data = self.dataset[index]
#         code = data['code']
#         doc = data['docstring']
#         return {'input': code, 'target': doc}
#
#     def __getitem_text_to_code__(self, index):
#         pass
#
#     def __getitem_NL_code_search__(self, index):
#         pass
#
#     def __getitem_Clone_detection_POJ_104__(self, index):
#         pass
#
#     def __getitem_CodeCompletion_line_python__(self, index):
#         pass
#
#     def __getitem_CodeCompletion_line_java__(self, index):
#         pass
#
#     def __getitem_CodeCompletion_token_python__(self, index):
#         pass
#
#     def __getitem_CodeCompletion_token_java__(self, index):
#         pass
#
#     def __getitem_Method_Generation__(self, index):
#         pass
#
#     def __getitem_codetrans__(self, index):
#         pass
#
#     def __init__(self, dataset, task_name, lang):
#         self.task_name = task_name
#         self.dataset = dataset
#         self.lang = lang
#
#
#
#     def __getitem__(self, index):
#         if self.task_name == 'Code-Text/code-to-text':
#             return self.__getitem_code_to_text__(index)
#
#         elif self.task_name == 'Text-Code/text-to-code':
#             return self.__getitem_text_to_code__(index)
#
#         elif self.task_name == 'Text-Code/NL-code-search-Adv':
#             return self.__getitem_NL_code_search__(index)
#
#         elif self.task_name == 'Code-Code/Clone-detection-POJ-104':
#             return self.__getitem_Clone_detection_POJ_104__(index)
#
#         elif self.task_name == 'Code-Code/CodeCompletion-line':
#             if self.lang == 'python':
#                 return self.__getitem_CodeCompletion_line_python__(index)
#             elif self.lang == 'java':
#                 return self.__getitem_CodeCompletion_line_java__(index)
#             else:
#                 raise NotImplementedError
#
#         elif self.task_name == 'Code-Code/CodeCompletion-token':
#             if self.lang == 'python':
#                 return self.__getitem_CodeCompletion_token_python__(index)
#             elif self.lang == 'java':
#                 return self.__getitem_CodeCompletion_token_java__(index)
#             else:
#                 raise NotImplementedError
#
#         elif self.task_name == 'Code-Code/Method-Generation':
#             return self.__getitem_Method_Generation__(index)
#
#         elif self.task_name == 'Code-Code/code-to-code-trans':
#             return self.__getitem_codetrans__(index)
#
#         else:
#             raise NotImplementedError
#
#     def __len__(self):
#         return len(self.dataset)

def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=False)
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids'],
    }
    return encodings


class CodeXGLUEModule:
    def __init__(self, root_dir, task_name, lang, tokenizer):
        self.root_dir = root_dir
        self.task_name = task_name
        self.lang = lang
        self.tokenizer = tokenizer

        if task_name == 'Code-Text/code-to-text':
            # code search net
            data_list = self.load_CodeSerchNet(lang)

        elif task_name == 'Text-Code/text-to-code':
            # concode
            data_list = self.load_concode(lang=None)

        elif task_name == 'Text-Code/NL-code-search-Adv':
            # code search net
            data_list = self.load_CodeSerchNet(lang)

        elif task_name == 'Code-Code/Clone-detection-POJ-104':
            data_list = self.load_POJ_104d(lang=None)

        elif task_name == 'Code-Code/CodeCompletion-line':
            if lang == 'python':
                data_list = self.load_py150()
            elif lang == 'java':
                data_list = self.load_javaCorpus()
            else:
                raise NotImplementedError

        elif task_name == 'Code-Code/CodeCompletion-token':
            if lang == 'python':
                data_list = self.load_py150()
            elif lang == 'java':
                data_list = self.load_javaCorpus()
            else:
                raise NotImplementedError

        elif task_name == 'Code-Code/Method-Generation':
            data_list = self.load_CodeSerchNet(lang)

        elif task_name == 'Code-Code/code-to-code-trans':
            data_list = self.load_codetrans(lang)

        else:
            raise NotImplementedError
        self.ori_train_data, self.ori_valid_data, self.ori_test_data = data_list

    def convert_examples(self, example):
        raise NotImplementedError

    def _load(self, dataset) -> DataSet:
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)

        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask")
        ds.set_target("labels")
        return ds

    @staticmethod
    def load_json_file(file_path):
        with open(file_path, 'r') as f:
            data = f.readlines()
            data = [json.loads(d) for d in data]
        return data

    @staticmethod
    def load_txt(file_path):
        with open(file_path, 'r') as f:
            data = f.readlines()
        return data


    def load_CodeSerchNet(self, lang):
        assert lang in ['go', 'java', 'javascript', 'php', 'python', 'ruby']

        data_dir = '%s/Code-Text/code-to-text/dataset/%s' % (self.root_dir, lang)
        train_path = os.path.join(data_dir, 'train.jsonl')
        valid_path = os.path.join(data_dir, 'valid.jsonl')
        test_path = os.path.join(data_dir, 'test.jsonl')

        train_data = self.load_json_file(train_path)
        valid_data = self.load_json_file(valid_path)
        test_data = self.load_json_file(test_path)

        return train_data, valid_data, test_data

    def load_concode(self, lang=None):

        data_dir = '%s/Text-Code/text-to-code/dataset/concode' % self.root_dir
        train_path = os.path.join(data_dir, 'train.json')
        test_path = os.path.join(data_dir, 'test.json')
        valid_path = os.path.join(data_dir, 'dev.json')

        train_data = self.load_json_file(train_path)
        valid_data = self.load_json_file(valid_path)
        test_data = self.load_json_file(test_path)

        return train_data, valid_data, test_data

    def load_WebQueryTest(self, lang=None):
        pass

    def load_POJ_104d(self, lang):
        data_dir = '%s/Code-Code/Clone-detection-POJ-104/dataset' % self.root_dir
        train_path = os.path.join(data_dir, 'train.jsonl')
        test_path = os.path.join(data_dir, 'test.jsonl')
        valid_path = os.path.join(data_dir, 'valid.jsonl')

        train_data = self.load_json_file(train_path)
        valid_data = self.load_json_file(valid_path)
        test_data = self.load_json_file(test_path)
        return train_data, valid_data, test_data

    def load_javaCorpus(self):
        data_dir = '%s/Code-Code/CodeCompletion-token/dataset/javaCorpus/token_completion' % self.root_dir
        train_path = os.path.join(data_dir, 'train.txt')
        test_path = os.path.join(data_dir, 'test.txt')
        valid_path = os.path.join(data_dir, 'dev.txt')

        train_data = self.load_txt(train_path)
        valid_data = self.load_txt(valid_path)
        test_data = self.load_txt(test_path)
        return train_data, valid_data, test_data

    def load_py150(self):
        data_dir = '%s/Code-Code/CodeCompletion-token/dataset/py150/token_completion' % self.root_dir
        train_path = os.path.join(data_dir, 'train.txt')
        test_path = os.path.join(data_dir, 'test.txt')
        valid_path = os.path.join(data_dir, 'dev.txt')

        train_data = self.load_txt(train_path)
        valid_data = self.load_txt(valid_path)
        test_data = self.load_txt(test_path)
        return train_data, valid_data, test_data


    def load_codetrans(self, lang):
        def combine(src_code_list, tgt_code_list):
            new_code = []
            for src_code, tgt_code in zip(src_code_list, tgt_code_list):
                new_code.append([src_code, tgt_code])
            return new_code

        data_dir = '%s/Code-Code/code-to-code-trans/data' % self.root_dir
        train_cs = self.load_txt(os.path.join(data_dir, 'train.java-cs.txt.cs'))
        train_java = self.load_txt(os.path.join(data_dir, 'train.java-cs.txt.java'))

        valid_cs = self.load_txt(os.path.join(data_dir, 'valid.java-cs.txt.cs'))
        valid_java = self.load_txt(os.path.join(data_dir, 'valid.java-cs.txt.java'))

        test_cs = self.load_txt(os.path.join(data_dir, 'test.java-cs.txt.cs'))
        test_java = self.load_txt(os.path.join(data_dir, 'test.java-cs.txt.java'))


        if lang == 'java':
            train_data = combine(train_java, train_cs)
            valid_data = combine(valid_java, valid_cs)
            test_data = combine(test_java, test_cs)
        elif lang == 'cs':
            train_data = combine(train_cs, train_java)
            valid_data = combine(valid_cs, valid_java)
            test_data = combine(test_cs, test_java)
        else:
            raise NotImplementedError
        return train_data, valid_data, test_data


def _test_():
    CODEXGLUE_DIR = './Dataset/CodeXGLUE'

    for (task_name, lang) in AVAILABLE_TASK_LIST:
        m = CodeXGLUEModule(CODEXGLUE_DIR, task_name, lang)
        print(len(m.test_data))
        print(m.test_data[0])


if __name__ == '__main__':
    _test_()


