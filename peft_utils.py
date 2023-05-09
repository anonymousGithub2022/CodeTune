import torch
from datasets import Dataset


#TODO: suitable for two text


def preprocess_seq2seq(
        dataset: Dataset, tokenizer, is_compute_loss, max_length=256
):
    def preprocess_func(examples):
        model_inputs = tokenizer(
            examples['input_text_a'], max_length=max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        targets = tokenizer(
            examples['target_text'], max_length=max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = targets['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs['labels'] = labels
        return model_inputs

    preprocessed_dataset = dataset.map(
        preprocess_func,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        desc='Running tokenizer on dataset'
    )
    return preprocessed_dataset


def preprocess_clm(
        dataset: Dataset, tokenizer, is_compute_loss, max_length=256,
):
    def train_preprocess_function(examples):
        batch_size = len(examples['input_text_a'])

        model_inputs = tokenizer(examples['input_text_a'])
        labels = tokenizer(examples['target_text'])
        ori_inputs = model_inputs['input_ids']
        ori_labels = labels['input_ids']

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = \
                [tokenizer.pad_token_id] * (max_length - len(sample_input_ids)) \
                + sample_input_ids

            model_inputs["attention_mask"][i] = \
                [0] * (max_length - len(sample_input_ids)) + \
                model_inputs["attention_mask"][i]

            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids

            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def test_preprocess_function(examples):
        batch_size = len(examples['input_text_a'])
        model_inputs = tokenizer(examples['input_text_a'])
        labels = tokenizer(examples['target_text'])

        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                    max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]

            label_input_ids = labels["input_ids"][i]
            labels["input_ids"][i] = [-100] * (max_length - len(label_input_ids)) + label_input_ids

            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if is_compute_loss:
        preprocess_func = train_preprocess_function
    else:
        preprocess_func = test_preprocess_function

    preprocessed_dataset = dataset.map(
        preprocess_func,
        batched=True,
        num_proc=1,
        remove_columns=dataset.column_names,
        desc='Running tokenizer on dataset'
    )
    return preprocessed_dataset

