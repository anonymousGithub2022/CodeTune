from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM

model_list = [
    ("t5", "Salesforce/codet5-small", AutoModelForSeq2SeqLM),
    ("t5", "Salesforce/codet5-base", AutoModelForSeq2SeqLM),
    ("t5", "Salesforce/codet5-large", AutoModelForSeq2SeqLM),

    ("gpt2", "Salesforce/codegen-350M-multi",AutoModelForCausalLM ),
    ("gpt2", "Salesforce/codegen-350M-mono", AutoModelForCausalLM),
    ("gpt2", "Salesforce/codegen-350M-nl", AutoModelForCausalLM),
]

for (_, model_name, class_name) in model_list:
    model = class_name.from_pretrained(model_name)

    sum_p = 0
    for p in model.parameters():
        sum_p += p.numel()
    print(model_name, len(list(model.modules())), sum_p / 10e6)


