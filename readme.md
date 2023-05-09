

openprompt
human-eval

need to be installed from source instead of the command ``pip``

1. run `python preprocess_codexglue.py` to prepare dataset


## Parameter range for reference
### Model tuning
code-t5-small
code-t5-base
code-t5-large
codegen-multi 1e-5 3e-5 5e-5 (1e-4)
codegen-mono  1e-5 3e-5 5e-5 (1e-4)
codegen-nl    1e-5 3e-5 5e-5 (1e-4)

### Prefix tuning
code-t5-small 1e-2 2e-2 5e-2
code-t5-base 1e-2 2e-2 5e-2
code-t5-large 1e-2 2e-2 5e-2

### P-tuning
codegen-multi >1e-4
codegen-mono >1e-4
codegen-nl >1e-4