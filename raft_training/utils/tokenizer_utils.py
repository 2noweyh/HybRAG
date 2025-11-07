import transformers
from transformers import (
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoTokenizer,
    AutoModelForCausalLM,
)


# utils/tokenizer_utils.py
def load_and_prepare_tokenizer(model_name_or_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path,
                                               fast_tokenizer=True,
                                               local_files_only=True)
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "[PAD]"

    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "</s>"

    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)

    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, special_tokens_dict
