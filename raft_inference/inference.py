import os
import sys
import time
import pandas as pd

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from raft_training.utils.callbacks import Iteratorize, Stream
from raft_training.utils.prompter import Prompter

import gc
import pdb
from tqdm import tqdm
import shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def extract_question(prompt_input: str) -> str:
    lines = prompt_input.split("\n")
    for i, line in enumerate(lines):
        if line.strip() == "Question:" and i + 1 < len(lines):
            return lines[i + 1].strip()
    return ""

def truncate_prompt(tokenizer, prompt, max_new_tokens=32, model_max_len=4096):
    max_input_tokens = model_max_len - max_new_tokens
    tokens = tokenizer(prompt, truncation=True, max_length=max_input_tokens, return_tensors="pt")
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

def main(
    load_8bit: bool = False,
    use_lora: bool = True,
    base_model: str = "../llama30B_hf",
    lora_weights: str = "",
    prompt_template: str = "mistral",
    data_path: str = "",
    task: str = "",
    setting: str = "",
    output_data_path: str = ""

):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":

        if base_model not in ['microsoft/phi-2', 'NingLab/eCeLLM-S']:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                # trust_remote_code=True,
                # attn_implementation='flash_attention_2',
                attn_implementation='eager',
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                # trust_remote_code=True,
            )

        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.bfloat16,
            )

    if not load_8bit:
        model.bfloat16()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if not model.config.eos_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        model.config.eos_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = model.config.eos_token_id
        tokenizer.padding_side = 'left'

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.float16, 
        device_map="auto",
    )

    dataset = load_dataset("json", data_files=data_path)['train']

    instructions, inputs, options, ids = [], [], [], []

    for data in dataset:
        instructions.append(data["instruction"])
        inputs.append(data["input"])
        options.append(data.get("options", None))  
        ids.append(data["id"] if "id" in data else str(len(ids)))

    output_dir = os.path.dirname(output_data_path)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    skipped_ids = set()
    if os.path.exists(output_data_path):
        backup_path = output_data_path + ".bak"
        print(f"[INFO] Found existing output. Backing up to {backup_path}")
        shutil.move(output_data_path, backup_path)

        with open(backup_path, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    skipped_ids.add(str(example["id"]))
                except:
                    continue

    print(f"[INFO] Skipping {len(skipped_ids)} previously processed examples.")

    checkpoint_interval = 10 

    output_mode = "w"
    # results = []
    max_batch_size = 1
    for i in tqdm(range(0, len(instructions), max_batch_size), desc="Running batches"):
        instruction_batch = instructions[i:i + max_batch_size]
        input_batch = inputs[i:i + max_batch_size]
        options_batch = options[i:i + max_batch_size]
        ids_batch = ids[i:i + max_batch_size]

        filtered_batch = [
            (inst, inp, opt, idx) for inst, inp, opt, idx in zip(instruction_batch, input_batch, options_batch, ids_batch)
            if str(idx) not in skipped_ids
        ]

        if len(filtered_batch) == 0:
            continue 

        prompts = [prompter.generate_prompt(inst, inp, opt) for inst, inp, opt, _ in filtered_batch]
        batch_results = evaluate(prompter, prompts, tokenizer, pipe, len(filtered_batch))
    
        id2output = {str(example["id"]): example["output"] for example in dataset}

        with open(output_data_path, output_mode, encoding='utf-8') as f:
            for (inst, inp, _, idx), response in zip(filtered_batch, batch_results):
                    result = {
                        "id": idx,
                        "question": extract_question(inp),
                        "output": id2output[str(idx)],
                        "prediction": response
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")


        output_mode = "a"
        gc.collect()
        torch.cuda.empty_cache()
        step = i // max_batch_size
        if step > 0 and step % checkpoint_interval == 0:
            backup_path = output_data_path + ".bak"
            shutil.copy(output_data_path, backup_path)


def extract_query(input_text):
    for line in input_text.split("\n"):
        if line.lower().startswith("query:"):
            return line.split(":", 1)[1].strip()
    return ""

def extract_product(input_text):
    for line in input_text.split("\n"):
        if line.lower().startswith("product:"):
            return line.split(":", 1)[1].strip()
    return ""


def evaluate(prompter, prompts, tokenizer, pipe, batch_size):
    batch_outputs = []

    generation_output = pipe(
        prompts,
        do_sample=True,
        max_new_tokens=32, #base 5
        temperature=0.7, #base 0.15
        top_p=0.9, #base 0.95
        num_return_sequences=1,
        # num_beams=1,
        top_k=40,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id, 
        batch_size=batch_size,
        no_repeat_ngram_size=3, 

    )

    for i in range(len(generation_output)):    
        resp = prompter.get_response(generation_output[i][0]['generated_text'])
        batch_outputs.append(resp)

    return batch_outputs


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)

