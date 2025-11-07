import json
import re
from pathlib import Path
from typing import Callable
import random
import torch
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, Sequence, List
import argparse
import fire
import gc

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text) 
    res = re.findall(r"(\d+(\.\d+)?)", text)  
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0

def get_last_index(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if not lines:
            return 0
        try:
            last_line = json.loads(lines[-1])
            return last_line.get("index", len(lines))
        except:
            return len(lines)  # fallback

def main(
    args,
    # is_bf16: bool = True,
):
    batch_size = args.batch_size
    print(f"main start, is_bf16:{args.is_bf16}, batch_size:{batch_size}")
    
    model_path = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = get_model(model_path, is_bf16=args.is_bf16, device=device)
    print("model loaded")
    print("Model device:", next(model.parameters()).device)

    batch_llama = get_batch_llama(model, tokenizer, args)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
                
    model_dirname = Path(model_path).name
    model_suffix = model_dirname[-2:]
    gen_datas_jsonl = Path(args.save_dir) / f"gen_datas_{model_suffix}.jsonl"
    # start_index = (
    #     len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
    # )
    start_index = get_last_index(gen_datas_jsonl)
    print(f"start_index: {start_index}")

    test_file = args.save_dir + "/rerank_test.json"
    datas = []
    with open(test_file, "r") as input_file:
        for line in input_file:
            datas.append(line)
    
    rec_datas = [json.loads(item) for item in datas]
    
    for i in tqdm(range(start_index, len(rec_datas), batch_size)):
        cur_gsm8k_batch = rec_datas[i : i + batch_size]
        input_str_list, output_str_list = gsm8k_batch_gen(
            cur_gsm8k_batch, batch_llama, args
        )
        for j, (gsm8k_data, input_str, output_str) in enumerate(
            zip(cur_gsm8k_batch, input_str_list, output_str_list)
        ):
            with open(gen_datas_jsonl, "a") as f:
                json.dump(
                    dict(
                        index=i + j,
                        source_data=gsm8k_data,
                        input_str=input_str,
                        output_str=output_str,
                    ),
                    f,
                )
                f.write("\n")

def gsm8k_batch_gen(
    cur_gsm8k_batch, batch_llm, args
):
    try:
        curs_gsm8k_questions = [v['prompt'] for v in cur_gsm8k_batch]
    except:
        curs_gsm8k_questions = [v['input_prompt'] for v in cur_gsm8k_batch]
    # prompt_no_input = PROMPT_DICTS['normal_prompt']
    input_str_list = [q for q in curs_gsm8k_questions]
    output_str_list = batch_llm(input_str_list)
    return input_str_list, output_str_list

def get_batch_llama(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, args):

    @torch.inference_mode()
    def batch_llama(input_strs):
        input_ids_w_attnmask = tokenizer(
            input_strs,
            padding=True,
            truncation=True,  # 요거 꼭 넣기
            max_length=1024, 
            return_tensors="pt",
        ).to(model.device)
        # ).to(dtype=model.dtype, device=model.device)
        
        with torch.cuda.amp.autocast(enabled=args.is_bf16):
            output_ids = model.generate(
                input_ids=input_ids_w_attnmask.input_ids,
                attention_mask=input_ids_w_attnmask.attention_mask,
                generation_config=GenerationConfig(
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                    temperature=0.0,  # t=0.0 raise error if do_sample=True
                    use_cache=False, 
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                ),

            ).tolist()
        torch.cuda.empty_cache()
        gc.collect()

        real_output_ids = [
            output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)
        ]
        output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True) # ###으로 끊는 과정 넣기

        # print("output_ids:", output_ids)
        # print("eos_token_id:", tokenizer.eos_token_id)
        # print("Decoded output:", tokenizer.decode(output_ids[0], skip_special_tokens=False))
        # print("real_output_ids:", real_output_ids)
        print("output_strs:", output_strs) # 코드 확인하기

        return output_strs

    return batch_llama

def get_model(model_path: str, is_bf16: bool = False, device=None):
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left") #, local_files_only=True
    print(tokenizer.pad_token)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    print('new pad ', tokenizer.pad_token)
    print(tokenizer.bos_token)
    print(tokenizer.unk_token)
    print(tokenizer.eos_token)
    print(tokenizer.eos_token_id)
    print(tokenizer.truncation_side)
    print(tokenizer.padding_side)

    if is_bf16:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            #  device_map='auto' #hw_edit
        )#.cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
        )#.cuda()
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    model.eval()
    print(model.dtype)

    return model, tokenizer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="which strategy to evaluate the model",
        required=True,
        choices=['Parallel','Cross']
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batchsize",
        required=True
    )
    parser.add_argument(
        "--lang_only",
        type=str,
        help="specific language to test",
        default = ''
    )
    parser.add_argument(
        "--shot",
        type=int,
        help="how many examples in your prompts",
        default=4
    )
    parser.add_argument(
        "--shuffle",
        type= bool,
        help="whether to shuffle your choices",
        default = True
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="maximum output tokens",
        default = 1024
    )
    parser.add_argument(
        "--is_bf16",
        action="store_true",
        help="Use bfloat16 inference"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed",
        default = 0
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="specific language to test",
        default = ''
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="file to store",
        default=""
    )

    args = parser.parse_args()
    main(args)
    # fire.Fire(main(args=args))