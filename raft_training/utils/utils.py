# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import torch
import random
import numpy as np
from transformers import set_seed, AutoTokenizer, LlamaTokenizer
import json
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import re
import string
import pandas as pd
from tqdm import tqdm

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean

def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "llama" in model_name_or_path:
        from transformers.models.llama import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        if tokenizer.pad_token is None:
            # assert tokenizer.eos_token is not None
            # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'right'
    return tokenizer


def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    # if os.path.exists(model_name_or_path):
    #     # Locally tokenizer loading has some issue, so we need to force download
    #     model_json = os.path.join(model_name_or_path, "config.json")
    #     if os.path.exists(model_json):
    #         model_json_file = json.load(open(model_json))
    #         model_name = model_json_file["_name_or_path"]
    #         # tokenizer = AutoTokenizer.from_pretrained(model_name,
    #         #                                           fast_tokenizer=True)
    #         tokenizer = LlamaTokenizer.from_pretrained(model_name,
    #                                                    padding_side = 'left',
    #                                               fast_tokenizer=True)
    #         print('i am loading here')
    # else:
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path,
                                                padding_side = 'left',
                                                fast_tokenizer=True)
   
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
    #                                               fast_tokenizer=True)
    return tokenizer


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


# def get_optimizer_grouped_parameters(model,
#                                      weight_decay,
#                                      no_decay_name_list=[
#                                          "bias", "LayerNorm.weight"
#                                      ]):
#     optimizer_grouped_parameters = [
#         {
#             "params": [
#                 p for n, p in model.named_parameters()
#                 if (not any(nd in n
#                             for nd in no_decay_name_list) and p.requires_grad)
#             ],
#             "weight_decay":
#             weight_decay,
#         },
#         {
#             "params": [
#                 p for n, p in model.named_parameters()
#                 if (any(nd in n
#                         for nd in no_decay_name_list) and p.requires_grad)
#             ],
#             "weight_decay":
#             0.0,
#         },
#     ]
#     return optimizer_grouped_parameters


## for dpo, ref: https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/dschat/utils/utils.py
def get_optimizer_grouped_parameters( model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower()
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups



def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, tokenizer, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    CONFIG_NAME = 'config.json'
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_dir, CONFIG_NAME)
    
    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            
            torch.save(output_state_dict, output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_pretrained(save_dir)
        del output_state_dict

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


# from utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer


# utils/eval_utils.py
def evaluate_model(model, eval_dataloader, device):
    model.eval()
    losses = 0
    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses += loss.float()

    losses = losses / (step + 1)
    perplexity = torch.exp(losses) if not torch.isnan(losses) else float("inf")

    # try:
    #     perplexity = get_all_reduce_mean(perplexity).item()
    #     loss = get_all_reduce_mean(losses).item()
    # except:
    #     loss = float("inf")

    # return loss, perplexity
    return losses.item(), perplexity.item() if isinstance(perplexity, torch.Tensor) else perplexity



        # try:
        #     perplexity = torch.exp(losses)
        # except OverflowError:
        #     perplexity = float("inf")

        # try:
        #     perplexity = get_all_reduce_mean(perplexity).item()
        # except Exception as e:
        #     print_rank_0(f"[WARNING] get_all_reduce_mean failed: {e}", args.global_rank)

        # try:
        #     loss = get_all_reduce_mean(losses).item()
        # except:
        #     loss = float("inf")
def evaluate_model_webqsp(model, eval_dataloader, tokenizer, device, save_path=None, max_samples=None):
    model.eval()
    eval_output = []

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        if max_samples and step >= max_samples:
            break

        batch = to_device(batch, device)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,
                temperature=0.0,
                use_cache=True,  # ← use_cache=True로 바꾸면 속도 더 빨라짐
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=False  # ← dict 변환 방지
            )

        for i in range(len(outputs)):
            # pred_str = tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            # pred_str = pred_str.replace('[PAD]', '').strip() # 새로 추가
            pred_ids = outputs[i].tolist()
            if tokenizer.eos_token_id in pred_ids:
                pred_ids = pred_ids[:pred_ids.index(tokenizer.eos_token_id)+1]  # eos까지 slice
            pred_str = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()

            # 필요할 때만 label decode 수행
            if save_path:
                label_str = tokenizer.decode(
                    [x for x in labels[i].tolist() if x >= 0],
                    skip_special_tokens=True
                ).strip()
                eval_output.append({"pred": pred_str, "label": label_str})
            else:
                label = [tokenizer.decode([x], skip_special_tokens=True).strip()
                         for x in labels[i].tolist() if x >= 0]
                pred = pred_str.replace("|", "\n").split("\n")

                f1, prec, recall = eval_f1(pred, label)
                acc = eval_acc(pred, label)
                hit = eval_hit(pred, label)

                acc_list.append(acc)
                hit_list.append(hit)
                f1_list.append(f1)
                precision_list.append(prec)
                recall_list.append(recall)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            for row in eval_output:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # metrics 계산 후 저장된 eval_output 기반 평가
        df = pd.DataFrame(eval_output)
        acc_list, hit_list, f1_list, precision_list, recall_list = [], [], [], [], []
        for _, row in df.iterrows():
            pred = row["pred"].replace("|", "\n").split("\n")
            label = row["label"].split("|")
            f1, prec, recall = eval_f1(pred, label)
            acc = eval_acc(pred, label)
            hit = eval_hit(pred, label)
            acc_list.append(acc)
            hit_list.append(hit)
            f1_list.append(f1)
            precision_list.append(prec)
            recall_list.append(recall)

    metrics = {
        "Accuracy": np.mean(acc_list) * 100,
        "Hit": np.mean(hit_list) * 100,
        "F1": np.mean(f1_list) * 100,
        "Precision": np.mean(precision_list) * 100,
        "Recall": np.mean(recall_list) * 100,
    }

    print("=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics

# def evaluate_model_webqsp(model, eval_dataloader, tokenizer, device, save_path=None):
#     model.eval()
#     eval_output = []

#     for batch in tqdm(eval_dataloader, desc="Evaluating"):
#         batch = to_device(batch, device)
#         input_ids = batch["input_ids"]
#         attention_mask = batch["attention_mask"]
#         labels = batch["labels"]

#         with torch.no_grad():
#             outputs = model.generate(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=64,
#                 do_sample=False,
#                 temperature=0.0,
#                 use_cache=False,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id,
#             )

#         for i in range(len(outputs)):
#             pred_str = tokenizer.decode(outputs[i], skip_special_tokens=True)
#             # label_str = tokenizer.decode(labels[i], skip_special_tokens=True)
#             label_str = tokenizer.decode([x for x in labels[i].tolist() if x >= 0], skip_special_tokens=True)

#             eval_output.append({
#                 "pred": pred_str.strip(),
#                 "label": label_str.strip()
#             })
#     for i in range(5):
#         print("Pred:", eval_output[i]["pred"])
#         print("Label:", eval_output[i]["label"])

#     # save to file
#     if save_path:
#         with open(save_path, "w") as f:
#             for row in eval_output:
#                 f.write(json.dumps(row, ensure_ascii=False) + "\n")

#     # compute metrics
#     df = pd.DataFrame(eval_output)
#     acc_list, hit_list, f1_list, precision_list, recall_list = [], [], [], [], []

#     for _, row in df.iterrows():
#         pred = row["pred"].replace("|", "\n").split("\n")
#         label = row["label"].split("|")

#         pred_str = " ".join(pred)

#         f1, prec, recall = eval_f1(pred, label)
#         acc = eval_acc(pred_str, label)
#         hit = eval_hit(pred_str, label)

#         f1_list.append(f1)
#         precision_list.append(prec)
#         recall_list.append(recall)
#         acc_list.append(acc)
#         hit_list.append(hit)

#     metrics = {
#         "Accuracy": sum(acc_list) * 100 / len(acc_list),
#         "Hit": sum(hit_list) * 100 / len(hit_list),
#         "F1": sum(f1_list) * 100 / len(f1_list),
#         "Precision": sum(precision_list) * 100 / len(precision_list),
#         "Recall": sum(recall_list) * 100 / len(recall_list),
#     }

#     print("=== Evaluation Results ===")
#     for k, v in metrics.items():
#         print(f"{k}: {v:.4f}")

#     return metrics  # 혹은 원하는 하나만 return

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

# def match(s1: str, s2: str) -> bool:
#     s1 = normalize(s1)
#     s2 = normalize(s2)
#     return s2 in s1


def match(s1, s2) -> bool:
    def to_str(x):
        if isinstance(x, list):
            return " ".join(x)
        return str(x)
    s1 = normalize(to_str(s1))
    s2 = normalize(to_str(s2))
    return s2 in s1

def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = " ".join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def eval_acc(prediction, answer):
    matched = 0.0
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0
