import os
import json
import string
import re
import numpy as np
from tqdm import tqdm
import torch
import argparse
import pandas as pd


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s

def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="webqsp")
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--tb_logdir", type=str, default="./runs/eval")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    input_path = args.input_path # or f"./data/{args.dataset}/gen_datas.jsonl"
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f.readlines()]

    preds = [l["prediction"] for l in lines]
    refs = [l["output"] for l in lines]  

    eval_output = []

    acc_list, hit_list, f1_list, precision_list, recall_list = [], [], [], [], []

    for i, (pred, label) in enumerate(zip(preds, refs)):
        pred = pred.replace(' ->', ',')
        
        pred_list = [p.strip() for p in pred.split(',') if p.strip()]
        label_list = [a.strip() for a in label.split(',') if a.strip()]

        pred_str = " ".join(pred_list)

        f1, prec, recall = eval_f1(pred_list, label_list)
        acc = eval_acc(pred_str, label_list)
        hit = eval_hit(pred_str, label_list)

        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(recall)
        acc_list.append(acc)
        hit_list.append(hit)

        eval_output.append({
            "pred": pred,
            "label": label,
        })
        if i < 5:
            print(f"[{i+1}]")
            print(f"Prediction: {pred.strip()}")
            print(f"Label: {label.strip()}")
            print("-" * 50)

    print(f"Total Smaple: {len(acc_list)}")  # = 1600
    metrics = {
        "Accuracy": np.mean(acc_list) * 100,
        "Hit": np.mean(hit_list) * 100,
        "F1": np.mean(f1_list) * 100,
        "Precision": np.mean(precision_list) * 100,
        "Recall": np.mean(recall_list) * 100,
    }

    print(f"==== Evaluation results on {args.dataset} ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if args.save_path:
        with open(args.save_path, "w", encoding="utf-8") as f:
            for row in eval_output:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.tb_logdir, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "metrics": metrics
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
