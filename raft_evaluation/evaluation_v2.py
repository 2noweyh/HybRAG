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
    """
    Checks if two strings match based on several flexible criteria.
    s1: prediction, s2: answer
    """
    s1 = normalize(s1)
    s2 = normalize(s2)

    # 1. 띄어쓰기 차이 해결 ("MeanTime" vs "Mean Time")
    # 공백을 모두 제거한 후 문자열이 완전히 일치하는지 확인합니다.
    if s1.replace(" ", "") == s2.replace(" ", ""):
        return True

    # # 2. 단어(토큰) 기반 F1 점수로 유사도 측정
    # prediction_tokens = set(s1.split())
    # answer_tokens = set(s2.split())

    # if not prediction_tokens or not answer_tokens:
    #     return False

    # common_tokens = prediction_tokens.intersection(answer_tokens)
    
    # # 공통 단어가 없으면 더 이상 계산하지 않음
    # if not common_tokens:
    #     return False

    # precision = len(common_tokens) / len(prediction_tokens)
    # recall = len(common_tokens) / len(answer_tokens)
    
    # f1 = 2 * precision * recall / (precision + recall)

    # # F1 점수가 0.5 이상이면 의미적으로 유사하다고 판단 (임계값)
    # # "Youngs Memorial Park" vs "Youngs Memorial Cemetery" 같은 경우 처리
    # if f1 >= 0.5:
    #     return True
        
    # 3. 기존의 포함 관계 확인 (Fallback)
    # "Delaware" vs "University of Delaware" 같은 경우 처리
    if s1 in s2 or s2 in s1:
        return True

    return False

# def match(s1: str, s2: str) -> bool:
#     s1 = normalize(s1)
#     s2 = normalize(s2)
#     return s2 in s1

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
        # 앞의 5건만 출력
        if i < 5:
            print(f"[{i+1}]")
            print(f"Prediction: {pred.strip()}")
            print(f"Label: {label.strip()}")
            print("-" * 50)

    # 계산된 결과 정리
    print(f"총 샘플 수: {len(acc_list)}")  # = 1600
    metrics = {
        "Accuracy": np.mean(acc_list) * 100,
        "Hit": np.mean(hit_list) * 100,
        "F1": np.mean(f1_list) * 100,
        "Precision": np.mean(precision_list) * 100,
        "Recall": np.mean(recall_list) * 100,
    }

    # 결과 출력
    print(f"==== Evaluation results on {args.dataset} ====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 결과 저장
    if args.save_path:
        with open(args.save_path, "w", encoding="utf-8") as f:
            for row in eval_output:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # metrics 저장
    with open(args.tb_logdir, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "metrics": metrics
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
