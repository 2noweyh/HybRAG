import argparse
import json
import os
import re
import string
import numpy as np
from tqdm import tqdm

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
    """Checks if normalized s2 is a substring of normalized s1."""
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1

# [수정] 함수 시그니처 및 로직 변경: 개별 예측/정답 아이템을 비교합니다.
def eval_acc(prediction_list, answer_list):
    """
    정답 리스트의 각 아이템이 예측 리스트에 포함되는지 확인하여 정확도를 계산합니다.
    (얼마나 많은 정답을 맞췄는가)
    """
    if not answer_list:
        return 1.0 if not prediction_list else 0.0
    if not prediction_list:
        return 0.0
        
    matched = 0
    for a in answer_list:
        # 하나의 정답(a)이라도 어떤 예측(p)과 일치하면 matched로 간주
        for p in prediction_list:
            if match(p, a):
                matched += 1
                break # 다음 정답으로 넘어감
    return matched / len(answer_list)

# [수정] 함수 시그니처 및 로직 변경: 개별 예측/정답 아이템을 비교합니다.
def eval_hit(prediction_list, answer_list):
    """
    정답 리스트의 아이템 중 하나라도 예측 리스트에 포함되면 1을 반환합니다.
    """
    if not prediction_list or not answer_list:
        return 0
    
    for a in answer_list:
        for p in prediction_list:
            if match(p, a):
                return 1
    return 0

# [수정] 안전성 강화: 예측 리스트가 비어있을 경우를 대비합니다.
def eval_hit1(prediction_list, answer_list):
    """
    가장 첫번째 예측이 정답 리스트의 아이템 중 하나라도 포함하면 1을 반환합니다.
    """
    if not prediction_list or not answer_list:
        return 0
    
    for a in answer_list:
        if match(prediction_list[0], a):
            return 1
    return 0

# [수정] F1 계산 로직을 더 표준적인 방식으로 변경합니다.
def eval_f1(prediction_list, answer_list):
    """
    예측과 정답 리스트 간의 F1 스코어를 계산합니다.
    """
    if not prediction_list or not answer_list:
        return 0, 0, 0
    
    # 예측된 아이템 중 정답과 매치되는 것들의 수 (True Positives)
    true_positives = 0
    # 중복 카운팅을 방지하기 위해 매치된 정답을 기록
    matched_answers = set()
    for p in prediction_list:
        for a in answer_list:
            if a not in matched_answers and match(p, a):
                true_positives += 1
                matched_answers.add(a)
                break
    
    precision = true_positives / len(prediction_list)
    recall = true_positives / len(answer_list)
    
    if precision + recall == 0:
        return 0, precision, recall
    else:
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall


def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]


def eval_result(predict_file, save_path=None, tb_logdir=None, cal_f1=True, top_k=-1):
    if not predict_file:
        print("Error: Input path is not provided.")
        return

    with open(predict_file, 'r', encoding='utf-8') as f:
        # 각 줄이 JSON 객체이므로 json.load가 아닌 한 줄씩 파싱
        lines = [json.loads(line) for line in f]

    acc_list, hit1_list, hit_list, f1_list, precision_list, recall_list = [], [], [], [], [], []
    
    for data in tqdm(lines, desc="Evaluating"):
        prediction_str_raw = data.get('prediction', '')
        answer_str_raw = data.get('output', '')

        # 쉼표로 분리하고 양쪽 공백 제거
        prediction_list = [p.strip() for p in prediction_str_raw.split(',') if p.strip()]
        answer_list = [a.strip() for a in answer_str_raw.split(',') if a.strip()]

        if top_k > 0:
            prediction_list = extract_topk_prediction(prediction_list, top_k)
        
        # [수정] 변경된 함수에 맞게 prediction_list를 직접 전달
        hit = eval_hit(prediction_list, answer_list)
        acc = eval_acc(prediction_list, answer_list)
        hit1 = eval_hit1(prediction_list, answer_list)
        
        hit_list.append(hit)
        acc_list.append(acc)
        hit1_list.append(hit1)
        
        if cal_f1:
            f1_score, precision, recall = eval_f1(prediction_list, answer_list)
            f1_list.append(f1_score)
            precision_list.append(precision)
            recall_list.append(recall)

    result_metrics = {
        "Accuracy": np.mean(acc_list) * 100,
        "Hit": np.mean(hit_list) * 100,
        "Hit@1": np.mean(hit1_list) * 100,
    }
    if cal_f1:
        result_metrics["F1"] = np.mean(f1_list) * 100
        result_metrics["Precision"] = np.mean(precision_list) * 100
        result_metrics["Recall"] = np.mean(recall_list) * 100

    print("\n==== Evaluation Results ====")
    for k, v in result_metrics.items():
        print(f"{k}: {v:.4f}")

    if save_path:
        # 결과 파일 저장 시에는 평가 메트릭을 함께 저장하면 더 유용할 수 있습니다.
        # 여기서는 원본 데이터를 그대로 저장합니다.
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(lines, f, indent=2, ensure_ascii=False)
        print(f"\nOriginal predictions saved to: {save_path}")

    if tb_logdir:
        with open(tb_logdir, "w", encoding="utf-8") as f:
            json.dump(result_metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {tb_logdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="webqsp")
    # input_path의 기본값을 None에서 실제 파일 경로로 바꾸거나, 실행 시 꼭 지정해야 합니다.
    parser.add_argument("--input_path", type=str, required=True, help="Path to the prediction file (jsonl format)")
    parser.add_argument("--tb_logdir", type=str, default="./runs/eval.json")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    
    eval_result(args.input_path, args.save_path, args.tb_logdir, cal_f1=True, top_k=-1)