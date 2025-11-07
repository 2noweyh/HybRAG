import argparse
import json
import pickle
import os
from utils import load_pickle, load_data
from model import TextModel
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict
from collections import Counter
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer
# 가장 기본 방법 + 선택적 정보 입력 추가
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal --mode both --split trn
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal --mode both --split tst
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal --mode both --split val
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal --mode dense --split trn
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal --mode dense --split tst
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset webqsp --version vfinal --mode dense --split val
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v9.py --dataset cwq --version vfinal --mode both --split trn --few_shot


# {
#     "description": "Template used by Alpaca-LoRA.",
#     "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Examples:\n{options}\n\n### Response:\n",
#     "prompt_no_options": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
#     "response_split": "### Response:"    
# }


def parse_args():
    parser = argparse.ArgumentParser(description='Flatten retrieval results')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use (e.g., yelp)')
    parser.add_argument('--split', type=str, required=True, help='Data split to use (e.g., trn)')
    parser.add_argument('--version', type=str, required=True, help='Data version to use (e.g., v1)')
    parser.add_argument('--text_encoder', default='SentenceBert', type=str, choices=['Bert', 'Roberta', 'SentenceBert', 'SimCSE', 'e5', 't5'],
                        help='Text encoder to use (Bert, Roberta, SentenceBert, SimCSE, e5, or t5)')
    parser.add_argument('--base_model', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--mode',  default='both', type=str, choices=['both', 'path', 'dense'])
    parser.add_argument('--max_token_limit', type=int, default=2048)
    parser.add_argument('--few_shot', action="store_true", help='Use few-shot examples (default: enabled)')
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    return parser.parse_args()

def check_answer_coverage(dic_graph, flattened_dense, flattened_path):
    """
    각 질문별로 dense/path retrieval 결과에서 정답이 포함되어 있는지 여부를 체크한다.
    """
    coverage_results = []

    for i in range(len(dic_graph["query_id"])):
        qid = dic_graph["query_id"][i]
        question = dic_graph["query"][i]
        gold_answers = dic_graph["tgt_id"][i]
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        dense_text = flattened_dense.get(qid, "")
        path_text = flattened_path.get(qid, "")

        dense_hit = any(ans.lower() in dense_text.lower() for ans in gold_answers)
        path_hit = any(ans.lower() in path_text.lower() for ans in gold_answers)

        coverage_results.append({
            "id": qid,
            "question": question,
            "answers": gold_answers,
            "dense_hit": dense_hit,
            "path_hit": path_hit,
            "dense_context": dense_text[:300],  # 너무 길면 앞부분만 기록
            "path_context": path_text[:300]
        })
        total = len(coverage_results)
        dense_hits = sum(1 for x in coverage_results if x["dense_hit"])
        path_hits = sum(1 for x in coverage_results if x["path_hit"])
    print(f"Dense coverage: {dense_hits}/{total} ({dense_hits/total:.2%})")
    print(f"Path coverage: {path_hits}/{total} ({path_hits/total:.2%})")

    return coverage_results


def load_rel_prefix_group_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def batched_encode(text_model, texts, device, batch_size=16):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_output = text_model(batch).to(device)
            embs.append(batch_output.cpu())
    return torch.cat(embs, dim=0)

def resolve_relation_text(matched_rels, rel_text, question, group_to_prefixes, model, device):
    matched_set = set(matched_rels)
    if len(matched_set) == 1:
        return False, list(matched_set)[0]

    matched_rels2 = [rel for rel in matched_set if rel.split(".")[0] in group_to_prefixes[rel_text]]
    
    if len(set(matched_rels2)) == 1:
        return True, matched_rels2[0]

    if not matched_rels2:
        matched_rels2 = list(matched_set)

    with torch.no_grad():
        rel_embeds = batched_encode(model, matched_rels2, device=device)
    sims = F.cosine_similarity(rel_embeds, question.unsqueeze(0), dim=1)
    return True, matched_rels2[torch.argmax(sims).item()]    

def write_to_file(file_name, data):
    """
    Write data to a JSON file, creating directories if they don't exist.
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'a', encoding='utf-8') as f_converted:
        json_str = json.dumps(data)
        f_converted.write(json_str + '\n')

# def flatten_dense_retrieval_results(dense_retrieval_results, max_lines=20):
#     flattened_results = {}
#     for qid, subgraph_text in dense_retrieval_results.items():
#         triples = [t.strip() for t in subgraph_text.split("\n") if t.strip()]
#         # triples = triples[:max_lines]
#         flattened_results[qid] = "\n".join(triples)
#     return flattened_results

def flatten_dense_retrieval_results(dense_retrieval_results, top_k=5, bottom_k=10):
    flattened_results = {}
    for qid, subgraph_text in dense_retrieval_results.items():
        triples = [t.strip() for t in subgraph_text.split("\n") if t.strip()]
        
        # 상위 5개 + 하위 10개 선택
        selected = triples[:top_k] + triples[-bottom_k:] if len(triples) > (top_k + bottom_k) else triples
        
        flattened_results[qid] = "\n".join(selected)
    return flattened_results

# def flatten_pagelink_retrieval_results(args, pred_edge_to_paths, max_paths=3):
#     flattened_results = defaultdict(list)

#     for (qid, src_info, tgt_info), paths in pred_edge_to_paths.items():
#         src_id, tgt_id = int(src_info[1]), int(tgt_info[1])
#         path_strs = []
#         for path in paths:
#             if not path:
#                 continue
#             segs = []
#             for i, (u_text, rel_text, v_text) in enumerate(path):
#                 if i == 0:
#                     segs.append(f"{u_text} -> {rel_text} -> {v_text}")
#                 else:
#                     segs.append(f"{rel_text} -> {v_text}")
#             full_path = " -> ".join(segs)
#             if full_path.strip():
#                 path_strs.append(full_path)

#         if path_strs:
#             # 중복 제거
#             path_strs = list(dict.fromkeys(path_strs))
#             flattened_results[(qid, tgt_id)].extend(path_strs)

#     finalized_results = {}
#     for (qid, tgt_id), path_list in flattened_results.items():
#         if not path_list:
#             continue
#         if qid not in finalized_results:
#             finalized_results[qid] = []

#         # 질문당 최대 max_paths 개만 유지
#         finalized_results[qid].extend(path_list[:max_paths])

#     # 한 줄에 여러 개 있으면 합쳐서 문자열로 변환
#     for qid, paths in finalized_results.items():
#         finalized_results[qid] = "\n".join(paths)

#     return finalized_results


import json

def flatten_gnn_retriever_result(file_path: str) -> dict:
    """
    .jsonl 파일에서 각 ID별로 'Reasoning Paths' 내용을 추출합니다.

    Args:
        file_path (str): 처리할 .jsonl 파일의 경로.

    Returns:
        dict: 키는 'id', 값은 추출된 'Reasoning Paths' 문자열인 딕셔너리.
    """
    retrieved_paths = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 각 줄을 JSON 객체로 파싱
                    data = json.loads(line)
                    qid = data.get("id")
                    input_text = data.get("input")

                    # id나 input 필드가 없는 경우 건너뛰기
                    if not qid or not input_text:
                        continue

                    # 시작과 끝 구분자 정의
                    start_delimiter = "Reasoning Paths:\n"
                    end_delimiter = "\n\nQuestion:"

                    # 문자열에서 시작 위치 찾기
                    start_index = input_text.find(start_delimiter)
                    
                    # 시작 구분자를 찾았을 경우에만 계속 진행
                    if start_index != -1:
                        # 실제 내용이 시작되는 위치로 인덱스 조정
                        start_index += len(start_delimiter)
                        
                        # 내용 시작 위치부터 끝 구분자 찾기
                        end_index = input_text.find(end_delimiter, start_index)
                        
                        if end_index != -1:
                            # 시작과 끝 인덱스 사이의 텍스트를 추출하고 공백 제거
                            reasoning_text = input_text[start_index:end_index].strip()
                            retrieved_paths[qid] = reasoning_text

                except json.JSONDecodeError:
                    print(f"경고: JSON 파싱 오류가 발생한 라인을 건너뜁니다: {line.strip()}")
                    continue
                    
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 - {file_path}")
        return {}

    return retrieved_paths

def flatten_pagelink_retrieval_results(args, pred_edge_to_paths, max_paths=10):
    """
    PaGE-Link retrieval 결과를 평탄화.
    각 질문(qid)마다 reasoning path를 최대 max_paths 개까지만 유지.
    """
    flattened_results = defaultdict(list)

    for (qid, src_info, tgt_info), paths in pred_edge_to_paths.items():
        src_id, tgt_id = int(src_info[1]), int(tgt_info[1])
        path_strs = []
        for path in paths:
            if not path:
                continue
            segs = []
            for i, (u_text, rel_text, v_text) in enumerate(path):
                if i == 0:
                    segs.append(f"{u_text} -> {rel_text} -> {v_text}")
                else:
                    segs.append(f"{rel_text} -> {v_text}")
            full_path = " -> ".join(segs)
            if full_path.strip():
                path_strs.append(full_path)

        if path_strs:
            # 중복 제거
            path_strs = list(dict.fromkeys(path_strs))
            # qid + target 기준으로 모으기
            flattened_results[(qid, tgt_id)].extend(path_strs)

    finalized_results = {}
    for (qid, tgt_id), path_list in flattened_results.items():
        if not path_list:
            continue
        if qid not in finalized_results:
            finalized_results[qid] = []

        # 질문당 reasoning path 개수 제한
        limited_paths = path_list[:max_paths]
        finalized_results[qid].extend(limited_paths)

    # 한 줄에 여러 개 있으면 합쳐서 문자열로 변환
    for qid, paths in finalized_results.items():
        finalized_results[qid] = "\n".join(paths)

    return finalized_results

# def merge_flattened_results(flattened_dense, flattened_pagelink, mode="both"):
#     merged_results = {}
#     for key, dense in flattened_dense.items():
#         final_lines = []

#         if mode in ("both", "dense") and dense.strip():
#             final_lines.append(dense.strip())

#         if mode in ("both", "path"):
#             pagelink_text = flattened_pagelink.get(key, "").strip()
#             if pagelink_text:
#                 final_lines.append(pagelink_text)

#         merged_results[key] = "\n".join(final_lines).strip()
#     return merged_results

def merge_flattened_results(flattened_dense, flattened_pagelink, mode="both"):
    merged_results = {}
    for key, dense in flattened_dense.items():
        final_lines = []

        # dense retrieval 결과
        if mode in ("both", "dense") and dense.strip():
            final_lines.append("### Subgraph around relevant entities:\n" + dense.strip())

        # path retrieval 결과
        if mode in ("both", "path"):
            pagelink_text = flattened_pagelink.get(key, "").strip()
            if pagelink_text:
                final_lines.append("### Reasoning paths supporting the answer(s):\n" + pagelink_text)

        # 최종 합치기
        if final_lines:
            merged_text = "The following context was retrieved to explain the question-answer connections:\n\n"
            merged_text += "\n\n".join(final_lines)
            merged_results[key] = merged_text.strip()
        else:
            merged_results[key] = ""
    return merged_results


# def sample_generation(args, merged_results, dic_graph):
#     all_samples = []

#     few_shot_examples = [
#         {
#             "question": "what was walt disney 's first cartoon called",
#             "context": (
#                 "The Walt Disney Company -> organization.organization.founders -> Walt Disney\n"
#                 "The Walt Disney Company -> symbols.namesake.named_after -> Walt Disney"
#             ),
#             "answer": "Walt Disney's Wonderful World of Color"
#         },
#         {
#             "question": "who was joseph pulitzer and what did he do",
#             "context": (
#                 "Joseph Pulitzer -> people.person.profession -> Journalist\n"
#                 "Joseph Pulitzer -> symbols.name_source.namesakes -> Pulitzer Prize"
#             ),
#             "answer": "Lawyer\nJournalist\nPolitician\nPublisher"
#         }
#     ]

#     for i in range(len(dic_graph["query_id"])):
#         qid = dic_graph["query_id"][i]
#         if qid not in merged_results:
#             continue

#         question = dic_graph["query"][i]
#         context_text = merged_results[qid]
#         output = dic_graph["tgt_id"][i]
#         if isinstance(output, list):
#             output = ", ".join(output)  # 쉼표 대신 줄바꿈             공백

#         example_block = ""
#         if args.few_shot:
#             for ex in few_shot_examples:
#                 example_block += (
#                     f"### Question:\n{ex['question']}\n\n"
#                     f"### Knowledge Graph Context:\n{ex['context']}\n\n"
#                     f"### Response:\n{ex['answer']}\n\n"
#                 )

#         instruction = (
#             # "Answer the question using the knowledge graph context as hints. "
#             # "Answer the question"
#             # "Given a question and its knowledge graph context, extract all correct answer entities.\n"
#             # "Respond with the entity names only, separated by commas if there are multiple."
#             "You are given a question and a knowledge graph context.\n"
#             "Extract the correct answer strictly from the given context.\n"
#             "Return only the answer entity/entities and do not include explanations or extra text."
#         )

#         # prompt_input = f"### Question:\n{question}\n\n### Knowledge Graph Context:\n{context_text}"
#         prompt_input = (
#                         f"### Question:\n{question}\n\n"
#                         f"### Knowledge Graph Context:\n{context_text}\n\n"
#                         # f"### Response:"
#                     )
#         # prompt_input = f"### Question:\n{question}\n\n"

#         sample = {
#             "id": qid,
#             "instruction": instruction,
#             "input": prompt_input,
#             "output": output,
#             "options": example_block if args.few_shot else None,
#         }
#         all_samples.append(sample)

#     return all_samples


def sample_generation(args, merged_results, dic_graph):
    all_samples = []
   
    for i in range(len(dic_graph["query_id"])):
        qid = dic_graph["query_id"][i]
        if qid not in merged_results:
            continue

        question = dic_graph["query"][i]
        context_text = merged_results[qid]
        outputs = dic_graph["tgt_id"][i]

        # 정답이 리스트면 ','로 합치기 (LLaMA friendly)
        if isinstance(outputs, list):
            # output_text = ", ".join([str(o) for o in outputs])
            output_text = ", ".join(outputs)
        else:
            output_text = str(outputs)

        instruction = (
            "Based on the reasoning paths, please answer the given question. "
            "Please keep the answer as simple as possible and return all the possible answers as a list."
        )

        prompt_input = (
            # f"[INST] <<SYS>>\n<</SYS>>\n"
            # f"{instruction}\n\n"
            f"Reasoning Paths:\n{context_text}\n\n"
            f"Question:\n{question}"
        )

        sample = {
            "id": qid,
            "instruction": instruction,
            "input": prompt_input,
            "output": output_text,  # 이제 항상 콤마로 구분된 문자열
            "options": None,
        }
        all_samples.append(sample)

    return all_samples



def rerank_flattened_results(args, all_samples, write_file, device):
    if args.split == "tst":
        for sample in all_samples:
            write_to_file(write_file, sample)
        return

    model = TextModel(args.text_encoder).to(device)
    similarities = []

    for sample in tqdm(all_samples, desc="Reranking by context relevance"):
        with torch.no_grad():
            # 질문/컨텍스트 분리
            if "### Knowledge Graph Context:" in sample['input']:
                q_text, c_text = sample['input'].split("### Knowledge Graph Context:")
            else:
                q_text, c_text = sample['input'], ""

            q_emb = model([q_text.strip()]).cpu().numpy()
            c_emb = model([c_text.strip()]).cpu().numpy()

            # 무조건 2차원으로 reshape
            q_emb = q_emb.reshape(1, -1)
            c_emb = c_emb.reshape(1, -1)

        sim = cosine_similarity(q_emb, c_emb)[0][0]
        sample['similarity_score'] = float(sim)
        similarities.append(sample)

    sorted_samples = sorted(similarities, key=lambda x: -x['similarity_score'])
    for sample in sorted_samples:
        write_to_file(write_file, sample)

def main(): 
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("few_shot example: ", args.few_shot)
    
    # Define mappings
    split_map = {"trn": "train", "val": "eval", "tst": "test"}
    
    # load dataset
    path_file = f"./saved_explanations/distmult/pagelink_{args.dataset}_{args.split}_inductive_paths.pkl"
    dense_file = f"./data/{args.dataset}/dense_retrieval_results_{args.split}_v1.json"
    dic_graph_file = f"./data/{args.dataset}/{args.dataset}_dic_graph_v1.pt"


    with open(path_file, "rb") as f:
        pagelink_retrieval_results = pickle.load(f)
    with open(dense_file, "r") as f:
        dense_retrieval_results = json.load(f)
    dic_graph = torch.load(dic_graph_file)[args.split]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)  # args에 base_model 추가 필요
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
        
    flattened_pagelink_retrieval_results = flatten_pagelink_retrieval_results(args, pagelink_retrieval_results)
    flattened_dense_retrieval_results = flatten_dense_retrieval_results(dense_retrieval_results)
    print('dense_retriever: ',len(flattened_dense_retrieval_results), 'pagelink_retriever: ',len(flattened_pagelink_retrieval_results))
    merged_results = merge_flattened_results(
        flattened_dense_retrieval_results,
        flattened_pagelink_retrieval_results,
        args.mode
    )

    # === Coverage check ===
    coverage_results = check_answer_coverage(
        dic_graph,
        flattened_dense_retrieval_results,
        flattened_pagelink_retrieval_results
    )
    coverage_file = f"./data/{args.dataset}/coverage_{split_map[args.split]}_{args.version}.json"
    with open(coverage_file, "w", encoding="utf-8") as f:
        json.dump(coverage_results, f, indent=2, ensure_ascii=False)
    print(f"Coverage results saved to {coverage_file}")

    all_samples = sample_generation(args, merged_results, dic_graph)
    
    
    write_file = f"./data/{args.dataset}/rerank_{split_map[args.split]}_{args.version}_{args.mode}.json"
    rerank_flattened_results(args, all_samples, write_file, device)

if __name__ == "__main__":
    main()



'''

### Instruction:
You are given a question and a knowledge graph context.
Extract the correct answer strictly from the given context.
Return only the answer entity/entities (one per line) and do not include explanations or extra text.

### Input:
### Question:
what is the zip code for morgantown in

### Knowledge Graph Context:
The following context was retrieved to explain the question-answer connections:


### Subgraph around relevant entities:

Morgantown -> location.hud_county_place.place -> Morgantown
Morgantown -> location.hud_county_place.county -> Monongalia County
Morgantown -> location.location.containedby -> Monongalia County
Morgantown -> common.topic.notable_types -> City/Town/Village
Morgantown -> location.location.containedby -> United States of America
Morgantown -> base.biblioness.bibs_location.country -> United States of America
Morgantown -> location.location.containedby -> West Virginia
Morgantown -> base.biblioness.bibs_location.state -> West Virginia
Morgantown -> location.location.contains -> West Virginia University
Morgantown -> location.location.contains -> West Virginia Junior College, Morgantown
Morgantown -> location.location.contains -> West Virginia University College of Law
Morgantown -> location.location.contains -> Downtown Morgantown Historic District
Morgantown -> location.location.people_born_here -> Don Knotts
Morgantown -> location.location.contains -> Morgantown Beauty College Inc
Morgantown -> sports.sports_team_location.teams -> West Virginia Mountaineers men's basketball
Morgantown -> base.wikipedia_infobox.settlement.area_code -> Area codes 304 and 681
Morgantown -> location.location.people_born_here -> Robert P. George
Morgantown -> location.location.contains -> Morgantown Wharf and Warehouse Historic District
Morgantown -> location.location.contains -> Monongalia County Technical Education Center
Morgantown -> location.location.nearby_airports -> Morgantown Municipal Airport
Morgantown -> location.citytown.postal_codes -> 26508
Morgantown -> location.citytown.postal_codes -> 26507
Morgantown -> location.citytown.postal_codes -> 26505
Morgantown -> location.citytown.postal_codes -> 26506
Morgantown -> location.citytown.postal_codes -> 26504
Morgantown -> location.citytown.postal_codes -> 26502
Morgantown -> location.citytown.postal_codes -> 26501
Morgantown -> location.statistical_region.population -> m.066gmfd
Morgantown -> location.statistical_region.population -> m.066gmf0
Morgantown -> location.statistical_region.population -> m.066gmdv

### Reasoning paths supporting the answer(s):
Morgantown -> location.location.events -> 26504

### Response:
'''