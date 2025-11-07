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
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v5.py --dataset webqsp --version v5 --mode both --split trn --few_shot
# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning_v5.py --dataset cwq --version v5 --mode both --split trn --few_shot

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

# 결과 생성
def flatten_pagelink_retrieval_results(args, pred_edge_to_paths):  
    flattened_results = defaultdict(list)
    not_found = 0

    for (qid, src_info, tgt_info), paths in pred_edge_to_paths.items():
        src_id = int(src_info[1])
        tgt_id = int(tgt_info[1])

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

        if not path_strs:
            not_found += 1
            print(f"❗ path_strs 비어있음 for qid={qid}, src_id={src_id}, tgt_id={tgt_id}")
            continue

        path_strs = list(dict.fromkeys(path_strs))
        flattened_results[(qid, tgt_id)].extend(path_strs)

    print("unmatched paths :", not_found)

    finalized_results = {}
    for (qid, tgt_id), path_list in flattened_results.items():
        clean_paths = [p for p in path_list if p and p.strip()]
        if not clean_paths:
            continue

        if qid not in finalized_results:
            finalized_results[qid] = []

        section = "\n".join(clean_paths).strip()
        if section:
            finalized_results[qid].append(section)

    for qid in list(finalized_results.keys()):
        prompt = "\n\n".join(finalized_results[qid]).strip()
        finalized_results[qid] = prompt

    print('original len', len(pred_edge_to_paths))
    print('final len', len(finalized_results))

    return finalized_results

def flatten_dense_retrieval_results(dense_retrieval_results):
    flattened_results = {}

    for qid, subgraph_text in dense_retrieval_results.items():
        flattened_results[qid] = subgraph_text.strip()

    return flattened_results

def merge_flattened_results(flattened_dense, flattened_pagelink, mode="both"):
    """
    mode:
      - "both"  : dense + path 정보 모두 포함
      - "dense" : dense 정보만 포함
      - "path"  : path 정보만 포함
    """
    merged_results = {}

    for key, dense in flattened_dense.items():
        final_lines = []

        header = "The following context was retrieved to explain the question-answer connections:\n\n"
        final_lines.append(header)

        if mode in ("both", "dense"):
            dense_header = "### Subgraph around relevant entities:\n"
            dense_lines = dense.strip().split("\n")
            final_lines.append(dense_header)
            final_lines.extend(dense_lines)

        if mode in ("both", "path"):
            pagelink_text = flattened_pagelink.get(key, "").strip()
            if pagelink_text:
                pagelink_lines = pagelink_text.split("\n")
                final_lines.append("")
                final_lines.append("### Reasoning paths supporting the answer(s):")
                final_lines.extend(pagelink_lines)

        merged_results[key] = "\n".join(final_lines)

    print(f"[merge_flattened_results] Total merged: {len(merged_results)}")
    return merged_results

def write_to_file(file_name, data):
    """
    Write data to a JSON file, creating directories if they don't exist.
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'a', encoding='utf-8') as f_converted:
        json_str = json.dumps(data)
        f_converted.write(json_str + '\n')

def sample_generation(args, merged_results, dic_graph):
    all_samples = []

    few_shot_examples = [
        {
            "question": "what was walt disney 's first cartoon called",
            "context": (
                "### Subgraph around relevant entities:\n"
                "The Walt Disney Company -> symbols.namesake.named_after -> Walt Disney\n"
                "The Walt Disney Company -> organization.organization.founders -> Walt Disney"
            ),
            "answer": "Walt Disney's Wonderful World of Color"
        },
        {
            "question": "who was joseph pulitzer and what did he do",
            "context": (
                "### Subgraph around relevant entities:\n"
                "Joseph Pulitzer -> people.person.profession -> Journalist\n"
                "Joseph Pulitzer -> symbols.name_source.namesakes -> Pulitzer Prize"
            ),
            "answer": "Lawyer, Journalist, Politician, Publisher"
        }
    ]

    # for qid, context_text in merged_results.items():
    for i in range(len(dic_graph["query_id"])):
        qid = dic_graph["query_id"][i]
        if qid not in merged_results:
            continue
        question = dic_graph["query"][i]
        context_text = merged_results[qid]
        output = dic_graph["tgt_id"][i]
        if not isinstance(output, str):
            output = ", ".join(output)

        example_block = ""
        if args.few_shot:
            for ex in few_shot_examples:
                example_block += (
                    "### Question:\n" + ex["question"] + "\n\n"
                    "### Knowledge Graph Context:\n" + ex["context"] + "\n\n"
                    "### Response:\n" + ex["answer"] + "\n\n"
                )

        # 본 문제 구성
        # instruction = "You are given a question and its knowledge graph context.\nBased on the context, extract the correct answer entities."
        instruction = "You are given a question and a knowledge graph context.\nExtract the correct answer strictly from the given context.\nReturn only the answer entity/entities (one per line) and do not include explanations or extra text."
        # instruction = "Questions are asked.\nExtract the correct answer from the given context.\nAnswer Only return entity/entity (one per line) and do not include comments or additional text."
        # instruction = "Questions are asked."
        prompt_input = (
            f"### Question:\n{question}\n\n"
            f"### Knowledge Graph Context:\n{context_text}"
        )

        sample = {
            "id": qid,
            "instruction": instruction,
            "input": prompt_input,
            "output": output,
            "options": example_block if args.few_shot else None,
        }

        all_samples.append(sample)

    print(f"[sample_generation] Generated {len(all_samples)} samples.")
    return all_samples

def rerank_flattened_results(args, all_samples, write_file, device):
    if args.split == "tst":
        for sample in all_samples:
            write_to_file(write_file, sample)
        return

    model = TextModel(args.text_encoder)
    model = model.to(device)

    batch_size = 256 # 64정도로 조절
    similarities = []
    original_order = {sample['id']: i for i, sample in enumerate(all_samples)}

    for i in tqdm(range(0, len(all_samples), batch_size), desc="Reranking samples"):
        batch = all_samples[i:i+batch_size]
        prompt_texts = [s['input'] for s in batch]
        answer_texts = [s['output'] for s in batch]

        with torch.no_grad():
            prompt_embs = model(prompt_texts).cpu().numpy()
            answer_embs = model(answer_texts).cpu().numpy()

        sim = cosine_similarity(prompt_embs, answer_embs)
        for j, sample in enumerate(batch):
            sample['similarity_score'] = float(sim[j][j])
            similarities.append(sample)

    sorted_samples = sorted(similarities, key=lambda x: x['similarity_score'])
    order_changes = sum(1 for i, s in enumerate(sorted_samples) if original_order[s['id']] != i)

    print(f"[rerank_flattened_results] rerank sample number: {order_changes}")

    for sample in sorted_samples:
        write_to_file(write_file, sample)

    print(f"Reranked and wrote {len(sorted_samples)} samples to {write_file}")

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
    all_samples = sample_generation(args, merged_results, dic_graph)
    
    
    write_file = f"./data/{args.dataset}/rerank_{split_map[args.split]}_{args.version}_{args.mode}.json"
    rerank_flattened_results(args, all_samples, write_file, device)

if __name__ == "__main__":
    main()