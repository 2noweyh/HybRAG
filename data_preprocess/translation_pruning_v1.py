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

def parse_args():
    parser = argparse.ArgumentParser(description='Flatten retrieval results')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use (e.g., yelp)')
    parser.add_argument('--split', type=str, required=True, help='Data split to use (e.g., trn)')
    parser.add_argument('--text_encoder', default='SentenceBert', type=str, choices=['Bert', 'Roberta', 'SentenceBert', 'SimCSE', 'e5', 't5'],
                        help='Text encoder to use (Bert, Roberta, SentenceBert, SimCSE, e5, or t5)')
    parser.add_argument('--shot', type=int, default=2, help='Number of few‑shot examples to prepend in the prompt')
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

def flatten_pagelink_retrieval_results(args, pred_edge_to_paths, dic, id2node, node2id, device):    
    text_pair_to_qid = {}
    for qid, src_id, tgt_ids in zip(dic["query_id"], dic["src_id"], dic["tgt_id"]):
        src_id = node2id[str(src_id)]
        if not isinstance(tgt_ids, list):
            tgt_ids = [tgt_ids]

        for tgt_id in tgt_ids:
            tgt_text = node2id[str(tgt_id)]
            text_pair_to_qid[(src_id, tgt_text)] = qid

    rel_prefix_group_map = load_rel_prefix_group_map(f'./data/{args.dataset}/rel_prefix_group_map.json')
    group_to_prefixes = defaultdict(list)
    for prefix, group in rel_prefix_group_map.items():
        group_to_prefixes[group].append(prefix)

    flattened_results = {}
    k = 0
    model = TextModel(args.text_encoder)
    model = model.to(device)

    for (src_info, tgt_info), paths in pred_edge_to_paths.items():
        src_id, tgt_id = src_info[1], tgt_info[1]
        qid = text_pair_to_qid.get((src_id, tgt_id))

        triple_list = dic['triplet_names'][dic['query_id'].index(qid)]
        question = dic['query_embedding'][dic['query_id'].index(qid)]

        path_strs = []
        for path in paths:
            path_text = []
            for i, (etype, u, v) in enumerate(path):
                u_text = id2node.get(str(u), f"[{u}]")
                v_text = id2node.get(str(v), f"[{v}]")
                rel_text = etype[1]

                # matched_rels = []
                # for triple_str in triple_list:
                #     parts = triple_str.split('-')
                #     h, r, t = parts[0], parts[1], parts[2]

                #     if r.split(".")[0] in group_to_prefixes[rel_text]:
                #         if h == u_text and t == v_text:
                #             matched_rels.append(r)

                # if matched_rels:
                #     matched_set = set(matched_rels)
                #     if len(matched_set) == 1:
                #         rel_text = list(matched_set)[0]

                #     else:
                #         with torch.no_grad():
                #             rel_embeds = batched_encode(model, list(matched_set), device=device)
                #         sims = F.cosine_similarity(rel_embeds, question.unsqueeze(0), dim=1)
                #         rel_text = list(matched_set)[torch.argmax(sims).item()]
                #         print(f"path_strs similarity for qid={qid}, src_id={src_id}, tgt_id={tgt_id}, ===> rel_text = {rel_text}")

                # path_text.append(f"{u_text} -> {rel_text} -> {v_text}")

                # matched_rels = []
                # for triple_str in triple_list:
                #     parts = triple_str.split(' -> ')
                #     h, r, t = parts[0], parts[1], parts[2]

                #     if h == u_text and t == v_text:
                #         matched_rels.append(r)

                # if matched_rels:
                #     was_filtered, rel_text = resolve_relation_text(matched_rels, rel_text, question, group_to_prefixes, model, device)
                #     if was_filtered:
                #         print(f"[Filtered] qid={qid}, src_id={src_id}, tgt_id={tgt_id}, ===> rel_text = {rel_text}") # 후보가 여러개여서 유사도기반 핕터링

                # path_text.append(f"{u_text} -> {rel_text} -> {v_text}")
                if i == 0:
                    path_text.append(f"{u_text} -> {rel_text} -> {v_text}")
                else:
                    path_text.append(f"{rel_text} -> {v_text}")

            path_strs.append(" -> ".join(path_text))

        if path_strs == ['']:
            k += 1

        key = (qid, tgt_id)
        if key not in flattened_results:
            flattened_results[key] = []

        flattened_results[key].extend(path_strs)
    print(f'No path total : {k}')

    finalized_results = {}
    for (qid, tgt_id), path_list in flattened_results.items():
        if qid not in finalized_results:
            finalized_results[qid] = []

        section = "\n".join(path_list)
        finalized_results[qid].append(section) 

    for qid in finalized_results:
        prompt = "\n\n".join(finalized_results[qid])
        finalized_results[qid] = prompt

    # print("="*40, "Sample finalized PaGE-Link prompts", "="*40)
    # for i, (qid, prompt) in enumerate(finalized_results.items()):
    #     print(f"{i+1}. QID: {qid}\n{prompt}\n")
    #     if i == 4:
    #         break

    return finalized_results

def flatten_dense_retrieval_results(dense_retrieval_results):
    flattened_results = {}

    for qid, subgraph_text in dense_retrieval_results.items():
        flattened_results[qid] = subgraph_text.strip()

    # print("="*40, "Sample from flattened_results", "="*40 )
    # for i, (qid, text) in enumerate(flattened_results.items()):
    #     print(f"{i+1}. QID: {qid}\n{text}\n")
    #     if i == 4:  
    #         break

    return flattened_results

def merge_flattened_results(flattened_dense_retrieval_results, flattened_pagelink_retrieval_results, split):
    merged_results = {}

    for key, dense in flattened_dense_retrieval_results.items():
        merged_text = "The following context was retrieved to explain the question-answer connections:\n\n"
        merged_text += "### Subgraph around relevant entities:\n" + dense.strip()
        if split == "trn":
            pagelink = flattened_pagelink_retrieval_results.get(key, "").strip()
            if pagelink:
                merged_text += "\n\n### Reasoning paths supporting the answer(s):\n" + pagelink

        merged_results[key] = merged_text

    # print("="*40, "Sample merged prompts", "="*40)
    # for i, (qid, prompt) in enumerate(merged_results.items()):
    #     print(f"\n{i+1}. QID: {qid}\n{prompt}\n")
    #     if i == 1:
    #         break

    print('merged result number', len(merged_results))

    return merged_results

def write_to_file(file_name, data):
    """
    Write data to a JSON file, creating directories if they don't exist.
    """
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'a', encoding='utf-8') as f_converted:
        json_str = json.dumps(data)
        f_converted.write(json_str + '\n')

# def sample_generation(args, merged_results, dict_data):
#     """
#     Generate prompt-answer format data for KGQA task with multi-answer support.

#     Args:
#         args: argparse.Namespace containing configuration like dataset name.
#         merged_results: dict mapping question_id to retrieval-based prompt string.
#         dict_data: dictionary with keys like 'query_id', 'query', 'tgt_id', etc.

#     Returns:
#         List of samples formatted for instruction tuning.
#     """
#     count_no_merge = 0
#     all_samples = []

#     for i in range(len(dict_data["query_id"])):
#         qid = dict_data["query_id"][i]
#         question = dict_data["query"][i]


#         user_message = (
#             "Given a question and its knowledge graph context, extract all correct answer entities.\n"
#             "Respond with the entity names only, separated by commas if there are multiple.\n"
#             f"Question:\n{question}"
#         )

#         if qid in merged_results and merged_results[qid].strip():
#             user_message += f"\n\n### knowledge graph Context:\n{merged_results[qid]}"
#         else:
#             count_no_merge += 1
#             continue #hw_edit

#         user_message += "\n\n### Correct answer(s): " #(Write only the answer entity names. Separate them with commas. Do not include any extra explanation.)

#         # Answer
#         answers = dict_data["tgt_id"][i]  # list or single string
#         if not isinstance(answers, list):
#             answers = [answers]        
#         user_response = ", ".join(answers) 

#         sample = {
#             "query_id": qid,
#             "prompt": user_message,
#             "chosen": user_response,  
#             "reject": "I DO NOT KNOW"
#         }
#         all_samples.append(sample)

#     print(f"[sample_generation] No merged context for {count_no_merge} samples")
#     return all_samples


def sample_generation(args, merged_results, dict_data, k_shot=2, rng_seed=0):
    rng = np.random.default_rng(rng_seed)

    cache = {}
    for i in range(len(dict_data["query_id"])):
        qid = dict_data["query_id"][i]
        if qid not in merged_results:
            continue
        answers = dict_data["tgt_id"][i]
        if not isinstance(answers, list):
            answers = [answers]
        qa_block = (
            f"\n### Question:\n{dict_data['query'][i]}\n"
            f"\n### knowledge Graph Context:\n{merged_results[qid]}\n"
            f"\n### Answer: {', '.join(answers)}" # Correct answer(s)
        )
        cache[qid] = qa_block

    count_no_merge = 0
    all_samples = []

    for i in range(len(dict_data["query_id"])):
        qid = dict_data["query_id"][i]
        if qid not in cache:          
            count_no_merge += 1
            continue

        # ---------- ② few‑shot 예시 선택 ----------
        pool = [v for k, v in cache.items() if k != qid]
        few_examples = []
        if k_shot > 0 and pool:
            k_actual = min(k_shot, len(pool))
            sampled_blocks = rng.choice(pool, size=k_actual, replace=False)
            for idx, ex in enumerate(sampled_blocks):
                few_examples.append(f"\n\n### Example{idx+1}\n{ex}")
            # few_examples = "\n\n\n".join(rng.choice(pool, size=k_actual, replace=False))

        # ---------- ③ 현재 질문(정답 비움) ----------
        question = dict_data["query"][i]
        context = merged_results[qid]
        # user_message = ""
        # if few_examples:
        #     user_message += few_examples + "\n\n\n"

        problem_block = (
            "\n### Problem\n"
            f"\n### Question:\n {question}\n"
            f"\n### Knowledge Graph Context:\n{context}\n"
            "\n### Answer:"
        )

        instruction_header = (
            "You are given a question and its knowledge graph context.\n"
            "Based on the context, extract the correct answer entities.\n"
            "Respond with only the entity names (no explanation), separated by commas if multiple.\n"
        )

        user_message = instruction_header
        if few_examples:
            user_message += "\n".join(few_examples)
        user_message += "\n\n" + problem_block

        # user_message += (
        #     "You are given a question and its knowledge graph context.\n"
        #     "Based on the context, extract all correct answer entities.\n"
        #     "Provide only the entity names, separated by commas if multiple.\n\n"
        #     f"Question:\n{question}"
        #     f"\n\n### knowledge graph Context:\n{merged_results[qid]}"
        #     "\n\n### Answer(s): "          # ← 정답 칸 비워 둠 Correct answer(s)
        # )

        # 라벨
        answers = dict_data["tgt_id"][i]
        if not isinstance(answers, list):
            answers = [answers]
        user_response = ", ".join(answers)

        all_samples.append({
            "query_id": qid,
            "prompt":  user_message,
            "chosen":  user_response,
            "reject":  "I DO NOT KNOW"
        })

    print(f"[sample_generation] No merged context for {count_no_merge} samples")

    print("="*40, "Sample instruction-tuning data", "="*40)
    for i, sample in enumerate(all_samples):
        print(f"\n{i+1}. Query ID: {sample['query_id']}")
        print(f"Prompt:\n{sample['prompt']}\n")
        print(f"Chosen: {sample['chosen']}")
        print(f"Reject: {sample['reject']}")
        if i == 1:
            break

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
    original_order = {sample['query_id']: i for i, sample in enumerate(all_samples)}

    for i in tqdm(range(0, len(all_samples), batch_size), desc="Reranking samples"):
        batch = all_samples[i:i+batch_size]
        prompt_texts = [s['prompt'] for s in batch]
        answer_texts = [s['chosen'] for s in batch]

        with torch.no_grad():
            prompt_embs = model(prompt_texts).cpu().numpy()
            answer_embs = model(answer_texts).cpu().numpy()

        sim = cosine_similarity(prompt_embs, answer_embs)
        for j, sample in enumerate(batch):
            sample['similarity_score'] = float(sim[j][j])
            similarities.append(sample)

    sorted_samples = sorted(similarities, key=lambda x: x['similarity_score'])
    order_changes = sum(1 for i, s in enumerate(sorted_samples) if original_order[s['query_id']] != i)

    print(f"[rerank_flattened_results] rerank sample number: {order_changes}")

    for sample in sorted_samples:
        write_to_file(write_file, sample)

    print(f"Reranked and wrote {len(sorted_samples)} samples to {write_file}")

def main(): 
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define mappings
    split_map = {"trn": "train", "val": "eval", "tst": "test"}
    
    # load dataset
    path_file = f"./saved_explanations/pagelink_{args.dataset}_model_{args.split}_pred_edge_to_paths"
    dense_file = f"./data/{args.dataset}/dense_retrieval_results_{args.split}.json"
    dic_graph_file = f"./data/{args.dataset}/{args.dataset}_dic_graph.pt"
    id2node_file = f"./data/{args.dataset}/id2node.json"
    node2id_file = f"./data/{args.dataset}/node2id.json"

    with open(path_file, "rb") as f:
        pagelink_retrieval_results = pickle.load(f)
    with open(dense_file, "r") as f:
        dense_retrieval_results = json.load(f)
    dic_graph = torch.load(dic_graph_file)[args.split]
    with open(id2node_file, "r") as f:
        id2node = json.load(f)
    with open(node2id_file, "r") as f:
        node2id = json.load(f)

    flattened_pagelink_retrieval_results = flatten_pagelink_retrieval_results(args, pagelink_retrieval_results, dic_graph, id2node, node2id, device)
    flattened_dense_retrieval_results = flatten_dense_retrieval_results(dense_retrieval_results)
    print('dense_retriever: ',len(flattened_dense_retrieval_results), 'pagelink_retriever: ',len(flattened_pagelink_retrieval_results))

    merged_results = merge_flattened_results(flattened_dense_retrieval_results, flattened_pagelink_retrieval_results, args.split)
    all_samples = sample_generation(args, merged_results, dic_graph,
                            k_shot=args.shot,     
                            rng_seed=args.seed)
    
    write_file = f"./data/{args.dataset}/rerank_{split_map[args.split]}.json"
    rerank_flattened_results(args, all_samples, write_file, device)

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 python data_preprocess/translation_pruning.py --dataset webqsp --split tst