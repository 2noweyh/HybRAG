import dgl
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
from model import TextModel

# CUDA_VISIBLE_DEVICES=1 python data_preprocess/dic_extractor.py --dataset webqsp --version v1

def parse_args():
    parser = argparse.ArgumentParser(description='Convert KG-QA dataset to DGL graph')
    parser.add_argument('--dataset', type=str, choices=['webqsp', 'cwq'], required=True, help='Dataset to convert (webqsp, vwq)')
    parser.add_argument('--version', type=str, required=True, help='Version tag used to distinguish retrieval result files (e.g., v1, rerun, ablation1)')
    parser.add_argument('--text_encoder', default='SentenceBert', type=str,
                        choices=['Bert', 'Roberta', 'SentenceBert', 'SimCSE', 'e5', 't5'],
                        help='Text encoder to use')
    return parser.parse_args()

def load_split_dataframe(dataset, split):
    base_path = f"../datasets/RoG-{dataset}/data/{split}"
    files = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".parquet")])
    df_list = [pd.read_parquet(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

def batched_encode(text_model, texts, batch_size=32, device='cuda'):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_output = text_model(batch.to(device) if isinstance(batch, torch.Tensor) else batch)

            if batch_output.dim() == 1:
                batch_output = batch_output.unsqueeze(0)

            embeddings.append(batch_output.detach().cpu())
    return torch.cat(embeddings, dim=0)

def create_kgqa_dic_graph(df, text_encoder):
    text_model = TextModel(text_encoder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model = text_model.to(device)
    text_model.eval()

    samples = {
        "query_id": [],
        "query": [],
        "query_embedding": [],
        "src_id": [],
        "src_attr": [],
        "tgt_id": [],
        "tgt_attr": [],
        "node_attr": [],
        "node_name": [],
        "edges": [],
        "edge_attr": [],
        "triplet_attr": [],
        "triplet_names": [],
    }

    # for i, row in df.iterrows():
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
        question = row['question']
        graph_triples = row['graph']

        q_entity = row["q_entity"]
        if isinstance(q_entity, (list, np.ndarray)):
            q_entity = q_entity[0]

        a_entity = row["a_entity"]
        if isinstance(a_entity, (list, np.ndarray)):
            a_entity = list(set(a_entity))
        else:
            a_entity = [a_entity]

        # === 1. Query, Entity 임베딩 ===
        # with torch.no_grad():
            # query_embed = text_model(question).detach().cpu()
            # src_embed = text_model(q_entity).detach().cpu()
            # tgt_embed = text_model(a_entity).detach().cpu()

        samples["query_id"].append(row["id"])
        samples["query"].append(question)
        # samples["query_embedding"].append(query_embed)
        samples["src_id"].append(q_entity)
        # samples["src_attr"].append(src_embed)
        samples["tgt_id"].append(a_entity)
        # samples["tgt_attr"].append(tgt_embed)

        # === 2. Node 정보 ===
        node_set = set()
        edge_list = []
        triplet_strings = []

        for h, r, t in graph_triples:
            node_set.update([h, t])
            edge_list.append(r)
            triplet_strings.append(f"{h} -> {r} -> {t}") #{u_text} -> {rel_text} -> {v_text}

        # node_list = sorted(node_set)
        node_list = sorted(node_set.union([q_entity] + a_entity))
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        # with torch.no_grad():
            # node_embeds = text_model(node_list).detach().cpu()
            # edge_embeds = text_model(edge_list).detach().cpu()
            # # triplet_embeds = text_model(triplet_strings).detach().cpu()
            # triplet_embeds = batched_encode(text_model, triplet_strings, batch_size=32, device=device)

        samples["node_name"].append(node_list)
        # samples["node_attr"].append(node_embeds)
        samples["edges"].append([(node_to_idx[h], node_to_idx[t]) for h, _, t in graph_triples])
        # samples["edge_attr"].append(edge_embeds)
        # samples["triplet_attr"].append(triplet_embeds)
        samples["triplet_names"].append(triplet_strings)

    return samples

def convert_graph(arr):
    return [tuple(triple.tolist()) for triple in arr]

def main():
    args = parse_args()
    split_list = ['trn', 'tst', 'val']

    print("Loading all split files for full KG...")
    split_dfs = {split: load_split_dataframe(args.dataset, split) for split in split_list}
    
    # No Graph remove
    for split in split_list:
        orig_len = len(split_dfs[split])
        split_dfs[split] = split_dfs[split][
                split_dfs[split]['graph'].apply(lambda g: isinstance(g, (list, np.ndarray)) and len(g) > 0)] 
        new_len = len(split_dfs[split])
        print(f"{split}: removed {orig_len - new_len} rows with empty graph")

    # print("Building unified DGL KG graph...")
    # full_df = pd.concat(split_dfs.values(), ignore_index=True)
    # full_df['graph'] = full_df['graph'].apply(convert_graph)
    # dic_graph = create_kgqa_dic_graph(full_df, args.text_encoder)

    # print("Saving unified graph...")
    # torch.save(dic_graph, f'./data/{args.dataset}_dic_graph.pt')
    # print(f"  - Graph: ./data/{args.dataset}_dic_graph.pt")

    dic = {}
    for split in split_list:
        print(f"Building DIC for {split} split...")
        df = split_dfs[split]
        df['graph'] = df['graph'].apply(convert_graph)
        dic[split] = create_kgqa_dic_graph(df, args.text_encoder)

    torch.save(dic, f'./data/{args.dataset}/{args.dataset}_dic_graph_{args.version}.pt')
    print(f"  - Saved split DIC graph dict: ./data/{args.dataset}_dic_graph_{args.version}.pt")

if __name__ == "__main__":
    main()