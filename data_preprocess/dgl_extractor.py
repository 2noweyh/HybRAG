import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import dgl
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import json
from model import TextModel


def parse_args():
    parser = argparse.ArgumentParser(description='Convert KG-QA dataset to DGL graph')
    parser.add_argument('--dataset', type=str, choices=['webqsp', 'cwq'], required=True, help='Dataset to convert (webqsp, vwq)')
    parser.add_argument('--group_map_path', type=str, default='./data/webqsp/rel_prefix_group_map.json')
    parser.add_argument('--text_encoder', default='SentenceBert', type=str,
                        choices=['Bert', 'Roberta', 'SentenceBert', 'SimCSE', 'e5', 't5'],
                        help='Text encoder to use')
    return parser.parse_args()

def load_split_dataframe(dataset, split):
    base_path = f"../datasets/RoG-{dataset}/data/{split}"
    files = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".parquet")])
    df_list = [pd.read_parquet(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

def load_rel_prefix_group_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_prefix(relation, max_depth=1):
    """'a.b.c.d' → 'a' (depth=1), 'a.b' (depth=2), ..."""
    return ".".join(relation.split(".")[:max_depth])

def create_kgqa_dgl_graph(df, group_map_path, text_encoder, max_depth=1):
    rel_prefix_group_map = load_rel_prefix_group_map(group_map_path)

    all_entities = set()
    for _, data in df.iterrows():
        q_entity = data['q_entity']
        a_entities = data['a_entity']

        if isinstance(q_entity, (list, np.ndarray)):
            q_entity = q_entity[0] 

        if isinstance(a_entities, (list, np.ndarray)):
            a_entity = list(a_entities)
        else:
            a_entities = [a_entities]

        all_entities.add(q_entity)
        all_entities.update(a_entity)

        for h, r, t in data['graph']:
            all_entities.update([h, t])

    node2id = {e: i for i, e in enumerate(sorted(all_entities))}
    id2node = {i: e for e, i in node2id.items()}  
    edge_dict = defaultdict(lambda: ([], []))
    edge_relation_map = defaultdict(list)

    for _, data in df.iterrows():
        for h, r, t in data['graph']:
            prefix = extract_prefix(r, max_depth=max_depth)
            group = rel_prefix_group_map.get(prefix, 'other')
            etype = ('entity', group, 'entity')
            edge_dict[etype][0].append(node2id[h])
            edge_dict[etype][1].append(node2id[t])
            edge_relation_map[etype].append(r)
            
    edge_dict_tensor = {
        k: (torch.tensor(v[0], dtype=torch.int64), torch.tensor(v[1], dtype=torch.int64))
        for k, v in edge_dict.items()
    }

    g = dgl.heterograph(edge_dict_tensor, num_nodes_dict={'entity': len(node2id)})
    
    text_model = TextModel(text_encoder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model = text_model.to(device)
    text_model.eval()

    print("Encoding node text...")
    id_list = [id2node[i] for i in range(len(id2node))]
    node_texts = [str(e) for e in id_list]

    node_feats = []
    batch_size = 128
    for i in tqdm(range(0, len(node_texts), batch_size), desc="Node Text Encoding"):
        batch = node_texts[i:i+batch_size]
        batch = [str(t) for t in batch]


        emb = text_model(batch)              # [b, D] on device
        emb = emb.to('cpu').to(torch.float16)
        node_feats.append(emb)
    g.nodes['entity'].data['feat'] = torch.cat(node_feats, dim=0)

    print("Encoding edge relation text (edge-wise)...")
    
    
    eweight_dict = {}



    print(f"Canonical edge types: {g.canonical_etypes}")
    print(f"Graph num nodes: {g.num_nodes()}")
    print(f"Graph num edges: {g.num_edges()}")
 
    return g, node2id, id2node, eweight_dict

def extract_positive_edges(df, node2id):
    positive_edges = []
    for _, data in df.iterrows():
        q_entity = data['q_entity']
        a_entities = data['a_entity']

        if isinstance(q_entity, (list, np.ndarray)):
            q_entity = q_entity[0]

        if isinstance(a_entities, (list, np.ndarray)):
            a_entities = list(a_entities)

        q = node2id[q_entity]
        for a in a_entities:
            a_id = node2id[a]
            positive_edges.append((q, a_id))
    return torch.tensor(positive_edges)

def convert_graph(arr):
    return [tuple(triple.tolist()) for triple in arr]

def main():
    args = parse_args()
    split = 'trn'

    print("Loading all split files for full KG...")
    df = load_split_dataframe(args.dataset, split)
    
    orig_len = len(df)
    df = df[df['graph'].apply(lambda g: isinstance(g, (list, np.ndarray)) and len(g) > 0)]
    new_len = len(df)
    print(f"{split}: removed {orig_len - new_len} rows with empty graph")

    print("Building unified DGL KG graph...")
    df['graph'] = df['graph'].apply(convert_graph)
    dgl_graph, node2id, id2node, eweight_dict = create_kgqa_dgl_graph(df, args.group_map_path, args.text_encoder)

    
    print("Extracting split-wise positive QA edges...")
    positive_edges = extract_positive_edges(df, node2id)

    save_dir = f'./data/{args.dataset}'

    print("Saving node to id...")
    with open(f'{save_dir}/node2id.json', 'w') as f:
        json.dump(node2id, f)
    with open(f'{save_dir}/id2node.json', 'w') as f:
        json.dump(id2node, f)
    
    print("Saving edge weight...")

    print("Saving unified graph...")
    dgl.save_graphs(f'{save_dir}/{args.dataset}_dgl_graph.bin', [dgl_graph])

    print("Saving supervision edges...")
    torch.save({'trn': positive_edges}, f'{save_dir}/{args.dataset}_qa_edges.pt')


    print("✅ All files saved.")
    print(f"  - Graph: {save_dir}/{args.dataset}_graph.bin")
    print(f"  - QA supervision edges: {save_dir}/{args.dataset}_qa_edges.pt")

if __name__ == "__main__":
    main()
    