import dgl
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
import json
from model import TextModel

# python data_preprocess/dgl_extractor.py --dataset webqsp

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
            edge_relation_map[etype].append(r) # 혜원 수정
    
    edge_dict_tensor = {
        k: (torch.tensor(v[0]), torch.tensor(v[1]))
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
        with torch.no_grad():
            batch_features = text_model(batch).cpu()
        node_feats.append(batch_features)
    g.nodes['entity'].data['feat'] = torch.cat(node_feats, dim=0)

    print("Encoding edge relation text (edge-wise)...")
    all_relation_texts = []
    edge_offset_map = {}  # etype → (start_idx, count)

    for etype in g.canonical_etypes:
        rel_texts = edge_relation_map.get(etype, [])
        if not rel_texts:
            continue
        start = len(all_relation_texts)
        all_relation_texts.extend(rel_texts)
        edge_offset_map[etype] = (start, len(rel_texts))

    rel_features = []
    for i in tqdm(range(0, len(all_relation_texts), batch_size), desc="Relation Text Encoding"):
        batch = all_relation_texts[i:i+batch_size]
        with torch.no_grad():
            batch_emb = text_model(batch).cpu()
        rel_features.append(batch_emb)
    rel_features = torch.cat(rel_features, dim=0)

    all_weights = rel_features.norm(dim=1)
    eweight_dict = {}

    for etype in g.canonical_etypes:
        if etype not in edge_offset_map:
            continue
        start, count = edge_offset_map[etype]
        eweight_dict[etype] = all_weights[start:start+count]


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

    print("Building unified DGL KG graph...")
    # full_df = pd.concat(split_dfs.values(), ignore_index=True)
    full_df = split_dfs['trn']
    full_df['graph'] = full_df['graph'].apply(convert_graph)
    dgl_graph, node2id, id2node, eweight_dict = create_kgqa_dgl_graph(full_df, args.group_map_path, args.text_encoder)

    
    print("Extracting split-wise positive QA edges...")
    split_edges = {
        split: extract_positive_edges(split_dfs[split], node2id)
        for split in split_list}


    # save_dir = f'./data/{args.dataset}'
    save_dir = f'./data/{args.dataset}'

    print("Saving node to id...")
    with open(f'{save_dir}/node2id.json', 'w') as f:
        json.dump(node2id, f)
    with open(f'{save_dir}/id2node.json', 'w') as f:
        json.dump(id2node, f)
    
    print("Saving edge weight...")
    torch.save(eweight_dict, f"{save_dir}/eweight_dict.pt")

    print("Saving unified graph...")
    dgl.save_graphs(f'{save_dir}/{args.dataset}_dgl_graph.bin', [dgl_graph])

    print("Saving supervision edges...")
    torch.save(split_edges, f'{save_dir}/{args.dataset}_qa_edges.pt')

    print("✅ All files saved.")
    print(f"  - Graph: {save_dir}/{args.dataset}_graph.bin")
    print(f"  - QA supervision edges: {save_dir}/{args.dataset}_qa_edges.pt")

if __name__ == "__main__":
    main()

# python dgl_extractor.py --dataset webqsp




#-----------------------BEFORE-----------------------------

# def parse_args():
#     parser = argparse.ArgumentParser(description='Convert KG-QA dataset to DGL graph')
#     parser.add_argument('--dataset', type=str, choices=['webqsp', 'cwq'], required=True, help='Dataset to convert (webqsp, vwq)')
#     return parser.parse_args()

# def load_split_dataframe(dataset, split):
#     base_path = f"../datasets/RoG-{dataset}/data/{split}"
#     files = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(".parquet")])
#     df_list = [pd.read_parquet(f) for f in files]
#     return pd.concat(df_list, ignore_index=True)


# def create_kgqa_dgl_graph(df):
#     all_entities = set()
#     for _, data in df.iterrows():
#         q_entity = data['q_entity']
#         a_entities = data['a_entity']

#         if isinstance(q_entity, (list, np.ndarray)):
#             q_entity = q_entity[0] 

#         if isinstance(a_entities, (list, np.ndarray)):
#             a_entity = list(a_entities)

#         all_entities.add(q_entity)
#         all_entities.update(a_entity)

#         for h, r, t in data['graph']:
#             all_entities.update([h, t])

#     node2id = {e: i for i, e in enumerate(sorted(all_entities))}
#     id2node = {i: e for e, i in node2id.items()}  
#     edge_dict = defaultdict(lambda: ([], []))  # key: (src_type, relation, dst_type)

#     # 수정 가능
#     for _, data in df.iterrows():
#         for h, r, t in data['graph']:
#             edge_dict[('entity', r, 'entity')][0].append(node2id[h])
#             edge_dict[('entity', r, 'entity')][1].append(node2id[t])
    
#     edge_dict_tensor = {
#         k: (torch.tensor(v[0]), torch.tensor(v[1]))
#         for k, v in edge_dict.items()
#     }

#     g = dgl.heterograph(edge_dict_tensor, num_nodes_dict={'entity': len(node2id)})

#     # # 4. 정답 링크 supervision용 edge 생성
#     # positive_edges = []
#     # for data in df:
#     #     qid = data['id']
#     #     q = node2id[data['q_entity']]
#     #     for a in data['a_entitity']:
#     #         a_id = node2id[a]
#     #         positive_edges.append((q, a_id))  # 하나의 질문에 여러 정답도 OK

#     # positive_edges_tensor = torch.tensor(positive_edges)

#     # entity_set = set()
#     # relation_set = set()

#     # edge_dict = defaultdict(list)  # relation -> list of (head, tail)

#     # for triples in tqdm(df['graph'], desc="Processing triples"):
#     #     for h, r, t in triples:
#     #         entity_set.update([h, t])
#     #         relation_set.add(r)
#     #         edge_dict[r].append((h, t))

#     # entity_list = sorted(entity_set)
#     # relation_list = sorted(relation_set)

#     # entity_id_map = {e: i for i, e in enumerate(entity_list)}
#     # num_nodes_dict = {'entity': len(entity_list)}

#     # hetero_data_dict = {}
#     # for rel in relation_list:
#     #     edges = edge_dict[rel]
#     #     src_ids = [entity_id_map[h] for h, t in edges]
#     #     dst_ids = [entity_id_map[t] for h, t in edges]
#     #     hetero_data_dict[('entity', rel, 'entity')] = (torch.tensor(src_ids), torch.tensor(dst_ids))

#     # g = dgl.heterograph(hetero_data_dict, num_nodes_dict=num_nodes_dict)
    
#     # print(f"Canonical edge types: {g.canonical_etypes}")
#     print(f"Graph num nodes: {g.num_nodes()}")
#     print(f"Graph num edges: {g.num_edges()}")
#     # print(f"Positive QA supervision edges: {len(positive_edges_tensor)}")

#     return g, node2id, id2node #, positive_edges_tensor

# def extract_positive_edges(df, node2id):
#     positive_edges = []
#     for _, data in df.iterrows():
#         q_entity = data['q_entity']
#         a_entities = data['a_entity']

#         if isinstance(q_entity, (list, np.ndarray)):
#             q_entity = q_entity[0]

#         if isinstance(a_entities, (list, np.ndarray)):
#             a_entities = list(a_entities)

#         q = node2id[q_entity]
#         for a in a_entities:
#             a_id = node2id[a]
#             positive_edges.append((q, a_id))
#     return torch.tensor(positive_edges)

# def convert_graph(arr):
#     return [tuple(triple.tolist()) for triple in arr]

# def main():
#     args = parse_args()
#     split_list = ['trn', 'tst', 'val']

#     print("Loading all split files for full KG...")
#     split_dfs = {split: load_split_dataframe(args.dataset, split) for split in split_list}
    
#     # No Graph remove
#     for split in split_list:
#         orig_len = len(split_dfs[split])
#         split_dfs[split] = split_dfs[split][
#                 split_dfs[split]['graph'].apply(lambda g: isinstance(g, (list, np.ndarray)) and len(g) > 0)] 
#         new_len = len(split_dfs[split])
#         print(f"{split}: removed {orig_len - new_len} rows with empty graph")

#     print("Building unified DGL KG graph...")
#     full_df = pd.concat(split_dfs.values(), ignore_index=True)
#     full_df['graph'] = full_df['graph'].apply(convert_graph)
#     dgl_graph, node2id, id2node = create_kgqa_dgl_graph(full_df)
    
#     print("Extracting split-wise positive QA edges...")
#     split_edges = {
#         split: extract_positive_edges(split_dfs[split], node2id)
#         for split in split_list}

#     print("Saving unified graph...")
#     dgl.save_graphs(f'./data/{args.dataset}_graph.bin', [dgl_graph])

#     print("Saving supervision edges...")
#     torch.save(split_edges, f'./data/{args.dataset}_qa_edges.pt')

#     print("✅ All files saved.")
#     print(f"  - Graph: ./data/{args.dataset}_graph.bin")
#     print(f"  - QA supervision edges: ./data/{args.dataset}_qa_edges.pt")

# if __name__ == "__main__":
#     main()

