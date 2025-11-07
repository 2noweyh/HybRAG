# data_preprocess/inductive_extractor_v2.py
# 사용 예: CUDA_VISIBLE_DEVICES=1 python data_preprocess/inductive_extractor_v2.py --dataset cwq

import os, json, argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import dgl

from model import TextModel  # 기존 텍스트 임베더

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['webqsp','cwq'])
    p.add_argument('--text_encoder', default='SentenceBert',
                   choices=['Bert','Roberta','SentenceBert','SimCSE','e5','t5'])
    p.add_argument('--splits', nargs='+', default=['trn','val','tst'])
    p.add_argument('--base_dir', default='../datasets') # RoG-{dataset}/data/{split}
    p.add_argument('--save_dir', default='./data')      # ./data/{dataset}/... 로 저장
    p.add_argument('--group_map_path', default=None,    # 기본값: ./data/{dataset}/rel_prefix_group_map.json
                   help='prefix -> group 이름 매핑 JSON 경로')
    p.add_argument('--prefix_depth', type=int, default=1,
                   help='관계 접두(prefix) 추출 depth (매핑 전에 접두를 뽑을 때 사용)')
    return p.parse_args()

def load_split_dataframe(dataset, split, base_dir):
    base_path = f"{base_dir}/RoG-{dataset}/data/{split}"
    files = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.parquet')])
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    df = df[df['graph'].apply(lambda g: isinstance(g, (list, np.ndarray)) and len(g)>0)]
    return df

def convert_graph(arr):
    return [tuple(tri.tolist()) for tri in arr]

def extract_prefix(rel: str, depth: int) -> str:
    return '.'.join(str(rel).split('.')[:depth])

def build_rel_vocab_from_groupmap(all_rel_strings, group_map, prefix_depth=1):
    """
    전체 데이터에서 등장한 원문 관계 문자열을 (prefix -> group) 매핑으로 정규화하여
    실제로 등장한 group 키만 vocab으로 만든다.
    """
    groups = set()
    for r in all_rel_strings:
        p = extract_prefix(r, prefix_depth)
        gkey = group_map.get(p, 'other')
        groups.add(gkey)
    rel_names = sorted(groups)
    rel_vocab = {name: i for i, name in enumerate(rel_names)}
    if '__UNK_REL__' not in rel_vocab:
        rel_vocab['__UNK_REL__'] = len(rel_vocab)
    return rel_vocab

@torch.no_grad()
def encode_texts_in_batches(text_model, strings, batch_size=128, to_dtype=torch.float16, device='cuda'):
    text_model = text_model.to(device).eval()
    feats = []
    for i in tqdm(range(0, len(strings), batch_size), desc='Encoding'):
        batch = [str(s) for s in strings[i:i+batch_size]]
        emb = text_model(batch)  # [B, D]
        feats.append(emb.to('cpu').to(to_dtype))
        del emb
    return torch.cat(feats, dim=0) if feats else torch.empty(0)

def make_question_subgraph(row, node2id, rel_vocab, group_map, prefix_depth):
    # --- 노드 집합 ---
    q_ent = row['q_entity'][0] if isinstance(row['q_entity'], (list,np.ndarray)) else row['q_entity']
    a_ents = row['a_entity']; 
    if not isinstance(a_ents, (list,np.ndarray)): a_ents = [a_ents]

    entities = set([q_ent] + list(a_ents))
    for h, r, t in row['graph']:
        entities.add(h); entities.add(t)

    # --- 로컬 인덱싱 ---
    local_nodes = sorted(entities)
    local2global = [node2id[s] for s in local_nodes]
    gl2loc = {gid: i for i, gid in enumerate(local2global)}

    # --- 에지(그룹 키 etype) 구성 ---
    # etype = ('entity', group_key, 'entity')
    edge = defaultdict(lambda: ([], []))
    relid_per_et = defaultdict(list)
    relstr_per_et = defaultdict(list)  # 원문 복원용

    for h, r, t in row['graph']:
        if h not in entities or t not in entities:
            continue
        src = gl2loc[node2id[h]]
        dst = gl2loc[node2id[t]]

        p = extract_prefix(r, prefix_depth)
        gkey = group_map.get(p, 'other')
        et = ('entity', gkey, 'entity')

        edge[et][0].append(src)
        edge[et][1].append(dst)
        relid_per_et[et].append(rel_vocab.get(gkey, rel_vocab['__UNK_REL__']))
        relstr_per_et[et].append(r)  # 원문 텍스트는 기록

    # --- 그래프 빌드 ---
    num_nodes_dict = {'entity': len(local_nodes)}
    edges_t = {et: (torch.tensor(u, dtype=torch.int64),
                    torch.tensor(v, dtype=torch.int64))
               for et, (u, v) in edge.items()}
    if len(edges_t) == 0:
        # 비어있으면 self-loop 하나라도 둬서 저장 에러 방지(선택)
        edges_t = {('entity','__empty__','entity'): (torch.tensor([0]), torch.tensor([0]))}

    g = dgl.heterograph(edges_t, num_nodes_dict=num_nodes_dict)

    for et in g.canonical_etypes:
        ids = relid_per_et.get(et, [])
        g.edges[et].data['rel_id'] = torch.tensor(ids, dtype=torch.int32) if len(ids)==g.num_edges(et) \
                                      else torch.zeros(g.num_edges(et), dtype=torch.int32)

    # --- 메타(경로 복원/피처 gather용) ---
    meta = {
        'qid': str(row['id']),
        'q_entity_str': str(q_ent),
        'a_entity_strs': [str(x) for x in a_ents],
        'node_strs': [str(x) for x in local_nodes],
        'local_q_entity': int(local_nodes.index(q_ent)),
        'local_a_entities': [int(local_nodes.index(a)) for a in a_ents if a in local_nodes],
        'local2global': [int(x) for x in local2global],
        'edge_rel_str': {f'{et[0]}::{et[1]}::{et[2]}': [str(x) for x in relstr_per_et[et]]
                         for et in relstr_per_et}  # etype별 원문 관계 텍스트
    }
    return g, meta

def main():
    args = parse_args()
    ds_dir = f"{args.save_dir}/{args.dataset}"
    os.makedirs(ds_dir, exist_ok=True)

    # group_map 로드 (prefix -> group)
    group_map_path = args.group_map_path or f"{ds_dir}/rel_prefix_group_map.json"
    assert os.path.exists(group_map_path), f"Missing group_map: {group_map_path}"
    with open(group_map_path, 'r', encoding='utf-8') as f:
        group_map = json.load(f)

    # split별 데이터프레임 로드 및 정규화
    splits_data = {}
    all_rel_strings = []
    for sp in args.splits:
        df = load_split_dataframe(args.dataset, sp, args.base_dir)
        if len(df)>0:
            df = df.copy()
            df['graph'] = df['graph'].apply(convert_graph)
            # 전체 vocab 생성을 위해 원문 관계 문자열 수집
            for _, row in df.iterrows():
                for _, r, _ in row['graph']:
                    all_rel_strings.append(r)
        splits_data[sp] = df

    # 관계 vocab(그룹 키) 생성: 실제 등장한 group만 포함 + __UNK_REL__
    rel_vocab = build_rel_vocab_from_groupmap(all_rel_strings, group_map, prefix_depth=args.prefix_depth)
    with open(f"{ds_dir}/rel_vocab.json", 'w', encoding='utf-8') as f:
        json.dump(rel_vocab, f, ensure_ascii=False)
    print(f"✅ Saved rel_vocab.json with {len(rel_vocab)} entries at {ds_dir}")

    # 텍스트 임베더
    text_model = TextModel(args.text_encoder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_model = text_model.to(device).eval()

    base_out = f"{ds_dir}/inductive"
    os.makedirs(base_out, exist_ok=True)

    for sp, df in splits_data.items():
        if df is None or len(df)==0:
            continue

        # split 전용 entity vocab
        all_entities = set()
        for _, row in df.iterrows():
            q = row['q_entity'][0] if isinstance(row['q_entity'], (list,np.ndarray)) else row['q_entity']
            a = row['a_entity'] if isinstance(row['a_entity'], (list,np.ndarray)) else [row['a_entity']]
            all_entities.add(q); all_entities.update(a)
            for h, r, t in row['graph']:
                all_entities.add(h); all_entities.add(t)

        node2id = {e:i for i,e in enumerate(sorted(all_entities))}
        id2node = {i:e for e,i in node2id.items()}

        # 노드 임베딩(글로벌)
        node_texts = [id2node[i] for i in range(len(id2node))]
        full_feat = encode_texts_in_batches(text_model, node_texts, batch_size=128, to_dtype=torch.float16, device=device)
        assert full_feat.shape[0] == len(id2node)

        graphs, metas = [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Build subgraphs [{sp}]'):
            g, meta = make_question_subgraph(row, node2id, rel_vocab, group_map, args.prefix_depth)
            graphs.append(g); metas.append(meta)

        # 공통 자원 저장
        with open(f"{base_out}/{sp}_node2id.json", 'w', encoding='utf-8') as f:
            json.dump(node2id, f, ensure_ascii=False)
        with open(f"{base_out}/{sp}_id2node.json", 'w', encoding='utf-8') as f:
            json.dump(id2node, f, ensure_ascii=False)
        torch.save(full_feat, f"{base_out}/{sp}_node_full_feat.pt")

        # 그래프/메타 저장(단일 버전)
        dgl.save_graphs(f"{base_out}/{sp}_graphs.bin", graphs)
        with open(f"{base_out}/{sp}_metas.jsonl", 'w', encoding='utf-8') as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

        print(f"✅ Saved [{sp}] graphs/metas/full_feat to {base_out}")

    print("✅ Done.")

if __name__ == "__main__":
    main()
