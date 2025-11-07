# CUDA_VISIBLE_DEVICES=1 python data_preprocess/inductive_extractor.py --dataset cwq --rel_mode grouped
# rel_mode 인자는 의미상 남겨두되, 이 스크립트는 항상 full과 grouped를 둘 다 생성/저장합니다.

import os, json, argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import dgl

from model import TextModel  # 워니가 쓰던 텍스트 임베더

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['webqsp','cwq'])
    p.add_argument('--text_encoder', default='SentenceBert',
                   choices=['Bert','Roberta','SentenceBert','SimCSE','e5','t5'])
    p.add_argument('--rel_mode', default='full', choices=['full','grouped'])  # 호환성 유지용(실제론 둘 다 생성)
    p.add_argument('--group_map_path', default='./data/webqsp/rel_prefix_group_map.json')
    p.add_argument('--max_depth', type=int, default=1)  # grouped 시 prefix depth
    p.add_argument('--splits', nargs='+', default=['trn','val','tst'])
    p.add_argument('--base_dir', default='../datasets') # RoG-{dataset}/data/{split}
    p.add_argument('--save_dir', default='./data')      # ./data/{dataset}/... 로 저장
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

def extract_prefix(rel, max_depth=1):
    return '.'.join(rel.split('.')[:max_depth])

def build_rel_vocab(splits_data, rel_mode, group_map=None, max_depth=1):
    vocab = {}
    for df in splits_data.values():
        if df is None or len(df)==0: continue
        for _, row in df.iterrows():
            for _, r, _ in row['graph']:
                key = extract_prefix(r, max_depth) if rel_mode=='grouped' else r
                if rel_mode=='grouped' and group_map is not None:
                    key = group_map.get(key, 'other')
                if key not in vocab:
                    vocab[key] = len(vocab)
    return vocab

@torch.no_grad()
def encode_texts_in_batches(text_model, strings, batch_size=128, to_dtype=torch.float16, device='cuda'):
    text_model = text_model.to(device).eval()
    feats = []
    for i in tqdm(range(0, len(strings), batch_size), desc='Encoding'):
        batch = [str(s) for s in strings[i:i+batch_size]]
        emb = text_model(batch)  # [B, D] on device
        feats.append(emb.to('cpu').to(to_dtype))
        del emb
    return torch.cat(feats, dim=0) if feats else torch.empty(0)

def make_question_subgraph(row, node2id,
                           rel_vocab_full,           # dict: rel(str) -> id
                           rel_vocab_grouped,        # dict: grouped_key(str) -> id
                           group_map, max_depth):
    # 노드 모음
    q_ent = row['q_entity'][0] if isinstance(row['q_entity'], (list,np.ndarray)) else row['q_entity']
    a_ents = row['a_entity']; 
    if not isinstance(a_ents, (list,np.ndarray)): a_ents = [a_ents]
    entities = set([q_ent] + list(a_ents))
    for h, r, t in row['graph']:
        entities.add(h); entities.add(t)

    # 로컬 인덱싱
    local_nodes = sorted(entities)
    local2global = [node2id[s] for s in local_nodes]  # 학습 시 full_feat[local2global]로 gather 가능

    # 에지 축적: 두 버전 동시 생성
    # keys: etype=('entity', rel_key, 'entity')
    edge_full = defaultdict(lambda: ([], []))
    edge_group = defaultdict(lambda: ([], []))

    relid_full_per_et = defaultdict(list)
    relid_group_per_et = defaultdict(list)

    # 원문 rel 문자열은 메타에 etype별 리스트로 저장해 경로 복원 가능
    relstr_full_per_et = defaultdict(list)
    relstr_group_per_et = defaultdict(list)

    # 로컬 인덱스 조회용
    gl2loc = {gid: i for i, gid in enumerate(local2global)}

    for h, r, t in row['graph']:
        if h not in entities or t not in entities:
            continue
        src = gl2loc[node2id[h]]
        dst = gl2loc[node2id[t]]

        # full
        et_full = ('entity', r, 'entity')
        edge_full[et_full][0].append(src)
        edge_full[et_full][1].append(dst)
        relid_full_per_et[et_full].append(rel_vocab_full[r])
        relstr_full_per_et[et_full].append(r)

        # grouped
        gkey = extract_prefix(r, max_depth)
        if group_map is not None:
            gkey = group_map.get(gkey, 'other')
        et_group = ('entity', gkey, 'entity')
        edge_group[et_group][0].append(src)
        edge_group[et_group][1].append(dst)
        relid_group_per_et[et_group].append(rel_vocab_grouped[gkey])
        relstr_group_per_et[et_group].append(r)  # 원문 문자열은 그대로 유지(복원용)

    # 그래프 빌드
    num_nodes_dict = {'entity': len(local_nodes)}

    def build_graph(edge_dict):
        edges_t = {et: (torch.tensor(u, dtype=torch.int64),
                        torch.tensor(v, dtype=torch.int64))
                   for et, (u, v) in edge_dict.items()}
        if len(edges_t) == 0:
            # 비어 있으면 self-loop 하나라도 둬서 DGL 저장 에러 방지(선택)
            edges_t = {('entity','__empty__','entity'): (torch.tensor([0]), torch.tensor([0]))}
        return dgl.heterograph(edges_t, num_nodes_dict=num_nodes_dict)

    g_full = build_graph(edge_full)
    g_group = build_graph(edge_group)

    # edata: rel_id 저장(숫자만 저장해 직렬화 가볍게)
    for et in g_full.canonical_etypes:
        ids = relid_full_per_et.get(et, [])
        g_full.edges[et].data['rel_id'] = torch.tensor(ids, dtype=torch.int32) if len(ids)==g_full.num_edges(et) \
                                          else torch.zeros(g_full.num_edges(et), dtype=torch.int32)
    for et in g_group.canonical_etypes:
        ids = relid_group_per_et.get(et, [])
        g_group.edges[et].data['rel_id'] = torch.tensor(ids, dtype=torch.int32) if len(ids)==g_group.num_edges(et) \
                                           else torch.zeros(g_group.num_edges(et), dtype=torch.int32)

    # 메타: 노드 문자열/로컬→글로벌, 에지 원문관계 문자열(경로 복원용)
    meta_shared = {
        'qid': str(row['id']),
        'q_entity_str': str(q_ent),
        'a_entity_strs': [str(x) for x in a_ents],
        'node_strs': [str(x) for x in local_nodes],
        'local_q_entity': int(local_nodes.index(q_ent)),
        'local_a_entities': [int(local_nodes.index(a)) for a in a_ents if a in local_nodes],
        'local2global': [int(x) for x in local2global],  # 학습시 full_feat[local2global]로 gather
    }
    meta_full = dict(meta_shared)
    meta_full['edge_rel_str'] = {f'{et[0]}::{et[1]}::{et[2]}': [str(x) for x in relstr_full_per_et[et]]
                                 for et in relstr_full_per_et}
    meta_group = dict(meta_shared)
    meta_group['edge_rel_str'] = {f'{et[0]}::{et[1]}::{et[2]}': [str(x) for x in relstr_group_per_et[et]]
                                  for et in relstr_group_per_et}

    return (g_full, meta_full), (g_group, meta_group)

def main():
    args = parse_args()
    os.makedirs(f"{args.save_dir}/{args.dataset}", exist_ok=True)

    # grouped용 prefix→group 매핑
    group_map = None
    if os.path.exists(args.group_map_path):
        with open(args.group_map_path, 'r', encoding='utf-8') as f:
            group_map = json.load(f)

    # split별 데이터 적재 및 graph 컬럼 정규화
    splits_data = {}
    for sp in args.splits:
        df = load_split_dataframe(args.dataset, sp, args.base_dir)
        if len(df)>0:
            df = df.copy()
            df['graph'] = df['graph'].apply(convert_graph)
        splits_data[sp] = df

    # relation vocab 두 버전 모두 생성(전체 split 합집합 기반)
    rel_vocab_full    = build_rel_vocab(splits_data, rel_mode='full',    group_map=None,      max_depth=args.max_depth)
    rel_vocab_grouped = build_rel_vocab(splits_data, rel_mode='grouped', group_map=group_map, max_depth=args.max_depth)

    with open(f"{args.save_dir}/{args.dataset}/rel_vocab_full.json", 'w', encoding='utf-8') as f:
        json.dump(rel_vocab_full, f, ensure_ascii=False)
    with open(f"{args.save_dir}/{args.dataset}/rel_vocab_grouped.json", 'w', encoding='utf-8') as f:
        json.dump(rel_vocab_grouped, f, ensure_ascii=False)

    # 텍스트 임베더 준비
    text_model = TextModel(args.text_encoder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    text_model = text_model.to(device).eval()

    # split별 처리
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

        # 고정 vocab(글로벌 ID): 이 순서로 full_feat를 만든다
        node2id = {e:i for i,e in enumerate(sorted(all_entities))}
        id2node = {i:e for e,i in node2id.items()}

        # 노드 텍스트 임베딩(한 번만 계산, 그래프에는 저장하지 않음)
        node_texts = [id2node[i] for i in range(len(id2node))]
        full_feat = encode_texts_in_batches(text_model, node_texts, batch_size=128, to_dtype=torch.float16, device=device)
        # 안전용 assert: vocab 사이즈와 일치
        assert full_feat.shape[0] == len(id2node)

        # 그래프/메타 생성(두 버전 동시)
        graphs_full, metas_full = [], []
        graphs_group, metas_group = [], []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Build subgraphs [{sp}]'):
            (g_full, meta_full), (g_group, meta_group) = make_question_subgraph(
                row, node2id, rel_vocab_full, rel_vocab_grouped, group_map, args.max_depth
            )
            graphs_full.append(g_full);   metas_full.append(meta_full)
            graphs_group.append(g_group); metas_group.append(meta_group)

        # 출력 디렉토리 구성
        base_out = f"{args.save_dir}/{args.dataset}/inductive"
        out_full    = f"{base_out}/full"
        out_grouped = f"{base_out}/grouped"
        os.makedirs(out_full, exist_ok=True)
        os.makedirs(out_grouped, exist_ok=True)

        # 공통 자원 저장(노드 vocab과 full_feat)은 split마다 한 번만 저장
        # 노드 vocab
        with open(f"{base_out}/{sp}_node2id.json", 'w', encoding='utf-8') as f:
            json.dump(node2id, f, ensure_ascii=False)
        with open(f"{base_out}/{sp}_id2node.json", 'w', encoding='utf-8') as f:
            json.dump(id2node, f, ensure_ascii=False)
        # 노드 임베딩(글로벌): 학습 시 meta['local2global']로 인덱싱해서 gather
        torch.save(full_feat, f"{base_out}/{sp}_node_full_feat.pt")

        # full 버전 저장
        dgl.save_graphs(f"{out_full}/{sp}_graphs.bin", graphs_full)
        with open(f"{out_full}/{sp}_metas.jsonl", 'w', encoding='utf-8') as f:
            for m in metas_full:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

        # grouped 버전 저장
        dgl.save_graphs(f"{out_grouped}/{sp}_graphs.bin", graphs_group)
        with open(f"{out_grouped}/{sp}_metas.jsonl", 'w', encoding='utf-8') as f:
            for m in metas_group:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')

        print(f"✅ Saved [{sp}] full/grouped graphs and shared resources to: {base_out}")

    print("✅ Done.")

if __name__ == "__main__":
    main()




# # CUDA_VISIBLE_DEVICES=1 python data_preprocess/inductive_extractor.py --dataset cwq --rel_mode grouped

# import os, json, argparse
# import pandas as pd
# from collections import defaultdict
# from tqdm import tqdm
# import numpy as np
# import torch
# import dgl
# import io

# from model import TextModel  # 워니가 쓰던 텍스트 임베더

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--dataset', required=True, choices=['webqsp','cwq'])
#     p.add_argument('--text_encoder', default='SentenceBert',
#                    choices=['Bert','Roberta','SentenceBert','SimCSE','e5','t5'])
#     p.add_argument('--rel_mode', default='full', choices=['full','grouped'])
#     p.add_argument('--group_map_path', default='./data/webqsp/rel_prefix_group_map.json')
#     p.add_argument('--max_depth', type=int, default=1)  # grouped일 때 prefix depth
#     p.add_argument('--splits', nargs='+', default=['trn','val','tst'])
#     p.add_argument('--base_dir', default='../datasets') # RoG-{dataset}/data/{split}
#     p.add_argument('--save_dir', default='./data')      # ./data/{dataset}/... 로 저장
#     return p.parse_args()

# def load_split_dataframe(dataset, split, base_dir):
#     base_path = f"{base_dir}/RoG-{dataset}/data/{split}"
#     files = sorted([os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith('.parquet')])
#     df_list = [pd.read_parquet(f) for f in files]
#     df = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
#     # 유효 그래프만
#     df = df[df['graph'].apply(lambda g: isinstance(g, (list, np.ndarray)) and len(g)>0)]
#     return df

# def convert_graph(arr):
#     # parquet에서 numpy로 올 수도 있으므로 tuple화
#     return [tuple(tri.tolist()) for tri in arr]

# def extract_prefix(rel, max_depth=1):
#     return '.'.join(rel.split('.')[:max_depth])

# def build_rel_vocab(splits_data, rel_mode, group_map=None, max_depth=1):
#     vocab = {}  # rel_str(or group) -> rel_id
#     for split, df in splits_data.items():
#         if df is None or len(df)==0: continue
#         for _, row in df.iterrows():
#             for h, r, t in row['graph']:
#                 key = extract_prefix(r, max_depth) if rel_mode=='grouped' else r
#                 if rel_mode=='grouped' and group_map is not None:
#                     key = group_map.get(key, 'other')
#                 if key not in vocab:
#                     vocab[key] = len(vocab)
#     return vocab  # 빈도 기반 정렬 필요없음

# def encode_texts_in_batches(text_model, strings, batch_size=128, to_dtype=torch.float16, device='cuda'):
#     text_model = text_model.to(device).eval()
#     feats = []
#     for i in tqdm(range(0, len(strings), batch_size), desc='Encoding'):
#         batch = [str(s) for s in strings[i:i+batch_size]]
#         with torch.no_grad():
#             emb = text_model(batch)            # [B, D] on device
#         feats.append(emb.to('cpu').to(to_dtype))
#         del emb
#     return torch.cat(feats, dim=0) if feats else torch.empty(0)

# def make_question_subgraph(row, node2id, rel_vocab, rel_mode, group_map, max_depth):
#     # 노드 수집
#     q_ent = row['q_entity'][0] if isinstance(row['q_entity'], (list,np.ndarray)) else row['q_entity']
#     a_ents = row['a_entity']
#     if not isinstance(a_ents, (list,np.ndarray)): a_ents = [a_ents]

#     entities = set([q_ent] + list(a_ents))
#     for h,r,t in row['graph']:
#         entities.add(h); entities.add(t)

#     # 로컬 노드 인덱싱 (질문 서브그래프 전용)
#     local_nodes = sorted(entities)
#     local2global = {i: node2id[ent] for i, ent in enumerate(local_nodes)}
#     global2local = {node2id[ent]: i for i, ent in enumerate(local_nodes)}

#     # 에지 빌드 (etype: ('entity', rel_key, 'entity'))
#     edge_dict = defaultdict(lambda: ([],[]))
#     rel_id_per_etype = defaultdict(list)  # edata 저장용
#     rel_str_per_etype = defaultdict(list)

#     for h,r,t in row['graph']:
#         if h not in entities or t not in entities:
#             continue
#         src = global2local[node2id[h]]
#         dst = global2local[node2id[t]]
#         rel_key = extract_prefix(r, max_depth) if rel_mode=='grouped' else r
#         if rel_mode=='grouped' and group_map is not None:
#             rel_key = group_map.get(rel_key, 'other')
#         etype = ('entity', rel_key, 'entity')
#         edge_dict[etype][0].append(src)
#         edge_dict[etype][1].append(dst)
#         rel_id_per_etype[etype].append(rel_vocab[rel_key])
#         rel_str_per_etype[etype].append(r)  # 원본 문자열

#     # 그래프 생성
#     num_nodes_dict = {'entity': len(local_nodes)}
#     edges_t = {et: (torch.tensor(u, dtype=torch.int64), torch.tensor(v, dtype=torch.int64))
#                for et, (u,v) in edge_dict.items()}
#     g = dgl.heterograph(edges_t, num_nodes_dict=num_nodes_dict)

#     # edata에 relation id/원문 저장 → 경로 설명 시 그대로 복원 가능
#     for et in g.canonical_etypes:
#         if len(rel_id_per_etype[et]) == g.num_edges(et):
#             g.edges[et].data['rel_id']  = torch.tensor(rel_id_per_etype[et], dtype=torch.int32)
#             # 원문 문자열 리스트를 그대로 저장하면 커질 수 있으니 필요 시 인덱스로
#             # 여기선 직렬화를 위해 json 문자열을 저장하지 않고, 별도 메타에 저장
#         else:
#             # 안전장치
#             g.edges[et].data['rel_id']  = torch.zeros(g.num_edges(et), dtype=torch.int32)

#     # 메타 정보 반환(문자열은 그래프 외부 json에 저장)
#     # meta = {
#     #     'qid': row['id'],
#     #     'q_entity_str': q_ent,
#     #     'a_entity_strs': a_ents,  # train에서만 평가용, dev/tst에선 비워도 됨
#     #     'node_strs': local_nodes,  # 로컬 인덱스 → 문자열
#     #     'local_q_entity': local_nodes.index(q_ent),
#     #     'local_a_entities': [local_nodes.index(a) for a in a_ents if a in local_nodes],
#     #     # 각 etype별 edge 원문 relation 문자열(경로 복원용)
#     #     'edge_rel_str': {f'{et[0]}::{et[1]}::{et[2]}': rel_str_per_etype[et] for et in rel_str_per_etype}
#     # }
#     meta = {
#     'qid': row['id'],
#     'q_entity_str': str(q_ent),
#     'a_entity_strs': [str(x) for x in a_ents],
#     'node_strs': [str(x) for x in local_nodes],
#     'local_q_entity': int(local_nodes.index(q_ent)),
#     'local_a_entities': [int(local_nodes.index(a)) for a in a_ents if a in local_nodes],
#     'edge_rel_str': {f'{et[0]}::{et[1]}::{et[2]}': [str(x) for x in rel_str_per_etype[et]]
#                      for et in rel_str_per_etype}
#     }
#     return g, meta

# def main():
#     args = parse_args()
#     os.makedirs(f"{args.save_dir}/{args.dataset}", exist_ok=True)

#     # grouped 모드면 prefix→group 매핑 로드
#     group_map = None
#     if args.rel_mode == 'grouped':
#         with open(args.group_map_path, 'r', encoding='utf-8') as f:
#             group_map = json.load(f)

#     # split별 데이터 적재
#     splits_data = {}
#     for sp in args.splits:
#         df = load_split_dataframe(args.dataset, sp, args.base_dir)
#         if len(df)>0:
#             df = df.copy()
#             df['graph'] = df['graph'].apply(convert_graph)
#         splits_data[sp] = df

#     # entity 사전은 split별로 독립적으로 구성 가능(완전 inductive)
#     # 또는 전 split 합집합으로 하나의 vocab을 만들 수도 있음.
#     # 여기서는 split마다 독립 vocab을 만든다. (유출 방지, 메모리 절약)
#     text_model = TextModel(args.text_encoder)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     text_model = text_model.to(device).eval()

#     # relation vocab은 schema 수준이라 전체 split 합집합을 사용해도 유출 위험 낮음
#     rel_vocab = build_rel_vocab(splits_data, args.rel_mode, group_map, args.max_depth)
#     with open(f"{args.save_dir}/{args.dataset}/rel_vocab_{args.rel_mode}.json", 'w', encoding='utf-8') as f:
#         json.dump(rel_vocab, f, ensure_ascii=False)

#     for sp, df in splits_data.items():
#         if df is None or len(df)==0:
#             continue

#         # split 전용 entity vocab
#         all_entities = set()
#         for _, row in df.iterrows():
#             q = row['q_entity'][0] if isinstance(row['q_entity'], (list,np.ndarray)) else row['q_entity']
#             a = row['a_entity'] if isinstance(row['a_entity'], (list,np.ndarray)) else [row['a_entity']]
#             all_entities.add(q); all_entities.update(a)
#             for h,r,t in row['graph']:
#                 all_entities.add(h); all_entities.add(t)
#         node2id = {e:i for i,e in enumerate(sorted(all_entities))}
#         id2node = {i:e for e,i in node2id.items()}

#         # 노드 텍스트 임베딩
#         node_texts = [id2node[i] for i in range(len(id2node))]
#         feats = encode_texts_in_batches(text_model, node_texts, batch_size=128, to_dtype=torch.float16, device=device)

#         graphs, metas = [], []
#         for _, row in tqdm(df.iterrows(), total=len(df), desc=f'Build subgraphs [{sp}]'):
#             g, meta = make_question_subgraph(row, node2id, rel_vocab, args.rel_mode, group_map, args.max_depth)
#             # 노드 피처 부여(인덱스 정렬과 맞음)
#             # g.nodes['entity'].data['feat'] = feats  # [N_local, D]에서 N_local만큼 slice 해야 할까?
#             # 주의: feats는 전체 split 엔티티 순서. 로컬은 subset이므로 인덱싱 필요
#             # 고정 feats를 그대로 넣으면 크기 불일치. 아래처럼 로컬 인덱스에 맞춰 선택.
#             local_idx = torch.tensor([node2id[s] for s in meta['node_strs']], dtype=torch.long)
#             g.nodes['entity'].data['feat'] = feats[local_idx]
#             graphs.append(g); metas.append(meta)

#         # 저장: 그래프 리스트(.bin)와 메타(.jsonl)
#         out_dir = f"{args.save_dir}/{args.dataset}/inductive_{args.rel_mode}"
#         os.makedirs(out_dir, exist_ok=True)
#         dgl.save_graphs(f"{out_dir}/{sp}_graphs.bin", graphs)
#         with open(f"{out_dir}/{sp}_metas.jsonl", 'w', encoding='utf-8') as f:
#             for m in metas:
#                 f.write(json.dumps(m, ensure_ascii=False) + '\n')

#         # 옵션: split vocab 저장
#         with open(f"{out_dir}/{sp}_node2id.json", 'w', encoding='utf-8') as f:
#             json.dump(node2id, f, ensure_ascii=False)
#         with open(f"{out_dir}/{sp}_id2node.json", 'w', encoding='utf-8') as f:
#             json.dump(id2node, f, ensure_ascii=False)

#     print("✅ Done.")

# if __name__ == "__main__":
#     main()
