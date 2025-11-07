import torch
import torch.nn.functional as F
import networkx as nx
import argparse
import json
from tqdm import tqdm
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocess.model import TextModel

def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve top-k relevant triples from KG')
    parser.add_argument('--dataset', type=str, choices=['webqsp', 'cwq'], required=True, help='Dataset to convert (webqsp, cwq)')
    parser.add_argument('--split', type=str, choices=['trn', 'val', 'tst'], required=True, help='Data split to use (trn, val, or tst)')
    parser.add_argument('--version', type=str, required=True, help='Version tag used to distinguish retrieval result files (e.g., v1, rerun, ablation1)')
    parser.add_argument('--text_encoder', default='SentenceBert', type=str,
                    choices=['Bert', 'Roberta', 'SentenceBert', 'SimCSE', 'e5', 't5'],
                    help='Text encoder to use')
    return parser.parse_args()

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

class DenseRetriever:
    def __init__(self, device='cuda'):
        self.device = device

    def g_encoder(self, samples, text_model, k_1=10, k_2=20, lambda_=0.8, triple_sim_topk=10, hop=2):
        node_names = samples["node_name"]
        triplet_names = samples["triplet_names"]
        edges = samples["edges"]
        query_id = samples['query_id']
        query_texts = samples["query"]

        textual_subgraph_dic = {}

        for q_id, query_text, node_name, edge_list, triplet_name in tqdm(
            zip(query_id, query_texts, node_names, edges, triplet_names), total=len(query_texts), desc="Retrieving subgraphs"):
    
            # 임베딩 계산
            with torch.no_grad():
                # query = text_model(query_text).detach().cpu()
                query = batched_encode(text_model, [query_text], batch_size=1, device=self.device).squeeze(0)
                # node_feats = text_model(node_name).detach().cpu()
                node_feats = batched_encode(text_model, node_name, batch_size=32, device=self.device) 
                # triplet_embeds = text_model(triplet_strings).detach().cpu()
                triplet_feats = batched_encode(text_model, triplet_name, batch_size=32, device=self.device)
  
            query = query.unsqueeze(0).to(self.device)
            node_feats = torch.FloatTensor(node_feats).to(self.device)

            # (1) Node similarity top-k
            node_sims = torch.stack([F.cosine_similarity(query, feats) for feats in node_feats], 0).squeeze(1)
            topk_node_indices = torch.topk(node_sims, k=min(k_1, len(node_sims))).indices.tolist()
            topk_nodes = [node_name[idx] for idx in topk_node_indices]

            # (3) Expand to 2-hop neighbors
            G = nx.Graph()
            G.add_nodes_from(range(len(node_name)))
            G.add_edges_from(edge_list)
            expanded_nodes = set(topk_node_indices)
            for node in topk_node_indices:
                if hop > 0:
                    for neighbor in nx.single_source_shortest_path_length(G, node, cutoff=hop):
                        expanded_nodes.add(neighbor)
            expanded_nodes = list(expanded_nodes)
            expanded_node_names = [node_name[idx] for idx in expanded_nodes]

            # (4) Score-based triplet filtering
            normalized_degree = [G.degree[i] / len(G.nodes) for i in range(len(G.nodes))]
            triplet_scores = []
            candidate_triplets = []
            for idx, (t, e) in enumerate(zip(triplet_name, edge_list)):
                if node_name[e[0]] in expanded_node_names or node_name[e[1]] in expanded_node_names:
                    score = lambda_ * (normalized_degree[e[0]] * node_sims[e[0]]) + \
                            (1 - lambda_) * (normalized_degree[e[1]] * node_sims[e[1]])
                    # triplet_scores.append(score)
                    triplet_scores.append(score.item())
                    candidate_triplets.append((idx, t))

            # Select top-k triplets by score
            if triplet_scores:
                topk = min(k_2, len(triplet_scores))
                sorted_indices = np.argsort(triplet_scores)[::-1][:topk]
                topk_triplets_by_node = [candidate_triplets[i][1] for i in sorted_indices]
            else:
                topk_triplets_by_node = []

            # (2) Triple-level similarity top-k (based on precomputed embeddings) # 이거 제거
            query_norm = query / query.norm(dim=-1, keepdim=True)
            triplet_tensor = torch.FloatTensor(triplet_feats).to(self.device)
            triplet_norm = triplet_tensor / triplet_tensor.norm(dim=-1, keepdim=True)
            sim = F.cosine_similarity(query_norm, triplet_norm)
            topk_sim_idx = torch.topk(sim, min(triple_sim_topk, len(sim))).indices.tolist()
            topk_triplets_by_text = [triplet_name[i] for i in topk_sim_idx]

            # (5) Merge + deduplication
            merged = list(dict.fromkeys(topk_triplets_by_node + topk_triplets_by_text))
            # textual_subgraph_dic[q_id] = "\n".join(merged)
            triplet_objects = []
            for idx, t in enumerate(merged):
                triplet_objects.append({
                    "triple": t,
                    "score": float(triplet_scores[idx]) if idx < len(triplet_scores) else float(sim[idx])
                })
            textual_subgraph_dic[q_id] = triplet_objects

        return textual_subgraph_dic

def main():
    args = parse_args()
    print(f"Processing {args.dataset} {args.split} data...")
    dic_all = torch.load(f'data/{args.dataset}/{args.dataset}_dic_graph_v1.pt')
    dic_data = dic_all[args.split]

    text_model = TextModel(args.text_encoder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model = text_model.to(device)
    text_model.eval()

    retriever = DenseRetriever(device='cuda')
    retrieval_results = retriever.g_encoder(dic_data, text_model)

    with open(f'data/{args.dataset}/dense_retrieval_results_{args.split}_{args.version}.json', 'w') as f:
        json.dump(retrieval_results, f, indent=2, ensure_ascii=False)

    print(f"Retrieval results saved to data/{args.dataset}/dense_retrieval_results_{args.split}_{args.version}.json")

if __name__ == "__main__":
    main()