import torch
import torch.nn.functional as F
import networkx as nx
import argparse
import json
from tqdm import tqdm

# python node_retriever/dense_retriever.py --dataset webqsp --split trn

def parse_args():
    parser = argparse.ArgumentParser(description='Retrieve top-k similar users and items')
    parser.add_argument('--dataset', type=str, choices=['webqsp', 'cwq'], required=True, help='Dataset to convert (webqsp, vwq)')
    parser.add_argument('--split', type=str, choices=['trn', 'val', 'tst'], required=True,
                        help='Data split to use (trn, val, or tst)')
    return parser.parse_args()

class DenseRetriever:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

    def g_encoder(self, samples, k_1=3, k_2=5, lambda_=0.8):
        query_embedding_list = samples["query_embedding"]
        node_attr = samples["node_attr"]
        edge_attr = samples["edge_attr"]
        node_names = samples["node_name"]
        triplet_attr = samples["triplet_attr"]
        triplet_names = samples["triplet_names"]
        edges = samples["edges"]
        query_id  = samples['query_id']

        # textual_subgraph_list = []
        textual_subgraph_dic = {}
        # for query, node_feats, node_name, edge_list, edge_feats, triplet_feats, triplet_name in zip(query_embedding_list, node_attr, node_names, edges, edge_attr, triplet_attr, triplet_names):
        for q_id, query, node_feats, node_name, edge_list, edge_feats, triplet_feats, triplet_name in tqdm(
                                                                                                        zip(query_id, query_embedding_list, node_attr, node_names, edges, edge_attr, triplet_attr, triplet_names),
                                                                                                        total=len(query_embedding_list),
                                                                                                        desc="Retrieving subgraphs"
                                                                                                    ):
                                                                                                                    
            query = query.unsqueeze(dim=0).to(self.device)
            node_feats = torch.FloatTensor(node_feats).to(self.device)

            node_sims = [F.cosine_similarity(query, feats) for feats in node_feats]
            node_sims = torch.stack(node_sims, 0).squeeze(dim=1).to(self.device)

            topk_nodes = torch.topk(node_sims, k=k_1)
            retrieved_node_names = [node_name[idx] for idx in topk_nodes.indices]
            # retrieved_node_names = list(set(retrieved_node_names))
            retrieved_node_names = list(dict.fromkeys([node_name[idx] for idx in topk_nodes.indices]))

            
            retrived_triplet_indices = []
            for index, sentence in enumerate(triplet_name):
                for node in retrieved_node_names:
                    if node in sentence:
                        retrived_triplet_indices.append(index)

            G = nx.Graph()
            G.add_edges_from(edge_list)
            normalized_degree = [G.degree[i]/len(G.nodes) for i in range(len(G.nodes))]

            score_list_ = []
            retrived_triplet_list = []
            retrived_edge_list = []
            for index in retrived_triplet_indices:
                edge = edge_list[index]
                score = (lambda_*(normalized_degree[edge[0]]*node_sims[edge[0]]))+((1-lambda_)*(normalized_degree[edge[1]]*node_sims[edge[1]]))
                score_list_.append(score)

                retrived_triplet_list.append(triplet_name[index])
                retrived_edge_list.append(edge_list[index])

            if len(score_list_) >= k_2:
                score_list = torch.stack(score_list_, 0).to(self.device)
                topk_triplets = torch.topk(score_list, k_2)
            else:
                score_list = torch.stack(score_list_, 0).to(self.device)
                topk_triplets = torch.topk(score_list, len(score_list_))

            ranked_edges = [retrived_edge_list[idx] for idx in topk_triplets.indices]

            # 주변 트리플 추가 (context expansion)
            # subgraph_node_set = set([n for e in ranked_edges for n in e])
            # context_triplets = []
            # for triplet_str in triplet_name:
            #     if any(node in triplet_str for node in subgraph_node_set):
            #         context_triplets.append(triplet_str)

            node_description = retrieved_node_names
            # triplet_description = ["\n".join(context_triplets)]
            triplet_description = ["\n".join([retrived_triplet_list[idx] for idx in topk_triplets.indices])]

            # textual_subgraph = node_description + triplet_description
            textual_subgraph = "\n".join(triplet_description)    # 서브그래프 양식 통일 및 노드 빼기 

            # textual_subgraph_list.append(textual_subgraph)
            textual_subgraph_dic[q_id] = textual_subgraph
            
        return textual_subgraph_dic

def main():
    args = parse_args()
    print(f"Processing {args.dataset} {args.split} data...")
    dic_all = torch.load(f'data/{args.dataset}/{args.dataset}_dic_graph.pt')  # Load PyG data
    dic_data = dic_all[args.split]
    
    # retrieval_results = DenseRetriever.g_encoder(dic_data)
    retriever = DenseRetriever(model=None, device='cuda')
    retrieval_results = retriever.g_encoder(dic_data)

    # Save retrieval results
    # with open(f'data/{args.dataset}/dense_retrieval_results_{args.split}.json', 'w') as f:
    #     json.dump(retrieval_results, f)
    # with open(f'data/{args.dataset}/dense_retrieval_results_{args.split}.json', 'w') as f:
        # json.dump([str(text) for text in retrieval_results], f, indent=2)
    with open(f'data/{args.dataset}/dense_retrieval_results_{args.split}.json', 'w') as f:
        json.dump(retrieval_results, f, indent=2, ensure_ascii=False)

    print(f"Retrieval results saved to data/{args.dataset}/dense_retrieval_results_{args.split}.json")

if __name__ == "__main__":
    main()
