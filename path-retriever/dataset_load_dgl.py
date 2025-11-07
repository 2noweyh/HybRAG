# dataset_load_dgl.py
import dgl, torch, json

def load_data_dgl(args):
    graph_path = f"{args['data_folder']}/{args['dataset']}_dgl_graph.bin"
    edge_path = f"{args['data_folder']}/{args['dataset']}_qa_edges.pt"
    node2id_path = f"{args['data_folder']}/node2id.json"
    id2node_path = f"{args['data_folder']}/id2node.json"

    g, _ = dgl.load_graphs(graph_path)
    g = g[0]
    qa_edges = torch.load(edge_path)
    with open(node2id_path) as f: node2id = json.load(f)
    with open(id2node_path) as f: id2node = json.load(f)

    dataset = {
        "graph": g,
        "qa_edges": qa_edges,
        "node2id": node2id,
        "id2node": id2node,
    }
    return dataset
