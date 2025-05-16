
import networkx as nx
import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj, to_networkx

def get_targets(batch, max_num_nodes):
    device = batch.edge_index.device
    x_size = batch.x.size(-1)
    x_target, mask = to_dense_batch(batch.x[..., :x_size], batch=batch.batch, max_num_nodes=max_num_nodes)
    x_target = x_target.argmax(-1) * mask

    edge_attr = torch.cat((batch.edge_attr.sum(-1, keepdims=True) == 0, batch.edge_attr), dim=-1)
    edge_target = to_dense_adj(batch.edge_index, edge_attr=edge_attr, batch=batch.batch,
                               max_num_nodes=max_num_nodes)
    edge_target = edge_target.argmax(-1)
    edge_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2) * \
                (1 - torch.eye(max_num_nodes, device=device)).unsqueeze(0)

    return x_target, edge_target, mask, edge_mask

def get_networkx_from_dense(X, A, mask, x_min=None):
    graphs = []
    if X is not None:
        X.squeeze_()
        assert X.dim() == 2, 'The annotation matrix must be of shape (batch_size, n)'
        assert mask.size() == X.size(), 'The annotation matrix and the mask must be of same shape'
    assert A.dim() == 3, 'The adjacency matrix must be of shape (batch_size, n, n)'

    batch_size = A.size(0)
    for b in range(batch_size):
        G = nx.Graph()

        # Add nodes with annotations
        for i, node in enumerate(X[b, mask[b]]):
            if x_min is not None:
                node += x_min
            G.add_node(i, label=node.int().item())

        # Add edges with attributes
        for i in range(mask[b].sum()):
            for j in range(i + 1, mask[b].sum()):  # Only consider the upper triangle for undirected graphs
                if A[b, i, j] > 0:  # Check if there is an edge
                    G.add_edge(i, j, label=A[b, i, j].int().item())
                    G.add_edge(j, i, label=A[b, i, j].int().item())  # Add reverse for undirected

        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        graphs.append(G)
    return graphs

def batch_to_networkx(batch):
    data_list = batch.to_data_list()
    nx_list = []
    for j, data in enumerate(data_list):

        if data.edge_attr is not None:
            is_edge = data.edge_attr[..., :-1].sum(-1) > 0
            assert data.edge_attr.dim() == 2, 'Edge_attr should be 2-dimensional'
            data.edge_index = data.edge_index[:, is_edge]
        nx_graph = to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True,remove_self_loops=True)

        if data.x is not None and data.x.size(-1) != 0:
            node_attrs = {i: {'label': x.int().item()} for i, x in enumerate(data.x[..., 0])}
            nx.set_node_attributes(nx_graph, node_attrs)
        if data.edge_attr is not None and data.edge_attr.size(-1) > 2:
            edge_attr = data.edge_attr[is_edge].argmax(-1) + 1
            edge_attrs = {(x.item(), y.item()): {'label': z.item()} for (x, y), z in zip(data.edge_index.T, edge_attr)}
        else:
            edge_attrs = {(x.item(), y.item()): {'label': 1} for x, y in data.edge_index.T}
        nx.set_edge_attributes(nx_graph, edge_attrs)

        nx_list.append(nx_graph)
    return nx_list

def get_batch_positional_encoding(n):
    d = 3
    device = n.device
    arange = torch.arange(n.max()).unsqueeze(0).to(device)
    mask = arange < n.unsqueeze(1)
    positions = arange.repeat(n.shape[0], 1)[mask].unsqueeze(1)

    arange_d = torch.arange(d).to(device).unsqueeze(0)

    arange_batch = torch.arange(n.size(0)).to(device)
    batch_idx = arange_batch.repeat_interleave(n)
    frequencies = torch.pi / torch.pow(n.unsqueeze(1), 2 * arange_d / d) #n x d
    frequencies = frequencies[batch_idx]
    sines = torch.sin(positions * frequencies)  # N, d
    cosines = torch.cos(positions * frequencies)
    encoding = torch.hstack((sines, cosines))
    return encoding
