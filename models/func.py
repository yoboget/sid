from torch_geometric.utils import sort_edge_index

def transpose_edge_attr(edge_index, edge_attr):
    edge_index = sort_edge_index(edge_index)
    edge_index = edge_index.flip(0)
    edge_index, edge_attr = sort_edge_index(edge_index, edge_attr)
    return edge_attr