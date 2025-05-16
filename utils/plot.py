import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_batch_networkx_graphs(graphs, layout='spring', positions=None, use_colormap = True, filename=None,
                               node_size=40, node_color='blue', edge_color='gray'):
    """
    Plot a batch of NetworkX graphs.

    Args:
        graphs (list of networkx.Graph): A batch of NetworkX graphs to plot.
        layout (str): The layout algorithm to use for positioning the nodes ('spring', 'circular', 'random', etc.).
        node_size (int): Size of the nodes.
        node_color (str or list): Color of the nodes.
        edge_color (str or list): Color of the edges.
    """
    num_graphs = len(graphs)
    if num_graphs == 0:
        print("No graphs to plot.")
        return

    # # Calculate the number of rows and columns for subplots
    # num_cols = int(torch.ceil(torch.sqrt(torch.tensor(num_graphs))))#.item()
    # num_rows = int(torch.ceil(torch.tensor(num_graphs / num_cols)))#.item()
    #
    # fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    # axes = axes.flatten() if num_graphs > 1 else [axes]

    saved_positions = []
    for i, G in enumerate(graphs):
        #ax.set_title(f'Graph {i + 1}')
        #ax.axis('off')

        # Choose the layout for the graph
        if positions is None:
            if layout == 'spring':
                pos = nx.spring_layout(G)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'random':
                pos = nx.random_layout(G)
            elif layout == 'shell':
                pos = nx.shell_layout(G)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G) # default
            elif layout == 'planar':
                pos = nx.planar_layout(G)
            else:
                pos = nx.spring_layout(G)
        else:
            pos = positions[i]
        saved_positions.append(pos)

        # Apply colormap if required
        if use_colormap:
            # Normalize positions to determine colors
            spectral_pos = pos
            spectral_array = np.array(list(spectral_pos.values()))
            min_spectral = spectral_array.min(axis=0)
            max_spectral = spectral_array.max(axis=0)
            norm_spectral = (spectral_array - min_spectral) / (max_spectral - min_spectral + 1e-6)
            color_values = (norm_spectral[:, 0] + norm_spectral[:, 1])/2
            cmap = cm.get_cmap('viridis')
            node_color = [cmap(value) for value in color_values]

        nx.draw_networkx(G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color, with_labels=False)

        plt.tight_layout()
        plt.box(False)
        if filename is not None:
            filename_ = filename + '_' + str(i)
            plt.savefig(filename_)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()
    return saved_positions