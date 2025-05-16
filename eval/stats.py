import concurrent.futures
import os
import subprocess as sp
from datetime import datetime

from scipy.linalg import eigvalsh
import networkx as nx
import numpy as np
# import graph_tool.all as gt

from eval.mmd import process_tensor, compute_mmd, gaussian, gaussian_emd, gaussian_tv, compute_nspdk_mmd
from scipy.stats import chi2

PRINT_TIME = True
ORCA_DIR = './eval/orca'  # the relative path to the orca dir


def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def add_tensor(x, y):
    x, y = process_tensor(x, y)
    return x + y




def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=True):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist

###############################################################################

def spectral_worker(G):
    # eigs = nx.laplacian_spectrum(G)
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    # from scipy import stats
    # kernel = stats.gaussian_kde(eigs)
    # positions = np.arange(0.0, 2.0, 0.1)
    # spectral_density = kernel(positions)

    # import pdb; pdb.set_trace()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_ref_list):
        #     sample_ref.append(spectral_density)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
        #     sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


###############################################################################

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
        # check non-zero elements in hist
        # total = 0
        # for i in range(len(sample_pred)):
        #    nz = np.nonzero(sample_pred[i])[0].shape[0]
        #    total += nz
        # print(total)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv,
    #                       sigma=1.0 / 10,)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd,
                           sigma=1.0 / 10, distance_scaling=bins)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts:'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_file_path = os.path.join(ORCA_DIR, 'tmp.txt')
    #print(tmp_file_path)
    f = open(tmp_file_path, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()
    output = sp.check_output([os.path.join(ORCA_DIR, 'orca'), 'node', '4', tmp_file_path, 'std'])
    #output = sp.check_output([os.path.join(ORCA_DIR), 'node', '4', tmp_file_path, 'std'])
    #print(output)
    output = output.decode('utf8').strip()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_file_path)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list):
    total_counts_ref = []
    total_counts_pred = []

    # graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G)
            #print(orbit_counts)
        except Exception as e:
            print(e)
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian_tv,
                           is_hist=False, sigma=30.0)

    # print('-------------------------')
    # print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    # print('...')
    # print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    # print('-------------------------')
    return mmd_dist

#######

def node_attr_stats(graph_ref_list, graph_pred_list, n_node_attr):
    sample_ref = get_node_attr_samples(graph_ref_list, n_node_attr)
    sample_pred = get_node_attr_samples(graph_pred_list, n_node_attr)
    return compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

def pmf_node_attr(graph, n_node_attr):
    attr = np.asarray([*nx.get_node_attributes(graph, "label").values()])
    attr = np.eye(n_node_attr+1)[attr]
    pmf = attr.sum(axis=0) / attr.sum()
    return pmf

def get_node_attr_samples(graph_list, n_node_attr):
    samples = []
    for graph in graph_list:
        pmf = pmf_node_attr(graph, n_node_attr)
        samples.append(pmf)
    return samples

def adjs_to_graphs(adjs):
    graph_list = []
    for adj in adjs:
        G = nx.from_numpy_array(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if not nx.is_empty(G):
            largest_component = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_component).copy()
        else:
            G.add_node(1)
        graph_list.append(G)
    return graph_list

# def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=100):
#     """
#     Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
#     """
#
#     adj = nx.adjacency_matrix(G).toarray()
#     idx = adj.nonzero()
#     g = gt.Graph()
#     g.add_edge_list(np.transpose(idx))
#     try:
#         state = gt.minimize_blockmodel_dl(g)
#     except ValueError:
#         if strict:
#             return False
#         else:
#             return 0.0
#
#     # Refine using merge-split MCMC
#     for i in range(refinement_steps):
#         state.multiflip_mcmc_sweep(beta=np.inf, niter=10)
#
#     b = state.get_blocks()
#     b = gt.contiguous_map(state.get_blocks())
#     state = state.copy(b=b)
#     e = state.get_matrix()
#     n_blocks = state.get_nonempty_B()
#     node_counts = state.get_nr().get_array()[:n_blocks]
#     edge_counts = e.todense()[:n_blocks, :n_blocks]
#     if strict:
#         if (node_counts > 40).sum() > 0 or (node_counts < 20).sum() > 0 or n_blocks > 5 or n_blocks < 2:
#             return False
#
#     max_intra_edges = node_counts * (node_counts - 1)
#     est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)
#
#     max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
#     np.fill_diagonal(edge_counts, 0)
#     est_p_inter = edge_counts / (max_inter_edges + 1e-6)
#
#     W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
#     W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)
#
#     W = W_p_inter.copy()
#     np.fill_diagonal(W, W_p_intra)
#     p = 1 - chi2.cdf(abs(W), 1)
#     p = p.mean()
#     if strict:
#         return p > 0.9  # p value < 10 %
#     else:
#         return p

def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]

def eval_acc_sbm_graph(G_list, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=100, is_parallel=True):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(is_sbm_graph,
                                     [gg for gg in G_list], [p_intra for i in range(len(G_list))],
                                     [p_inter for i in range(len(G_list))],
                                     [strict for i in range(len(G_list))],
                                     [refinement_steps for i in range(len(G_list))]):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(gg, p_intra=p_intra, p_inter=p_inter, strict=strict,
                                  refinement_steps=refinement_steps)
    return count / float(len(G_list))


def eval_fraction_unique(fake_graphs, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True
        if not fake_g.number_of_nodes() == 0:
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.is_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs

    return frac_unique

def eval_fraction_unique_non_isomorphic_valid(fake_graphs, train_graphs, validity_func=(lambda x: True)):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True

        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (float(len(fake_graphs)) - count_non_unique - count_isomorphic) / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs))  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


def nspdk_stats(graph_ref_list, graph_pred_list):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, metric='nspdk', is_hist=False, n_jobs=-1)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


METHOD_NAME_TO_FUNC = {
    'degree': degree_stats,
    'cluster': clustering_stats,
    'orbit': orbit_stats_all,
    'spectral': spectral_stats,
    'nspdk': nspdk_stats,
    'node_attr': node_attr_stats
}


def eval_torch_batch(ref_batch, pred_batch, methods=None):
    graph_ref_list = adjs_to_graphs(ref_batch.detach().cpu().numpy())
    graph_pred_list = adjs_to_graphs(pred_batch.detach().cpu().numpy())
    results = eval_graph_list(graph_ref_list, graph_pred_list, methods=methods)
    return results


# -------- Evaluate generated generic graphs --------
def eval_graph_list(graph_ref_list, graph_pred_list, methods=None, kernels=None, n_node_attr=None):
    if methods is None:
        methods = ['degree', 'cluster', 'orbit', 'spectral']
    results = {}
    kernels = {'degree': gaussian_emd,
               'cluster': gaussian_emd,
               'orbit': gaussian,
               'spectral': gaussian_emd}
    for method in methods:
        if len(graph_pred_list)>0:
            if method == 'nspdk':
                results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list)
            elif method == 'node_attr':
                results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list, n_node_attr)
            else:
                results[method] = round(METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list, is_parallel=True), 6)
            print('\033[91m' + f'{method:9s}' + '\033[0m' + ' : ' + '\033[94m' +  f'{results[method]:.6f}' + '\033[0m')

    return results