import osmnx as ox
import networkx as nx
from multiprocessing import Pool
import itertools

import pandas as pd


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(G, processes=None, weight=None, path=None):
    """Parallel betweenness centrality  function"""
    G1 = nx.DiGraph(G)
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(G1.nodes(), int(G1.order() / node_divisor)))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_subset,
        zip(
            [G1] * num_chunks,
            node_chunks,
            [list(G)] * num_chunks,
            [True] * num_chunks,
            [weight] * num_chunks,
        ),
    )
    p.close()
    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    if path:
        df = pd.DataFrame([[k, v] for k, v in bt_c.items()], columns=["node_id", "value"])
        df.to_csv(path)
    return bt_c


def pagerank(G, weights=None, path=None):
    wpr_dict = nx.pagerank_numpy(G, weight=weights)
    if path:
        df = pd.DataFrame([[k, v] for k, v in wpr_dict.items()], columns=["node_id", "value"])
        df.to_csv(path)
    return wpr_dict


def clustering_coefficient(G, weight=None, path=None):
    g_not_multi = ox.get_digraph(G)
    c_dict = nx.clustering(g_not_multi, weight=weight)
    if path:
        df = pd.DataFrame([[k, v] for k, v in c_dict.items()], columns=["node_id", "value"])
        df.to_csv(path)
    return c_dict


def eigenvector_centrality(G, weight=None, path=None):
    wec_dict = nx.eigenvector_centrality(G, weight=weight)
    if path:
        df = pd.DataFrame([[k, v] for k, v in wec_dict.items()], columns=["node_id", "value"])
        df.to_csv(path)
    return wec_dict

def weighted_eccentricity(graph, weight=None, path=None):
    ecc_dict = nx.eccentricity(graph)
    if path:
        df = pd.DataFrame([[k, v] for k, v in ecc_dict.items()], columns=["node_id", "value"])
        df.to_csv(path)
    return ecc_dict

if __name__ == '__main__':

    g = ox.graph_from_place("Kraków, Poland", network_type="drive")
    clustering_coefficient(g, path="test.csv")