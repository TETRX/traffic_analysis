from typing import (
    List,
    Tuple,
    Set,
)

import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point, LineString


def add_edge(g: nx.MultiDiGraph, node1, node2, weight=None):
    g.add_edge(node1, node2, geometry=LineString([Point(g.nodes[node1]["x"], g.nodes[node1]["y"]), Point(g.nodes[
                                                                                                             node2][
                                                                                                             "x"],
                                                                                                         g.nodes[node2][
                                                                                                             "y"])]),
               weight=weight)


def get_center(g, property_dict, threshold=0.1):
    max_val = max(property_dict.values())
    min_val = min(property_dict.values())

    center = set()
    for key, value in property_dict.items():
        if value < min_val + (max_val - min_val) * threshold:
            center.add(key)
    return g.subgraph(center)


def _get_nodes_as_pd_df(graph):
    return pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')


def get_delta_graph(g_new, g_old, *delta_attributes):
    df_new = _get_nodes_as_pd_df(g_new)
    df_old = _get_nodes_as_pd_df(g_old)

    df_new = df_new[df_new.columns.intersection(delta_attributes)]
    df_old = df_old[df_old.columns.intersection(delta_attributes)]

    delta_df = df_new - df_old

    nodes = delta_df.to_dict('index')

    delta_graph = nx.MultiDiGraph(g_new)
    nx.set_node_attributes(delta_graph, nodes)

    return delta_graph


def add_edge(g: nx.MultiDiGraph, node1, node2, weight=None):
    g.add_edge(node1, node2, geometry=LineString([Point(g.nodes[node1]["x"], g.nodes[node1]["y"]), Point(g.nodes[
                                                                                                             node2][
                                                                                                             "x"],
                                                                                                         g.nodes[node2][
                                                                                                             "y"])]),
               weight=weight)


def get_crossroad_nodes(edges, street_name1: str, street_name2: str) -> Set[int]:
    """
    :param edges:
    :param street_name1: Doesn't need to be the full name, just a fragment of the street name will be enough if its unique, e.x. part of the surname could suffice
    :param street_name2:
    :return: Set of all nodes that belong simultaneously to both streets
    """
    relevant_edges1 = edges[edges['name'].str.contains(street_name1).fillna(False)]
    relevant_edges2 = edges[edges['name'].str.contains(street_name2).fillna(False)]

    index1 = relevant_edges1.index.to_list()
    index2 = relevant_edges2.index.to_list()

    crossroad_nodes = set()
    for u1, v1, _ in index1:
        for u2, v2, __ in index2:
            if u1 in (u2, v2):
                crossroad_nodes.add(u1)
            elif v1 in (u2, v2):
                crossroad_nodes.add(v1)
    return crossroad_nodes


def get_graph_from_place(place: str):
    g = ox.graph_from_place(place, network_type="drive")
    g = get_sc(g)
    g = nx.MultiDiGraph(g)  # unfreeze: IMPORTANT
    nodes, edges = ox.graph_to_gdfs(g)
    return g, nodes, edges


def get_graph_from_bbox(north: float, south: float, east: float, west: float):
    g = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    g = get_sc(g)
    g = nx.MultiDiGraph(g)  # unfreeze: IMPORTANT
    nodes, edges = ox.graph_to_gdfs(g)
    return g, nodes, edges


def get_sc(g):
    scc = nx.strongly_connected_components(g)
    scc = list(scc)

    main_component = max(scc, key=len)

    return g.subgraph(main_component)


def remove_edge(g: nx.MultiDiGraph, node1, node2):
    g.remove_edge(node1, node2)


def remove_shortest_path(g: nx.MultiDiGraph, source_node: int, target_node: int):
    path: List[int] = nx.shortest_path(g, source_node, target_node)

    for i in range(len(path) - 1):
        remove_edge(g, path[i], path[i + 1])


def remove_street(g: nx.MultiDiGraph, edges, street_name: str):
    edges_to_remove = edges[edges['name'].str.contains(
        street_name
    ).fillna(
        False
    ) | edges['ref'].str.contains(street_name).fillna(False)].index.to_list()

    removed_edges: List[Tuple[int, int]] = []
    for (u, v, _) in edges_to_remove:
        remove_edge(g, u, v)
        removed_edges.append((u, v))
    return removed_edges


def read_nodes_dict_from_csv(path: str):
    values_df = pd.read_csv(path, index_col=0)
    values_dict = {row[0]: row[1] for i, row in values_df.iterrows()}
    return values_dict
