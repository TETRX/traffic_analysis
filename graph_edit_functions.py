from typing import (
    List,
    Tuple,
)

import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString


def get_sc(g):
    scc = nx.strongly_connected_components(g)
    scc = list(scc)

    main_component = max(scc, key = len)

    return g.subgraph(main_component)

def get_center(g, property_dict, threshold=0.1):
    max_val = max(property_dict.values())
    min_val = min(property_dict.values())

    center = set()
    for key, value in property_dict.items():
        if value < min_val + (max_val-min_val)*threshold:
            center.add(key)
    return g.subgraph(center)

def add_edge(g: nx.MultiDiGraph, node1, node2, weight=None):
    g.add_edge(node1, node2, geometry=LineString([Point(g.nodes[node1]["x"], g.nodes[node1]["y"]), Point(g.nodes[
                                                                                                             node2]["x"], g.nodes[node2]["y"])]), weight=weight)

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
