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

def add_edge(g, node1, node2, weight=None):
    g.add_edge(node1,node2, geometry=LineString([Point(g.nodes[node1]["x"], g.nodes[node1]["y"]), Point(g.nodes[node2]["x"], g.nodes[node2]["y"])]), weight=weight)
