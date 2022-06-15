from collections import namedtuple
from typing import (
    List,
    Tuple,
)

import folium
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd

from src.centrality_algorithms import betweenness_centrality_parallel, pagerank, \
    weighted_eccentricity, clustering_coefficient
from src.config import HIGHWAY_WEIGHTS


def basic_plot(
        G,
        node_weights,
        title,
        path=None,
        special_edges=None,
        change_sizes=True,
        show_plot=True
):
    edge_color = "white"
    edge_width = 1
    if special_edges is not None:
        edge_color = ["red" if (u, v) in special_edges or (v, u) in special_edges else "white" for u, v, k in G.edges(
            keys=True
        )]
        edge_width = [3 if (u, v) in special_edges or (v, u) in special_edges else 1 for u, v, k in G.edges(
            keys=True
        )]
    if isinstance(
            node_weights,
            pd.DataFrame
    ):
        node_weights = {row[0]: row[1] for i, row in node_weights.iterrows()}

    min_val = min(
        node_weights.values()
    )
    max_val = max(
        node_weights.values()
    )
    print(
        f"Range: {min_val} - {max_val}"
    )
    node_weights = {key: (val - min_val) / (max_val - min_val) for key, val in node_weights.items()}

    nx.set_node_attributes(
        G,
        values=node_weights,
        name=title
    )
    color_map = ox.plot.get_node_colors_by_attr(
        G,
        title,
        cmap="jet"
    )

    if change_sizes:
        node_size = [val * 50 + 15 for val in node_weights.values()]
    else:
        node_size = 15

    ox.plot_graph(
        G,
        edge_color=edge_color,
        edge_linewidth=edge_width,
        node_color=color_map,
        node_size=node_size,
        figsize=(55, 55),
        show=False,
        save=True,
        filepath=path
    )

    if show_plot:
        plt.show()


Coords = namedtuple("Coords", "long_min long_max lati_min lati_max")


class ObservableArea:
    """
    Class to represent abailable fixed areas to observe,
    coordinates copied from https://www.openstreetmap.org/
    """
    lagiewnicka_street = Coords(19.9279, 19.9473, 50.0335, 50.0262)
    zabiniec_district = Coords(19.9380, 19.9560, 50.0801, 50.0881)
    agh_area = Coords(19.9073, 19.9275, 50.0625, 50.0698)


def _get_color_map(vmin: int, vmax: int):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.hot
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    def color_map(weight: float):
        return mcolors.to_hex(m.to_rgba(weight))

    return color_map


def plot_graph_on_map(g, bc_weights, area, path=None):
    bc_weights = {row[0]: row[1] for i, row in bc_weights.iterrows()}

    nodes, edges = ox.graph_to_gdfs(g)

    nodes = nodes.loc[(nodes['x'] > area.long_min)
                      & (nodes['x'] < area.long_max)
                      & (nodes['y'] > area.lati_min)
                      & (nodes['y'] < area.lati_max)]

    bc_weights = {k: v for k, v in bc_weights.items() if int(k) in nodes.index}

    node_color_map = _get_color_map(min(bc_weights.values()), max(bc_weights.values()))
    edge_color_map = _get_color_map(min(HIGHWAY_WEIGHTS.values()), max(HIGHWAY_WEIGHTS.values()))

    f = folium.Figure(width=800, height=500)
    m = folium.Map(location=[(area.lati_min + area.lati_max) / 2, (area.long_min + area.long_max) / 2],
                   min_zoom=15, max_zoom=15).add_to(f)

    for index, row in nodes.iterrows():
        folium.CircleMarker(
            location=[row['y'], row['x']],
            radius=5,
            color=node_color_map(bc_weights[index]),
            fill=True,
            fill_color=node_color_map(bc_weights[index]),
        ).add_to(m)

    for index, row in edges.iterrows():
        start_node_id, end_node_id, _ = index

        if start_node_id in nodes.index and end_node_id in nodes.index:
            start_point = nodes.iloc[nodes.index == start_node_id]['y'].values[0], \
                          nodes.iloc[nodes.index == start_node_id]['x'].values[0]
            end_point = nodes.iloc[nodes.index == end_node_id]['y'].values[0], \
                        nodes.iloc[nodes.index == end_node_id]['x'].values[0]

            color = edge_color_map(HIGHWAY_WEIGHTS[row['highway']])
            folium.PolyLine([start_point, end_point], color=color, weight=2, opacity=0.5).add_to(m)
    return m


def plot_new_roads(g, special_edges: List[Tuple[int, int]], plot_title):
    special_nodes = set()
    for (u, v) in special_edges:
        special_nodes.add(u)
        special_nodes.add(v)

    path_title = plot_title.lower().replace(' ', '_')

    _dict = {node: int(node in special_nodes) for node in g.nodes}
    basic_plot(g, _dict, title=plot_title, special_edges=special_edges, path=f'images/{path_title}.png')


def plot_centralities(g, title: str, show_plots=False):
    path_title = title.lower().replace(' ', '_')

    bc_dict = betweenness_centrality_parallel(g, processes=16, path=f'values/betweenness_{path_title}.csv')
    basic_plot(g, bc_dict, f'Betweenness Centrality - with {title}', path=f'images/betweenness_{path_title}.png',
               show_plot=show_plots)
    del bc_dict

    cc_dict = clustering_coefficient(g, path=f'values/clustering_{path_title}.csv')
    basic_plot(g, cc_dict, f'Clustering Coefficient - with {title}', path=f'images/clustering_{path_title}.png',
               show_plot=show_plots)
    del cc_dict

    if nx.is_strongly_connected(g):
        we_dict = weighted_eccentricity(g, path=f'values/eccentricity_{path_title}.csv')
        basic_plot(g, we_dict, f'Weighted Eccentricity - with {title}', path=f'images/eccentricity_{path_title}.png',
                   show_plot=show_plots)
        del we_dict

    pagerank_dict = pagerank(g, path=f'values/pagerank_{path_title}.csv')
    basic_plot(g, pagerank_dict, f'Pagerank - with {title}', path=f'images/pagerank_{path_title}.png',
               show_plot=show_plots)
    del pagerank_dict


def plot_centralities_diff(g, dicts_unchanged: List[dict], dicts_changed: List[dict], title: str):
    path_title = title.lower().replace(' ', '_')

    bc_dict = get_weights_diff(dicts_unchanged[0], dicts_changed[0])
    basic_plot(g, bc_dict, f'Betweenness Centrality - with {title}', path=f'images/betweenness_diff_{path_title}.png')

    cc_dict = get_weights_diff(dicts_unchanged[1], dicts_changed[1])
    basic_plot(g, cc_dict, f'Clustering Coefficient - with {title}', path=f'images/clustering_diff_{path_title}.png')

    we_dict = get_weights_diff(dicts_unchanged[2], dicts_changed[2])
    basic_plot(g, we_dict, f'Weighted Eccentricity - with {title}', path=f'images/eccentricity_diff_{path_title}.png')

    pagerank_dict = get_weights_diff(dicts_unchanged[3], dicts_changed[3])
    basic_plot(g, pagerank_dict, f'Pagerank - with {title}', path=f'images/pagerank_diff_{path_title}.png')


def get_weights_diff(dict_unchanged, dict_changed):
    dict_diff = {}

    for node_id in set([*dict_unchanged.keys(), *dict_changed.keys()]):
        unchanged_weight = 0
        if node_id in dict_unchanged:
            unchanged_weight = dict_unchanged[node_id]

        changed_weight = 0
        if node_id in dict_changed:
            changed_weight = dict_changed[node_id]

        dict_diff[node_id] = fabs(changed_weight - unchanged_weight)

    return dict_diff


if __name__ == '__main__':
    weights = pd.read_csv("test.csv", index_col=0)
    g = ox.graph_from_place("Kraków, Poland", network_type="drive")

    area = ObservableArea.agh_area
    map = plot_graph_on_map(g, weights, area)
