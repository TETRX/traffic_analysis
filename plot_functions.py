
import osmnx as ox
import networkx as nx
import folium

import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors



def basic_plot(G, node_weights, title, path=None, special_edges = None, change_sizes=True):
    edge_color="white"
    edge_width = 1
    if special_edges is not None:
        edge_color = ["red" if (u,v) in special_edges or (v,u) in special_edges else "white" for u,v,k in G.edges(keys=True)]
        edge_width = [3 if (u,v) in special_edges or (v,u) in special_edges else 1 for u,v,k in G.edges(keys=True)]
    if isinstance(node_weights, pd.DataFrame):
        node_weights = {row[0]: row[1] for i, row in node_weights.iterrows()}

    
    min_val = min(node_weights.values())
    max_val = max(node_weights.values())
    print(f"Range: {min_val} - {max_val}")
    node_weights = {key: (val-min_val) / (max_val-min_val) for key, val in node_weights.items()}

    nx.set_node_attributes(G, values=node_weights, name=title)
    color_map = ox.plot.get_node_colors_by_attr(G, title, cmap="jet")

    if change_sizes:
        node_size = [val*50+15 for val in node_weights.values()]
    else:
        node_size = 15
    
    ox.plot_graph(G, edge_color=edge_color, edge_linewidth=edge_width, node_color=color_map, node_size=node_size, figsize=(55, 55), show=False, save=True, filepath=path)
    if path:
        plt.savefig(path)
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
    highway_weights = {
        'residential': 2,
        'tertiary': 2,
        'living_street': 4,
        'secondary': 6,
        'unclassified': 1,
        'primary': 8,
        'secondary_link': 5,
        'tertiary_link': 5,
        'primary_link': 5,
        'motorway_link': 5,
        'motorway': 10,
        'trunk_link': 5,
        'trunk': 10,
    }

    bc_weights = {row[0]: row[1] for i, row in bc_weights.iterrows()}

    nodes, edges = ox.graph_to_gdfs(g)

    nodes = nodes.loc[(nodes['x'] > area.long_min)
                      & (nodes['x'] < area.long_max)
                      & (nodes['y'] > area.lati_min)
                      & (nodes['y'] < area.lati_max)]

    bc_weights = {k: v for k, v in bc_weights.items() if int(k) in nodes.index}

    node_color_map = _get_color_map(min(bc_weights.values()), max(bc_weights.values()))
    edge_color_map = _get_color_map(min(highway_weights.values()), max(highway_weights.values()))

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

            color = edge_color_map(highway_weights[row['highway']])
            folium.PolyLine([start_point, end_point], color=color, weight=2, opacity=0.5).add_to(m)
    return m


if __name__ == '__main__':
    weights = pd.read_csv("test.csv", index_col=0)
    g = ox.graph_from_place("Kraków, Poland", network_type="drive")

    area = ObservableArea.agh_area
    map = plot_graph_on_map(g, weights, area)

