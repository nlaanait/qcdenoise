"""module for graph data and graph database
"""
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import networkx as nx

from .config import _plots, nx_plot_options, get_module_logger

if _plots:
    import matplotlib.pyplot as plt

__all__ = [
    "GraphData",
    "HeinGraphData",
    "partition_GraphData",
    "GraphDB"]

# module logger
logger = get_module_logger(__name__)


def offset(edge_list: list) -> list:
    """offset edge ordering

    Args:
        edge_list (list): list of tuples defining edges

    Returns:
        new_l: offset list of tuples defining edges
    """
    new_l = [(edge[0] - 1, edge[1] - 1, edge[-1])
             for edge in edge_list]
    return new_l


@dataclass(init=True)
class GraphData:
    """Stores data to define a graph via edges and their weights
    """
    data: Dict[str, List[Tuple]]

    def groupby_edges(self) -> None:
        """Groups data based on the # of edges per entry
        """
        grouped_g_data = {}
        for _, itm in self.data.items():
            n_nodes = len(itm)
            if str(n_nodes) in grouped_g_data.keys():
                grouped_g_data[str(n_nodes)].append(itm)
            else:
                grouped_g_data[str(n_nodes)] = [itm]
        self.data = grouped_g_data

    def __getattr__(self, name: str) -> Any:
        if name == "keys":
            return self.data.keys
        elif name == "items":
            return self.data.items


# Graph data from Hein et al. arXiv:060296.
# @article{undefined,
# year = {2006},
# title = {{Entanglement in Graph States and its Applications}},
# author = {Hein, M and Dür, W and Eisert, J and Raussendorf, R and Nest, M Van den and Briegel, H -J},
# journal = {arXiv},
# eprint = {quant-ph/0602096},
# keywords = {},
# }
# The keys correspond to the graph numbers in Table V
HeinGraphData = GraphData(data={'1': [(1, 2, 1.0)],
                                '2': [(1, 2, 1.0), (1, 3, 1.0)],
                                '3': [
    (1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0)],
    '4': [
    (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)],
    '5': [
    (1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)],
    '6': [
    (1, 2, 1.0), (2, 3, 1.0), (2, 5, 1.0), (3, 4, 1.0)],
    '7': [
    (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0)],
    '8': [
    (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 1, 1.0)],
    '9': [
    (1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 4, 1.0), (1, 6, 1.0)],
    '10': [
    (1, 6, 1.0), (2, 6, 1.0), (3, 6, 1.0), (4, 5, 1.0), (5, 6, 1.0)],
    '11': [
    (1, 6, 1.0), (2, 6, 1.0), (3, 5, 1.0), (4, 5, 1.0), (5, 6, 1.0)],
    '12': [
    (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (6, 2, 1.0)],
    '13': [
    (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (3, 6, 1.0)],
    '14': [
    (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0)],
    '15': [
    (1, 6, 1.0), (2, 4, 1.0), (3, 4,
                               1.0), (3, 6, 1.0), (4, 5, 1.0),
    (5, 6, 1.0)],
    '16': [
    (1, 2, 1.0), (2, 3, 1.0), (2, 4,
                               1.0), (3, 4, 1.0), (2, 6, 1.0),
    (4, 5, 1.0)],
    '17': [
    (1, 2, 1.0), (1, 5, 1.0), (1, 6,
                               1.0), (2, 3, 1.0), (3, 4, 1.0),
    (4, 5, 1.0)],
    '18': [
    (1, 2, 1.0), (2, 3, 1.0), (3, 4,
                               1.0), (4, 5, 1.0), (5, 6, 1.0),
    (6, 1, 1.0)],
    '19': [(1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0), (2, 5, 1.0), (
        3, 4, 1.0), (4, 5, 1.0), (4, 6, 1.0), (5, 6, 1.0), (6, 1, 1.0)],
    '20': [
    (1, 2, 1.0), (1, 3, 1.0), (1, 4,
                               1.0), (1, 5, 1.0), (1, 6, 1.0),
    (1, 7, 1.0)],
    '21': [
    (1, 7, 1.0), (7, 2, 1.0), (7, 3,
                               1.0), (7, 4, 1.0), (5, 6, 1.0),
    (6, 7, 1.0)],
    '22': [
    (1, 7, 1.0), (7, 2, 1.0), (7, 3,
                               1.0), (7, 6, 1.0), (6, 4, 1.0),
    (6, 5, 1.0)],
    '23': [
    (1, 7, 1.0), (7, 2, 1.0), (7, 3,
                               1.0), (7, 6, 1.0), (6, 5, 1.0),
    (5, 4, 1.0)],
    '24': [
    (1, 7, 1.0), (7, 6, 1.0), (7, 2,
                               1.0), (6, 5, 1.0), (5, 3, 1.0),
    (5, 4, 1.0)],
    '25': [
    (1, 2, 1.0), (1, 7, 1.0), (7, 3,
                               1.0), (7, 4, 1.0), (7, 6, 1.0),
    (6, 5, 1.0)],
    '26': [
    (1, 7, 1.0), (7, 2, 1.0), (7, 6,
                               1.0), (6, 3, 1.0), (6, 5, 1.0),
    (5, 4, 1.0)],
    '27': [
    (1, 2, 1.0), (2, 7, 1.0), (2, 3,
                               1.0), (3, 4, 1.0), (4, 5, 1.0),
    (5, 6, 1.0)],
    '28': [
    (1, 2, 1.0), (2, 3, 1.0), (3, 4,
                               1.0), (3, 5, 1.0), (5, 6, 1.0),
    (6, 7, 1.0)],
    '29': [
    (1, 2, 1.0), (2, 3, 1.0), (3, 4,
                               1.0), (4, 5, 1.0), (3, 6, 1.0),
    (6, 7, 1.0)],
    '30': [
    (1, 2, 1.0), (2, 3, 1.0), (3, 4,
                               1.0), (4, 5, 1.0), (5, 6, 1.0),
    (6, 7, 1.0)],
    '31': [(1, 3, 1.0), (2, 3, 1.0), (3, 4, 1.0), (
        3, 6, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 3, 1.0), (5, 7, 1.0)],
    '32': [
    (1, 7, 1.0), (2, 7, 1.0), (3, 6,
                               1.0), (4, 5, 1.0), (5, 6, 1.0),
    (6, 7, 1.0)],
    '33': [(1, 3, 1.0), (2, 3, 1.0), (3, 4, 1.0),
           (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (7, 3, 1.0)],
    '34': [(1, 4, 1.0), (2, 3, 1.0), (3, 4, 1.0),
           (3, 6, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)],
    '35': [(1, 6, 1.0), (2, 3, 1.0), (3, 4, 1.0),
           (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (7, 3, 1.0)],
    '36': [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
           (3, 5, 1.0), (4, 5, 1.0), (4, 7, 1.0), (5, 6, 1.0)],
    '37': [(1, 7, 1.0), (7, 6, 1.0), (6, 5, 1.0),
           (5, 4, 1.0), (4, 3, 1.0), (3, 2, 1.0), (3, 7, 1.0)],
    '38': [(1, 6, 1.0), (1, 2, 1.0), (2, 3, 1.0),
           (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)],
    '39': [(1, 5, 1.0), (1, 2, 1.0), (2, 3, 1.0),
           (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)],
    '40': [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0),
           (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (7, 1, 1.0)],
    '41': [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (
        4, 5, 1.0), (5, 6, 1.0), (5, 1, 1.0), (6, 1, 1.0), (6, 7, 1.0)],
    '42': [(1, 3, 1.0), (1, 7, 1.0), (2, 3, 1.0), (
        2, 6, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)],
    '43': [(1, 2, 1.0), (1, 4, 1.0), (2, 3, 1.0), (
        3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 3, 1.0), (7, 1, 1.0)],
    '44': [(1, 4, 1.0), (1, 7, 1.0), (2, 3, 1.0), (2, 7, 1.0), (
        3, 4, 1.0), (3, 5, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)],
    '45': [(1, 2, 1.0), (2, 3, 1.0), (2, 5, 1.0), (2, 7, 1.0), (
        3, 4, 1.0), (3, 7, 1.0), (4, 5, 1.0), (5, 6, 1.0), (4, 6, 1.0),
    (6, 7, 1.0)]})


def partition_GraphData(
        graph_data: GraphData,
        ratio: float = 0.2,
        not_empty: bool = True) -> Tuple[GraphData, GraphData]:
    """Two-way split of GraphData. Split is balanced across # of edges per graph

    Args:
        graph_data (GraphData): input GraphData to be split
        ratio (float) : ratio of test to train (default: {0.8})
        not_empty (bool) : ensures that each partition has at least 1 element

    Returns:
        (GraphData, GraphData) -- GraphData for the train and test datasets
    """
    test_g_data = []
    train_g_data = []
    graph_data.groupby_edges()
    # grouped_g_data = graph_data.groupby_edges()

    # shuffle each category and 2-way split
    for _, itm in graph_data.items():
        if len(itm) > 2:
            random.shuffle(itm)
            part = math.ceil(len(itm) * ratio)
            test = itm[: part]
            train = itm[part:]
            if len(train) > 1:
                test_g_data.append(test)
                train_g_data.append(train)
            else:
                test_g_data.append(test)
                train_g_data.append(test)
        elif not_empty:
            test_g_data.append(itm)
            train_g_data.append(itm)
        else:
            part = random.choices(
                ["test", "train"], weights=[0.5, 0.5])
            if part == "test":
                test_g_data.append(itm)
            else:
                train_g_data.append(itm)
    logger.debug("Test Data -  # of graph examples per # of edges in graph:" +
                 f"{[len(itm) for itm in test_g_data]}")
    logger.debug("Train Data- # of graph examples per # of edges in graph:" +
                 f"{[len(itm) for itm in train_g_data]}")

    # collapse categories into a single list
    test_data = []
    for items in test_g_data:
        if len(items) > 1 and isinstance(items[0], list):
            for itm in items:
                test_data.append(itm)
        else:
            test_data.append(items[0])

    train_data = []
    for items in train_g_data:
        if len(items) > 1 and isinstance(items[0], list):
            for itm in items:
                train_data.append(itm)
        else:
            train_data.append(items[0])
    logger.info(
        f"# of Graphs in Train Partition: {len(train_data)}. " +
        f"# of Graphs in Test Partition: {len(test_data)}")
    train_data = GraphData(
        data={
            key: item for key,
            item in enumerate(train_data)})
    test_data = GraphData(
        data={
            key: item for key,
            item in enumerate(test_data)})
    return train_data, test_data


class GraphDB:
    def __init__(self, graph_data: GraphData = HeinGraphData,
                 directed: bool = False):
        self.directed = directed
        self.graph_data = graph_data
        if graph_data == HeinGraphData:
            logger.info(
                "Building database with Graphs from Hein et al. ")
        self.db = self._build_graphDB()

    def _build_graphDB(self):
        """populate a dictionary out of networkx graphs

        Returns:
            graph_db (dict): built dictionary
        """
        graph_db = dict([('%d' % d, {'G': None, 'V': None, 'LUclass': None, '2Color': None})
                         for d in range(1, len(self.graph_data.keys()) + 1)])
        for (_, g_entry), (key, g_data) in zip(
                graph_db.items(), self.graph_data.items()):
            if g_data:
                g_data = offset(g_data)
                if self.directed:
                    G = nx.DiGraph()
                else:
                    G = nx.Graph()
                G.add_weighted_edges_from(g_data)
                g_entry['G'] = G
                g_entry['V'] = len(G.nodes)
                logger.debug(
                    f"loading graph {key} with # of vertices={g_entry['V']}")
        return graph_db

    def plot_graph(self, graph_number: List[int] = [1]) -> None:
        """plot the graph

        Args:
            graph_number (list, optional): list of keys whose graphs will be
            plotted. Defaults to [1].
        """
        graph_number = list(
            self.db.keys()) if graph_number is None else graph_number
        if _plots:
            plt.figure(figsize=(2, 2))
            for g_num in graph_number:
                plt.clf()
                G = self.db[str(g_num)]['G']
                if _plots:
                    nx.draw_circular(G, **nx_plot_options)
                    plt.title('No. %s' % g_num, loc='right')
                    plt.show()
        else:
            logger.warning(
                "matplotlib could not be imported- skipping plots.")

    def test_graph_build(self, graph_number: int = 1) -> None:
        """Tests graph building. This is useful when testing a new GraphData object
           1. Building the graph
           2. Printing nodes, neighbors and weight values
           3. Plotting the built graph.

           if `graph_number` is None then all graphs in graph_data are built and tested

        Keyword Arguments:
            graph_number {int} -- [description] (default: {1})
        """
        if _plots:
            plt.figure(figsize=(2, 2))
            for g_num, g_data in self.graph_data.items():
                plt.clf()
                cond = True
                if graph_number is not None:
                    cond = g_num == str(graph_number)
                if g_data and cond:
                    if self.directed:
                        G = nx.DiGraph()
                    else:
                        G = nx.Graph()
                    G.add_weighted_edges_from(g_data)
                    # log nodes and neighbors
                    for node, ngbrs in G.adjacency():
                        for ngbr, edge_attr in ngbrs.items():
                            logger.info(f"Node:{node}, Neighbor:{ngbr}," +
                                        f"Weight:{edge_attr['weight']}")
                    # plot
                    nx.draw_circular(G, **nx_plot_options)
                    plt.title('No. %s' % g_num, loc='right')
                    plt.show()
        else:
            logger.warning(
                "matplotlib could not be imported- skipping plots.")

    def get_edge_sorted_graphs(self) -> Dict:
        """return a dictionary with keys/vals assigned to graphs with
        same # of edges

        Returns:
            dict: sorted graph dictionary
        """
        sorted_graphs = {
            i: [] for i in range(
                1, len(
                    self.db.keys()))}
        for _, itm in self.db.items():
            if itm["G"] is not None:
                sorted_key = len(itm["G"].nodes)
                sorted_graphs[sorted_key].append(itm["G"])
        return sorted_graphs

    def __getitem__(self, key: str) -> dict:
        return self.db[key]

    def __getattr__(self, name: str) -> Any:
        if name == "keys":
            return self.db.keys
        elif name == "items":
            return self.db.items
