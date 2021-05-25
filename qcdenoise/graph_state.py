"""Module for generating random graph states from a graph database
"""
import logging
import random
from typing import Union

import networkx as nx
import numpy as np
from networkx.algorithms.approximation import vertex_cover


from .graph_data import GraphDB
from .config import _plots, nx_plot_options

__all__ = ["GraphState"]

if _plots:
    import matplotlib.pyplot as plt

# module logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    f"{__name__}- %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def partitions(n, start=2):
    """Return all possible partitions of integer n

    Arguments:
        n {int} -- integer

    Keyword Arguments:
        start {int} -- starting position (default: {2})

    Yields:
        tuple -- returns tuples of partitions
    """
    yield(n,)
    for i in range(start, n // 2 + 1):
        for p in partitions(n - i, i):
            yield (i,) + p


class GraphState:
    """Construct a Graph State from a Graph Database
    """

    def __init__(self, graph_db: GraphDB,
                 n_qubits: int = 5,
                 min_num_edges: int = 2,
                 max_num_edges: int = 7,
                 directed: bool = False) -> None:
        """initialize

        Args:
            graph_db (GraphDB): graph database
            n_qubits (int, optional): # of qubits. Defaults to 5.
            min_num_edges (int, optional): minimum graph order. Defaults to 3.
            max_num_edges (int, optional): maximum graph order. Defaults to 5.
            directed (bool, optional): construct a directed graph.
            Defaults to False.
        """
        self.graph_db = graph_db
        assert n_qubits >= 2, logger.error(f"n_qubits={n_qubits} < 2")
        self.n_qubits = n_qubits
        self.all_graphs = graph_db.get_edge_sorted_graphs()
        self.largest_subgraph = self._check_largest(max_num_edges)
        if min_num_edges > self.n_qubits:
            logger.warning(
                f"min_num_edges({min_num_edges}) > n_qubits({self.n_qubits})" +
                ", resetting min_num_edges.")
            min_num_edges = int((self.n_qubits // 2) - 1)
        self.smallest_subgraph = max(2, min_num_edges)
        self.graph_combs = self._generate_all_subgraphs()
        self.directed = directed

    def sample(self,
               min_vertex_cover: int = 1,
               max_itr: int = 10,
               graph_plot: bool = False) -> Union[nx.Graph, nx.DiGraph]:
        """Sample a graph state subject to vertex cover constraints

        Args:
            min_vertex_cover (int, optional): minimum vertex cover for returned
            graph state. Defaults to 1.
            max_itr (int, optional): maximum # of tries to sample a graph state
             satisfying min_vertex_cover condition  . Defaults to 10.
            graph_plot (bool, optional): plot the constructed graph.
            Defaults to False.

        Returns:
            Union[nx.Graph, nx.DiGraph]: graph state
        """
        itr = 0
        while itr < max_itr:
            # 1. Pick a random combination of subgraphs
            sub_graphs = self.pick_subgraphs()

            # 2. Combine subgraphs into a single circuit graph
            graph_state = self.combine_subgraphs(sub_graphs)

            # 3. Screen based on size of minimum vertex cover
            min_cover = vertex_cover.min_weighted_vertex_cover(
                graph_state, weight="weight")

            # FIXME: only valid when weights of all nodes are 1
            min_cover = len(min_cover) // 2
            logger.debug(
                f"graph state iteration={itr}, vertex cover={min_cover}")

            # repeat (up to `max_itr`) to find a graph with a minimum
            if min_cover >= min_vertex_cover:
                break
            elif itr == max_itr:
                logger.warn(
                    "sampled graph state does not satisfy min_vertex_cover.")
                break
            itr += 1

        if graph_plot and _plots:
            plt.figure(figsize=(2, 2))
            nx.draw_circular(graph_state, **nx_plot_options)
            plt.title('Graph State', loc='right')
            plt.show()
        return graph_state

    def _check_largest(self, val: int) -> int:
        """Check that input value matches the largest # of graph nodes in the database

        Args:
            val (int): input value to check

        Returns:
            int: input value or largest # of graph nodes in the database
            if different
        """
        for key, itm in self.all_graphs.items():
            if len(itm) != 0:
                max_subgraph = key
        if val is None:
            return max_subgraph
        elif val > max_subgraph:
            logger.warning(
                "The largest possible subgraph in the database has" +
                f"{max_subgraph} nodes")
            logger.warning(
                f"Resetting largest subgraph: {val} --> {max_subgraph}")
            return max_subgraph
        return val

    def _generate_all_subgraphs(self) -> list:
        """Return all possible (unique) subgraph combinations to build the
        desired graph

        Raises:
            ValueError: if there are no combinations of subgraphs which
            generate the desired graph

        Returns:
            list: subgraph combinations
        """
        combs = list(
            set(partitions(self.n_qubits, start=self.smallest_subgraph)))
        valid_combs = []
        if len(combs) == 0:
            raise ValueError(
                "Empty list of subgraph combinations." +
                "Circuit cannot be constructed as specified.")
        for comb in combs:
            cond_bigger = all(
                [itm <= self.largest_subgraph for itm in comb])
            cond_smaller = all(
                [itm >= self.smallest_subgraph for itm in comb])
            if cond_smaller and cond_bigger:
                valid_combs.append(comb)
        logger.debug(f"subgraphs combinations {valid_combs}")
        return valid_combs

    def combine_subgraphs(
            self, sub_graphs: list) -> Union[nx.Graph, nx.DiGraph]:
        if self.directed:
            union_graph = nx.DiGraph()
        else:
            union_graph = nx.Graph()
        for sub_g in sub_graphs:
            union_nodes = len(union_graph.nodes)
            if union_nodes > 1:
                first_node = random.randint(
                    0, union_graph.order() - 1)
                low_order = union_graph.order()
                high_order = sub_g.order() + union_graph.order() - 1
                second_nodes = np.random.randint(low=low_order,
                                                 high=high_order,
                                                 size=sub_g.order())
                union_graph = nx.disjoint_union(union_graph, sub_g)
                for idx in second_nodes:
                    union_graph.add_weighted_edges_from(
                        [(first_node, idx, 1.0)])
            else:
                union_graph = nx.disjoint_union(union_graph, sub_g)
        return union_graph

    def pick_subgraphs(self):
        """Generate a combination of subgraphs

        Returns:
            list: list of subgraphs
        """
        comb_idx = random.randint(0, len(self.graph_combs) - 1)
        comb = self.graph_combs[comb_idx]
        logger.debug(
            f"Sampled configuration:{len(comb)} Subgraphs with # nodes:{comb}")
        sub_graphs = []
        for num_nodes in comb:
            sub_g = self.all_graphs[num_nodes]
            idx = random.randint(0, len(sub_g) - 1)
            sub_graphs.append(sub_g[idx])
        return sub_graphs
