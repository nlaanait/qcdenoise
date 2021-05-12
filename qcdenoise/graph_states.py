import logging

import networkx as nx

from .graph_data import GraphDB
import qcdenoise


__all__ = ["GraphState"]

# module logger
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    "dataset- %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# module plotting options
_plots = qcdenoise.__plots__
if _plots:
    import matplotlib.pyplot as plt
    nx_plot_options = {
        'with_labels': True,
        'node_color': 'red',
        'node_size': 175,
        'width': 2,
        'font_weight': 'bold',
        'font_color': 'white',
    }
else:
    nx_plot_options = None


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
    def __init__(self, graph_db: GraphDB, n_qubits: int) -> None:
        self.graph_db = graph_db
        assert n_qubits >= 2, logger.error(f"n_qubits={n_qubits} < 2")
        self.n_qubits = n_qubits
        self.all_graphs = graph_db.get_edge_sorted_graphs()
        self.largest_subgraph = self._check_largest()

    def build_graph(self, circuit_graph=None, graph_plot=False):
        if circuit_graph is None:
            # 1. Pick a random combination of subgraphs
            sub_graphs = self.pick_subgraphs()
            # 2. Combine subgraphs into a single circuit graph
            circuit_graph = self.combine_subgraphs(sub_graphs)
        self.circuit_graph = circuit_graph
        if self.edge_list is None:
            self.edge_list = self.circuit_graph.edges()
        if graph_plot and _plots:
            nx.draw_circular(circuit_graph, **nx_plot_options)

    def _check_largest(self, val):
        for key, itm in self.all_graphs.items():
            if len(itm) != 0:
                max_subgraph = key
        if val is None:
            return max_subgraph
        elif val > max_subgraph:
            .warn(
                "The largest possible subgraph in the database has %s nodes" %
                max_subgraph)
            warnings.warn(
                "Resetting largest possible subgraph: %s --> %s" %
                (val, max_subgraph))
            return max_subgraph
        return val

    def generate_all_subgraphs(self):
        combs = list(
            set(partitions(self.n_qubits, start=self.smallest_subgraph)))
        for (itm, comb) in enumerate(combs):
            if any([itm > self.largest_subgraph for itm in comb]):
                combs.pop(itm)
        if len(combs) == 0:
            raise ValueError(
                "Empty list of subgraph combinations. Circuit cannot be constructed as specified.")
        return combs

    def combine_subgraphs(self, sub_graphs):
        if self.directed:
            union_graph = nx.DiGraph()
        else:
            union_graph = nx.Graph()
        for sub_g in sub_graphs:
            union_nodes = len(union_graph.nodes)
            if union_nodes > 1:
                first_node = random.randint(
                    0, union_graph.order() - 1)
                second_nodes = np.random.randint(union_graph.order(), sub_g.order() + union_graph.order() - 1,
                                                 sub_g.order())
                union_graph = nx.disjoint_union(union_graph, sub_g)
                for idx in second_nodes:
                    union_graph.add_weighted_edges_from(
                        [(first_node, idx, 1.0)])
            else:
                union_graph = nx.disjoint_union(union_graph, sub_g)
        return union_graph

    def pick_subgraphs(self):
        comb_idx = random.randint(0, len(self.graph_combs) - 1)
        comb = self.graph_combs[comb_idx]
        self.print_verbose(
            "Configuration with {} Subgraphs with # nodes:{}".format(
                len(comb), comb))
        sub_graphs = []
        for num_nodes in comb:
            sub_g = self.all_graphs[num_nodes]
            idx = random.randint(0, len(sub_g) - 1)
            sub_graphs.append(sub_g[idx])
        return sub_graphs
