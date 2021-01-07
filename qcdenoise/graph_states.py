import warnings
import random
import math

import networkx as nx
import numpy as np

warnings.simplefilter("ignore")

try: 
    import matplotlib.pyplot as plt
    _plots = True
except ImportError:
    warnings.warn("matplotlib could not be imported- skipping plotting.")
    _plots = False

def offset(l):
    l = [(edge[0]-1, edge[1]-1, edge[-1]) for edge in l]
    return l

nx_plot_options = {
            'with_labels': True,
            'node_color': 'red',
            'node_size': 175,
            'width': 2,
            'font_weight':'bold',
            'font_color': 'white',
            }
global seed
seed = 1234
np.random.seed(seed)

class GraphData:
    """The keys correspond to the graph numbers in Table V in arXiv:060296 (Hein et al.)
    """
    def __init__(self, data=None):
        if data is None:
            self.data = dict([('%d' %d, None) for d in range(1,46)])
            self.data['1'] =  [(1, 2, 1.0)]
            self.data['2'] =  [(1, 2, 1.0), (1, 3, 1.0)]
            self.data['3'] =  [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0)]
            self.data['4'] =  [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
            self.data['5'] =  [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)]
            self.data['6'] =  [(1, 2, 1.0), (2, 3, 1.0), (2, 5, 1.0), (3, 4, 1.0)]
            self.data['7'] =  [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0)]
            self.data['8'] =  [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 1, 1.0)]
            self.data['9'] =  [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 4, 1.0), (1, 6, 1.0)]
            self.data['10'] = [(1, 6, 1.0), (2, 6, 1.0), (3, 6, 1.0), (4, 5, 1.0), (5, 6, 1.0)]
            self.data['11'] = [(1, 6, 1.0), (2, 6, 1.0), (3, 5, 1.0), (4, 5, 1.0), (5, 6, 1.0)]
            self.data['12'] = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (6, 2, 1.0)]
            self.data['13'] = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (3, 6, 1.0)]
            self.data['14'] = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0)]
            self.data['15'] = [(1, 6, 1.0), (2, 4, 1.0), (3, 4, 1.0), (3, 6, 1.0), (4, 5, 1.0), (5, 6, 1.0)]
            self.data['16'] = [(1, 2, 1.0), (2, 3, 1.0), (2, 4, 1.0), (3, 4, 1.0), (2, 6, 1.0), (4, 5, 1.0)]
            self.data['17'] = [(1, 2, 1.0), (1, 5, 1.0), (1, 6, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0)]
            self.data['18'] = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 1, 1.0)]
            self.data['19'] = [(1, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0), (2, 5, 1.0), (3, 4, 1.0), (4, 5, 1.0), (4, 6, 1.0), (5, 6, 1.0), (6, 1, 1.0)]
            self.data['20'] = [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0), (1, 6, 1.0), (1, 7, 1.0)]
            self.data['21'] = [(1, 7, 1.0), (7, 2, 1.0), (7, 3, 1.0), (7, 4, 1.0), (5, 6, 1.0), (6, 7, 1.0)]
            self.data['22'] = [(1, 7, 1.0), (7, 2, 1.0), (7, 3, 1.0), (7, 6, 1.0), (6, 4, 1.0), (6, 5, 1.0)]
            self.data['23'] = [(1, 7, 1.0), (7, 2, 1.0), (7, 3, 1.0), (7, 6, 1.0), (6, 5, 1.0), (5, 4, 1.0)]
            self.data['24'] = [(1, 7, 1.0), (7, 6, 1.0), (7, 2, 1.0), (6, 5, 1.0), (5, 3, 1.0), (5, 4, 1.0)]
            self.data['25'] = [(1, 2, 1.0), (1, 7, 1.0), (7, 3, 1.0), (7, 4, 1.0), (7, 6, 1.0), (6, 5, 1.0)]
            self.data['26'] = [(1, 7, 1.0), (7, 2, 1.0), (7, 6, 1.0), (6, 3, 1.0), (6, 5, 1.0), (5, 4, 1.0)]
            self.data['27'] = [(1, 2, 1.0), (2, 7, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0)]
            self.data['28'] = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (3, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)]
            self.data['29'] = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (3, 6, 1.0), (6, 7, 1.0)]
            self.data['30'] = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)]
            self.data['31'] = [(1, 3, 1.0), (2, 3, 1.0), (3, 4, 1.0), (3, 6, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 3, 1.0), (5, 7 , 1.0)]
            self.data['32'] = [(1, 7, 1.0), (2, 7, 1.0), (3, 6, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)]
            self.data['33'] = [(1, 3, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (7, 3, 1.0)]          
            self.data['34'] = [(1, 4, 1.0), (2, 3, 1.0), (3, 4, 1.0), (3, 6, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)]
            self.data['35'] = [(1, 6, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (7, 3, 1.0)]
            self.data['36'] = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (3, 5, 1.0), (4, 5, 1.0), (4, 7, 1.0), (5, 6, 1.0)]
            self.data['37'] = [(1, 7, 1.0), (7, 6, 1.0), (6, 5, 1.0), (5, 4, 1.0), (4, 3, 1.0), (3, 2, 1.0), (3, 7, 1.0)]
            self.data['38'] = [(1, 6, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)]
            self.data['39'] = [(1, 5, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)]
            self.data['40'] = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (7, 1, 1.0)]
            self.data['41'] = [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (5, 1, 1.0), (6, 1, 1.0), (6, 7, 1.0)]
            self.data['42'] = [(1, 3, 1.0), (1, 7, 1.0), (2, 3, 1.0), (2, 6, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)]
            self.data['43'] = [(1, 2, 1.0), (1, 4, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 3, 1.0), (7, 1, 1.0)]
            self.data['44'] = [(1, 4, 1.0), (1, 7, 1.0), (2, 3, 1.0), (2, 7, 1.0), (3, 4, 1.0), (3, 5, 1.0), (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0)]
            self.data['45'] = [(1, 2, 1.0), (2, 3, 1.0), (2, 5, 1.0), (2, 7, 1.0), (3, 4, 1.0), (3, 7, 1.0), (4, 5, 1.0), (5, 6, 1.0), (4, 6, 1.0), (6, 7, 1.0)]                    
        else:
            self.data = dict([(d, None) for d in range(len(data))])
            for (idx, datum) in enumerate(data):
                self.data[idx] = datum

    def partition(self, ratio=0.2, nonzero=True):
        """Two-way split of database
        
        Keyword Arguments:
            ratio {float} -- ratio between test and train (default: {0.8})
            nonzero {bool} -- ensures that each partition has at least 1 element per # of edges category (default: {True})
        
        Returns:
            (GraphData, GraphData) -- The GraphData for the train and test datasets
        """
        # group data into categories defined by the # of edges
        test_g_data = []
        train_g_data = []
        grouped_g_data = {}
        for key, itm in self.data.items():
            n_nodes = len(itm)
            if n_nodes in grouped_g_data.keys():
                grouped_g_data[n_nodes].append(itm)
            else:
                grouped_g_data[n_nodes] = [itm]

        # shuffle each category and 2-way split
        for idx, (key, itm) in enumerate(grouped_g_data.items()):
            if len(itm)> 2:
                random.shuffle(itm)
                part = math.ceil(len(itm) * ratio)
                test = itm[:part]
                train = itm[part:]
                if len(train) > 1 :
                    test_g_data.append(test)
                    train_g_data.append(train)
                else:
                    test_g_data.append(test)
                    train_g_data.append(test)
            else:
                test_g_data.append(itm)
                train_g_data.append(itm)
        print("Test Data- Number of graph examples per # of edges in graph: ", [len(itm) for itm in test_g_data])
        print("Train Data- Number of graph examples per # of edges in graph: ",[len(itm) for itm in train_g_data]) 
        
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
        print("Number of Graphs in Train Database: ", len(train_data))
        print("Number of Graphs in Test Database: ", len(test_data))
        return GraphData(data=train_data), GraphData(data=test_data)

class GraphDB:
    def __init__(self, graph_data=None):
        self.graph_data = graph_data if graph_data is not None else GraphData().data
        self.graph = self._build_graphDB()

    def _build_graphDB(self):
        graph_db= dict([('%d' %d, {'G':None, 'V':None, 'LUclass':None, '2Color':None}) 
                            for d in range(1,len(self.graph_data.keys()) + 1)])
        for (_, g_entry), (_, g_data) in zip(graph_db.items(), self.graph_data.items()):
            if g_data:
                g_data = offset(g_data)
                G = nx.DiGraph()
                G.add_weighted_edges_from(g_data)
                g_entry['G'] = G
                g_entry['V'] = len(G.nodes)
        return graph_db

    def plot_graph(self, graph_number=[1]):
        graph_number = graph_number if graph_number is not None else list(self.graph.keys())
        if _plots:
            plt.figure(figsize=(2,2))
            for g_num in graph_number:
                plt.clf()
                G = self.graph[str(g_num)]['G']
                if _plots:
                    nx.draw_circular(G, **nx_plot_options)
                    plt.title('No. %s' % g_num, loc='right')
                    plt.show()

    def test_graph_build(self, graph_number=1):
        """Tests graph building. This is useful when testing a new GraphData object
           1. Building the graph
           2. Printing nodes, neighbors and weight values
           3. Plotting the built graph.     

           if `graph_number` is None then all graphs in graph_data are built and tested
        
        Keyword Arguments:
            graph_number {int} -- [description] (default: {1})
        """
        if _plots:
            plt.figure(figsize=(2,2))
        for g_num, g_data in self.graph_data.items():
            plt.clf()
            cond = True
            if graph_number is not None:
                cond = g_num == str(graph_number)
            if g_data and cond:
                G = nx.DiGraph()
                G.add_weighted_edges_from(g_data)
                # print nodes and neighbors
                for node, ngbrs in G.adjacency():
                    for ngbr, edge_attr in ngbrs.items():
                        print("Node:{}, Neighbor:{}, Weight:{}".format(node, ngbr, edge_attr["weight"]))
                # plot
                if _plots:
                    nx.draw_circular(G, **nx_plot_options)
                    plt.title('No. %s' % g_num, loc='right')
                    plt.show()

    def __getitem__(self, key):
        return self.graph[key]
    
    def __getattr__(self, keys):
        return self.graph.keys
    
    def __getattr__(self, items):
        return self.graph.items
