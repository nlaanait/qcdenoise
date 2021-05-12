import pytest

from qcdenoise import GraphData, partition_GraphData, GraphDB


@pytest.fixture()
def get_data():
    data = {'1': [(1, 2, 1.0)],
            '2': [(1, 2, 1.0), (1, 3, 1.0)],
            '3': [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0)],
            '4': [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)],
            '5': [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)],
            '6': [(1, 2, 1.0), (2, 3, 1.0), (2, 5, 1.0), (3, 4, 1.0)],
            '7': [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0)],
            '8': [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 1, 1.0)],
            '9': [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 4, 1.0), (1, 6, 1.0)],
            '10': [(1, 6, 1.0), (2, 6, 1.0), (3, 6, 1.0), (4, 5, 1.0), (5, 6, 1.0)],
            '11': [(1, 6, 1.0), (2, 6, 1.0), (3, 5, 1.0), (4, 5, 1.0), (5, 6, 1.0)]
            }
    grouped_data = {'1': [data['1']],
                    '2': [data['2']],
                    '3': [data['3'], data['4']],
                    '4': [data['5'], data['6'], data['7']],
                    '5': [data['8'], data['9'], data['10'], data['11']]
                    }
    data_part_1 = {
        0: [(1, 2, 1.0)],
        1: [(1, 2, 1.0), (1, 3, 1.0)],
        2: [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0)],
        3: [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)],
        4: [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0)],
        5: [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)],
        6: [(1, 6, 1.0), (2, 6, 1.0), (3, 5, 1.0), (4, 5, 1.0), (5, 6, 1.0)],
        7: [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 1, 1.0)]}

    data_part_2 = {
        0: [(1, 2, 1.0)],
        1: [(1, 2, 1.0), (1, 3, 1.0)],
        2: [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0)],
        3: [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)],
        4: [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0)],
        5: [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 5, 1.0)],
        6: [(1, 2, 1.0), (1, 3, 1.0), (1, 4, 1.0), (1, 4, 1.0), (1, 6, 1.0)],
        7: [(1, 6, 1.0), (2, 6, 1.0), (3, 6, 1.0), (4, 5, 1.0), (5, 6, 1.0)]}

    return data, grouped_data, data_part_1, data_part_2


@ pytest.mark.dependency()
def test_graph_data(get_data):
    data, grouped_data, _, _ = get_data
    graph_data = GraphData(
        data=data)
    # test attributes
    assert graph_data.data == data
    assert list(graph_data.keys()) == list(data.keys())
    assert list(graph_data.items()) == list(data.items())

    # test methods
    graph_data.groupby_edges()
    assert graph_data.data == grouped_data


@ pytest.mark.dependency()
def test_graph_partition(get_data):
    data, _, part_1, part_2 = get_data
    graph_data = GraphData(data=data)
    graph_data_1, graph_data_2 = partition_GraphData(
        graph_data, ratio=0.5)
    assert graph_data_1.data == part_1
    assert graph_data_2.data == part_2


@ pytest.mark.dependency(depends=["test_graph_data",
                                  "test_graph_partition"])
def test_graph_db(get_data):
    data, _, _, _ = get_data
    graph_data = GraphData(data=data)
    graph_db = GraphDB(graph_data=graph_data)
    assert list(graph_db.keys()) == list(data.keys())
