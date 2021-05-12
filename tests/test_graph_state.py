import pytest

import qcdenoise as qcd

# @pytest.fixture()


def get_graphDB():
    graph_db = qcd.GraphDB()
    return graph_db


@pytest.fixture()
def get_test_graph_state():
    pass


@pytest.mark.dependency()
def test_graph_state(get_graphDB):
    graph_db = get_graphDB
    graph_state = qcd.GraphState(graph_db=graph_db, n_qubits=7)
    graph = graph_state.sample()
    if graph:
        assert True


if __name__ == "__main__":
    graph_db = get_graphDB()
    test_graph_state(graph_db)
