from copy import deepcopy
from qiskit.quantum_info.operators import Operator
from uncertainties import unumpy
from uncertainties.umath import *
from uncertainties import ufloat
import qiskit.ignis.mitigation as igmit
import qiskit as qk
from itertools import combinations, product
import random
from typing import List
import warnings
from datetime import datetime
import networkx as nx
import numpy as np
import pandas as pd

from .graph_states import GraphDB, _plots, nx_plot_options
from .stabilizer_measurement import sigma_prod, build_stabilizer_meas, get_unique_operators
# from .graph_functions import partitions, d2_neighborhood
d2_neighborhood = None


# for stochastic noise operator

global seed
seed = 1234
np.random.seed(seed)


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


class CircuitConstructor:
    """Parent class of constructing circuits

    Raises:
        NotImplementedError: build_circuit() must be overriden by child class
        NotImplementedError: estimate_entanglement() must be overriden by child class

        options for layout_method: [‘trivial’, ‘dense’, ‘noise_adaptive’, ‘sabre’]
    """

    def __init__(self, n_qubits=2, n_shots=1024, verbose=False,
                 state_simulation=False, backend=None, backend_name=None,
                 layout=None, layout_method=None, track_gates=True, **kwargs):
        assert n_qubits >= 2, "# of qubits must be 2 or larger"
        self.n_qubits = n_qubits
        self.n_shots = n_shots
        self.verbose = verbose
        self.runtime = datetime.now().strftime("%Y_%m_%d")
        self.statevec = None
        self.circuit = None
        self.state_sim = state_simulation
        self.generators = None
        self.stabilizers = None
        self.stab_circuits = None
        self.diags = None
        self.backend = backend
        self.backend_name = backend_name
        self.hardware_qubits = layout
        self.layout_method = layout_method
        self.track_gates = track_gates
        self.gate_count = {}
        self.opt_level = 0
        self.cached_circuit = None

        if self.backend is None and state_simulation == False:
            self.backend_name = 'sim'
            self.device = qk.Aer.get_backend('qasm_simulator')
        elif self.backend == 'noisy_sim' and state_simulation == False:
            import qiskit.test.mock as mock
            fake_devices = {'Boebligen': mock.FakeBoeblingen(),
                            'Bogota': mock.FakeBogota(),
                            'Cambridge': mock.FakeCambridge(),
                            'Essex': mock.FakeEssex(),
                            'Johannesburg': mock.FakeJohannesburg(),
                            'Melbourne': mock.FakeMelbourne(),
                            'Montreal': mock.FakeMontreal(),
                            'Poughkeepsie': mock.FakePoughkeepsie(),
                            'Tenerife': mock.FakeTenerife(),
                            'Rueschlikon': mock.FakeRueschlikon(),
                            'Singapore': mock.FakeSingapore(),
                            'Tokyo': mock.FakeTokyo()}
            if self.backend_name is None:
                raise ValueError(
                    "Using noisy simulator requires backend_name")
            if self.backend_name not in fake_devices.keys():
                raise ValueError("Unrecognized name, here are the options: " +
                                 "Boebligen\nBogota\nCambridge\nEssex\nJohannesburg\nMelbourne\n" +
                                 "Poughkeepsie\nTenerife\nRueschlikon\nSingapore\nTokyo")
            if fake_devices[self.backend_name].configuration(
            ).n_qubits < self.n_qubits:
                raise ValueError("not enough qubits in backend")
            self.device = qk.providers.aer.QasmSimulator.from_backend(
                fake_devices[self.backend_name])
        elif backend == 'hardware':
            raise NotImplementedError
        elif backend == 'my_device':
            import qiskit.providers.aer.noise as noise
            self.backend_name = 'Frankie'
            self.device = qk.Aer.get_backend('qasm_simulator')
            # Error probabilities
            if 'p1' not in kwargs.keys():
                prob_1 = 0.001  # 1-qubit gate
            else:
                prob_1 = kwargs['p1']
            # prob_2 = 0.01   # 2-qubit gate

            # Depolarizing quantum errors
            error_1 = noise.depolarizing_error(prob_1, 1)
            #error_2 = noise.depolarizing_error(prob_2, 2)

            # Add errors to noise model
            noise_model = noise.NoiseModel()
            noise_model.add_all_qubit_quantum_error(
                error_1, ['u1', 'u2', 'u3'])
            #noise_model.add_all_qubit_quantum_error(error_2, ['cx'])

            # Get basis gates from noise model
            basis_gates = noise_model.basis_gates
            self.noise_model = noise_model
            self.basis_gates = basis_gates

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def build_circuit(self):
        raise NotImplementedError

    def construct_diagonals(self):
        from copy import copy
        sgn_convert = {'-': -1., '+': 1.0}
        a = [1., 1.]
        b = [1., -1.]
        diags = {}
        if self.stab_circuits is None:
            self.build_stabilizer_circuits()
        for idx in range(len(self.stab_circuits)):
            stab_sgn, stab_ops = sgn_convert[self.stab_circuits[idx].name[0]], list(
                self.stab_circuits[idx].name[1:])
            stab_ops_temp = stab_ops.copy()
            pop_op = stab_ops_temp.pop()
            D = [a if pop_op == 'I' else b][0]
            while len(stab_ops_temp) > 0:
                pop_op = stab_ops_temp.pop()
                d = [a if pop_op == 'I' else b][0]
                D = np.kron(d, D)
            diags[self.stab_circuits[idx].name] = np.multiply(
                stab_sgn, D)
            # diags[self.stab_circuits[idx].name]=np.ones(2**self.n_qubits)
            self.diags = diags

    def build_stabilizer_circuits(self):
        raise NotImplementedError

    def get_statevector(self):
        """execute circuit with state_vector sim."""
        self.statevec = qk.execute(self.circuit,
                                   backend=qk.Aer.get_backend(
                                       'statevector_simulator'),
                                   optimization_level=self.opt_level).result().get_statevector()
        return

    def get_stabilizer_statevectors(self):
        """build ALL stabilizers and execute circuit with state_vector sim."""
        if self.stab_circuits is None:
            self.build_stabilizer_circuits()
        if self.diags is None:
            self.construct_diagonals()
        stabilizer_statevecs = {}
        for idx in range(len(self.stab_circuits)):
            temp_svec = qk.execute(self.stab_circuits[idx],
                                   backend=qk.Aer.get_backend(
                                       'statevector_simulator'),
                                   optimization_level=self.opt_level).result().get_statevector()
            stabilizer_statevecs[idx] = temp_svec
        return stabilizer_statevecs

    def get_measurement(self):
        """execute circuit with qasm sim."""
        if self.backend_name is not 'Frankie':
            self.counts = qk.execute(self.circuit,
                                     backend=self.device, initial_layout=self.hardware_qubits,
                                     shots=self.n_shots, optimization_level=self.opt_level).result().get_counts()
        elif self.backend_name is 'Frankie':
            self.counts = qk.execute(self.circuit,
                                     backend=self.device,
                                     basis_gates=self.basis_gates,
                                     noise_model=self.noise_model).result().get_counts()
        return

    def get_stabilizer_measurements(self, noise_model=None):
        """measure ALL stabilizers: execute circuits with qasm sim."""
        if self.stab_circuits is None:
            self.build_stabilizer_circuits()
        self.stab_circuits = [self.cached_circuit.append(circ.to_gate(label="stab"))
                              for circ in self.stab_circuits]
        for circ in self.stab_circuits:
            circ.measure_all()
        if self.diags is None:
            self.construct_diagonals()
        stab_counts = qk.execute(self.stab_circuits,
                                 noise_model=noise_model,
                                 shots=self.n_shots,
                                 optimization_level=self.opt_level)
        # if self.backend_name is not 'Frankie':
        #     qk_outcome = self.stab_counts = qk.execute(self.stab_circuits,
        #                                                backend=self.device,
        #                                                noise_model=noise_model,
        #                                                initial_layout=self.hardware_qubits,
        #                                                shots=self.n_shots,
        #                                                optimization_level=self.opt_level)
        # elif self.backend_name is 'Frankie':
        #     qk_outcome = self.stab_counts = qk.execute(self.stab_circuits,
        #                                                backend=self.device,
        #                                                basis_gates=self.basis_gates,
        #                                                noise_model=self.noise_model)
        if self.track_gates == False:
            self.stab_counts = stab_counts.result().get_counts()
        if self.track_gates == True:
            if self.hardware_qubits is not None:
                hw_tup = tuple(self.hardware_qubits)
                #print('counting 2 qubit gates in hardware')
                # qk_outcome = qk.execute(self.stab_circuits,\
                #        backend=self.device,initial_layout=self.hardware_qubits,\
                #        shots=self.n_shots,optimization_level=self.opt_level)
                self.stab_counts = stab_counts.result().get_counts()
                # count number of cx gates in transpiled circuits
                qo_dict = stab_counts.qobj().to_dict()
                count_list = {}
                for ic in range(len(self.stab_circuits)):
                    ndx = self.stab_circuits[ic].name
                    cx_counter = 0
                    instr = qo_dict['experiments'][ic]['instructions']
                    for gate in instr:
                        if gate['name'] == 'cx':
                            cx_counter += 1
                    count_list[ndx] = cx_counter
                self.gate_count[hw_tup] = self.gate_count.get(
                    hw_tup, [])
                self.gate_count[hw_tup].append(count_list)
            else:
                hw_tup = tuple(())
                #print('counting 2 qubit gates in simulated')
                # qk_outcome = qk.execute(self.stab_circuits,\
                #        backend=self.device,initial_layout=self.hardware_qubits,\
                #        shots=self.n_shots,optimization_level=self.opt_level)
                self.stab_counts = stab_counts.result().get_counts()
                # count number of cx gates in transpiled circuits
                qo_dict = stab_counts.qobj().to_dict()
                count_list = {}
                for ic in range(len(self.stab_circuits)):
                    ndx = self.stab_circuits[ic].name
                    cx_counter = 0
                    instr = qo_dict['experiments'][ic]['instructions']
                    for gate in instr:
                        if gate['name'] == 'cx':
                            cx_counter += 1
                    count_list[ndx] = cx_counter
                self.gate_count[hw_tup] = self.gate_count.get(
                    hw_tup, [])
                self.gate_count[hw_tup].append(count_list)
                # print(self.gate_count)
        stab_exp_vals = {}
        for idx in range(len(self.stab_circuits)):
            test_counts = self.stab_counts[idx]
            dvec = self.diags[self.stab_circuits[idx].name]
            stab_exp_vals[self.stab_circuits[idx].name] = igmit.expectation_value(
                test_counts, diagonal=dvec)
        self.stabilizer_measurements = stab_exp_vals
        return

    def get_witness_measurements(self, witness_list, coef_list=[]):
        """execute circuit with qasm sim.
        witness_list = list of stabilizers by circuit index
        c0 = coefficient (float)"""
        if self.stab_circuits is None:
            self.build_stabilizer_circuits()
        if self.diags is None:
            self.construct_diagonals()
        witness_circuits = [self.stab_circuits[x]
                            for x in witness_list]
        # witness_counts=qk.execute(witness_circuits,\
        #            backend=self.device,initial_layout=self.hardware_qubits,\
        #            shots=self.n_shots,optimization_level=self.opt_level).result().get_counts()
        if self.backend_name is not 'Frankie':
            witness_counts = qk.execute(witness_circuits,
                                        backend=self.device, initial_layout=self.hardware_qubits,
                                        shots=self.n_shots, optimization_level=self.opt_level).result().get_counts()
        elif self.backend_name is 'Frankie':
            witness_counts = qk.execute(witness_circuits,
                                        backend=self.device,
                                        basis_gates=self.basis_gates,
                                        noise_model=self.noise_model).result().get_counts()
        operator_measurements = []
        operator_variances = []
        for idx in range(len(witness_circuits)):
            test_counts = witness_counts[idx]
            dvec = self.diags[witness_circuits[idx].name]
            op_exp, op_var = igmit.expectation_value(
                test_counts, diagonal=dvec)
            operator_measurements.append(op_exp)
            operator_variances.append(op_var)
        return operator_measurements, operator_variances, np.sum(
            np.multiply(coef_list, operator_measurements))

    def estimate_entanglement(self):
        """estimate entanglement of final state using n-qubit entanglement
        winessess if circuit was prepared as GHZ state then assume maximally
        entangled if circuit was prepared as random graph state then use
        witnesses."""
        raise NotImplementedError


class GraphCircuit(CircuitConstructor):
    """Class to construct circuits from graph states.

    Arguments:
        CircuitConstructor {Parent class} -- abstract class

    gate_types: ['CPHASE': control phase gate with some angle (passed as keyword rot_angle),
                'Control_Z': control Z gate implemented with qiskit 'cz' gate]
                'Hgate_CZ': (default) control Z gate implemented with H-CNOT-H decomposition]
    """

    def __init__(self, graph_data=None, gate_type="Hgate_CZ",
                 stochastic=False, smallest_subgraph=2,
                 largest_subgraph=None, directed=False, noise_robust=0,
                 **kwargs):
        super(GraphCircuit, self).__init__(**kwargs)
        if (gate_type == 'CPHASE'):
            if ('rot_angle' not in kwargs.keys()):
                print(
                    'building controlled phase gates with default rotation pi/2')
                self.rot_angle = np.pi / 2.0
            else:
                self.rot_angle = kwargs['rot_angle']
        if graph_data is None:
            self.graph_db = GraphDB(directed=directed)
        else:
            self.graph_db = GraphDB(
                graph_data=graph_data, directed=directed)
        self.gate_type = gate_type
        self.all_graphs = self.get_sorted_db()
        self.largest_subgraph = self.check_largest(largest_subgraph)
        if smallest_subgraph > self.n_qubits:
            print('resetting smallest subgraph order')
            smallest_subgraph = int((self.n_qubits // 2) - 1)
        self.smallest_subgraph = max(2, smallest_subgraph)
        self.graph_combs = self.generate_all_subgraphs()
        self.ops_labels = None
        self.stochastic = stochastic
        self.name = "GraphState"
        self.circuit_graph = None
        self.directed = directed
        self.generators = None
        self.stabilizers = None
        self.stab_circuits = None
        self.stabilizer_measurements = None
        self.diags = None
        self.min_set_B = None
        self.max_set_B = None
        self.edge_list = None
        self.noise_robust = noise_robust
        self.store_defaults()

    def store_defaults(self):
        self._circuit_graph = None
        self._edge_list = None
        self._generators = None
        self._stabilizers = None
        self._stab_circuits = None
        self._diags = None

    def reset_stabilizers(self):
        if self.generators is not None:
            self.circuit_graph = self._circuit_graph
            self.edge_list = self._edge_list
            self.generators = self._generators
            self.stabilizers = self._stabilizers
            self.stab_circuits = self._stab_circuits
            self.diags = self._diags

    def check_largest(self, val):
        for key, itm in self.all_graphs.items():
            if len(itm) != 0:
                max_subgraph = key
        if val is None:
            return max_subgraph
        elif val > max_subgraph:
            warnings.warn(
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

    def build_circuit(self):
        if self.circuit_graph is None:
            self.build_graph()
            print('calling build graph')
            # return
        else:
            # 3. Build a circuit from the graph state
            if self.gate_type == "Hgate_CZ":
                self.print_verbose(
                    "Assigning a Decomposed CZ gate (H-CNOT-H) to Node Edges")
                self._build_Hgate_CZ_gate()
            if self.gate_type == "CPHASE":
                self.print_verbose(
                    "Assigning a Controlled Phase Gate to Node Edges with rotation angle", str(
                        self.rot_angle))
                self._build_CPHASE_gate()
            # same as controlled phase gate but w/ Stochastic unitary
            # gates after CNOT
            elif self.gate_type == "SControlled_Phase":
                self.print_verbose(
                    "Assigning a Stochastic Controlled Phase Gate (H-CNOT-P(U)-H) to Node Edges")
                self._build_Scontrolled_phase_gate()
            # same as controlled phase gate but w/ Stochastic unitary
            # gates after CNOT
            elif self.gate_type == "Controlled_Z":
                self.print_verbose(
                    "Assigning a Controlled Z gate to Node Edges")
                self._build_controlled_Z_gate()

    def update_edge_orientation(self, newlist=None):
        if self.edge_list is None:
            print('build circuit first')
            return
        if newlist is None:
            temp = []
            for edx in self.circuit_graph.edges():
                if bool(np.random.choice(2)):
                    # print('flip')
                    temp.append(edx[::-1])
                else:
                    temp.append(edx)
            self.edge_list = temp
        else:
            self.edge_list = newlist

    def get_generators(self):
        """ get generators of the graph state stabilizers
        generators are n-length strings (n = number of qubits)
        X operator at vertex, Z operators on neighbors, I everywhere else
        generator is first constructed with leftmost index = 0
        then flipped s.t. rightmost entry corresponds to qubit 0"""
        generators = []
        if self.n_qubits > 10:
            print("not implemented for circuits with more than 10 qubits")
            raise NotImplementedError
        else:
            for idx in self.circuit_graph.nodes():
                temp = list('I' * self.n_qubits)
                temp[idx] = 'X'
                for jdx in self.circuit_graph.neighbors(idx):
                    temp[jdx] = 'Z'
                temp = "".join(temp)
                generators.append(temp[::-1])
        self.generators = generators

    def get_stabilizers(self, approach='Toth'):
        """ get the stabilizer operators for an arbitrary graph state
        noise_robust level is passed if approach='Jung'
        noise_robust=0: no additional terms added to genuine entanglement witness
        noise_robust=1: single B vertex set
        noise_robust=2: all B vertex sets used """
        stab_label = []
        noise_robust = 0
        if self.n_qubits > 10:
            print("will not run for circuits with more than 10 qubits")
            raise NotImplementedError
        else:
            if self.generators is None:
                self.get_generators()
            if approach == 'Toth':
                self.stabilizers = ['+' + x for x in self.generators]
                self.stabilizers.append('+' + 'I' * self.n_qubits)
            elif approach == 'Jung':
                if noise_robust == 0:
                    binary_keys = [
                        np.binary_repr(
                            x, self.n_qubits) for x in range(
                            2**self.n_qubits)]
                    for idx in binary_keys:
                        coefs = [int(x) for x in list(idx)]
                        op_mat = []
                        for jdx in range(len(coefs)):
                            if coefs[jdx] == 0:
                                op_mat.append(
                                    list('I' * self.n_qubits))
                            elif coefs[jdx] == 1:
                                op_mat.append(
                                    list(self.generators[jdx]))
                        op_mat = np.asarray(op_mat)
                        cf_arr = []
                        lb_arr = []
                        for kdx in range(op_mat.shape[0]):
                            cf, lb = sigma_prod(
                                ''.join(op_mat[:, kdx]))
                            cf_arr.append(cf)
                            lb_arr.append(lb)
                        if np.iscomplex(np.prod(cf_arr)):
                            print(
                                "Flag-error, coefficient cannot be complex")
                            return
                        else:
                            val = np.prod(cf_arr)
                            if np.real(val) == 1:
                                stab_label.append(
                                    '+' + ''.join(lb_arr))
                            else:
                                stab_label.append(
                                    '-' + ''.join(lb_arr))
                self.stabilizers = stab_label

    def get_biseparable_witnesses(self):
        """ get witness of the graph state stabilizers
        generators are n-length strings (n = number of qubits)"""
        if self.stabilizer_measurements is None:
            self.get_stabilizer_measurements()
        bs_witness = {}
        for idx, edx in enumerate(self.circuit_graph.edges()):
            e0, e1 = edx
            gi = [k for k in self.stabilizer_measurements.keys()
                  if k[::-1][e0] == 'X'][0]
            gj = [k for k in self.stabilizer_measurements.keys()
                  if k[::-1][e1] == 'X'][0]
            val = 1 - \
                self.stabilizer_measurements[gi][0] - \
                self.stabilizer_measurements[gj][0]
            meas_vals = [1.0, -
                         1. *
                         self.stabilizer_measurements[gi][0], -
                         1.0 *
                         self.stabilizer_measurements[gj][0]]
            std_vals = [
                0.0,
                self.stabilizer_measurements[gi][1],
                self.stabilizer_measurements[gj][1]]
            arr = unumpy.uarray(meas_vals, std_vals)
            #print('W: ',val,unumpy.nominal_values(arr.sum()).tolist())
            # print(unumpy.std_devs(arr.sum()).tolist())
            if self.hardware_qubits is None:
                bs_witness[idx] = [tuple(
                    [e0, e1]), val, 'TBD', self.noise_robust, self.backend_name, tuple(())]

            elif self.hardware_qubits is not None:
                bs_witness[idx] = [tuple([e0, e1]), val, unumpy.std_devs(arr.sum()).tolist(), self.noise_robust, self.backend_name,
                                   tuple([self.hardware_qubits[e0], self.hardware_qubits[e1]])]
        bisep_wit_df = pd.DataFrame.from_dict(
            bs_witness, orient='index')
        if len(bisep_wit_df.columns) == 0:
            self.bisep_witnesses = bisep_wit_df
        else:
            bisep_wit_df.columns = [
                'W(i,j)',
                'value',
                'var',
                'noise_robust_level',
                'backend',
                'qubits']
            self.bisep_witnesses = bisep_wit_df

    def get_genuine_witnesses(self):
        """ get witness of the graph state stabilizers
        generators are n-length strings (n = number of qubits)"""

        witness_list = []
        if self.stabilizers is None:
            if self.generators is None:
                self.get_generators()
            self.get_stabilizers()
        if self.noise_robust == 0:
            if self.stabilizer_measurements is None:
                self.get_stabilizer_measurements()
            df = pd.DataFrame.from_dict(
                self.stabilizer_measurements, orient='index')
            df.columns = ['meas', 'var']
            W_ = (self.circuit_graph.order() - 1) * (df['meas'].loc['+' + 'I' * self.n_qubits]) \
                - (df['meas'].sum() -
                   df['meas'].loc['+' + 'I' * self.n_qubits])
            #print('propagation of error not yet implemented, someone should do that')
            nl = len(self.stabilizers)
            ns = len([x for x in self.stabilizers if x !=
                      '+' + 'I' * self.n_qubits])
            wit_coef = np.multiply(np.ones(nl), -1.0)
            iden_idx = self.stabilizers.index(
                '+' + 'I' * self.n_qubits)
            wit_coef[iden_idx] = (ns - 1)

            meas_vals = df['meas'].values
            var_vals = df['var'].values

            witness_terms = np.multiply(wit_coef, meas_vals)
            arr = unumpy.uarray(witness_terms, var_vals)
            #print('W: ',W_,unumpy.nominal_values(arr.sum()).tolist())
            # print(unumpy.std_devs(arr.sum()).tolist())
            self.genuine_witness = (
                W_, unumpy.std_devs(
                    arr.sum()).tolist(), self.noise_robust)
        elif self.noise_robust == 1:
            self._build_Bvertices()

            print("WIP! not yet implemented")

    def build_unique_stabilizer_circuits(self):
        # Build all stabilizer circuits from the graph state and
        # append stabilizer measurements
        if self.circuit_graph is None:
            print('build graph first')
            return
        if self.gate_type == "Controlled_Phase":
            self.print_verbose(
                "Assigning a decomposed Controlled Z Gate (H-CNOT-H) to Node Edges")
            self._build_unique_Hgate_CZ_stabilizer()
        # same as controlled phase gate but w/ Stochastic unitary
        # gates after CNOT
        elif self.gate_type == "Controlled_Z":
            self.print_verbose(
                "Assigning a Controlled Z gate to Node Edges")
            self._build_unique_controlZ_stabilizer()
        elif self.gate_type == "CPHASE":  # same as controlled phase gate but w/ Stochastic unitary gates after CNOT
            self.print_verbose(
                "Assigning a Controlled Phase gate to Node Edges")
            self._build_unique_CPHASE_stabilizer()

    def build_stabilizer_circuits(self):
        # Build all stabilizer circuits from the graph state and
        # append stabilizer measurements
        if self.circuit_graph is None:
            print('build graph first')
            return
        if self.gate_type == "Hgate_CZ":
            self.print_verbose(
                "Assigning a decomposed Controlled Z Gate (H-CNOT-H) to Node Edges")
            self._build_Hgate_CZ_stabilizer()
        # same as controlled phase gate but w/ Stochastic unitary
        # gates after CNOT
        elif self.gate_type == "Controlled_Z":
            self.print_verbose(
                "Assigning a Controlled Z gate to Node Edges")
            self._build_controlZ_stabilizer()
        elif self.gate_type == "CPHASE":  # same as controlled phase gate but w/ Stochastic unitary gates after CNOT
            self.print_verbose(
                "Assigning a Controlled Phase gate to Node Edges")
            self._build_CPHASE_stabilizer()

    def _build_Bvertices(self):
        trials = {}
        # this is a kludge -- it will need to be refined as the graphs
        # get bigger
        for tries in range(5 * self.circuit_graph.order()):
            test_set = set(self.circuit_graph.nodes())
            s0 = []
            start = np.random.randint(self.circuit_graph.order())
            s0.append(start)
            d2_vert = d2_neighborhood(self.circuit_graph, start)
            test_set = test_set - d2_vert
            for idx in range(len(test_set)):
                start = [x for x in list(test_set) if x not in s0]
                if len(start) == 0:
                    break
                else:
                    s0.append(start[0])
                    d2_vert = d2_neighborhood(
                        self.circuit_graph, start[0])
                    test_set = test_set - d2_vert
            trials[tries] = test_set
        min_len = np.min([len(v) for k, v in trials.items()])
        max_len = np.max([len(v) for k, v in trials.items()])
        min_set_B = np.unique(np.asarray(
            [tuple(v) for k, v in trials.items() if v.__len__() == min_len]), axis=0)
        max_set_B = np.unique(np.asarray(
            [tuple(v) for k, v in trials.items() if v.__len__() == max_len]), axis=0)
        self.min_set_B = {min_len: min_set_B}
        self.max_set_B = {max_len: max_set_B}
        self.B_set = self.max_set_B[max_len][np.random.choice(
            range(len(self.max_set_B[max_len])))]

    def _build_svec_set(self, b):
        ''' return all s-vectors that have Hamming weight <= b-4 '''
        binary_keys = [np.binary_repr(x, b) for x in range(2**b)]
        s_vecs = []
        for idx in binary_keys:
            coefs = [(2 * int(x) - 1) for x in list(idx)]
            if np.sum(coefs) <= b - 4:
                s_vecs.append(coefs)
        self.s_vecs = s_vecs

    def _build_decomposable_witness(self):
        ''' build a completely decomposible witness '''
        max_count = 0
        if (self.min_set_B is None) and (self.max_set_B is None):
            self._build_Bvertices
        for kdx, vecs in enumerate(self.max_set_B):
            if len(vecs) >= max_count:
                max_count = len(vecs)
        if max_count < 2:
            print('As good as it gets, no additional terms')
        else:
            self._build_svec_set(max_count)

    def _build_Hgate_CZ_gate(self):
        unitary_op = Operator(np.identity(4))
        ops_labels = []
        q_reg = qk.QuantumRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg)
        for idx in range(self.n_qubits):
            circ.h(idx)
        if type(self.circuit_graph) == nx.DiGraph:
            gate_pairs = []
            for node, ngbrs in self.circuit_graph.adjacency():
                for ngbr, _ in ngbrs.items():
                    circ.h(node)
                    circ.cx(node, ngbr)
                    if bool(np.random.choice(2)) and self.stochastic:
                        label = 'unitary_{}_{}'.format(node, ngbr)
                        ops_labels.append(label)
                        circ.unitary(
                            unitary_op, [
                                node, ngbr], label=label)
                    circ.h(node)
            if self.stochastic:
                self.ops_labels = ops_labels
            if self.state_sim:
                self.circuit = circ
                return
            circ.measure_all()
            self.circuit = circ
            return
        elif type(self.circuit_graph) == nx.Graph:
            gate_pairs = []
            for node, ngbr in self.edge_list:
                circ.h(ngbr)
                circ.cx(node, ngbr)
                circ.h(ngbr)
                if bool(np.random.choice(2)) and self.stochastic:
                    label = 'unitary_{}_{}'.format(node, ngbr)
                    ops_labels.append(label)
                    circ.unitary(
                        unitary_op, [
                            node, ngbr], label=label)
            if self.stochastic:
                self.ops_labels = ops_labels
            if self.state_sim:
                self.circuit = circ
                return
            self.cached_circuit = deepcopy(circ)
            circ.measure_all()
            self.circuit = circ

    def _build_CPHASE_gate(self):
        lam = self.rot_angle
        q_reg = qk.QuantumRegister(self.n_qubits)
        c_reg = qk.ClassicalRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg, c_reg)
        for idx in range(self.n_qubits):
            circ.h(idx)
        for node, ngbr in self.edge_list:
            circ.ry(self.theta / 2, ngbr)
            circ.cx(node, ngbr)
            circ.ry(-self.theta / 2, ngbr)
            circ.cx(node, ngbr)
            circ.ry(self.rot_angle / 2, ngbr)

        if self.state_sim:
            self.circuit = circ
        else:
            circ.barrier()
            circ.measure(q_reg, c_reg)
            self.circuit = circ

    def _build_controlled_Z_gate(self):
        unitary_op = Operator(np.identity(4))
        q_reg = qk.QuantumRegister(self.n_qubits)
        c_reg = qk.ClassicalRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg, c_reg)
        ops_labels = []
        for idx in range(self.n_qubits):
            circ.h(idx)
        if type(self.circuit_graph) == nx.DiGraph:
            gate_pairs = []
            for node, ngbrs in self.circuit_graph.adjacency():
                for ngbr, _ in ngbrs.items():
                    circ.cz(node, ngbr)
                    #circ.cz(ngbr, node)
                    if bool(np.random.choice(2)) and self.stochastic:
                        label = 'unitary_{}_{}'.format(node, ngbr)
                        ops_labels.append(label)
                        circ.unitary(
                            unitary_op, [
                                node, ngbr], label=label)
            if self.stochastic:
                self.ops_labels = ops_labels
            if self.state_sim:
                self.circuit = circ
                return
            circ.barrier()
            circ.measure(q_reg, c_reg)
            self.circuit = circ
            return
        elif type(self.circuit_graph) == nx.Graph:
            for node, ngbr in self.edge_list:
                circ.cz(node, ngbr)
                if bool(np.random.choice(2)) and self.stochastic:
                    label = 'unitary_{}_{}'.format(node, ngbr)
                    ops_labels.append(label)
                    circ.unitary(
                        unitary_op, [
                            node, ngbr], label=label)
            if self.stochastic:
                self.ops_labels = ops_labels
            if self.state_sim:
                self.circuit = circ
                return
            circ.barrier()
            circ.measure(q_reg, c_reg)
            self.circuit = circ

    def _build_unique_Hgate_CZ_stabilizer(self):
        """
        build all circuits for specific stabilizers, store in a list
        """
        unitary_op = Operator(np.identity(4))
        ops_labels = []
        self.stab_circuits = []
        if self.stabilizers is None:
            if self.generators is None:
                self.get_generators()
            self.get_stabilizers()
        for sdx in get_unique_operators(self.stabilizers):
            q_reg = qk.QuantumRegister(self.n_qubits)
            c_reg = qk.ClassicalRegister(self.n_qubits)
            circ = qk.QuantumCircuit(q_reg, c_reg, name=sdx)
            for idx in range(self.n_qubits):
                circ.h(idx)
            if type(self.circuit_graph) == nx.DiGraph:
                print('something is not right with directed graphs')
                raise NotImplementedError
                return
            elif type(self.circuit_graph) == nx.Graph:
                for node, ngbr in self.edge_list:
                    circ.h(ngbr)
                    circ.cx(node, ngbr)
                    circ.h(ngbr)
                    if bool(np.random.choice(2)) and self.stochastic:
                        label = 'unitary_{}_{}'.format(node, ngbr)
                        ops_labels.append(label)
                        circ.unitary(
                            unitary_op, [
                                node, ngbr], label=label)
                    # circ.h(ngbr)
                if self.stochastic:
                    self.ops_labels = ops_labels
                # add operators for stabilizer measurements
                stab_ops = list(sdx)
                circ.barrier()
                circ = build_stabilizer_meas(
                    circ, sdx, drop_coef=False)
                if self.state_sim:
                    self.stab_circuits.append(circ)
                circ.barrier()
                circ.measure(q_reg, c_reg)
                self.stab_circuits.append(circ)

    def _build_Hgate_CZ_stabilizer(self):
        """
        build all circuits for all stabilizers, store in a list
        """
        unitary_op = Operator(np.identity(4))
        self.stab_circuits = []
        ops_labels = []
        if self.stabilizers is None:
            if self.generators is None:
                self.get_generators()
            self.get_stabilizers()
        for sdx in self.stabilizers:
            q_reg = qk.QuantumRegister(self.n_qubits)
            circ = qk.QuantumCircuit(q_reg, name=sdx)
            for idx in range(self.n_qubits):
                circ.h(idx)
            if type(self.circuit_graph) == nx.DiGraph:
                print('something is not right with directed graphs')
                raise NotImplementedError
                return
            elif type(self.circuit_graph) == nx.Graph:
                for node, ngbr in self.edge_list:
                    circ.h(ngbr)
                    circ.cx(node, ngbr)
                    circ.h(ngbr)
                    if bool(np.random.choice(2)) and self.stochastic:
                        label = 'unitary_{}_{}'.format(node, ngbr)
                        ops_labels.append(label)
                        circ.unitary(
                            unitary_op, [
                                node, ngbr], label=label)
                    # circ.h(ngbr)
                if self.stochastic:
                    self.ops_labels = ops_labels
                # add operators for stabilizer measurements
                stab_ops = list(sdx)
                circ = build_stabilizer_meas(circ, sdx)
                if self.state_sim:
                    self.stab_circuits.append(circ)
                # circ.measure_all(q_reg, c_reg)
                self.stab_circuits.append(circ)

    def _build_unique_controlZ_stabilizer(self):
        """
        build all circuits for all stabilizers, store in a list
        """
        unitary_op = Operator(np.identity(4))
        ops_labels = []
        self.stab_circuits = []
        if self.stabilizers is None:
            if self.generators is None:
                self.get_generators()
            self.get_stabilizers()
        for sdx in get_unique_operators(self.stabilizers):
            q_reg = qk.QuantumRegister(self.n_qubits)
            c_reg = qk.ClassicalRegister(self.n_qubits)
            circ = qk.QuantumCircuit(q_reg, c_reg, name=sdx)
            for idx in range(self.n_qubits):
                circ.h(idx)
            if type(self.circuit_graph) == nx.DiGraph:
                print('something is not right with directed graphs')
                raise NotImplementedError
                return
            elif type(self.circuit_graph) == nx.Graph:
                for node, ngbr in self.edge_list:
                    circ.cz(node, ngbr)
                    if bool(np.random.choice(2)) and self.stochastic:
                        label = 'unitary_{}_{}'.format(node, ngbr)
                        ops_labels.append(label)
                        circ.unitary(
                            unitary_op, [
                                node, ngbr], label=label)
                if self.stochastic:
                    self.ops_labels = ops_labels
                # add operators for stabilizer measurements
                stab_ops = list(sdx)
                circ.barrier()
                circ = build_stabilizer_meas(
                    circ, sdx, drop_coef=False)
                if self.state_sim:
                    self.stab_circuits.append(circ)
                circ.barrier()
                circ.measure(q_reg, c_reg)
                self.stab_circuits.append(circ)

    def _build_controlZ_stabilizer(self):
        """
        build all circuits for all stabilizers, store in a list
        """
        self.stab_circuits = []
        if self.stabilizers is None:
            if self.generators is None:
                self.get_generators()
            self.get_stabilizers()
        for sdx in self.stabilizers:
            q_reg = qk.QuantumRegister(self.n_qubits)
            c_reg = qk.ClassicalRegister(self.n_qubits)
            circ = qk.QuantumCircuit(q_reg, c_reg, name=sdx)
            for idx in range(self.n_qubits):
                circ.h(idx)
            if type(self.circuit_graph) == nx.DiGraph:
                print('something is not right with directed graphs')
                raise NotImplementedError
                return
            elif type(self.circuit_graph) == nx.Graph:
                for node, ngbr in self.edge_list:
                    circ.cz(node, ngbr)
                    # if bool(np.random.choice(2)) and self.stochastic:
                    #    label = 'unitary_{}_{}'.format(node, ngbr)
                    #    ops_labels.append(label)
                    #    circ.unitary(unitary_op, [node, ngbr], label=label)
                #if self.stochastic: self.ops_labels=ops_labels
                # add operators for stabilizer measurements
                stab_ops = list(sdx)
                circ.barrier()
                circ = build_stabilizer_meas(circ, sdx)
                if self.state_sim:
                    self.stab_circuits.append(circ)
                circ.barrier()
                circ.measure(q_reg, c_reg)
                self.stab_circuits.append(circ)

    def _build_unique_CPHASE_stabilizer(self):
        """
        build all circuits for all stabilizers, store in a list
        """
        self.stab_circuits = []
        if self.stabilizers is None:
            if self.generators is None:
                self.get_generators()
            self.get_stabilizers()
        for sdx in get_unique_operators(self.stabilizers):
            q_reg = qk.QuantumRegister(self.n_qubits)
            c_reg = qk.ClassicalRegister(self.n_qubits)
            circ = qk.QuantumCircuit(q_reg, c_reg, name=sdx)
            for idx in range(self.n_qubits):
                circ.h(idx)
            if type(self.circuit_graph) == nx.DiGraph:
                print('something is not right with directed graphs')
                raise NotImplementedError
                return
            elif type(self.circuit_graph) == nx.Graph:
                for node, ngbr in self.edge_list:
                    circ.ry(self.theta / 2, ngbr)
                    circ.cx(node, ngbr)
                    circ.ry(-self.theta / 2, ngbr)
                    circ.cx(node, ngbr)
                    circ.ry(self.rot_angle / 2, ngbr)
                    # if bool(np.random.choice(2)) and self.stochastic:
                    #    label = 'unitary_{}_{}'.format(node, ngbr)
                    #    ops_labels.append(label)
                    #    circ.unitary(unitary_op, [node, ngbr], label=label)
                #if self.stochastic: self.ops_labels=ops_labels
                # add operators for stabilizer measurements
                stab_ops = list(sdx)
                circ.barrier()
                circ = build_stabilizer_meas(
                    circ, sdx, drop_coef=False)
                if self.state_sim:
                    self.stab_circuits.append(circ)
                circ.barrier()
                circ.measure(q_reg, c_reg)
                self.stab_circuits.append(circ)

    def _build_CPHASE_stabilizer(self):
        """
        build all circuits for all stabilizers, store in a list
        """
        self.stab_circuits = []
        if self.stabilizers is None:
            if self.generators is None:
                self.get_generators()
            self.get_stabilizers()
        for sdx in self.stabilizers:
            q_reg = qk.QuantumRegister(self.n_qubits)
            c_reg = qk.ClassicalRegister(self.n_qubits)
            circ = qk.QuantumCircuit(q_reg, c_reg, name=sdx)
            for idx in range(self.n_qubits):
                circ.h(idx)
            if type(self.circuit_graph) == nx.DiGraph:
                print('something is not right with directed graphs')
                raise NotImplementedError
                return
            elif type(self.circuit_graph) == nx.Graph:
                for node, ngbr in self.edge_list:
                    circ.ry(self.theta / 2, ngbr)
                    circ.cx(node, ngbr)
                    circ.ry(-self.theta / 2, ngbr)
                    circ.cx(node, ngbr)
                    circ.ry(self.rot_angle / 2, ngbr)
                    # if bool(np.random.choice(2)) and self.stochastic:
                    #    label = 'unitary_{}_{}'.format(node, ngbr)
                    #    ops_labels.append(label)
                    #    circ.unitary(unitary_op, [node, ngbr], label=label)
                #if self.stochastic: self.ops_labels=ops_labels
                # add operators for stabilizer measurements
                stab_ops = list(sdx)
                circ.barrier()
                circ = build_stabilizer_meas(circ, sdx)
                if self.state_sim:
                    self.stab_circuits.append(circ)
                circ.barrier()
                circ.measure(q_reg, c_reg)
                self.stab_circuits.append(circ)

    def get_sorted_db(self):
        sorted_dict = dict([(i, []) for i in range(1, len(self.graph_db.keys())
                                                   + 1)])
        for _, itm in self.graph_db.items():
            if itm["G"] is not None:
                sorted_key = len(itm["G"].nodes)
                sorted_dict[sorted_key].append(itm["G"])
        return sorted_dict


class GHZCircuit(CircuitConstructor):
    def __init__(self, state_sim=False, stochastic=False, **kwargs):
        super(GHZCircuit, self).__init__(**kwargs)
        self.ops_labels = None
        self.state_sim = state_sim
        self.stochastic = stochastic
        self.name = "GHZ"
        self.generators = None
        self.stabilizers = None
        self.stab_circuits = None
        self.diags = None

    def build_circuit(self):
        #unitary_op = Operator(np.identity(4))
        q_reg = qk.QuantumRegister(self.n_qubits)
        c_reg = qk.ClassicalRegister(self.n_qubits)
        circ = qk.QuantumCircuit(q_reg, c_reg)
        #ops_labels = []
        circ.h(0)  # pylint: disable=no-member
        #self.ops_labels = []
        for q in range(self.n_qubits - 1):
            circ.cx(q, q + 1)  # pylint: disable=no-member
        """
            if self.stochastic and bool(np.random.choice(2)):
                label = 'unitary_{}_{}'.format(q,q+1)
                ops_labels.append(label)
                circ.unitary(unitary_op, [q, q+1], label=label) #pylint: disable=no-member

        self.ops_labels = ops_labels
        """
        if self.state_sim:
            self.circuit = circ
            return
        circ.barrier()
        circ.measure(q_reg, c_reg)  # pylint: disable=no-member
        self.circuit = circ

    def build_stabilizer_circuits(self):
        """
        build all circuits for all stabilizers, store in a list
        """
        self.stab_circuits = []
        if self.stabilizers is None:
            if self.generators is None:
                self.get_generators()
            self.get_stabilizers()
        for idx in range(len(self.stabilizers)):
            sdx = self.stabilizers[idx]
            #unitary_op = Operator(np.identity(4))
            q_reg = qk.QuantumRegister(self.n_qubits)
            c_reg = qk.ClassicalRegister(self.n_qubits)
            circ = qk.QuantumCircuit(q_reg, c_reg)
            circ.name = sdx
            #ops_labels = []
            circ.h(0)  # pylint: disable=no-member
            #self.ops_labels = []
            for q in range(self.n_qubits - 1):
                circ.cx(q, q + 1)  # pylint: disable=no-member
            """
                if self.stochastic and bool(np.random.choice(2)):
                    label = 'unitary_{}_{}'.format(q,q+1)
                    ops_labels.append(label)
                    circ.unitary(unitary_op, [q, q+1], label=label) #pylint: disable=no-member
            """
            # add operators for stabilizer measurements
            circ.barrier()
            circ = build_stabilizer_meas(circ, sdx)
            if self.state_sim:
                self.stab_circuits.append(circ)
            else:
                circ.barrier()
                circ.measure(q_reg, c_reg)
                self.stab_circuits.append(circ)

    def get_generators(self):
        """ get the stabilzier operators for a GHZ state """
        if self.n_qubits > 10:
            print("will not run for circuits with more than 10 qubits")
            raise NotImplementedError
        else:
            generators = []
            generators.append('X' * self.n_qubits)
            for idx in range(1, self.n_qubits):
                temp = list('I' * self.n_qubits)
                temp[idx - 1] = 'Z'
                temp[idx] = 'Z'
                temp = "".join(temp)
                generators.append(temp)
            self.generators = generators

    def get_stabilizers(self):
        """ get the stabilizer operators for a GHZ state """
        if self.n_qubits > 10:
            print("will not run for circuits with more than 10 qubits")
            raise NotImplementedError
        else:
            if self.generators is None:
                self.get_generators()
            binary_keys = [
                np.binary_repr(
                    x, self.n_qubits) for x in range(
                    2**self.n_qubits)]
            stab_label = []
            for idx in binary_keys:
                coefs = [int(x) for x in list(idx)]
                op_mat = []
                for jdx in range(len(coefs)):
                    if coefs[jdx] == 0:
                        op_mat.append(list('I' * self.n_qubits))
                    else:
                        op_mat.append(list(self.generators[jdx]))
                op_mat = np.asarray(op_mat)
                cf_arr = []
                lb_arr = []
                for kdx in range(op_mat.shape[0]):
                    cf, lb = sigma_prod(''.join(op_mat[:, kdx]))
                    cf_arr.append(cf)
                    lb_arr.append(lb)
                if np.iscomplex(np.prod(cf_arr)):
                    print("Flag-error, coefficient cannot be complex")
                    return
                else:
                    val = np.prod(cf_arr)
                    if np.real(val) == 1:
                        stab_label.append('+' + ''.join(lb_arr))
                    else:
                        stab_label.append('-' + ''.join(lb_arr))
            self.stabilizers = stab_label

    def build_biseparable_witnesses(self):
        """ build witnesses to detect biseparable entanglement from the
            GHZ state stabilizers
            Reference: Eqtn (11) in Toth 2005
            witnesses identified by circuit index

            witness[n]=[index for iden,index for circuit0, index for circuit n]"""
        if self.stab_circuits is None:
            self.build_stabilizer_circuits()
        witnesses = {}
        iden_wit = [x for x in range(len(self.stab_circuits)) if
                    self.stab_circuits[x].name == '+' + 'I' * self.n_qubits][0]
        S1_wit = [x for x in range(len(self.stab_circuits)) if
                  self.stab_circuits[x].name == '+' + 'X' * self.n_qubits][0]
        for idx in range(1, self.n_qubits):
            idx_gen = [
                x for x in self.generators if x[idx - 1:idx + 1] == 'ZZ'][0]
            Sm_wit = [x for x in range(len(self.stab_circuits)) if
                      self.stab_circuits[x].name == '+' + idx_gen][0]
            witnesses[idx] = tuple([iden_wit, S1_wit, Sm_wit])
        self.bisep_witnesses = witnesses
        self.bisep_coefs = [1.0, -1.0, -1.0]

    def build_genuine_witnesses(self, noise_robust=0):
        """ build witnesses to detect genuine entanglement from the
            GHZ state stabilizers
            noise_robustness: level flag
            (0: no noise robustness = Eqtn (21) in Toth 2005)
            (1: some noise robustness = Eqtn (23) in Toth 2005)"""
        ops_list = []

        iden_wit = [x for x in range(len(self.stab_circuits)) if
                    self.stab_circuits[x].name == '+' + 'I' * self.n_qubits][0]
        S1_wit = [x for x in range(len(self.stab_circuits)) if
                  self.stab_circuits[x].name == '+' + 'X' * self.n_qubits][0]
        if noise_robust == 0:
            ops_list.append(iden_wit)
            ops_list.append(S1_wit)
            for idx in range(1, self.n_qubits):
                idx_gen = [
                    x for x in self.generators if x[idx - 1:idx + 1] == 'ZZ'][0]
                Sm_wit = [x for x in range(len(self.stab_circuits)) if
                          self.stab_circuits[x].name == '+' + idx_gen][0]
                ops_list.append(Sm_wit)
            coefs = np.multiply(-1.0,
                                np.ones(len(self.generators) + 1))
            coefs[0] = len(self.generators) - 1
        elif noise_robust == 1:
            coefs = np.ones(len(self.generators) + 1)
            ops_list.append(iden_wit)
            ops_list.append(S1_wit)
            for idx in range(1, self.n_qubits):
                idx_gen = [
                    x for x in self.generators if x[idx - 1:idx + 1] == 'ZZ'][0]
                Sm_wit = [x for x in range(len(self.stab_circuits)) if
                          self.stab_circuits[x].name == '+' + idx_gen][0]
                ops_list.append(Sm_wit)
            # add in products of generators to improve
            # noise_robustness

        self.genuine_witnesses = tuple(ops_list)
        self.genuine_wit_coefs = coefs
