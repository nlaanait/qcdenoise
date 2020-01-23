from warnings import warn
import numpy as np
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel, errors
from qiskit.quantum_info.operators import Operator


class CircuitSampler:
    def __init__(self):
        pass

    def build_GHZ(self, n_qubits=2, insert_unitary=True, unitary_op=None):
        assert n_qubits >= 2, "# of qubits must be 2 or larger"
        # use identity for a unitary op if not passed
        if insert_unitary:
            if unitary_op is None:
                unitary_op = Operator(np.identity(4))
            else:
                assert isinstance(unitary_op, Operator), "unitary op must be instance of qiskit.quantum_info.operators.Operator"
        circ = QuantumCircuit(n_qubits, n_qubits)
        circ.h(0)
        ops_labels = []
        for q in range(n_qubits-1):
            circ.cx(q, q+1)
            if insert_unitary and bool(np.random.choice(2)):
                    label = 'unitary_{}_{}'.format(q,q+1)
                    ops_labels.append(label)
                    circ.unitary(unitary_op, [q, q+1], label=label)
        ## TODO: must check if built circuit is unitary
        # measure
        circ.measure(range(n_qubits), range(n_qubits))
        return circ, ops_labels
    

    def build_unitary_noise_model(self, unitary_ops, noise_specs={'type':'phase_amplitude_damping_error', 
                                                  'max_prob':0.1}):
        noise_model = NoiseModel()
        callbacks = {'phase_amplitude_damping_error': CircuitSampler.get_phase_amp_damp_error,
                    'amplitude_damping_error': CircuitSampler.get_amp_damp_error}
        error_type = noise_specs.pop('type')
        if unitary_ops:
            assert error_type in list(callbacks.keys()), "noise type must be of type {}".format(list(callbacks.keys()))
            try:
                error_call = getattr(errors, error_type)
            except AttributeError as e:
                print(e)
            for op in unitary_ops:
                error = callbacks[error_type](error_call, **noise_specs) 
                noise_model.add_all_qubit_quantum_error(error, op)
            noise_model.add_basis_gates(['unitary'])
            return noise_model
        return noise_model

    @staticmethod
    def get_phase_amp_damp_error(func, max_prob=0.1):
        phases = np.random.uniform(high=max_prob, size=2)
        amps = np.random.uniform(high=max_prob, size=2)
        amps = [min(amp, 1-phase) for amp, phase in zip(amps, phases)]
        q_error_1 = func(phases[0], amps[0])
        q_error_2 = func(phases[1], amps[1])
        unitary_error = q_error_1.tensor(q_error_2) 
        return unitary_error

    @staticmethod
    def get_amp_damp_error(func, max_prob=0.1):
        amps = np.random.uniform(high=max_prob, size=2)
        q_error_1 = func(amps[0])
        q_error_2 = func(amps[1])
        unitary_error = q_error_1.tensor(q_error_2) 
        return unitary_error


if __name__ == "__main__":
    circ_sampler = CircuitSampler()
    # Build 10-qubit GHZ with randomly inserted unitary operators
    circ, ops_labels = circ_sampler.build_GHZ(10, insert_unitary=True)
    # Randomly apply Krauss operators to unitary operators to generate 1-qbit errors
    noise_specs = {'type':'amplitude_damping_error', 'max_prob': 0.001}
    noise_specs = {'type':'phase_amplitude_damping_error', 'max_prob': 0.001}
    noise_model = circ_sampler.build_unitary_noise_model(ops_labels, noise_specs=noise_specs)
    # execute
    try:
        job = execute(circ, QasmSimulator(), noise_model=noise_model, seed_simulator=1234, memory=True)
    except Exception as e:
        print('Circuit execution failed!!')
        print(e)

