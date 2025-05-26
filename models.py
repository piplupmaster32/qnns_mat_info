import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import CXGate, CZGate
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import CXGate, CZGate
from qiskit.circuit import Parameter
from qiskit.circuit import QuantumRegister, ClassicalRegister

### QNN ENCODER
class QNNEncoder:
    def __init__(self, n_qubits, encoding_method = "arctan", redundant_encoding = "linear"):
        self.n_qubits = n_qubits
        self.encoding_method = encoding_method
        self.redundant_encoding = redundant_encoding

        if n_qubits not in [5, 10]:
            raise ValueError("n_qubits must be either 5 or 10.")
        if encoding_method not in ["arctan", "linear"]:
            raise ValueError("encoding_method must be either 'arctan' or 'linear'.")
        if n_qubits == 10 and redundant_encoding not in ["linear", "quadratic"]:
            raise ValueError("For 10 qubits, redundant_encoding must be either 'linear' or 'quadratic'.")
        
        self.n_features = 5 #TODO: Generalize
        self.features = ParameterVector("x", self.n_features)
    
    def _transform_feature(self, x):
        if self.encoding_method == "arctan":
            return np.arctan(x) + np.pi/2
        elif self.encoding_method == "linear":
            return np.pi *x
        else:
            raise ValueError("Invalid encoding method.")
    
    def _create_5qubit_circuit(self):
        qc = QuantumCircuit(self.n_qubits)

        for i in range(self.n_features):
            theta = self._transform_feature(self.features[i])
            qc.ry(theta, i)
            
        return qc

    def _create_10qubit_circuit(self, x):
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_features):
            theta1 = self._transform_feature(x[i])

            if self.redundant_encoding == "linear":
                theta2 = theta1
            else:
                theta2 = self._transform_feature(x[i]**2)
            
            qc.ry(theta1, 2*i)
            qc.ry(theta2, 2*i+1)
        
        return qc
    
    def create_encoding_circuit(self):
        if self.n_qubits == 5:
            return self._create_5qubit_circuit()
        elif self.n_qubits == 10:
            return self._create_10qubit_circuit()
        else:
            raise ValueError("Invalid number of qubits.")

### QNN ANSATZ
class QNNAnsatz:
    def __init__(self, n_qubits, depth = 1, entangler_type = 'linear', two_qubit_gate = 'cx'):
        self.n_qubits = n_qubits
        self.depth = depth
        self.entangler_type = entangler_type
        self.two_qubit_gate = two_qubit_gate

        if n_qubits not in [5, 10]:
            raise ValueError("n_qubits must be either 5 or 10.")
        if entangler_type not in ['linear', 'star']:
            raise ValueError("entangler_type must be either 'linear' or 'star'.")
        if two_qubit_gate not in ['cx', 'cz']:
            raise ValueError("two_qubit_gate must be either 'cx' or 'cz'.")
    
        self.params = ParameterVector("θ", self.n_qubits * self.depth)
    
    def create_ansatz(self):
        qc = QuantumCircuit(self.n_qubits)
        param_idx = 0
        
        for d in range(self.depth):
            # Add single-qubit rotations
            for q in range(self.n_qubits):
                qc.ry(self.params[param_idx], q)
                param_idx += 1
                
            # Add entangler
            self._add_entangler(qc)
                
        return qc
        
    def _add_entangler(self, qc):
        if self.two_qubit_gate == 'cx':
            two_qubit_gate = CXGate()
        elif self.two_qubit_gate == 'cz':
            two_qubit_gate = CZGate()
        
        if self.entangler_type == 'linear':
            for q in range(self.n_qubits - 1):
                qc.append(two_qubit_gate, [q, q + 1])
        
        elif self.entangler_type == 'circular':
            for q in range(self.n_qubits):
                qc.append(two_qubit_gate, [q, (q + 1) % self.n_qubits])
        
        elif self.entangler_type == 'circular2':
            # Circular connections up to 2nd neighbors
            for q in range(self.n_qubits):
                qc.append(two_qubit_gate, [q, (q+1)%self.n_qubits])
                qc.append(two_qubit_gate, [q, (q+2)%self.n_qubits])
        
        elif self.entangler_type == 'circular4':
            # Circular connections up to 4th neighbors
            for q in range(self.n_qubits):
                for offset in [1, 2, 3, 4]:
                    qc.append(two_qubit_gate, [q, (q+offset)%self.n_qubits])

### QUANTUM NEURAL NETWORK
class QNN:    
    def __init__(self, n_qubits, encoding_method="arctan", redundant_encoding="linear", 
                 depth=1, entangler_type='linear', two_qubit_gate='cx', backend=None):
        self.n_qubits = n_qubits
        self.encoder = QNNEncoder(n_qubits, encoding_method, redundant_encoding)
        self.ansatz = QNNAnsatz(n_qubits, depth, entangler_type, two_qubit_gate)
    
        if backend is None:
            self.sampler = StatevectorSampler()
            self.estimator = StatevectorEstimator()
            self.backend = None
        
        else:
            from qiskit.primitives import BackendSamplerV2, BackendEstimatorV2
            self.backend = backend
            self.sampler = BackendSamplerV2(backend)
            self.estimator = BackendEstimatorV2(backend)

            self.pass_manager = generate_preset_pass_manager(
                optimization_level= 1,
                backend= self.backend
            )

    def create_circuit(self, add_measurements=True):
        encoder_circuit = self.encoder.create_encoding_circuit()
        ansatz_circuit = self.ansatz.create_ansatz()

        qc = encoder_circuit.compose(ansatz_circuit)

        if add_measurements:
            qc = self._add_measurements(qc)

        return qc

    def _add_measurements(self, qc):
        if self.n_qubits == 5:
            cr = ClassicalRegister(1)
            qc.add_register(cr)
            qc.measure(4, 0)
        else:
            cr = ClassicalRegister(2)
            qc.add_register(cr)
            qc.measure([8, 0], [9, 1])
        return qc
    
    def get_observable(self):
        """Get the observable for expectation value calculation"""
        if self.n_qubits == 5:
            # Z measurement on qubit 4
            pauli_string = 'I' * 4 + 'Z'
            return SparsePauliOp.from_list([(pauli_string, 1.0)])
        else:
            # Z measurement on qubits 8 and 9
            pauli_string_8 = 'I' * 8 + 'Z' + 'I'
            pauli_string_9 = 'I' * 9 + 'Z'
            return SparsePauliOp.from_list([
                (pauli_string_8, 1.0),
                (pauli_string_9, 1.0)
            ])
    
    def run_sampling(self, parameter_values, shots = 1024):
        circuit = self.create_circuit(add_measurements=True)
        all_params = list(self.encoder.features) + list(self.ansatz.params)

        if len(parameter_values) != len(all_params):
            raise ValueError(f"Expected {len(all_params)} parameters, got {len(parameter_values)}")
        
        if self.backend is not None:
            circuit = self.pass_manager.run(circuit)
            bound_circuit = circuit.assign_parameters(dict(zip(all_params, parameter_values)))
            job = self.sampler.run(bound_circuit, shots=shots)
            result = job.result()

        return result
    
    def run_expectation(self, parameter_values):
        circuit = self.create_circuit( add_measurements=False)
        observable = self.get_observable()
        
        # Create parameter binding
        all_params = list(self.encoder.features) + list(self.ansatz.params)
        
        if len(parameter_values) != len(all_params):
            raise ValueError(f"Expected {len(all_params)} parameters, got {len(parameter_values)}")
        
        # Transpile circuit if using a backend
        if self.backend is not None:
            circuit = self.pass_manager.run(circuit)
        
        # Bind parameters
        bound_circuit = circuit.assign_parameters(
            dict(zip(all_params, parameter_values))
        )
        
        # Run estimation using EstimatorV2 interface
        job = self.estimator.run([(bound_circuit, observable)])
        result = job.result()

        return result

    def run_batch_expectation(self, parameter_batch):
        """Run expectation values for a batch of parameters"""
        circuit = self.create_circuit(add_measurements=False)
        observable = self.get_observable()
        
        all_params = list(self.encoder.features) + list(self.ansatz.params)
        
        # Prepare batch of bound circuits
        pubs = []
        for parameter_values in parameter_batch:
            if len(parameter_values) != len(all_params):
                raise ValueError(f"Expected {len(all_params)} parameters, got {len(parameter_values)}")
            
            # Transpile if needed
            if self.backend is not None:
                transpiled_circuit = self.pass_manager.run(circuit)
            else:
                transpiled_circuit = circuit
            
            # Bind parameters
            bound_circuit = transpiled_circuit.assign_parameters(
                dict(zip(all_params, parameter_values))
            )
            
            pubs.append((bound_circuit, observable))
        
        # Run batch estimation
        job = self.estimator.run(pubs)
        result = job.result()
        
        return result

    def get_parameter_count(self):
        """Get total number of parameters"""
        return len(self.encoder.features) + len(self.ansatz.params)
    
    def get_feature_parameters(self):
        """Get feature parameters for binding input data"""
        return self.encoder.features
    
    def get_trainable_parameters(self):
        """Get trainable parameters for optimization"""
        return self.ansatz.params
