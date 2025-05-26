from qiskit.circuit import ClassicalRegister

def add_measurement(qc, num_qubits):
    if num_qubits == 5:
        # Measure σ_z on the last qubit
        cr = ClassicalRegister(1)
        qc.add_register(cr)
        qc.measure(4, 0)  # Measuring qubit 4 to classical bit 0
    else:
        # Measure σ_z on qubits 8 and 9 and sum
        cr = ClassicalRegister(2)
        qc.add_register(cr)
        qc.measure(8, 0)
        qc.measure(9, 1)
        # Note: Actual summation would be done classically
    return qc