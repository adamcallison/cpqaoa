import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, transpile

def standard_mixer_eigenvalues(n):
    # returns diagonal of standard mixer that has been hadamard Hd_transformed
    # into Z eigenbasis
    N = 2**n
    eigvals = np.zeros(N)
    for j in range(N):
        jstr = bin(j)[2:]
        ones = jstr.count('1')
        eigvals[j] = 2*ones
    return eigvals

def standard_mixer_circuit(n, param):
    qc_qubits = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qc_qubits)
    for q1 in range(n):
        qc.rx(-2*param, q1)
    qc.global_phase = -(param*n)
    return qc
