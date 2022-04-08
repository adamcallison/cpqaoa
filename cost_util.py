import numpy as np

from qiskit import QuantumCircuit, QuantumRegister


def qubo_to_ising(M, v, quboc):
    n = v.shape[0]
    M_to_J = M/4.0
    M_to_h = np.zeros(n)
    for j in range(n):
        M_to_h[j] += np.sum(M[j]) + np.sum(M[:,j])
    M_to_h = -1.0*M_to_h/4.0
    M_to_c = np.sum(M)/4.0

    v_to_h = -v/2.0
    v_to_c = np.sum(v)/2.0

    quboc_to_c = quboc

    J = M_to_J
    h = M_to_h + v_to_h
    c = M_to_c + v_to_c + quboc_to_c

    return J, h, c

def cost_eigenvalues(J, h, c):
    n = h.shape[0]
    N = 2**n
    costs = np.ndarray(N)
    costs[:] = c
    states = (N - 1) -  np.arange(N)
    for q1 in range(n):
        q1bit = (states & (1 << q1)) >> q1
        costs += -((-1)**q1bit)*h[q1]
        for q2 in range(n):
            q2bit = (states & (1 << q2)) >> q2
            costs += ((-1)**(q1bit + q2bit))*J[q1, q2]
    return costs

def cost_circuit(J, h, c, param):
    n = h.shape[0]
    qc_qubits = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qc_qubits)
    for q1 in range(n):
        qc.rz(2*param*h[q1], q1)
    for q1 in range(n-1):
        for q2 in range(q1+1, n):
            Jcoeff = J[q1, q2] + J[q2, q1]
            if Jcoeff == 0.0:
                continue
            qc.rzz(2*param*Jcoeff, q1, q2)
    qc.global_phase = qc.global_phase - (param*c)
    return qc
