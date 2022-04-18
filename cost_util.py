import numpy as np
import mixsk_all

from qiskit import QuantumCircuit, QuantumRegister

def ising_assignment_cost_from_vec(J, h, c, assignment_vec):
    Jpart = np.dot(assignment_vec, np.dot(J, assignment_vec))
    hpart = np.dot(h, assignment_vec)
    return Jpart + hpart + c

def ising_assignment_cost_from_binary(J, h, c, assignment_bin):
    n = h.shape[0]
    assignment_bin = ('0'*(n-len(assignment_bin))) + assignment_bin
    assignment_vec = np.array(
        tuple((1 if x == '0' else -1) for x in assignment_bin[::-1])
        )
    return ising_assignment_cost_from_vec(J, h, c, assignment_vec)

def ising_assignment_cost_from_int(J, h, c, assignment_int):
    n = h.shape[0]
    assignment_bin = bin(assignment_int)[2:]
    assignment_bin = ('0'*(n-len(assignment_bin))) + assignment_bin
    return ising_assignment_cost_from_binary(J, h, c, assignment_bin)

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

def bnb_optimize(J, h, c, verbose=False):
    best_state, best_nrg, stats = mixsk_all.bnb(J, h, priority_order='lowest', \
    bound_type='recursive', verbose=verbose)
    best_nrg += c
    best_state = \
        int(''.join(['0' if x == 1 else '1' for x in best_state[0][::-1]]),2)
    return best_state, best_nrg, stats
