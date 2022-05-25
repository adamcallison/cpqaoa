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
        hval = h[q1]
        if not (hval == 0.0):
            qc.rz(2*param*hval, q1)
    for q1 in range(n-1):
        for q2 in range(q1+1, n):
            Jcoeff = J[q1, q2] + J[q2, q1]
            if Jcoeff == 0.0:
                continue
            qc.rzz(2*param*Jcoeff, q1, q2)
    if not (c == 0.0):
        qc.global_phase = qc.global_phase - (param*c)
    return qc

def invert_permutation(permutation):
    # maybe not needed here
    inverse_perm = np.zeros_like(permutation)
    for i, j in enumerate(permutation):
        inverse_perm[j] = i
    return inverse_perm

def cost_circuit_2qaoan(J_sequence, J, h, c, param):
    n = h.shape[0]
    qc_qubits = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qc_qubits)

    initial_permutation, inv_initial_permutation = None, None
    if J_sequence[0][0] == 'map':
        initial_permutation = np.array(J_sequence[0][1], dtype=int)
        inv_initial_permutation = invert_permutation(initial_permutation)
        h = h.copy()
        J_sequence = J_sequence[1:]

    for q1 in range(n):
        logical_qubit = q1
        if initial_permutation is None:
            physical_qubit = q1
        else:
            physical_qubit = inv_initial_permutation[q1]
        hval = h[physical_qubit]
        if not (hval == 0.0):
            qc.rz(2*param*hval, q1)

    final_permutation = None
    for i, instruction in enumerate(J_sequence):
        if instruction[0] == 'swap':
            qc.swap(instruction[1][0], instruction[1][1])
        elif instruction[0] == 'hamiltonian_gate':
            physical_pair = instruction[2]
            logical_pair = instruction[1]
            Jcoeff = J[logical_pair[0], logical_pair[1]] + J[logical_pair[1], logical_pair[0]]
            if Jcoeff == 0:
                continue
            qc.rzz(2*param*Jcoeff, physical_pair[0], physical_pair[1])
        elif instruction[0] == 'map':
            raise ValueError("map can only be done as first step")
        elif instruction[0] == 'unmap':
            if not (i == (len(J_sequence)) - 1):
                raise ValueError("unmap can only be done as last step")
            else:
                final_permutation = np.array(instruction[1])

    if not (c == 0.0):
        qc.global_phase = qc.global_phase - (param*c)
    return qc, initial_permutation, final_permutation

def bnb_optimize(J, h, c, verbose=False):
    best_state, best_nrg, stats = mixsk_all.bnb(J, h, priority_order='lowest', \
    bound_type='recursive', verbose=verbose)
    best_nrg += c
    best_state = \
        int(''.join(['0' if x == 1 else '1' for x in best_state[0][::-1]]),2)
    return best_state, best_nrg, stats
