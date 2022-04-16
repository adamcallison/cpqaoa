import numpy as np
import cost_util

def cp_qubo_mode1(adjacency, sparse_approx=False):
    n = adjacency.shape[0]
    N1 = int(np.round(np.sum(adjacency)/2))
    Nt = n*(n-1)//2
    N2 = Nt - N1
    rho = N1/N2

    M_p1 = rho*np.ones((n, n))
    np.fill_diagonal(M_p1, 0.0)
    M_p2 = -(1+rho)*adjacency
    if sparse_approx:
        M = M_p2
    else:
        M = M_p1 + M_p2

    v_p1 = -2.0*rho*(n-1)*np.ones(n)
    v_p2 = np.zeros(n)
    for j in range(n):
        v_p2[j] = np.sum(adjacency[j])
    v_p2 = 2.0*(1+rho)*v_p2
    v = v_p1 + v_p2

    quboc = rho*(n*(n-1) - np.sum(adjacency))

    return M, v, quboc

def cp_qubo_mode2(adjacency, sparse_approx=False):
    # need to check if this is correct
    n = adjacency.shape[0]
    N1 = int(np.round(np.sum(adjacency)/2))
    Nt = n*(n-1)//2
    N2 = Nt - N1
    rho = N1/N2

    M_p2 = (1-rho)*adjacency
    if sparse_approx:
        M = M_p2
    else:
        M_p1 = rho*np.ones((n, n))
        np.fill_diagonal(M_p1, 0.0)
        M = M_p1 + M_p2

    v_p1 = -2.0*rho*(n-1)*np.ones(n)
    v_p2 = np.zeros(n)
    for j in range(n):
        v_p2[j] = np.sum(adjacency[j])
    v_p2 = 2.0*rho*v_p2
    v = v_p1 + v_p2

    quboc = rho*(n*(n-1) - np.sum(adjacency))

    return M, v, quboc

def cp_qubo_mode3(adjacency, sparse_approx=False):
    # need to check if this is correct
    n = adjacency.shape[0]
    N1 = int(np.round(np.sum(adjacency)/2))
    Nt = n*(n-1)//2
    N2 = Nt - N1
    rho = N1/N2

    M_p2 = (1+rho)*adjacency
    if sparse_approx:
        M = M_p2
    else:
        M_p1 = -1.0*rho*np.ones((n, n))
        np.fill_diagonal(M_p1, 0.0)
        M = M_p1 + M_p2

    v = np.zeros(n)

    quboc = rho*(n*(n-1) - np.sum(adjacency))

    return M, v, quboc

def cp_qubo(adjacency, sparse_approx=False, mode=1):
    if mode == 1:
        return cp_qubo_mode1(adjacency, sparse_approx=sparse_approx)
    elif mode == 2:
        return cp_qubo_mode2(adjacency, sparse_approx=sparse_approx)
    elif mode == 3:
        return cp_qubo_mode3(adjacency, sparse_approx=sparse_approx)


def cp_ising(adjacency, sparse_approx=False, mode=1):
    M, v, quboc = cp_qubo(adjacency, sparse_approx=sparse_approx, mode=mode)
    J, h, c = cost_util.qubo_to_ising(-M, -v, -quboc)
    return J, h, c

def cp_ising_direct_mode1(A, sparse_approx=False):
    # tries to place at least one vertex of each edge in the core
    # tries to exclude both vertices of each non-edge from the core
    n = A.shape[0]
    N1 = int(np.round(np.sum(A)/2))
    Nt = n*(n-1)//2
    N2 = Nt - N1

    alpha, beta = (Nt/N1), (Nt/N2)

    J_p1 = (beta+alpha)*A/8
    if sparse_approx:
        J = J_p1
    else:
        J_p2 = -beta*np.ones((n, n))/8
        J = J_p1 + J_p2

    h_p1 = np.zeros(n)
    for j in range(n):
        h_p1[j] = np.sum(A[j])
    h_p1 *= (alpha+beta)/4
    h_p2 = -n*beta/4

    h = h_p1 + h_p2

    c = ( (beta-(3*alpha))*np.sum(A)/8 ) - ( n*n*beta/8 )

    return J, h, c

def cp_ising_direct_mode2(A, sparse_approx=False):
    # tries to place both vertices of each edge in the core
    # tries to exclude at least one vertex of each non-edge from the core
    n = A.shape[0]
    N1 = int(np.round(np.sum(A)/2))
    Nt = n*(n-1)//2
    N2 = Nt - N1

    alpha, beta = (Nt/N1), (Nt/N2)

    J_p1 = -(beta+alpha)*A/8
    if sparse_approx:
        J = J_p1
    else:
        J_p2 = beta*np.ones((n, n))/8
        J = J_p1 + J_p2

    h_p1 = np.zeros(n)
    for j in range(n):
        h_p1[j] = np.sum(A[j])
    h_p1 *= (alpha+beta)/4
    h_p2 = -n*beta/4

    h = h_p1 + h_p2

    c = ( ((3*beta)-alpha)*np.sum(A)/8 ) - ( 3*n*n*beta/8 )

    return J, h, c

def cp_ising_direct(A, sparse_approx=False, mode=1):
    if mode == 1:
        return cp_ising_direct_mode1(A, sparse_approx=sparse_approx)
    elif mode == 2:
        return cp_ising_direct_mode2(A, sparse_approx=sparse_approx)

def sample_to_cp_partition(sample, n):
    # takes a sample (in integer form) and returns the corresponding
    # core-periphery partition
    samplestr = bin(sample)[2:]
    samplestr = ('0'*(n-len(samplestr))) + samplestr
    samplestr = samplestr[::-1]
    core, periphery = [], []
    for i, val in enumerate(samplestr):
        if val == '1':
            core.append(i+1)
        else:
            periphery.append(i+1)
    core, periphery = np.array(core), np.array(periphery)
    return core, periphery

def stochastic_block_model(n_nodes, n_core, p_cc, p_cp, p_pp):
    A = np.zeros((n_nodes, n_nodes))
    for row in range(n_nodes-1):
        for col in range(row+1, n_nodes):
            rn = np.random.default_rng().uniform()
            if (row < n_core) and (col < n_core):
                A[row, col] = int(rn < p_cc)
                A[col, row] = int(rn < p_cc)
            elif (row >= n_core) and (col < n_core):
                A[row, col] = int(rn < p_cp)
                A[col, row] = int(rn < p_cp)
            elif (row < n_core) and (col >= n_core):
                A[row, col] = int(rn < p_cp)
                A[col, row] = int(rn < p_cp)
            elif (row >= n_core) and (col >= n_core):
                A[row, col] = int(rn < p_pp)
                A[col, row] = int(rn < p_pp)
    return A

def borgatti_etal():
    # returns example adjacency matrix for Borgatti et. al
    # https://www.sciencedirect.com/science/article/pii/S0378873399000192
    graph = {}
    graph[16] = [7,13,15,17, 8,2,6,12,18,20]
    graph[7] = [16,15,17, 4,9,14,5,6,20]
    graph[13] = [16,15,17, 2,3,6,18,19,20]
    graph[15] = [16,7,13,17, 4,8,9,11,2,5,6,18,19,20]
    graph[17] = [16,7,13,15, 4,1,8,9,10,11,2,3,14,6,18,19,20]
    graph[4] = [7,15,17, 14,6]
    graph[1] = [17]
    graph[8] = [16,15,17]
    graph[9] = [7,15,17]
    graph[10] = [17]
    graph[11] = [15,17]
    graph[2] = [16,13,15,17, 14,20]
    graph[3] = [13,17, 18]
    graph[14] = [7,17, 4,2,6]
    graph[5] = [7,15, 6]
    graph[6] = [16,7,13,15,17, 4,14,5,20]
    graph[12] = [16]
    graph[18] = [16,13,15,17, 3,19]
    graph[19] = [13,15,17, 18]
    graph[20] = [16,7,13,15,17, 2,6]

    n = 20
    A = np.zeros((n, n))
    for j in range(n):
        row = graph[j+1]
        for k in row:
            A[j, k-1] = 1.0
    return A
