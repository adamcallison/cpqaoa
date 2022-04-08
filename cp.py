import numpy as np
import cost_util

def cp_qubo(adjacency, sparse_approx=False):
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

    v_p1 = -2.0*rho*n*np.ones(n)
    v_p2 = np.zeros(n)
    for j in range(n):
        v_p2[j] = np.sum(adjacency[j])
    v_p2 = 2.0*(1+rho)*v_p2
    v = v_p1 + v_p2

    quboc = rho*(n*(n-1) - np.sum(adjacency))

    return M, v, quboc

def cp_ising(adjacency, sparse_approx=False):
    M, v, quboc = cp_qubo(adjacency, sparse_approx=sparse_approx)
    J, h, c = cost_util.qubo_to_ising(-M, -v, -quboc)
    return J, h, c

def sample_to_cp_partition(sample, n):
    # takes a sample (in integer form) and returns the corresponding
    # core-periphery partition
    samplestr = bin(sample)[2:]
    samplestr = ('0'*(n-len(samplestr))) + samplestr
    samplestr = samplestr[::-1]
    core, periphery = [], []
    for i, val in enumerate(samplestr):
        if val == '1':
            core.append(i)
        else:
            periphery.append(i)
    core, periphery = np.array(core), np.array(periphery)
    return core, periphery

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
