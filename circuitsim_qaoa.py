from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.test.mock import FakeTokyo
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit import Parameter
import numpy as np
import skopt


import cost_util, mix_util

def qaoa_circuit(J, h, c, params_or_layers, measurement=True):
    if not type(params_or_layers) == int:
        layers_double = len(params)
        layers = layers_double // 2
    else:
        layers = params_or_layers
        params = ()
        for layer in range(layers):
            params = params + (Parameter(f'param_p{layer}'), \
                               Parameter(f'param_d{layer}'))
    n = h.shape[0]

    qc_qubits = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qc_qubits)

    for q in range(n):
        qc.h(q)

    for layer in range(layers):
        param_p, param_d = params[2*layer], params[(2*layer)+1]
        qc_cost = cost_util.cost_circuit(J, h, c, param_p)
        qc_mix = mix_util.standard_mixer_circuit(n, param_d)
        qc = qc.compose(qc_cost)
        qc = qc.compose(qc_mix)
    if measurement:
        qc.measure_all()

    backend = FakeTokyo()
    qc = transpile(qc, backend, optimization_level=3)

    return qc

def circuitsim_qaoa_run(pqc, Jcost, hcost, ccost, params, shots, \
    cvar=False, sample_catcher=None, cost_catcher=None):

    pqc_params = pqc.parameters
    binds = {}
    for layer in range(2):
        for p in pqc_params:
            if p.name == f'param_p{layer}':
                pp = p
            if p.name == f'param_d{layer}':
                pd = p
        binds[pp] = params[2*layer]
        binds[pd] = params[(2*layer)+1]
    qc = pqc.bind_parameters(binds)

    simulator = QasmSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    samples = []
    nrgs = []
    for key, val in counts.items():
        sample = int(key, 2)
        samples = samples + ([sample]*val)
        sample_vec = np.array(tuple((1 if x == '0' else -1) for x in key[::-1]))
        nrg = np.dot(sample_vec, np.dot(Jcost, sample_vec)) + \
            np.dot(hcost, sample_vec) + ccost
        nrgs = nrgs + ([nrg]*val)
    samples, nrgs = np.array(samples), np.array(nrgs)
    if not (sample_catcher is None):
        sample_catcher.extend(list(samples))
        cost_catcher.extend(list(nrgs))
    if cvar:
        nrgs = np.sort(nrgs)
        obj = np.mean(nrgs[:int(np.ceil(0.5*shots))])
    else:
        obj = np.mean(nrgs)
    return obj

def circuitsim_qaoa_loop(J, h, c, Jcost, hcost, ccost, layers, shots, \
    cvar=False, extra_samples=0, minimizer_params=None, verbose=False):

    if minimizer_params is None:
        minimizer_params = {'n_calls': 100, 'n_random_starts':25}

    dims = [(0.0, 2*np.pi)]*2*layers

    sample_catcher, cost_catcher = [], []

    pqc = qaoa_circuit(J, h, c, layers, measurement=True)

    def func(params):
        params = tuple(params)
        obj = circuitsim_qaoa_run(pqc, Jcost, hcost, ccost, params, shots, \
            cvar=cvar, sample_catcher=sample_catcher, cost_catcher=cost_catcher)
        return obj

    res = skopt.gp_minimize(func,                  # the function to minimize
                      dims,      # the bounds on each dimension of x
                      acq_func="EI",      # the acquisition function
                      verbose=verbose,
                      **minimizer_params
    )
    value = res.fun
    params = tuple(res.x)
    success = True
    if extra_samples > 0:
        params = tuple(params)
        circuitsim_qaoa_run(pqc, Jcost, hcost, ccost, params, \
            extra_samples, cvar=cvar, sample_catcher=sample_catcher, \
            cost_catcher=cost_catcher)
    samples = np.array(sample_catcher)
    costs = np.array(cost_catcher)
    samples_unique, costs_unique, counts = [], [], []
    while len(samples) > 0:
        sample = samples[0]
        fil = (samples == sample)
        count = np.sum(fil)
        cost = costs[0]
        samples = samples[~fil]
        costs = costs[~fil]
        samples_unique.append(sample)
        costs_unique.append(cost)
        counts.append(count)
    samples_unique, costs_unique, counts = np.array(samples_unique), \
        np.array(costs_unique), np.array(counts)
    costs_unique_idx = np.argsort(costs_unique)
    samples_unique = samples_unique[costs_unique_idx]
    costs_unique = costs_unique[costs_unique_idx]
    counts = counts[costs_unique_idx]
    samples_dict = {}
    for i, su in enumerate(samples_unique):
        samples_dict[su] = {
            'cost': costs_unique[i],
            'count': counts[i]
            }

    return value, params, success, samples_dict
