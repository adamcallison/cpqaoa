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

def circuitsim_qaoa_run(pqc, Jcost, hcost, ccost, params, shots, cvar=False, \
    sample_catcher=None):

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
    counts_qc = result.get_counts(qc)

    samples, nrgs, counts = [], [], []

    # logic below has been somewhat tested, but could do with more, especially
    # cvar stuff...

    for sample_bin, count in counts_qc.items():
        sample = int(sample_bin, 2)
        nrg = cost_util.ising_assignment_cost_from_binary(Jcost, hcost, ccost, \
            sample_bin)
        samples.append(sample)
        nrgs.append(nrg)
        counts.append(count)
    samples, nrgs, counts = np.array(samples), np.array(nrgs), np.array(counts)
    nrgs_idx = np.argsort(nrgs)
    samples, nrgs, counts = samples[nrgs_idx], nrgs[nrgs_idx], counts[nrgs_idx]
    if not (sample_catcher is None):
        for i, sample in enumerate(samples):
            try:
                sample_stats = sample_catcher[sample]
                sample_stats[1] += counts[i]
            except KeyError:
                sample_stats = [nrgs[i], counts[i]]
            sample_catcher[sample] = sample_stats
    if cvar:
        thresh = int(np.ceil(0.5*shots))
        counts_cumsum = np.cumsum(counts)
        use = np.sum(counts_cumsum < thresh)
        samples_use = samples[:use+1]
        nrgs_use = nrgs[:use+1]
        counts_use = counts[:use+1]
        counts_use[-1] = thresh - np.sum(counts_use[:-1])
    else:
        samples_use = samples
        nrgs_use = nrgs
        counts_use = counts
        thresh = shots

    assert np.sum(counts_use) == thresh

    obj = np.dot(nrgs_use, counts_use)/thresh

    return obj

def circuitsim_qaoa_loop(J, h, c, Jcost, hcost, ccost, layers, shots, \
    cvar=False, extra_samples=0, minimizer_params=None, verbose=False):

    if minimizer_params is None:
        minimizer_params = {'n_calls': 100, 'n_random_starts':25}

    dims = [(0.0, 2*np.pi)]*2*layers

    sample_catcher = {}

    pqc = qaoa_circuit(J, h, c, layers, measurement=True)

    def func(params):
        params = tuple(params)
        obj = circuitsim_qaoa_run(pqc, Jcost, hcost, ccost, params, shots, \
            cvar=cvar, sample_catcher=sample_catcher)
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
        circuitsim_qaoa_run(pqc, Jcost, hcost, ccost, params, extra_samples, \
            cvar=cvar, sample_catcher=sample_catcher)
    samples = sample_catcher

    samples = \
        {k: v for k, v in sorted(samples.items(), key=lambda item: item[1][0])}

    return value, params, success, samples
