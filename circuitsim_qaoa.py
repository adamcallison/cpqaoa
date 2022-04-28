from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.test.mock import FakeTokyo
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit import Parameter
import numpy as np
import skopt

import cost_util, mix_util, generic_qaoa

from qiskit.providers.aer import AerSimulator

def qaoa_circuit_layer(J, h, c, params):
    try:
        params = tuple(params)
    except TypeError:
        raise ValueError("params must be iterable")
    if not len(params) == 2:
        raise ValueError("single QAOA layer has only 2 parameters")

    n = h.shape[0]

    qc_qubits = QuantumRegister(n, 'q')
    qc = QuantumCircuit(qc_qubits)

    param_p, param_d = params[0], params[1]
    qc_cost = cost_util.cost_circuit(J, h, c, param_p)
    qc_mix = mix_util.standard_mixer_circuit(n, param_d)
    qc = qc.compose(qc_cost)
    qc = qc.compose(qc_mix)

    return qc

def determine_qubit_assignment(qc):
    # not needed, will remove eventually
    wire_names = qc.draw().wire_names()
    logical_idxs, physical_idxs = [], []
    for wn in wire_names:
        if '->' in wn:
            logical_name, physical_name = \
                tuple(x.strip() for x in wn.split('->'))
            logical_name = logical_name.split('_')[1]
            logical_idx, physical_idx = int(logical_name), int(physical_name)
        else:
            logical_idx = wn.split('_')[1].strip().strip(':')
            logical_idx = int(logical_idx)
            physical_idx = logical_idx
        logical_idxs.append(logical_idx)
        physical_idxs.append(physical_idx)
    l_to_p = {logical_idxs[j]:physical_idxs[j] for j in \
        range(len(logical_idxs))}
    p_to_l = {physical_idxs[j]:logical_idxs[j] for j in \
        range(len(physical_idxs))}

    return l_to_p, p_to_l

def qaoa_circuit(J, h, c, params_or_layers, measurement=True, noise=False, \
    compile=True, optimization_level=3):
    # optimization_level only used if compile=True (passed to qiskit transpile)
    if not type(params_or_layers) == int:
        params = params_or_layers
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

    layer_template_params = (Parameter(f'param_p'), Parameter(f'param_d'))
    layer_template = qaoa_circuit_layer(J, h, c, layer_template_params)

    for layer in range(layers):
        param_p, param_d = params[2*layer], params[(2*layer)+1]
        qc_layer = layer_template.assign_parameters({
            layer_template_params[0]:param_p,
            layer_template_params[1]:param_d,
            }
        )
        qc = qc.compose(qc_layer)

    if measurement:
        qc.measure_all()

    if compile:
        device_backend = FakeTokyo()
        if noise:
            sim_tokyo = AerSimulator.from_backend(device_backend)
            qc = transpile(qc, sim_tokyo, optimization_level=optimization_level)
        else:
            qc = transpile(qc, device_backend, \
                optimization_level=optimization_level)
    return qc

def circuitsim_qaoa_objective(pqc, Jcost, hcost, ccost, params, shots, \
    cvar=False, noise=False, sample_catcher=None):
    run_inputs, cost_inputs = (pqc, (Jcost, hcost, ccost))
    res = generic_qaoa.qaoa_objective("circuitsim", run_inputs, cost_inputs, \
        params, shots, cvar, False, noise, sample_catcher)
    return res

def _circuitsim_qaoa_objective(pqc, Jcost, hcost, ccost, params, shots, \
    cvar=False, noise=False, sample_catcher=None):

    pqc_params = pqc.parameters
    layers = len(pqc_params)//2
    binds = {}
    for layer in range(layers):
        for p in pqc_params:
            if p.name == f'param_p{layer}':
                pp = p
            if p.name == f'param_d{layer}':
                pd = p
        binds[pp] = params[2*layer]
        binds[pd] = params[(2*layer)+1]
    qc = pqc.bind_parameters(binds)

    if noise:
        device_backend = FakeTokyo()
        sim_tokyo = AerSimulator.from_backend(device_backend)
        job = sim_tokyo.run(qc, shots=shots)
    else:
        simulator = QasmSimulator()
        job = simulator.run(qc, shots=shots)
    result = job.result()
    counts_qc = result.get_counts(qc)

    samples, nrgs, counts = [], [], []

    for sample_bin, count in counts_qc.items():
        sample = int(sample_bin, 2)
        nrg = cost_util.ising_assignment_cost_from_binary(Jcost, hcost, ccost, \
            sample_bin)
        samples.append(sample)
        nrgs.append(nrg)
        counts.append(count)
    samples, nrgs, counts = np.array(samples), np.array(nrgs), np.array(counts)
    return (samples, counts, nrgs)

def circuitsim_qaoa_loop(J, h, c, Jcost, hcost, ccost, layers, shots, \
    cvar=False, extra_samples=0, minimizer_params=None, param_max=(2*np.pi), \
    compile=False, noise=False, verbose=False):

    if minimizer_params is None:
        minimizer_params = {'n_calls': 100, 'n_random_starts':25}

    dims = [(0.0, param_max)]*2*layers

    sample_catcher = {}

    pqc = qaoa_circuit(J, h, c, layers, noise=noise, measurement=True, \
        compile=compile)

    def func(params):
        params = tuple(params)
        obj = circuitsim_qaoa_objective(pqc, Jcost, hcost, ccost, params, \
            shots, cvar=cvar, noise=noise, sample_catcher=sample_catcher)
        return obj

    if verbose:
        calls = 0
        best_obj = float('inf')
        def callback(res):
            nonlocal calls
            nonlocal best_obj
            obj = res.fun
            calls += 1
            if obj < best_obj:
                best_obj = obj
            print(
                f"Call {calls} of {minimizer_params['n_calls']}. "\
                f"Best Obj.: {best_obj}",
                end='\r')
    else:
        callback = None

    res = skopt.gp_minimize(func, dims, acq_func="EI", callback=callback, \
        **minimizer_params)
    value = res.fun
    params = tuple(res.x)

    if extra_samples > 0:
        params = tuple(params)
        circuitsim_qaoa_objective(pqc, Jcost, hcost, ccost, params, \
            extra_samples, cvar=cvar, sample_catcher=sample_catcher)
    samples = sample_catcher

    samples = \
        {k: v for k, v in sorted(samples.items(), key=lambda item: item[1][0])}

    return value, params, samples
