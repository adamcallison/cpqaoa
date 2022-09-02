from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.test.mock import FakeTokyo
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit import Parameter
import numpy as np
import skopt

import cost_util, mix_util, generic_qaoa

from qiskit.providers.aer import AerSimulator

def qaoa_circuit_layer(J, h, c, params, J_sequence=None, invert_cost_circuit=False):
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
    if J_sequence is None:
        qc_cost = cost_util.cost_circuit(J, h, c, param_p)
    else:
        qc_cost, tmp1, tmp2 = cost_util.cost_circuit_2qaoan(J_sequence, J, h, c, param_p)
    if invert_cost_circuit:
        qc_cost = qc_cost.inverse()
    qc_mix = mix_util.standard_mixer_circuit(n, param_d)
    qc = qc.compose(qc_cost)
    qc = qc.compose(qc_mix)

    return qc

def qaoa_circuit(J, h, c, params_or_layers, J_sequence=None, measurement=True, \
    noise=False, compile=True, optimization_level=3):
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
    layer_template = qaoa_circuit_layer(J, h, c, layer_template_params, \
        J_sequence=J_sequence)
    if not (J_sequence is None):
        layer_template2 = qaoa_circuit_layer(J, h, c, layer_template_params, \
            J_sequence=J_sequence, invert_cost_circuit=True)

    for layer in range(layers):
        param_p, param_d = params[2*layer], params[(2*layer)+1]
        if (not (J_sequence is None)) and (not (layer % 2 == 0)):
            qc_layer = layer_template2.assign_parameters({
                layer_template_params[0]:-param_p,
                layer_template_params[1]:param_d,
                }
            )
        else:
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
    physical_to_logical=None, cvar=False, noise=False, sample_catcher=None):
    run_inputs, cost_inputs = (pqc, (Jcost, hcost, ccost))
    res = generic_qaoa.qaoa_objective("circuitsim", run_inputs, cost_inputs, \
        params, shots, physical_to_logical, cvar, False, noise, sample_catcher)
    return res

def _circuitsim_qaoa_objective(pqc, Jcost, hcost, ccost, params, shots, \
    physical_to_logical=None, cvar=False, noise=False, sample_catcher=None):

    logical_to_physical_final = None
    if not (physical_to_logical is None):
        logical_to_physical_final = cost_util.invert_permutation(physical_to_logical)

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
        if not (logical_to_physical_final is None):
            n = len(logical_to_physical_final)
            tmp1 = ''
            tmp2 = sample_bin[::-1]
            for j in range(n):
                tmp1 = tmp1 + tmp2[logical_to_physical_final[j]]
            sample_bin_unmapped = tmp1[::-1]
            sample_bin = sample_bin_unmapped
        sample = int(sample_bin, 2)


        nrg = cost_util.ising_assignment_cost_from_binary(Jcost, hcost, ccost, \
            sample_bin)
        samples.append(sample)
        nrgs.append(nrg)
        counts.append(count)
    samples, nrgs, counts = np.array(samples), np.array(nrgs), np.array(counts)
    return (samples, counts, nrgs)

def circuitsim_qaoa_loop(J, h, c, Jcost, hcost, ccost, layers, shots, \
    J_sequence=None, cvar=False, extra_samples=0, minimizer_params=None, \
    compile=False, noise=False, get_pqc=False, verbose=False):

    if minimizer_params is None:
        minimizer_params = {'n_calls': 100, 'n_random_starts':25}
    else:
        minimizer_params = dict(minimizer_params)

    try:
        param_max = minimizer_params['param_max']
        del minimizer_params['param_max']
    except KeyError:
        param_max = 2*np.pi
    dims = [(0.0, param_max)]*2*layers

    sample_catcher = {}

    pqc = qaoa_circuit(J, h, c, layers, J_sequence=J_sequence, noise=noise, \
        measurement=True, compile=compile)

    #permutation = None
    #logical_to_physical = None
    if not (J_sequence is None):
        #if (layers % 2 == 0) and (J_sequence[0][0] == 'map'):
        if (layers % 2 == 0) and (J_sequence[0][0] == 'logical to physical'):
            physical_to_logical = cost_util.invert_permutation(J_sequence[0][1])
        #elif (layers % 2 == 1) and (J_sequence[-1][0] == 'unmap'):
        elif (layers % 2 == 1) and (J_sequence[-1][0] == 'physical to logical'):
            #permutation = J_sequence[-1][1]
            physical_to_logical = J_sequence[-1][1]
    else:
        #permutation = None
        physical_to_logical = None

    def func(params):
        params = tuple(params)
        #obj = circuitsim_qaoa_objective(pqc, Jcost, hcost, ccost, params, \
        #    shots, permutation=permutation, cvar=cvar, noise=noise, \
        #    sample_catcher=sample_catcher)
        obj = circuitsim_qaoa_objective(pqc, Jcost, hcost, ccost, params, \
            shots, physical_to_logical=physical_to_logical, cvar=cvar, noise=noise, \
            sample_catcher=sample_catcher)
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

    if get_pqc:
        return value, params, samples, pqc
    else:
        return value, params, samples
