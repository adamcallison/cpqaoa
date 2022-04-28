import numpy as np
import skopt
import sys
from numba import njit

import mix_util
import generic_qaoa

@njit
def fwht(a) -> None:
    """In-place Fast Walsh–Hadamard Transform of array a of size 2^n."""
    N = a.shape[0]
    sqrtN = np.sqrt(N)
    h = 1
    while h < N:
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = (x + y)
                a[j + h] = (x - y)
        h *= 2

    for j in range(len(a)):
        a[j] /= sqrtN


def unitary_propagator(H, state, time):
    return np.exp(-1.0j*time*H)*state

def abstract_qaoa_run(Hp, params):

    N = Hp.shape[0]
    n = int(np.round(np.log2(N)))

    Hd_transformed = mix_util.standard_mixer_eigenvalues(n)

    layers_double = len(params)
    if not layers_double % 2 == 0:
        raise ValueError
    layers = layers_double//2

    N = Hp.shape[0]
    state = np.ones(N)/np.sqrt(N)
    for layer in range(layers):
        param_p, param_d = params[2*layer], params[(2*layer)+1]
        state = unitary_propagator(Hp, state, param_p)
        fwht(state)
        state = unitary_propagator(Hd_transformed, state, param_d)
        fwht(state)
    return state

def abstract_qaoa_objective(Hp_run, Hp_cost, params, get_statevector=False, \
    shots=None, cvar=False, sample_catcher=None):
    run_inputs, cost_inputs = (Hp_run, Hp_cost)
    res = generic_qaoa.qaoa_objective("abstract", run_inputs, cost_inputs, \
        params, shots, cvar, get_statevector, False, sample_catcher)
    if get_statevector:
        obj, fstate = res
        return obj, fstate
    else:
        obj = res
        return obj

def _abstract_qaoa_objective(Hp_run, Hp_cost, params, shots=None, \
    get_statevector=False, cvar=False, sample_catcher=None):

    if (shots is None) and cvar:
        raise NotImplementedError
    if (shots is None) and ( not (sample_catcher is None) ):
        raise ValueError

    N = Hp_run.shape[0]
    fstate = abstract_qaoa_run(Hp_run, params)
    fprobs = np.abs(fstate)**2
    if shots is None:
        obj = np.dot( Hp_cost, fprobs )
        res =  obj
    else:
        samples = np.random.default_rng().choice(N, size=shots, p=fprobs)
        samples, counts = np.unique(samples, return_counts=True)
        nrgs = Hp_cost[samples]
        res = (samples, counts, nrgs)

    if get_statevector:
        return res, fstate
    else:
        return res

def abstract_qaoa_loop(Hp_run, Hp_cost, layers, shots=None, cvar=False, \
    extra_samples=0, minimizer_params=None, get_statevector=False, \
    param_max=(2*np.pi), verbose=False):

    if minimizer_params is None:
        minimizer_params = {'n_calls': 100, 'n_random_starts':25}

    dims = [(0.0, param_max)]*2*layers

    if not (shots is None):
        sample_catcher = {}
    else:
        sample_catcher = None

    def func(params):
        params = tuple(params)
        obj = abstract_qaoa_objective(Hp_run, Hp_cost, params, shots=shots, \
            cvar=cvar, sample_catcher=sample_catcher)
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

    res = skopt.gp_minimize(func,                  # the function to minimize
                      dims,      # the bounds on each dimension of x
                      acq_func="EI",      # the acquisition function
                      verbose=False,
                      callback=callback,
                      **minimizer_params
    )

    value = res.fun
    params = tuple(res.x)

    if shots is None:
        sample_catcher = {}
    if (extra_samples > 0) or get_statevector:
        _, fstate = abstract_qaoa_objective(Hp_run, Hp_cost, params, \
            get_statevector=True, shots=extra_samples, cvar=cvar, \
            sample_catcher=sample_catcher)

    samples = sample_catcher

    samples = \
        {k: v for k, v in sorted(samples.items(), key=lambda item: item[1][0])}

    if get_statevector:
        return value, params, samples, fstate
    else:
        return value, params, samples
