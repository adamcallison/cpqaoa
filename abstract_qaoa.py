import numpy as np
import skopt
import sys

from numba import njit

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

def abstract_qaoa_run(Hd_transformed, Hp, params):
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

def abstract_qao_objective(Hd_transformed, Hp_run, Hp_cost, params, \
    shots=None, cvar=False, sample_catcher=None):

    if (shots is None) and cvar:
        raise NotImplementedError
    if (shots is None) and ( not (sample_catcher is None) ):
        raise ValueError

    N = Hp_run.shape[0]
    fstate = abstract_qaoa_run(Hd_transformed, Hp_run, params)
    fprobs = np.abs(fstate)**2
    if shots is None:
        obj = np.dot( Hp_cost, fprobs )
    else:
        samples = np.random.default_rng().choice(N, size=shots, p=fprobs)
        if not (sample_catcher is None):
            sample_catcher.extend(list(samples))
        nrgs = Hp_cost[samples]
        if cvar:
            nrgs = np.sort(nrgs)
            obj = np.mean(nrgs[:int(np.ceil(0.5*shots))])
        else:
            obj = np.mean(nrgs)
    return obj

def abstract_qaoa_loop(Hd_transformed, Hp_run, Hp_cost, layers, shots=None, \
    cvar=False, extra_samples=0, minimizer_params=None, get_statevector=False, \
    param_max=(2*np.pi), verbose=False):

    if minimizer_params is None:
        minimizer_params = {'n_calls': 100, 'n_random_starts':25}

    dims = [(0.0, param_max)]*2*layers

    if not (shots is None):
        sample_catcher = []
    else:
        sample_catcher = None

    def func(params):
        params = tuple(params)
        obj = abstract_qao_objective(Hd_transformed, Hp_run, Hp_cost, params, \
            shots=shots, cvar=cvar, sample_catcher=sample_catcher)
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
    success = True
    if shots is None:
        sample_catcher = []
    if (extra_samples > 0) or get_statevector:
        fstate = abstract_qaoa_run(Hd_transformed, Hp_run, params)
    if extra_samples > 0:
        fprobs = np.abs(fstate)**2
        N = Hp_run.shape[0]
        new_samples = np.random.default_rng().choice(N, \
            size=extra_samples, p=fprobs)
        sample_catcher.extend(list(new_samples))
    samples = np.array(sample_catcher)
    samples_unique, counts = np.unique(samples, return_counts=True)
    costs_unique = Hp_cost[samples_unique]
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
    if get_statevector:
        return value, params, success, samples_dict, fstate
    else:
        return value, params, success, samples_dict
