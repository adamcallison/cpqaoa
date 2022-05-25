import numpy as np

import skopt
import spsa_optimization
import adam_optimization

def gaussian_process(objective_func, bounds, n_calls, n_random_starts, \
    verbose=False):
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
                f"Call {calls} of {n_calls}. "\
                f"Best Obj.: {best_obj}",
                end='\r')
    else:
        callback = None

    res = skopt.gp_minimize(objective_func, bounds, acq_func="EI", \
        verbose=False, callback=callback, n_calls=n_calls, \
        n_random_starts=n_random_starts)

    value = res.fun
    params = tuple(res.x)

    return value, params

def adam(objective_func, layers, n_calls, n_random_starts, verbose=False):
    grad_func = adam_optimization.grad_func
    outer_verbose = verbose and (n_random_starts > 1)
    inner_verbose = verbose and (n_random_starts == 1)

    best_value = float('inf')
    for nrs in range(n_random_starts):
        if outer_verbose:
            pc = 100*(nrs+1)/n_random_starts
            print(f"{pc:.2f}% complete", end="\r")

        initial_params = np.random.default_rng().uniform(size=layers*2)*2*np.pi
        iterations = n_calls//n_random_starts
        step = (2*np.pi/iterations)*100
        #step = np.min((0.1, step))
        grad_step = 0.005
        grad_args = (grad_step, objective_func)

        params = adam_optimization.adam_opt(grad_func, initial_params, step, \
            iterations, grad_args=grad_args, verbose=inner_verbose)

        value = objective_func(params)

        if value < best_value:
            best_value, best_params = value, params

    return best_value, best_params

def spsa(objective_func, initial_params, bounds, n_calls, verbose=False):

    func2 = lambda *params: objective_func(params)
    #stepsize = np.min((0.02, 0.02*(50/n_calls)))
    #stepsize = 0.01
    stepsize = 0.03
    spsa = spsa_optimization.SPSA(func2, initial_params, stepsize=stepsize, \
        bounds=bounds)
    spsa.optimize(n_calls, verbose=verbose)

    value = None
    params = tuple(spsa.parameters)

    return value, params
