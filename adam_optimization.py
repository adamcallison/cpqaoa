import numpy as np

def grad_func(params, grad_step, obj_func):
    n_params = len(params)
    grads = np.zeros_like(params)
    for j in range(n_params):
        curr_params_m, curr_params_p = np.array(params), np.array(params)
        curr_params_m[j] -= (grad_step/2)
        curr_params_p[j] += (grad_step/2)
        obj_m, obj_p =  obj_func(curr_params_m), obj_func(curr_params_p)
        grads[j] = (obj_p - obj_m)/grad_step

    return grads

def adam_opt(grad_func, params, step, iterations, beta1=0.9, beta2=0.999, \
    eps=1e-8, grad_args=None, grad_kwargs=None, verbose=False):
    if grad_args is None:
        grad_args = ()
    if grad_kwargs is None:
        grad_kwargs = {}

    params = np.array(params)
    n_params = params.shape[0]
    m = np.zeros(n_params, dtype=float) # initialize 1st moment vector
    v = np.zeros(n_params, dtype=float) # initialize 2nd moment vector
    t = 0 # initialize timestep
    for iteration in range(iterations):
        if verbose:
            pc = 100*(iteration+1)/iterations
            print(f"{pc:.2f}% complete", end="\r")
        t += 1
        g = grad_func(params, *grad_args, **grad_kwargs) # get gradients wrt stochastic objective function
        m = (beta1*m) + ((1-beta1)*g) # update biased first moment estimate
        v = (beta2*v) + ((1-beta2)*(g**2))
        #m_hat = m/(1-(beta1**t)) # Compute bias-corrected first moment estimate
        #v_hat = v/(1-(beta2**t)) # Compute bias-corrected first moment estimate
        #params = params - (step*m_hat/(np.sqrt(v_hat)+eps))
        step_curr = step*np.sqrt(1-(beta2**t))/(1-(beta1**t))
        params = params - (step_curr*m/(np.sqrt(v)+eps))
    return params
