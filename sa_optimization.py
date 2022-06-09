import numpy as np

def basic_search(extra_inputs, cost_function, next_point_finder, verbose=False):
    current_point = None
    best_cost = float('inf')
    current_point_num = 0
    while True:
        current_point, total_points = next_point_finder(extra_inputs, current_point)
        current_point_num += 1
        if verbose:
            pc = 100*(current_point_num/total_points)
            print(f"Basic search {pc:.2f}% done", end="\r")
        if current_point is None:
            break
        cost = cost_function(extra_inputs, current_point)
        if cost < best_cost:
            best_point, best_cost = current_point, cost
    return best_point, best_cost

def qaoa_grid_search_point_finder(extra_inputs, previous_point):
    dims = extra_inputs['dims']
    n_params = len(dims)
    ranges, inds = [], []
    for dim in dims:
        ranges.append(list(np.arange(*tuple(dim))))
        inds.append(list(np.arange(0, len(ranges[-1]), 1)))
    if previous_point is None:
        current_inds = [0]*n_params
        found = True
    else:
        previous_point = list(previous_point)
        current_inds = [ranges[j].index(x) for j, x in enumerate(previous_point)]
        current_dim = 0
        found = False
        while True:
            if current_dim >= n_params:
                break
            else:
                current_inds[current_dim] += 1
                if (current_inds[current_dim] > inds[current_dim][-1]):
                    current_inds[current_dim] = 0
                    current_dim += 1
                else:
                    found = True
                    break

    if found:
        try:
            current_point = np.array([ranges[j][x] for j, x, in enumerate(current_inds)])
        except IndexError:
            print(current_inds)
            raise
    else:
        current_point = None

    total_points = np.prod([len(x) for x in  ranges])
    return current_point, total_points

def simulated_annealing_run(extra_inputs, iterations,
                           initial_state_generator,
                           cost_function,
                           candidate_generator,
                           acceptance_rule,
                           acceptance_parameter_generator,
                           verbose=False):

    best_cost = float('inf')
    state = initial_state_generator(extra_inputs)
    cost = cost_function(extra_inputs, state)
    costs = np.zeros(iterations+1, dtype=float)
    costs[0] = cost
    for iteration in range(iterations):
        if verbose:
            pc = 100*(iteration)/iterations
            print(f"Simulated annealing {pc:.2f}% complete.", end="\r")
        acceptance_parameter = acceptance_parameter_generator(extra_inputs, iterations, iteration)
        candidate_state = candidate_generator(extra_inputs, state)
        candidate_cost = cost_function(extra_inputs, candidate_state)
        accept = acceptance_rule(cost, candidate_cost, acceptance_parameter)
        if accept:
            state, cost = candidate_state, candidate_cost
            if cost < best_cost:
                best_state, best_cost = state, cost
        costs[iteration+1] = cost
    return best_state, best_cost, costs

def simulated_annealing(extra_inputs, iterations, runs,
                        initial_state_generator,
                        cost_function,
                        candidate_generator,
                        acceptance_rule,
                        acceptance_parameter_generator,
                        verbose=False):
    if verbose:
        if runs == 1:
            outer_verbose, inner_verbose = False, True
        else:
            outer_verbose, inner_verbose = True, False
    else:
        outer_verbose, inner_verbose = False, False

    best_cost = float('inf')
    for run in range(runs):
        if outer_verbose:
            pc = 100*(run)/runs
            print(f"Simulated annealing {pc:.2f}% complete.", end="\r")

        state, cost, costs = simulated_annealing_run(extra_inputs, iterations,
                             initial_state_generator,
                             cost_function,
                             candidate_generator,
                             acceptance_rule,
                             acceptance_parameter_generator,
                             verbose=inner_verbose)
        if cost < best_cost:
            best_state, best_cost, best_costs = state, cost, costs
    return best_state, best_cost, best_costs

def initial_params_for_annealing(annealing_inputs):
    initial_params = annealing_inputs['initial_params']
    if initial_params == 'random':
        initial_params = []
        dims = annealing_inputs['dims']
        for dim in dims:
            val = (np.random.default_rng().uniform()*(dim[1] - dim[0])) + dim[0]
            initial_params.append(val)
    initial_params = np.array(initial_params)
    return initial_params

def cost_for_annealing(annealing_inputs, params):
    cost_function = annealing_inputs['cost_function']
    return cost_function(tuple(params))

def new_params_for_annealing(annealing_inputs, current_params):
    step = annealing_inputs['step']
    n_params = len(current_params)
    param_idx = np.random.default_rng().choice(n_params)
    sign = np.random.default_rng().choice([-1, 1])
    new_params = current_params.copy()
    new_params[param_idx] += (sign*step)
    return new_params

def boltzmann_acceptance_rule(current_cost, candidate_cost, temperature):
    if candidate_cost <= current_cost:
        accept = True
    else:
        acceptance_probability = np.exp(-(candidate_cost-current_cost)/temperature)
        test_probability = np.random.default_rng().uniform()
        accept = test_probability <= acceptance_probability
    return accept

def temperature_schedule(annealing_inputs, iterations, iteration):
    try:
        T_max = annealing_inputs['T_max']
    except KeyError:
        T_max = 1.0
    scale = np.log((iterations+1)/(iteration+1))/np.log(iterations+1)
    T = T_max*scale
    return T
