import numpy as np

import abstract_qaoa
import circuitsim_qaoa

def qaoa_objective(mode, run_inputs, cost_inputs, params, shots, physical_to_logical, \
    cvar, get_statevector, sample_catcher):

    mode = mode.lower()

    if not mode in ('abstract', 'circuitsim'):
        raise ValueError(f"unrecognised mode: {mode}")

    if mode == 'circuitsim' and (shots is None):
        raise ValueError("shots required for circuitsim mode")

    if mode == 'circuitsim' and (not (get_statevector is False)):
        raise ValueError("get_statevector must be False for abstract mode")

    if mode == 'abstract' and (not (physical_to_logical is None)):
        raise ValueError(" physical_to_logical must be None for abstract mode")

    if mode == 'abstract':
        Hp_run = run_inputs
        Hp_cost = cost_inputs

        res = abstract_qaoa._abstract_qaoa_objective(Hp_run, Hp_cost, params, \
            get_statevector=get_statevector, shots=shots, cvar=cvar, \
            sample_catcher=sample_catcher)

    if mode == 'circuitsim':
        pqc, noise = run_inputs
        Jcost, hcost, ccost = cost_inputs

        res = circuitsim_qaoa._circuitsim_qaoa_objective(pqc, Jcost, hcost, \
            ccost, params, shots=shots, physical_to_logical=physical_to_logical, \
            cvar=cvar, sample_catcher=sample_catcher, noise=noise)

    if get_statevector:
        res, fstate = res
    if shots is None:
        obj = res
    else:
        # logic below has been somewhat tested, but could do with more,
        # especially cvar stuff...
        samples, counts, nrgs = res
        nrgs_idx = np.argsort(nrgs)
        samples, nrgs, counts = samples[nrgs_idx], nrgs[nrgs_idx], \
            counts[nrgs_idx]
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

    if get_statevector:
        return obj, fstate
    else:
        return obj
