import numpy as np
from scipy.stats import geom


def disperse_obs_dual_geom(obs, max_range, p):

    '''
    In: observation array, maximum range away from observation dispersed, p_geometric
    Out: new observation array weighted by probability at that base
    '''

    T = np.size(obs, axis=0)

    mut_range = 50
    geom_parameter = 0.05
    half_prob_vec = np.zeros(mut_range + 1)
    mut_prob_vector = np.zeros(mut_range * 2 + 1)

    for i in range(1, mut_range + 2):
        half_prob_vec[i - 1] = geom.pmf(i, geom_parameter)

    mut_prob_vector[mut_range:] = half_prob_vec
    mut_prob_vector[0:mut_range] = np.flip(half_prob_vec[1:], axis=0)
    mut_prob_vector /= np.sum(mut_prob_vector)

    obs_out = np.zeros(T)

    for t in range(0, T):
        value = obs[t]
        obs_out[t - mut_range: t + mut_range] = value * mut_prob_vector

    return obs_out

def calc_likelihood(obs, start, states):

    '''
    In: observation array, base where array starts on genome, list of state objects
    Out: compressed likelihood array
    '''

    N_states = len(states)
    T = np.size(obs[0], axis=0)
    out_likelihood = np.zeros((T, N_states))
    end = start + T

    for i in range(0, N_states):

        # Adjust for what is possible (the likelihood is non-zero)
        if states[i].possible == None:
            out_likelihood[:, i] = states[i].likelihood(obs)
        else:
            for region in states[i].possible:
                if region[0] >= start and region[1] <= end:
                    s_idx = region[0] - start
                    f_idx = region[1] - start
                    out_likelihood[s_idx:f_idx, i] = states[i].likelihood(obs)

        # Adjust for the explicit locations (the likelihood is one for state, zero for all others)
        if states[i].locations == None:
            pass
        else:
            for location in states[i].locations:
                if location[0] >= start and location[1] <= end:
                    s_idx = location[0] - start
                    f_idx = location[1] - start
                    assert np.any(out_likelihood[s_idx:f_idx, :] == 1) is False, 'Overlapping states with probability 1'
                    out_likelihood[s_idx:f_idx, :] = 0
                    out_likelihood[s_idx:f_idx, i] = 1

    return out_likelihood

