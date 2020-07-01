import numpy as np
from decryptr.classify.HMM_functions import fwd_bwd_alg
from scipy.special import psi

def baum_welch_alg(pi_prior, tmat_prior, pi, tmat, states, likelihood, obs):

    '''
    In: compressed pi, tmat as well as original priors.
    Also takes list of state objects and list of likelihood arrays. (size of T for axis 0 for likelihood)

    Out: new tmat, list of state object
    '''

    '''EXPAND COMPRESSED INPUTS'''

    # Determine how many sub-states we are working with
    N_states = len(states)
    N_substates = 0

    # Examine properties of pi and tmat to make sure they make sense
    # assert np.size(tmat, axis=0) == len(states), 'tmat does not have N_states rows'
    # assert np.size(tmat, axis=0) == len(states), 'tmat does not have N_states columns'
    #
    # for i in range(0, np.size(tmat, axis=0)):
    #     assert np.sum(tmat[i, :]) == 1, 'Not all rows in compressed tmat add to one'
    # assert np.sum(pi) == 1, 'Pi vector does not add to one'

    # Make a list of emit names
    emit_names = []
    for i in range(0, len(states[0].emits)):
        emit_names.append(states[0].emits[i].emit_name)

    for i in range(0, N_states):
        block = states[i].block_tmat
        state_name = states[i].name
        out_prob = states[i].leaving_prob

        # Examine properties of state object to make sure they make sense
        # assert np.size(block, axis=0) == np.size(block, axis=1), 'Block is not square matrix for state: %r' %state_name
        # for j in range(0, np.size(block, axis=0) - 1):
        #     assert np.sum(block[j, :]) == 1, 'Not all rows in block add to one for state: %r' %state_name
        # assert np.sum(block[i, :]) + out_prob == 1, 'Last row in block does not add to one for state: %r' %state_name
        #
        # assert len(states[i].emissions) == len(emit_names), 'Num emission objects do not agree across states'
        # for j in range(0, len(states[i].emits)):
        #     assert states[i].emits[j].emit_name == emit_names[j], 'Emission types do not agree across states'

        # Increment N_substates counter
        N_substates += np.size(block, axis=0)

    # Declare empty array (N_substates x N_substates) for tmat expanded and vector (N_substates) for pi expanded
    expand_tmat = np.zeros((N_substates, N_substates))
    expand_pi = np.zeros(N_substates)

    # Fill expanded tmat with blocks (keep track of row of the first and last sub-state for each macro state)
    idx = 0
    s_idx = []
    f_idx = []
    for i in range(0, N_states):
        s_idx.append(idx)
        end_idx = idx + states[i].n
        expand_tmat[idx:end_idx, idx:end_idx] = states[i].block_tmat
        expand_pi[idx:end_idx] = pi[i] / states[i].n
        idx = end_idx
        f_idx.append(end_idx - 1)

    # Add transitions between macro states to the blocks
    idx = 0
    for i in range(0, N_states):
        for j in range(0, N_states):
            expand_tmat[f_idx[i], s_idx[j]] = states[i].leaving_prob * tmat[i, j]

    # Likelihood array calculations

    # Make sure likelihood array list makes sense
    T = np.size(likelihood, axis=0)
    assert np.size(likelihood, axis=1) == len(states), 'Likelihood list is not of same length as state list'

    # Create the expanded likelihood array
    expand_likelihood = np.zeros((T, N_substates))
    for i in range(0, N_states):
        for temp_idx in range(int(s_idx[i]), int(f_idx[i]+1)):
            expand_likelihood[:, temp_idx] = likelihood[:, i]

    # Normalize likelihood array

    for i in range(0, T):
        if np.sum(expand_likelihood[i, :]) == 0:
            expand_likelihood[i, :] = 1
            #print("Underflow, consider as missing observation")
        else:
            expand_likelihood[i, :] = expand_likelihood[i, :] / np.sum(expand_likelihood[i, :])

    '''RUN THE FORWARD BACKWARD ALGORITHM'''

    l_fMsg, l_bMsg, l_Marg, resp, respPair = fwd_bwd_alg(np.log(expand_pi), np.log(expand_tmat), np.log(expand_likelihood))

    # Update state variables (transitions within blocks and emissions from state)
    for i in range(0, N_states):
        states[i].update_parameters(obs, resp[:, s_idx[i]:f_idx[i]+1], respPair[:, s_idx[i]:f_idx[i]+1, s_idx[i]:f_idx[i]+1])

    # Update the compressed transition matrix
    compressed_count = np.zeros((N_states, N_states))

    for i in range(0, N_states):
        for j in range(0, N_states):
            compressed_count[i, j] = np.sum(respPair[f_idx[i], s_idx[j], :]) / np.sum(resp[:, f_idx[i]])

    tmat_posterior = tmat_prior + compressed_count

    # From posterior counts calculate the transition matrix
    for i in range(0, N_states):
        for j in range(0, N_states):
            tmat[i, j] = np.exp(psi(tmat_posterior[i, j]) - psi(np.sum(tmat_posterior[i, :])))
        tmat[i, :] /= np.sum(tmat[i, :])

    # From posterior count calculate initial probability vector

    compressed_pi_count = np.zeros((N_states))
    for i in range(0, N_states):
        compressed_pi_count = np.sum(resp[0, s_idx[i]:f_idx[i]])

    pi_posterior = pi_prior + compressed_pi_count

    for i in range(0, N_states):
        pi[i] = np.exp(psi(pi_posterior[i]) - psi(np.sum(pi_posterior)))

    return pi, tmat, states


def decode(pi_prior, tmat_prior, pi, tmat, states, likelihood, obs, epsilons=False):


    '''
    In: compressed pi, tmat as well as original priors.
    Also takes list of state objects and list of likelihood arrays. (size of T for axis 0 for likelihood)

    Out: compressed gammas
    '''

    '''EXPAND COMPRESSED INPUTS'''

    # Determine how many sub-states we are working with
    N_states = len(states)
    N_substates = 0

    # Examine properties of pi and tmat to make sure they make sense
    # assert np.size(tmat, axis=0) == len(states), 'tmat does not have N_states rows'
    # assert np.size(tmat, axis=0) == len(states), 'tmat does not have N_states columns'
    #
    # for i in range(0, np.size(tmat, axis=0)):
    #     assert np.sum(tmat[i, :]) == 1, 'Not all rows in compressed tmat add to one'
    # assert np.sum(pi) == 1, 'Pi vector does not add to one'

    # Make a list of emit names
    emit_names = []
    for i in range(0, len(states[0].emits)):
        emit_names.append(states[0].emits[i].emit_name)

    for i in range(0, N_states):
        block = states[i].block_tmat
        state_name = states[i].name
        out_prob = states[i].leaving_prob

        # Examine properties of state object to make sure they make sense
        # assert np.size(block, axis=0) == np.size(block, axis=1), 'Block is not square matrix for state: %r' %state_name
        # for j in range(0, np.size(block, axis=0) - 1):
        #     assert np.sum(block[j, :]) == 1, 'Not all rows in block add to one for state: %r' %state_name
        # assert np.sum(block[i, :]) + out_prob == 1, 'Last row in block does not add to one for state: %r' %state_name
        #
        # assert len(states[i].emissions) == len(emit_names), 'Num emission objects do not agree across states'
        # for j in range(0, len(states[i].emits)):
        #     assert states[i].emits[j].emit_name == emit_names[j], 'Emission types do not agree across states'

        # Increment N_substates counter
        N_substates += np.size(block, axis=0)

    # Declare empty array (N_substates x N_substates) for tmat expanded and vector (N_substates) for pi expanded
    expand_tmat = np.zeros((N_substates, N_substates))
    expand_pi = np.zeros(N_substates)

    # Fill expanded tmat with blocks (keep track of row of the first and last sub-state for each macro state)
    idx = 0
    s_idx = []
    f_idx = []
    for i in range(0, N_states):
        s_idx.append(idx)
        end_idx = idx + states[i].n
        expand_tmat[idx:end_idx, idx:end_idx] = states[i].block_tmat
        expand_pi[idx:end_idx] = pi[i] / states[i].n
        idx = end_idx
        f_idx.append(end_idx - 1)

    # Add transitions between macro states to the blocks
    idx = 0
    for i in range(0, N_states):
        for j in range(0, N_states):
            expand_tmat[f_idx[i], s_idx[j]] = states[i].leaving_prob * tmat[i, j]

    # Likelihood array calculations

    # Make sure likelihood array list makes sense
    T = np.size(likelihood, axis=0)
    assert np.size(likelihood, axis=1) == len(states), 'Likelihood list is not of same length as state list'

    # Create the expanded likelihood array
    expand_likelihood = np.zeros((T, N_substates))
    for i in range(0, N_states):
        for temp_idx in range(int(s_idx[i]), int(f_idx[i]+1)):
            expand_likelihood[:, temp_idx] = likelihood[:, i]

    # Normalize likelihood array

    for i in range(0, T):
        if np.sum(expand_likelihood[i, :]) == 0:
            expand_likelihood[i, :] = 1
            print("Underflow, consider as missing observation")
        else:
            expand_likelihood[i, :] = expand_likelihood[i, :] / np.sum(expand_likelihood[i, :])

    '''RUN THE FORWARD BACKWARD ALGORITHM'''

    l_fMsg, l_bMsg, l_Marg, resp, respPair = fwd_bwd_alg(np.log(expand_pi), np.log(expand_tmat), np.log(expand_likelihood))

    # Compress gammas

    compressed_gammas = np.zeros((N_states, T))

    for i in range(0, N_states):
        compressed_gammas[i] = np.sum(resp[:, s_idx[i]:f_idx[i] + 1], axis=1)

    state_change_prob = np.zeros(np.size(respPair, axis=0))

    for i in range(0, N_states):
        for j in range(0, N_states):
            if i != j:
                state_change_prob += respPair[:, f_idx[i], s_idx[j]]

    if epsilons == False:
        return compressed_gammas

    if epsilons == True:
        return compressed_gammas, state_change_prob
