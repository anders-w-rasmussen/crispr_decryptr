from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import psi


# Abstract state type model
class AbstractState(object):
    """ Abstract state class"""
    __meta__ = ABCMeta

    @abstractmethod
    def likelihood(self, obs):
        """ From observation vectors in a list, returns vector of state likelihoods. Each element in the list
            can be a vector of array that is n x T

        :param obs: observation vectors in a list:
        :return log_likelihood: log likelihood vector (or array if sub-states do not have identical emission dists)

        """
        raise NotImplementedError

    @abstractmethod
    def update_parameters(self, obs, gammas, epsilons):
        """ Updates state params from observation vectors, block of gammas, and block of epsilons (For that state)

            :param obs: observation vectors in a list:
            :param gammas: array (n_sub_states x T) of gammas for the sub-states
            :param epsilons: array (n_sub_states x n_sub_states x T) of epsilons for sub-states

        """
        raise NotImplementedError

class Negative_Binomial:
    def __init__(self, name, r, prior_pseudocounts, emissions, explicit_location=None, possible=None):

        # Name of state
        self.name = name

        # Number of expanded states (is r for negative binomial duration)
        self.n = r

        # NB parameter (p) successes and failures
        self.s_pseudo = prior_pseudocounts[0]
        self.f_pseudo = prior_pseudocounts[1]

        # Emission list for this state
        self.emits = emissions

        # List of locations on genome we know to be the state
        # (Note that this is handled in the model class when we calculate the likelihood)
        self.locations = explicit_location

        # List of locations on genome that are possible for this state (if not included are allowed anywhere)
        self.possible = possible

        # Initial block matrix of transitions from success and failure pseudocounts
        self.block_tmat = np.zeros((self.n, self.n))
        self.p = psi(self.s_pseudo) - psi(self.s_pseudo + self.f_pseudo)
        self.f = psi(self.f_pseudo) - psi(self.s_pseudo + self.f_pseudo)

        for i in range(0, self.n - 1):
            self.block_tmat[i, i] = np.exp(self.f)
            self.block_tmat[i, i + 1] = np.exp(self.p)
        self.block_tmat[self.n - 1, self.n - 1] = np.exp(self.f)

        # Note 'leaving prob' which is probability of leaving the the final substate and entire macro state
        self.leaving_prob = np.exp(self.p)

    def likelihood(self, obs):

        T = np.size(obs[0], axis=0)
        N_emits = len(self.emits)

        # Temp array to hold emission probabilities of individual emission types
        log_prob_array = np.zeros((N_emits, T))

        # Loop through emissions and calculate the log likelihood
        for i in range(0, N_emits):
            test = self.emits[i].likelihood(obs[i])
            log_prob_array[i, :] = self.emits[i].likelihood(obs[i])

        # Sum the log_prob_array to find joint probability (note all emits independent)
        log_likelihood = np.sum(log_prob_array, axis=0)

        return log_likelihood

    def update_parameters(self, obs, gammas, epsilons):

        T = np.size(obs[0], axis=0)
        N_emits = len(self.emits)

        # Update transition matrix
        s_ss = 0
        f_ss = 0

        for i in range(1, self.n):
            s_ss += np.sum(epsilons[i - 1, i, :])
            f_ss += np.sum(epsilons[i - 1, i - 1, :])

        s_ss += np.sum(epsilons[self.n - 1, :, :]) - np.sum(epsilons[self.n - 1, self.n - 1, :])
        f_ss += np.sum(epsilons[self.n - 1, self.n - 1, :])

        self.p = psi(self.s_pseudo + s_ss) - psi(self.s_pseudo + s_ss + self.f_pseudo + f_ss)
        self.f = psi(self.f_pseudo + f_ss) - psi(self.s_pseudo + s_ss + self.f_pseudo + f_ss)

        for i in range(0, self.n - 1):
            self.block_tmat[i, i] = np.exp(self.f)
            self.block_tmat[i, i + 1] = np.exp(self.p)
        self.block_tmat[self.n - 1, self.n - 1] = np.exp(self.f)

        # Note 'leaving prob' which is probability of leaving the the final substate and entire macro state
        self.leaving_prob = np.exp(self.p)

        # Update emissions
        for i in range(0, N_emits):
            self.emits[i].update_state_parameters(obs[i], np.sum(gammas, axis=1))