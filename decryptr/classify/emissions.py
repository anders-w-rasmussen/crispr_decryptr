from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import psi
from scipy.stats import norm
from scipy.stats import geom
from scipy.stats import invgamma
import copy as cp


# Abstract state type model
class AbstractEmission(object):
    """ Abstract emission class"""
    __meta__ = ABCMeta

    @abstractmethod
    def likelihood(self, obs):
        """ From observation vectors, returns vector of state likelihoods

        :param obs: observation vector:
        :return log_likelihood: log probability vector (or array if sub-states do not have identical emission dists)

        """
        raise NotImplementedError

    @abstractmethod
    def update_state_parameters(self, obs, gammas):
        """ Updates emit params from observation vector, block or vector of gammas

            :param obs: observation vector:
            :param gammas: array (n_sub_states x T) of gammas for the sub-states or vector if identical

        """
        raise NotImplementedError


class binary:
    ''' Bernoulli Emissions for 1 x T vector of ones and zeros.
        Good for binary flags, histone modifications '''

    def __init__(self, prior_pseudocounts, emit_name='bernoulli emit'):

        # Emission name
        self.emit_name = emit_name

        # Pseudocounts for emitting a 0 (failure) or 1 (success)
        self.pseudo0 = prior_pseudocounts[0]
        self.pseudo1 = prior_pseudocounts[1]

        # Emission probabilities from prior pseudocounts
        self.p0 = psi(self.pseudo0) - psi(self.pseudo0 + self.pseudo1)
        self.p1 = psi(self.pseudo1) - psi(self.pseudo0 + self.pseudo1)

    def likelihood(self, obs):

        T = np.size(obs)
        likelihood = np.zeros(T)
        p_vec = np.exp([self.p0, self.p1])

        # Calculate likelihood vector
        for i in range(0, T):
            likelihood[i] = p_vec[int(obs[i])]

        return likelihood

    def update_state_parameters(self, obs, gammas):

        T = np.size(obs)

        # Calculate expected counts
        p1_ss = np.dot(gammas, obs)
        p0_ss = np.dot(gammas, np.ones(T) - obs)

        # Update posterior
        self.p1 = psi(self.pseudo1 + p1_ss) - psi(self.pseudo0 + p0_ss + self.pseudo1 + p1_ss)
        self.p0 = psi(self.pseudo0 + p0_ss) - psi(self.pseudo0 + p0_ss + self.pseudo1 + p1_ss)


class normal_dist_emit:
    ''' Normally distributed observation with mean and variance inferred. '''

    def __init__(self, prior_parameters, emit_name='norm_effect'):

        # Emission name
        self.emit_name = emit_name

        # Prior parameters:
        # mu_0, sigma_0 (for normal prior of mean),
        # alpha, and beta (for gamma prior of precision)

        self.mu_0 = prior_parameters[0]
        self.lam_0 = prior_parameters[1]
        self.alpha_0 = prior_parameters[2]
        self.beta_0 = prior_parameters[3]

        self.mu = self.mu_0
        self.precision = (self.alpha_0 / self.beta_0) * self.lam_0

    def likelihood(self, obs):

        T = np.size(obs)
        likelihood = np.zeros(T)

        # Calculate likelihood vector
        # for i in range(0, T):
        #     if np.isnan(obs[i]) == False:
        #         likelihood[i] = norm.pdf(obs[i], self.mu, np.sqrt(float(self.precision)**(-1.0)))
        #     else:
        #         likelihood[i] = 1

        obs_no_nan = np.nan_to_num(obs)
        likelihood = norm.pdf(obs_no_nan, self.mu, np.sqrt(float(self.precision)**(-1.0)))
        likelihood[np.argwhere(np.isnan(obs))] = 1

        # likelihood[np.argwhere(np.isnan(obs))] = 1
        # likelihood[np.argwhere(np.isfinite(obs))] = norm.pdf(np.argwhere(np.isfinite(obs)), self.mu, np.sqrt(float(self.precision)**(-1.0)))
        # print("Calculated Likelihood")

        return likelihood

    def update_state_parameters(self, obs, gammas):

        # NEED TO REWRITE THIS TO HAVE A CONVERGENCE CRITERIA

        update_steps = 10
        T = np.size(obs)

        # Calculate sufficient statistics for normal distribution

        # COULD MAKE THIS FASTER (SLOW BECAUSE AM ACCOUNTING FOR NANS IN A LOOP)
        N = 0
        x_sum = 0
        x_sq_sum = 0

        for t in range(0, T):
            if np.isnan(obs[t]) == False:
                N += gammas[t]
                x_sum += obs[t] * gammas[t]
                x_sq_sum += obs[t]**2 * gammas[t]

        x_bar = x_sum / N

        # Calculate posterior hyperparameters

        n_iter = 20

        for i in range(0, n_iter):

            mu_N = (self.lam_0 * self.mu_0 + N * x_bar) / (self.lam_0 + N)
            lam_N = (self.lam_0 + N * self.precision)

            self.mu = mu_N

            alpha_N = self.alpha_0 + N/2
            beta_N = self.beta_0 + .5 * x_sq_sum + ((self.lam_0 * N)/(self.lam_0 + N)) * ((x_bar - self.mu)**2) / 2

            if alpha_N <= 1:
                raise Exception(
                    'Mean of Inverse Gamma posterior undefined with alpha < 1. The value of alpha was: {}'.format(
                        alpha_N))

            self.precision = alpha_N / beta_N

        # Update mu and tau as the expectation of the posterior


        # DONT NEED VARIATIONAL BAYES UPDATE RULES JUST USE NORMAL INVERSE GAMMA POSTERIOR HYPERPARAMETERS
        # Update posterior (see wikipedia page on variational bayes for math behind this)

        # lam_N = 1
        # mu_N = (self.lam_0 * self.mu_0 + N * x_bar) / (self.lam_0 + N)
        # alpha_N = self.alpha_0 + (N + 1)/2
        #
        # # Beta and lambda need to be updated in an iterative procedure
        # for i in range(0, update_steps):
        #     beta_N = self.beta_0 + (0.5) * ((self.lam_0 + N) * (lam_N ** (-1) + mu_N ** 2) - 2*(
        #         self.lam_0 * self.mu_0 + x_sum) * mu_N + x_sq_sum + self.lam_0 * self.mu_0 ** 2)
        #     lam_N = (self.lam_0 + N) * (alpha_N / beta_N)
        #
        # # Update the parameters
        # self.mu = mu_N
        # self.precision = (alpha_N / beta_N)


class normal_dist_known_precision_vector:
    ''' Normally distributed observation. Mean inferred for each base, precision known for each observation. '''

    def __init__(self, mu_0, tau_0, known_precisions, emit_name='norm_effect_fixed_precisions'):

        # Emission name
        self.emit_name = emit_name
        self.precisions = known_precisions

        # Prior Hyperparameters
        self.mu_0 = mu_0
        self.tau_0 = tau_0
        self.mu = cp.copy(self.mu_0)
        self.tau = cp.copy(self.tau_0)

    def likelihood(self, obs):

        T = np.size(obs)
        likelihood = np.zeros(T)

        # Calculate likelihood vector
        for i in range(0, T):
            if np.isnan(obs[i]) == False:
                likelihood[i] = norm.pdf(obs[i], self.mu, np.sqrt(float(self.precisions[i])**(-1.0)))
            else:
                likelihood[i] = 1

        return likelihood

    def update_state_parameters(self, obs, gammas):

        T = np.size(obs)

        ss_tx = 0
        ss_nt = 0

        for t in range(0, T):
            if np.isnan(obs[t]) == False:
                ss_tx += obs[t] * gammas[t] * self.precisions[t]
                ss_nt += gammas[t] * self.precisions[t]

        self.mu = (self.tau_0 * self.mu_0 + ss_tx) / (self.tau_0 + ss_nt)
        self.tau = self.tau_0 + ss_nt