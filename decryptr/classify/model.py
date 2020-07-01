from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.special import psi
from decryptr.classify.likelihood_calcs import calc_likelihood
from decryptr.classify.baum_welch import baum_welch_alg, decode
from time import time
from alive_progress import alive_bar, config_handler



# Abstract state type model
class AbstractModel(object):
    """ Abstract model class"""
    __meta__ = ABCMeta

    @abstractmethod
    def train(self, obs, obs_start, num_iterations):
        """ Train the model on data

        :param obs: observation array (Must be N_emits x T)
        :param obs_start: observation start base
        :param num_iterations: number of Baum-Welch iterations performed
        """
        raise NotImplementedError

    @abstractmethod
    def decode_write(self, obs, obs_start, out_directory, threshold, write_wig):
        """ Takes observations, runs a forward backward to calculate gammas, writes .bed (and .wig files
            which are optional) for each state (calls and probability track respectively).

            :param obs: observation array (Must be N_emits x T)
            :param gammas: array (n_sub_states x T) of gammas for the sub-states or vector if identical

        """
        raise NotImplementedError

class model_1:
    def __init__(self, pi_prior, tmat_prior, states):

        self.pi_prior = pi_prior
        self.tmat_prior = tmat_prior
        self.states = states

    def train(self, obs, obs_start, num_iterations):

        # Ignore div/0 warnings
        np.seterr(divide='ignore')

        # First iteration starts with pi and tmat from priors
        self.pi = np.zeros(np.size(self.pi_prior))
        for i in range(0, np.size(self.pi_prior)):
            self.pi[i] = psi(self.pi_prior[i]) - psi(np.sum(self.pi_prior))
        self.tmat = np.exp(np.zeros((np.size(self.tmat_prior, axis=0), np.size(self.tmat_prior, axis=1))))

        for i in range(0, np.size(self.tmat_prior, axis=0)):
            for j in range(0, np.size(self.tmat_prior, axis=0)):
                self.tmat[i, j] = np.exp(psi(self.tmat_prior[i, j]) - psi(np.sum(self.tmat_prior[i, :])))

        self.pi /= np.sum(self.pi)
        for i in range(0, np.size(self.tmat_prior, axis=0)):
            self.tmat[i, :] /= np.sum(self.tmat[i, :])

        config_handler.set_global(length=40, spinner='dots_reverse')

        # Begin iterations

        with alive_bar(num_iterations) as bar:
            for iteration in range(0, num_iterations):

                start_time = time()
                likelihood = calc_likelihood(obs, obs_start, self.states)
                self.pi, self.tmat, self.states = baum_welch_alg(self.pi_prior, self.tmat_prior, self.pi, self.tmat, self.states,
                                                        likelihood,
                                                        obs)
                iter_time = time() - start_time

                #print("Iteration {iter} completed in {time_val} seconds".format(iter=iteration, time_val=iter_time))
                #print("===========")

                bar()


    def give_gammas(self, obs, obs_start, state_change=True):
        likelihood = calc_likelihood(obs, obs_start, self.states)

        if state_change == False:
            # Return compressed gammas from a forward-backward pass
            resp = np.transpose(decode(self.pi_prior, self.tmat_prior, self.pi, self.tmat, self.states,
                                                        likelihood,
                                                        obs, state_change))
            return resp

        if state_change == True:
            resp, change_probs = np.transpose(decode(self.pi_prior, self.tmat_prior, self.pi, self.tmat, self.states,
                                                        likelihood,
                                                        obs, state_change))
            return resp, change_probs

    def decode_write(self, obs, obs_start, out_directory, threshold, state_num, write_wig=False):

        # Calculate likelihood
        likelihood = calc_likelihood(obs, obs_start, self.states)

        # Return compressed gammas from a forward-backward pass
        resp = np.transpose(decode(self.pi_prior, self.tmat_prior, self.pi, self.tmat, self.states,
                                                    likelihood,
                                                    obs))

        # Write a .bed file

        # Determine state 1 regions

        T = np.size(likelihood, axis=0)
        start_base = obs_start
        region_list = []
        flag = False
        t = 0

        while 1:
            if resp[t, state_num] >= threshold and flag == False:
                start = t + start_base
                flag = True

            if resp[t, state_num] < threshold and flag == True:
                end = t + start_base
                region_list.append([start, end])
                flag = False

            if resp[t, state_num] < threshold and flag == False:
                pass

            if resp[t, state_num] > threshold and flag == True:
                pass

            t += 1

            if t >= T:
                if flag == True:
                    end = t + start_base
                    region_list.append([start, end])

                break

        # Write bedfile

        with open(out_directory + "calls.bed", 'w') as f:
            for i in range(0, len(region_list)):

                if int(region_list[i][1]) - int(region_list[i][0]) >= 8:
                    f.write(str("chr2") + "\t" + str(region_list[i][0]) + "\t" + str(region_list[i][1]) + "\n")

        # FOR SPECIFIC CRISPR SCREEN NEED TO REPLACE
        if write_wig == True:
            with open(out_directory + "gammas.wig", 'w') as f:
                f.write("variableStep chrom=chr2" + "\n")
                for i in range(0, T-1):
                    f.write(str(start_base + i) + "\t" + str(resp[i, state_num]) + "\n")
