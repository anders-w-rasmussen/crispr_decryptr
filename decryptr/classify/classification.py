import numpy as np
import pickle
import logging
from decryptr.classify.model import model_1 as model
import decryptr.classify.states as states
import decryptr.classify.emissions as emits
import decryptr.classify.gp_utils as gp_utils
import csv
from halo import Halo
import os

def classify_procedure(effect_file, target_file, convolution_matrix, logfilename, prior=None, out_dir=None, alpha=None, rho=None,
                       sn=None, bed_threshold=0.8, flip_sign=None):

    logging.basicConfig(filename=logfilename, level=logging.DEBUG)

    if prior != None:
        prior_file = prior
        print("decryptr: using user defined hyperparameters")
    else:
        curpath = os.path.dirname(os.path.realpath(__file__))
        prior_file = curpath + '/default_prior_file.tsv'
        print("decryptr: using default hyperparameters (two-state model)")

    cmat = pickle.load(open(convolution_matrix, "rb"))

    size_region = np.size(cmat, axis=1)
    modifier = float(size_region) / 1000

    prior_array = np.loadtxt(prior_file, dtype=str, skiprows=1)

    logging.info("using the prior file located at " + str(prior_file))

    prior_names = np.asarray(prior_array[:, 0], dtype=str)
    prior_sigma_mus = np.asarray(prior_array[:, 1], dtype=float)
    prior_taus = np.asarray(prior_array[:, 2], dtype=float)
    prior_Rs = np.asarray(prior_array[:, 3], dtype=int)
    prior_success = np.asarray(prior_array[:, 4], dtype=float)
    prior_failures = np.asarray(prior_array[:, 5], dtype=float)

    if prior == None:
        prior_taus *= modifier
        prior_success *= modifier
        prior_failures *= modifier

        if flip_sign in ['True', 'true']:
            prior_sigma_mus *= -1
            print("decryptr: user specified that enhancers have negative regulatory effect")

    back_idx = np.argwhere(prior_names == 'background')
    if np.size(back_idx) == 0:
        back_idx = np.argwhere(prior_names == 'Background')
    if np.size(back_idx) != 1:
        assert np.size(back_idx) == 0, "error: there must be one state labeled background"
    num_states = np.size(prior_names)

    logging.info("loaded sparse convolution matrix")

    chrom = np.loadtxt(target_file, dtype=str, delimiter='\t').flatten()[0]
    assert chrom.isnumeric() == False, "error: the first entry in the targets file should be the chromosome. Looks like a number..."

    target_locs_original = np.loadtxt(target_file, skiprows=1, dtype=int)
    good_idxs = np.argwhere(np.sum(cmat, axis=1) != 0.0).flatten()
    cmat = cmat[good_idxs, :]
    target_locs_original = target_locs_original[good_idxs]
    sort_idxs = np.argsort(target_locs_original)
    target_locs_original = target_locs_original[sort_idxs]
    cmat = cmat[sort_idxs, :]

    # Figure out what the buffer beyond the targets is

    buffer = int((np.size(cmat, axis=1) - (np.max(target_locs_original) - np.min(target_locs_original))) / 2)
    target_locs = target_locs_original - (np.min(target_locs_original) - buffer)

    # Check the length of the target file and the convolution matrix
    assert np.size(cmat, axis=0) == np.size(target_locs,
                                            axis=0), "error: the number of guides (rows) in the convolution matrix is not in agreement with the number of targets in the target file. Was this the same target file used to make the convolution matrix?"

    T = np.size(cmat, axis=1)
    x = np.linspace(0, T, num=T + 1)

    logging.info("loaded target information")

    # Read effect posterior moments from file
    effect_mus = []
    effect_precisions = []

    column_names = np.loadtxt(effect_file, dtype=str, delimiter='\t')[0, :].flatten()
    effect_names = []

    data_array = np.loadtxt(effect_file, dtype=float, skiprows=1)
    num_effects = int(int(np.size(column_names)) / 2)

    logging.debug("there are" + str(num_effects) + "effects")

    for j in range(0, num_effects):
        effect_names.append(column_names[2 * j][:-10])

        # Check the length of the target file and the effect file
        assert np.size(data_array[:, 2 * j][good_idxs][sort_idxs], axis=0) == np.size(target_locs,
                                                                                      axis=0), "error: the number of guides from the effect file is not in agreement with the number of targets in the target file. Are there extra rows somewhere? Perhaps control guides?"
        effect_mus.append(data_array[:, 2 * j][good_idxs][sort_idxs])
        effect_precisions.append(1 / (data_array[:, 2 * j + 1][good_idxs][sort_idxs] ** 2))

    for j in range(0, num_effects):

        print("decryptr: analyzing the effect labeled " + str(effect_names[j]))
        logging.info("Beginning examination of effect" + str(effect_names[j]))

        convolved_signal = effect_mus[j] - np.median(effect_mus[j])
        prec_vec = effect_precisions[j]

        signal_sigma = np.std(convolved_signal)
        missing_obs = np.ones(T) * np.nan
        missing_obs[target_locs] = convolved_signal

        prec_expand = np.ones(T) * np.nan
        prec_expand[target_locs] = prec_vec

        missing_obs[np.argwhere(np.sum(cmat, axis=0) == 0)] = np.nan
        prec_expand[np.argwhere(np.sum(cmat, axis=0) == 0)] = np.nan

        state_list = []
        obs_list = []
        obs_list.append(missing_obs)

        for s in range(0, num_states):
            emit = [
                emits.normal_dist_known_precision_vector(np.mean(convolved_signal) + prior_sigma_mus[s] * signal_sigma,
                                                         prior_taus[s], prec_expand)]
            pseudo_dur = np.array([prior_success[s], prior_failures[s]])
            state_list.append(states.Negative_Binomial('state_' + str(s), prior_Rs[s], pseudo_dur, emit))

        pi_prior = np.ones(num_states)
        pi_prior[back_idx] = 100
        tmat_prior = np.ones((num_states, num_states)) - np.identity(num_states)
        hsmm_model = model(pi_prior, tmat_prior, state_list)


        hsmm_model.train(obs_list, 0, 5)

        logging.info("Trained the model")

        spinner = Halo(text='decryptr: calculating marginal probabilities of latent states', spinner='dots',
                      color='white', placement='right')
        spinner.start()

        marg_probs, state_change_prob = hsmm_model.give_gammas(obs_list, 0, state_change=True)

        max_distance = 50
        gp_deconvolution = gp_utils.GP_Deconvolution(maximum_distance=max_distance)

        # Set the GP parameters

        if alpha == None:
            alpha_opt = (np.mean(1 / prec_vec) + np.var(convolved_signal)) ** (1 / 2)
        else:
            alpha_opt = alpha

        if rho == None:
            rho_opt = np.mean(np.diff(target_locs))  # 6
        else:
            rho_opt = rho

        if sn == None:
            sn_opt = (np.mean(1 / prec_vec) + np.var(convolved_signal)) ** (1 / 2)
        else:
            sn_opt = sn

        print("signal_dev: " + str(alpha_opt ** 2) + "     length_scale: " + str(
            rho_opt) + "     process_noise (std. dev): " + str(sn_opt ** 2))

        mean_f, var_f, x_truncated = gp_deconvolution.pred([cmat], [convolved_signal], [x],
                                                           [target_locs], alpha_opt,
                                                           rho_opt, sn_opt,
                                                           [1 / prec_vec], full_pred=False, K_mod=None)

        x_vals = np.asarray(x[np.argwhere(x_truncated[0] == True)].flatten(), dtype=int)

        deconv_mean = np.zeros(T)
        deconv_mean[x_vals] += mean_f

        deconv_var = np.ones(T) * sn_opt
        deconv_var[x_vals] = var_f

        signal_sigma = np.std(deconv_mean[x_vals])

        logging.info("Finished deconvolving effect for iteration " + str(i))

        # HsMM

        state_list = []
        obs_list = []
        obs_list.append(deconv_mean)

        for s in range(0, num_states):
            emit = [emits.normal_dist_known_precision_vector(prior_sigma_mus[s] * signal_sigma,
                                                             prior_taus[s], 1 / deconv_var)]
            pseudo_dur = np.array([prior_success[s], prior_failures[s]])
            state_list.append(states.Negative_Binomial('state_' + str(s), prior_Rs[s], pseudo_dur, emit))

        pi_prior = np.ones(num_states)
        pi_prior[back_idx] = 100
        tmat_prior = np.ones((num_states, num_states)) - np.identity(num_states)
        hsmm_model = model(pi_prior, tmat_prior, state_list)

        print("decryptr: training Hidden semi-Markov Model on deconvolved effect")

        hsmm_model.train(obs_list, 0, 3)

        logging.info("Trained HsMM of iteration " + str(i))

        spinner = Halo(text='decryptr: calculating marginal probabilities of latent states', spinner='dots',
                       color='white', placement='right')
        spinner.start()

        marg_probs, state_change_prob = hsmm_model.give_gammas(obs_list, 0, state_change=True)

        spinner.stop()

        if out_dir != None:
            logging.debug("out_dir selected = " + str(out_dir))
            out_dir_name = out_dir + '/' + str(effect_names[j]) + '/'
            os.system('mkdir ' + out_dir_name + '/' + str(effect_names[j]))

        else:
            logging.debug("no out directory was specified. Saving to current directory")
            os.system('mkdir ' + str(effect_names[j]))
            out_dir_name = str(effect_names[j]) + '/'

        os.system('mkdir ' + out_dir_name + 'signals')

        with open(out_dir_name + 'signals/' + str(effect_names[j]) + "_deconvolved_mu.wig", 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')

            writer.writerow(['variableStep chrom=' + chrom])
            outbases = x + np.min(target_locs_original) - buffer
            for b in range(0, np.size(outbases, axis=0) - 1):
                writer.writerow([str(int(outbases[b])), str(deconv_mean[b])])

        with open(out_dir_name + 'signals/' + str(effect_names[j]) + "_deconvolved_dev.wig", 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')

            writer.writerow(['variableStep chrom=' + chrom])
            outbases = x + np.min(target_locs_original) - buffer
            for b in range(0, np.size(outbases, axis=0) - 1):
                writer.writerow([str(int(outbases[b])), str(deconv_var[b] ** (1 / 2))])

        # write the wig file

        os.system('mkdir ' + out_dir_name + 'state_probabilities')
        os.system('mkdir ' + out_dir_name + 'peak_calls')

        names = np.unique(prior_names)
        for u in range(0, np.size(names)):

            idxs = np.argwhere(prior_names == names[u])
            out_marg = np.sum(marg_probs[idxs, :], axis=0).flatten()

            with open(out_dir_name + 'state_probabilities/' + str(names[u]) + '_' + str(
                    effect_names[j]) + "_margprobs.wig", 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')

                writer.writerow(['variableStep chrom=' + chrom])
                outbases = x + np.min(target_locs_original) - buffer
                for b in range(0, np.size(outbases, axis=0) - 1):
                    writer.writerow([str(int(outbases[b])), str(out_marg[b])])


            idxs = np.argwhere(out_marg > bed_threshold).flatten()

            if np.size(idxs) == 0:
                continue

            num_idxs = np.size(idxs)
            diff_mat = idxs[1:num_idxs] - idxs[0:num_idxs - 1]
            idx_of_idx = np.argwhere(diff_mat != 1)
            starting_idxs = idxs[idx_of_idx + 1]
            ending_idxs = idxs[idx_of_idx]
            start_base = np.min(outbases)

            starting_idxs = np.concatenate(([[idxs[0]]], starting_idxs))
            ending_idxs = np.concatenate((ending_idxs, [[idxs[num_idxs - 1]]]))

            with open(out_dir_name + 'peak_calls/' + str(names[u]) + '_' + str(effect_names[j]) + "_peaks.bed", 'w',
                      newline='') as f:
                for i in range(0, np.size(starting_idxs, axis=0)):
                    f.write(str(chrom) + "\t" + str(int(starting_idxs[i][0] + start_base)) + "\t" + str(int(
                        ending_idxs[i][0] + start_base)) + "\n")

        print("decryptr: wrote output files for effect " + str(effect_names[j]))
