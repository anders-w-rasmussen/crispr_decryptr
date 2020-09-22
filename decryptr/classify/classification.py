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
from multiprocessing import Process, Manager

def classify_procedure(effect_file, target_file, convolution_matrix, logfilename, prior=None, out_dir=None, alpha=None, rho=None,
                       sn=None, bed_threshold=0.8, flip_sign=None, norm=None, hampel_filter=None, parallelize=False):

    logging.basicConfig(filename=logfilename, level=logging.DEBUG)

    if prior != None:
        prior_file = prior
        print("decryptr: using user defined hyperparameters")
    else:
        curpath = os.path.dirname(os.path.realpath(__file__))
        prior_file = curpath + '/default_prior_file.tsv'
        print("decryptr: using default hyperparameters (two-state model)")

    if parallelize in ['True', 'true']:
        parallelize = True
        print("decryptr: using parallelized implementation")
        print("please note this will discard distant off target binding events")
    else:
        parallelize = False
	

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

        convolved_signal = effect_mus[j]
	
        if norm in ['True', 'true']:
            convolved_signal -= np.mean(convolved_signal)
            convolved_signal /= np.std(convolved_signal)

        if hampel_filter in ['True', 'true']:
            for i in range(3, np.size(convolved_signal) - 3):
                win_med = np.median(convolved_signal[i - 3: i + 4])
                MAD = np.median(np.absolute(convolved_signal[i - 3: i + 4] - win_med))
                if np.absolute(convolved_signal[i] - win_med)  > 3 * MAD:
                    convolved_signal[i] = win_med

        prec_vec = effect_precisions[j]

        signal_sigma = np.std(convolved_signal)
        missing_obs = np.ones(T) * np.nan


        # Set the GP parameters

        if alpha == None:
            alpha_opt = (np.mean(1 / prec_vec) + np.var(convolved_signal)) ** (1 / 2)
        else:
            alpha_opt = alpha

        diff_arr = np.diff(target_locs)
        if rho == None:
            rho_opt = np.mean(diff_arr[diff_arr < 300])
        else:
            rho_opt = rho

        if sn == None:
            sn_opt = (np.mean(1 / prec_vec) + np.var(convolved_signal)) ** (1 / 2)
        else:
            sn_opt = sn

        print("decryptr: Gaussian process parameters")
        print("signal_dev: " + str(round(alpha_opt ** 2, 2)) + "   length_scale: " + str(
            round(rho_opt, 2)) + "   process_noise (std. dev): " + str(round(sn_opt ** 2, 2)))

        ###########

        if parallelize == False:

            mean_f, var_f, x_truncated = gp_deconvolution.pred([cmat], [convolved_signal], [x],
                                                               [target_locs], alpha_opt,
                                                               rho_opt, sn_opt,
                                                               [1 / prec_vec], full_pred=False)

            x_vals = np.asarray(x[np.argwhere(x_truncated[0] == True)].flatten(), dtype=int)

            deconv_mean = np.zeros(T)
            deconv_mean[x_vals] += mean_f

            deconv_var = np.ones(T) #* np.mean(var_f)
            deconv_var[x_vals] = var_f

            signal_sigma = np.std(deconv_mean[x_vals])

            logging.info("Finished deconvolving effect")

            # HsMM
            state_list = []
            obs_list = []
            obs_list.append(deconv_mean)

            for s in range(0, num_states):
                emit = [emits.normal_dist_known_precision_vector(np.median(deconv_mean) + prior_sigma_mus[s] * signal_sigma,
                                                                 prior_taus[s], 1 / deconv_var)]
                pseudo_dur = np.array([prior_success[s], prior_failures[s]])
                state_list.append(states.Negative_Binomial('state_' + str(s), prior_Rs[s], pseudo_dur, emit))

            pi_prior = np.ones(num_states)
            pi_prior[back_idx] = 10
            tmat_prior = np.ones((num_states, num_states)) - np.identity(num_states)
            hsmm_model = model(pi_prior, tmat_prior, state_list)

            print("decryptr: training Hidden semi-Markov Model on deconvolved effect")

            hsmm_model.train(obs_list, 0, 3)

            spinner = Halo(text='decryptr: calculating marginal probabilities of latent states', spinner='dots',
                           color='white', placement='right')
            spinner.start()

            marg_probs, state_change = hsmm_model.give_gammas(obs_list, 0, state_change=True)

            spinner.stop()

        if parallelize == True:

            cmat_list = []
            convolved_signal_list = []
            x_list = []
            target_locs_list = []
            precision_vec_list = []

            for gap in [5000, 2000, 1000, 500, 100, 50, 20]:
                terminal_idxs = np.append(target_locs[np.argwhere(diff_arr.flatten() > gap)], np.asarray([T], dtype = int))
                if np.size(terminal_idxs) >= 20:
                    break
		
            print(np.size(terminal_idxs))

            for i in range(0, np.size(terminal_idxs)):
                if i == 0:
                    last_idx = 0
                    last_guide_idx = 0

                terminal_idx = terminal_idxs[i]
                terminal_guide_idx = np.argmax(np.argwhere((target_locs <= terminal_idx).flatten())) + 1

                print(terminal_idx)
                print(terminal_guide_idx)
		
                if terminal_idx - last_idx <= 1000:
                    continue
                if terminal_guide_idx - last_guide_idx < 20:
                    continue

                cmat_list.append(cmat[last_guide_idx:terminal_guide_idx, last_idx:terminal_idx])
                convolved_signal_list.append(convolved_signal[last_guide_idx:terminal_guide_idx])
                x_list.append(x[last_idx:terminal_idx])
                target_locs_list.append(target_locs[last_guide_idx:terminal_guide_idx])
                precision_vec_list.append(prec_vec[last_guide_idx:terminal_guide_idx])


                last_idx = terminal_idx + 1
                last_guide_idx = terminal_guide_idx + 1
            
            print("cmat_list length")
            print(len(cmat_list))

            if __name__ != "__main__":
                with Manager() as manager:
                    master_list = manager.list()
                    processes = []
                    for k in range(len(cmat_list)):
                        p = Process(target=run_slice, args=(cmat_list[k],
                                                               convolved_signal_list[k], x_list[k],
                                                               target_locs_list[k], alpha_opt, rho_opt, sn_opt, precision_vec_list[k]))
                        p.start()
                        processes.append(p)
                        if k > 0:
                            break
			
                    for p in processes:
                        p.join()

                    saved_master_list = [x for x in master_list]


            marg_probs = np.concatenate(saved_master_list)

        ###############


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

def run_slice(cmat, convolved_signal, x, target_locs, alpha_opt, rho_opt, sn_opt, prec_vec):

    max_distance = 50
    gp_deconvolution = gp_utils.GP_Deconvolution(maximum_distance=max_distance)
    print(np.shape(cmat))
    print("convolved_sig")
    print(convolved_signal)
    print("x")
    print(x)
    print("targ_locs")
    print(target_locs)
    print("precision_vec")
    print(prec_vec)

	
    mean_f, var_f, x_truncated = gp_deconvolution.pred([cmat], [convolved_signal], [np.asarray(x, dtype=int)],
                                                       [target_locs], alpha_opt,
                                                       rho_opt, sn_opt,
                                                       [1 / prec_vec], full_pred=True)

    x_vals = np.asarray(x[np.argwhere(x_truncated[0] == True)].flatten(), dtype=int)

    deconv_mean = np.zeros(T)
    deconv_mean[x_vals] += mean_f

    deconv_var = np.ones(T)  # * np.mean(var_f)
    deconv_var[x_vals] = var_f

    signal_sigma = np.std(deconv_mean[x_vals])

    # HsMM
    state_list = []
    obs_list = []
    obs_list.append(deconv_mean)

    for s in range(0, num_states):
        emit = [emits.normal_dist_known_precision_vector(np.median(deconv_mean) + prior_sigma_mus[s] * signal_sigma,
                                                         prior_taus[s], 1 / deconv_var)]
        pseudo_dur = np.array([prior_success[s], prior_failures[s]])
        state_list.append(states.Negative_Binomial('state_' + str(s), prior_Rs[s], pseudo_dur, emit))

    pi_prior = np.ones(num_states)
    pi_prior[back_idx] = 10
    tmat_prior = np.ones((num_states, num_states)) - np.identity(num_states)
    hsmm_model = model(pi_prior, tmat_prior, state_list)

    hsmm_model.train(obs_list, 0, 3)

    marg_probs, state_change = hsmm_model.give_gammas(obs_list, 0, state_change=True)

    return marg_probs       # , state_change
