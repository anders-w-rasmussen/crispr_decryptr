import os
import re
from functools import reduce
import logging
import numpy
import pandas as pd
from multiprocessing import Pool, cpu_count
from decryptr.mcmc_glm import istarmap
from tqdm import tqdm
from cmdstanpy import cmdstan_path, CmdStanModel, install_cmdstan
from decryptr.mcmc_glm.filter_guides import filter_guides
import shutil


def analyze(count_filename, design_matrix_filename, replicate_information_filename,
            sample_file_prefix, n_chains, n_samples, batch_size,
            n_batches, output_filename, logfilename, outfile_devs, species, target_file, spacers_file, lambda_filename=None):

            
    install_cmdstan()

    if species != None:
        filter_guides(count_filename, target_file, spacers_file, 0.2, species)
        count_filename = count_filename.split(".")[0] + '_filtered.tsv'

    logging.basicConfig(filename='decryptr_infer.log', level=logging.DEBUG)


    data_dict = generate_stan_input_collapsed(count_filename, design_matrix_filename,
                                              replicate_information_filename, batch_size,
                                              lambda_filename)

    logging.info("stan input generated")

    moments = run_stan(data_dict, output_prefix=sample_file_prefix, n_chains=n_chains, n_samples=n_samples,
                       n_batches=n_batches)

    logging.info("moments calculated")

    save_output(moments, count_filename, design_matrix_filename, output_filename, outfile_devs)

    logging.info("output saved")


def read_stan_csv(filename, variables=['theta', 'mu', 's2']):

    data = pd.read_csv(filename, sep=',', index_col=False, comment='#', header=0, na_filter=False, dtype=numpy.float64,
                       usecols=lambda x: x.startswith(tuple(variables)))

    N_samples = data.shape[0]

    # initialize a dictionary for storing the samples
    samples = {}
    for variable in variables:
        # if the variable does not exist, let's skip it
        if len([foo for foo in list(data.columns) if foo.startswith(variable)]) == 0:
            continue

        if len([foo for foo in list(data.columns) if foo.startswith(variable + '.')]) == 0:
            dimensions = [N_samples, 1]
        # non scalar parameter
        else:
            dimensions = numpy.hstack((N_samples, numpy.array(
                [list(map(int, foo.split('.')[1:])) for foo in list(data.columns) if foo.startswith(variable)]).max(0)))

        samples[variable] = numpy.zeros(dimensions)

    for col in data.columns:
        # get the variable name
        variable = col.split('.')[0]
        # scalar parameter
        if len(col.split('.')) == 1:
            samples[variable][:, 0] = data[col].values
        # non scalar parameter
        else:
            # get the indices
            indices = list(map(lambda x: int(x) - 1, col.split('.')[1:]))
            # collect mean and standard deviation
            samples[variable][tuple([slice(0, N_samples)]) + tuple(indices)] = data[col].values

    return samples


def calculate_moments(sample_files, variables=['beta', 's']):
    seen_headers = []
    data = {}
    for variable in variables:
        data[variable] = []

    for sample_file in sample_files:
        with open(sample_file, 'r') as f:
            header = None
            for line in f:
                if not line.startswith('#') and header is None:
                    header = line.split(',')
                    seen_headers.append(header)

                    # check that the headers match
                    if not numpy.all([header == seen_header for seen_header in seen_headers]):
                        logging.critical("headers of the sample files don't match")
                        raise ValueError('Headers of the sample files do not match!')

                    if len(seen_headers) == 1:
                        indices = {}
                        for variable in variables:
                            indices[variable] = \
                            numpy.where([re.search('^%s(\.\d+)*$' % (variable), column) for column in header])[0]

                elif not line.startswith('#'):
                    for variable in variables:
                        data[variable].append(numpy.array(line.split(','))[indices[variable]].astype('float'))

    array_indices = {}
    for variable in variables:
        array_indices[variable] = numpy.array(
            [list(map(int, column.split('.')[1:])) for column in numpy.array(header)[indices[variable]]])

    output = {}
    for variable in variables:
        output[variable] = {}
        for moment in ['mean', 'std']:
            output[variable][moment] = numpy.zeros(numpy.max(array_indices[variable], axis=0))

    for variable in variables:
        output[variable]['mean'][tuple(array_indices[variable].T - 1)] = numpy.array(data[variable]).mean(0)
        output[variable]['std'][tuple(array_indices[variable].T - 1)] = numpy.array(data[variable]).std(0)

    return output


# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)


# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
def finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
    if count < 2:
        return float('nan')
    else:
        return (mean, variance, sampleVariance)


def calculate_moments_online(sample_files, variables=['beta', 's']):
    seen_headers = []
    data = {}
    for variable in variables:
        data[variable] = (0, 0, 0)

    for sample_file in sample_files:
        with open(sample_file, 'r') as f:
            header = None
            for line in f:
                if not line.startswith('#') and header is None:
                    header = line.split(',')
                    seen_headers.append(header)

                    # check that the headers math
                    if not numpy.all([header == seen_header for seen_header in seen_headers]):
                        logging.critical('headers of the sample files do not match')
                        raise ValueError('Headers of the sample files do not match!')

                    if len(seen_headers) == 1:
                        indices = {}
                        for variable in variables:
                            indices[variable] = \
                            numpy.where([re.search('^%s(\.\d+)*$' % (variable), column) for column in header])[0]

                elif not line.startswith('#'):
                    tmp = numpy.array(line.split(','))
                    for variable in variables:
                        data[variable] = update(data[variable], tmp[indices[variable]].astype('float'))

    array_indices = {}
    for variable in variables:
        array_indices[variable] = numpy.array(
            [list(map(int, column.split('.')[1:])) for column in numpy.array(header)[indices[variable]]])

    output = {}
    for variable in variables:
        output[variable] = {}
        for moment in ['mean', 'std']:
            output[variable][moment] = numpy.zeros(numpy.max(array_indices[variable], axis=0))

    for variable in variables:
        tmp = finalize(data[variable])
        output[variable]['mean'][tuple(array_indices[variable].T - 1)] = tmp[0]
        output[variable]['std'][tuple(array_indices[variable].T - 1)] = numpy.sqrt(tmp[1])

    return output


def to_rdump(data, filename):
    with open(filename, 'w') as f:
        for key in data:
            tmp = numpy.asarray(data[key])
            if len(tmp.shape) == 0:
                f.write('%s <- %s\n' % (key, str(tmp)))
            elif len(tmp.shape) == 1:
                f.write('%s <-\n c(%s)\n' % (key,
                                             ','.join(map(str, tmp))))
            else:
                f.write('%s <-\n structure(c(%s), .Dim=c(%s))\n' % (key,
                                                                    ','.join(map(str, tmp.flatten('F'))),
                                                                    ','.join(map(str, tmp.shape))))


def generate_stan_input_dictionary(data, N_replicates, replicate_mapping,
                                   design_matrix, lambda_prior=None):
    # convert the design matrix into a ragged array
    num_params = design_matrix.shape[1]
    num_params_per_sample = [len(list(numpy.where(row)[0] + 1)) for row in design_matrix]
    param_indices = reduce(lambda x, y: x + y,
                           [list(numpy.where(row)[0] + 1) for row in design_matrix], [])

    data_dict = {'N_guides': data.shape[0], 'N_samples': data.shape[1],
                 'N_replicates': N_replicates, 'replicate_mapping': replicate_mapping,
                 'counts': data.T.astype(int), 'num_params': design_matrix.shape[1],
                 'num_params_per_sample': num_params_per_sample,
                 'param_indices': numpy.asarray(param_indices).astype(int)}

    if lambda_prior is not None:
        data_dict['use_lambda'] = 1
        data_dict['lambda_mu'] = list(map(lambda x: float(x.split(',')[0]), lambda_prior[numpy.where(design_matrix)]))
        data_dict['lambda_std'] = list(map(lambda x: float(x.split(',')[1]), lambda_prior[numpy.where(design_matrix)]))
    else:
        data_dict['use_lambda'] = 0
        data_dict['lambda_mu'] = sum(num_params_per_sample) * [0]
        data_dict['lambda_std'] = sum(num_params_per_sample) * [0]

    return data_dict


def generate_stan_input(data_filename, design_matrix_filename,
                        replicate_information_filename):
    # read the count data file
    data_pd = pd.read_table(data_filename, sep='\t', header=0)
    logging.info('read data file: %s' % (data_filename))

    # read the design matrix file
    design_matrix_pd = pd.read_table(design_matrix_filename, sep='\t', header=0, index_col=0)
    logging.info('read design matrix file: %s' % (design_matrix_filename))

    # read the replicate information file
    replicate_information_pd = pd.read_table(replicate_information_filename,
                                             sep='\t', header=0, index_col=0)
    logging.info('read replicate information file: %s' % (replicate_information_filename))

    # extract the names of the samples
    sample_names = list(design_matrix_pd.index)
    # extract the design matrix
    design_matrix = design_matrix_pd.values
    # extract the guide RNA counts of each sample
    data_counts = data_pd[sample_names].values
    # extract the replicate information of each sample
    replicate_mapping = list(replicate_information_pd.loc[sample_names]['replicate'])
    # number of replicates
    N_replicates = max(replicate_mapping)

    data_dict = generate_stan_input_dictionary(data_counts, N_replicates,
                                               replicate_mapping, design_matrix)
    data_dict['log_size_factors'] = numpy.zeros(data_counts.shape[0] - 1)
    logging.info('generated stan input data dictionary')

    # write the stan input data file
    # to_rdump(data_dict,output_filename)

    return data_dict


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def generate_stan_input_collapsed(data_filename, design_matrix_filename,
                                  replicate_information_filename, batch_size=None,
                                  lambda_filename=None):

    # read the count data file
    data_pd = pd.read_table(data_filename, sep='\t', header=0)
    logging.info('read data file: %s' % (data_filename))

    # read the design matrix file
    design_matrix_pd = pd.read_table(design_matrix_filename, sep='\t', header=0, index_col=0)
    logging.info('read design matrix file: %s' % (design_matrix_filename))

    # read the replicate information file
    replicate_information_pd = pd.read_table(replicate_information_filename,
                                             sep='\t', header=0, index_col=0)
    logging.info('read replicate information file: %s' % (replicate_information_filename))

    # read the lambda prior information file
    if lambda_filename is not None:
        lambda_prior_information_pd = pd.read_table(lambda_filename,
                                                    sep='\t', header=0, index_col=0)
        lambda_prior = lambda_prior_information_pd.values
        logging.info('read lambda prior information file: %s' % (lambda_filename))
    else:
        lambda_prior = None

    # extract the names of the samples
    sample_names = list(design_matrix_pd.index)


    # print(sample_names)
    # print(list(data_pd.columns)[1:])
    # print(x in sample_names for x in list(data_pd.columns)[1:])
    #
    #
    # if not(all(x in sample_names for x in list(data_pd.columns)[1:])):
    #     logging.critical('Design matrix has a sample name not found in screen data file')
    #     raise ValueError('Design matrix has a sample name not found in screen data file')
    #
    #
    # if not(all(x in sample_names for x in list(replicate_information_pd.index))):
    #     logging.critical('Design matrix has a sample name not found in replicate information file')
    #     raise ValueError('Design matrix has a sample name not found in replicate information file')
    #

    # extract the design matrix
    design_matrix = design_matrix_pd.values
    # extract the guide RNA counts of each sample
    data_counts = data_pd[sample_names].values
    # extract the replicate information of each sample
    replicate_mapping = list(replicate_information_pd.loc[sample_names]['replicate'])
    # number of replicates
    N_replicates = max(replicate_mapping)

    N_guides = data_counts.shape[0]

    indices = list(range(0, N_guides - 1))

    baseline_index = N_guides - 1

    if batch_size is None:
        batch_size = len(indices)

    data_dicts = []
    for guide_indices in batch(indices, batch_size):
        collapsed_indices = numpy.sort(numpy.array(list(set(indices) - set(guide_indices)), dtype='int'))

        data_counts_batch = numpy.vstack((data_counts[guide_indices, :],
                                          data_counts[collapsed_indices, :].sum(0, keepdims=True),
                                          data_counts[[baseline_index], :]))

        # log size factor of the last element is 0
        if len(collapsed_indices) > 0:
            data_counts_batch = numpy.vstack((data_counts[guide_indices, :],
                                              data_counts[collapsed_indices, :].sum(0, keepdims=True),
                                              data_counts[[baseline_index], :]))
            log_size_factors = numpy.log(numpy.concatenate((numpy.ones(len(guide_indices), dtype=int),
                                                            [len(collapsed_indices)])))
        else:
            data_counts_batch = numpy.vstack((data_counts[guide_indices, :],
                                              data_counts[[baseline_index], :]))
            log_size_factors = numpy.log(numpy.ones(len(guide_indices), dtype=int))

        data_dict = generate_stan_input_dictionary(data_counts_batch, N_replicates,
                                                   replicate_mapping, design_matrix,
                                                   lambda_prior)

        data_dict['guide_indices'] = guide_indices
        data_dict['log_size_factors'] = log_size_factors
        data_dicts.append(data_dict)

    logging.info('genererated stan input data dictionary')

    # write the stan input data file
    # to_rdump(data_dict,output_filename)

    return data_dicts


def helper(data, crispr_decryptr_model, n_chains, n_samples, output_prefix, batch_idx):
    crispr_decryptr_model.compile()
    crispr_decryptr_fit = crispr_decryptr_model.sample(data=data, chains=n_chains,
                                                       iter_warmup=n_samples, iter_sampling=n_samples, save_warmup=False, show_progress=False)

    # save_csvfiles will fail if any of the target files exist, thus let's delete them
    sample_filenames = ['%s-%d-%d.csv' % (output_prefix, batch_idx, idx) for idx in range(1, n_chains + 1)]
    for sample_filename in sample_filenames:
        if os.path.exists(sample_filename):
            os.remove(sample_filename)

    crispr_decryptr_fit.save_csvfiles(dir='.', basename='%s-%d' % (output_prefix, batch_idx))
    logging.info('saved sample files: %s' % (', '.join(sample_filenames)))
            
    for filename in crispr_decryptr_fit.runset._csv_files:
            basename = os.path.basename(filename)
            shutil.move(filename,os.path.join('./','%s-%d-%s.csv'%(output_prefix,batch_idx,basename.split('-')[-2])))
    logger.info('saved sample files: %s'%(', '.join(sample_filenames)))        
            
  

    return True


def run_stan(data_dict, stan_filename='crispr_decryptr.stan', output_prefix='samples',
             n_chains=1, n_samples=200, n_batches=1):
    # compile once

    use_lambda = False

    dir_path = os.path.dirname(os.path.realpath(__file__))

    crispr_decryptr_model = CmdStanModel(stan_file=dir_path + '/' + stan_filename)
    crispr_decryptr_model.compile()
    logging.info('starting sampling')

    total = sum(1 for e in zip(data_dict,
                              [CmdStanModel(stan_file=dir_path + '/' + stan_filename) for _ in range(0, len(data_dict))],
                              [n_chains] * len(data_dict),
                              [n_samples] * len(data_dict),
                              [output_prefix] * len(data_dict),
                              list(range(1, len(data_dict) + 1))))

    print("decryptr: Beginning sampling")
    print("decryptr: " + str(n_batches * n_chains) + " processes")
    print("decryptr: " + str(total) + " batch(es) to complete")
    print("PROGRESS:")

    with Pool(n_batches) as p:
        for _ in tqdm(p.istarmap(helper, zip(data_dict,
                              [CmdStanModel(stan_file=dir_path + '/' + stan_filename) for _ in range(0, len(data_dict))],
                              [n_chains] * len(data_dict),
                              [n_samples] * len(data_dict),
                              [output_prefix] * len(data_dict),
                              list(range(1, len(data_dict) + 1)))), total=total, unit='Batch'):
            pass



    moments = {'beta': {'mean': [], 'std': []}, 's': {'mean': [], 'std': []}, 'lambda': {'mean': [], 'std': []}}

    for batch_idx, data in enumerate(data_dict, start=1):
        sample_filenames = ['%s-%d-%d.csv' % (output_prefix, batch_idx, idx) for idx in range(1, n_chains + 1)]

        if use_lambda:
            moments_batch = calculate_moments_online(sample_filenames, variables=['beta', 's', 'lambda'])
        else:
            moments_batch = calculate_moments_online(sample_filenames, variables=['beta', 's'])
        logging.info('read sample files: %s' % (', '.join(sample_filenames)))

        moments['beta']['mean'].append(moments_batch['beta']['mean'][:, 0:len(data['guide_indices'])])
        moments['beta']['std'].append(moments_batch['beta']['std'][:, 0:len(data['guide_indices'])])
        moments['s']['mean'].append(moments_batch['s']['mean'])
        moments['s']['std'].append(moments_batch['s']['std'])
        if use_lambda:
            moments['lambda']['mean'].append(moments_batch['lambda']['mean'])
            moments['lambda']['std'].append(moments_batch['lambda']['std'])

    moments['beta']['mean'] = numpy.hstack(moments['beta']['mean'])
    moments['beta']['std'] = numpy.hstack(moments['beta']['std'])
    moments['s']['mean'] = numpy.vstack(moments['s']['mean']).mean(0)
    moments['s']['std'] = numpy.vstack(moments['s']['std']).mean(0)
    if use_lambda:
        moments['lambda']['mean'] = numpy.vstack(moments['lambda']['mean'])
        moments['lambda']['std'] = numpy.vstack(moments['lambda']['std'])

    logging.info('finished sampling')
    return moments


def save_output(moments, data_filename, design_matrix_filename, output_filename, outfile_devs):

    data_pd = pd.DataFrame() #pd.read_table(data_filename, sep='\t', header=0)

    data_devs = pd.DataFrame()


    # read the design matrix file
    design_matrix_pd = pd.read_table(design_matrix_filename, sep='\t', header=0, index_col=0)

    if len(moments['lambda']['mean']) > 0:
        col_labels = []
        row_labels = []
        rows, cols = numpy.where(design_matrix_pd.values)
        for (row, col) in zip(rows, cols):
            col_labels.append('%s,%s' % (design_matrix_pd.index[row], design_matrix_pd.columns[col]))
        for idx in range(len(moments['lambda']['mean'])):
            row_labels.append('%s (mean) (batch %d)' % ('lambda', idx + 1))
        for idx in range(len(moments['lambda']['std'])):
            row_labels.append('%s (std) (batch %d)' % ('lambda', idx + 1))

        pd.DataFrame(data=numpy.vstack((moments['lambda']['mean'], moments['lambda']['std'])),
                     columns=col_labels).to_csv('lambda_%s' % (output_filename), sep='\t', header=True, index=False)
        logging.info('saved output file: %s' % ('lambda_%s' % output_filename))

        #index=row_labels,

    for beta_idx, beta_label in enumerate(list(design_matrix_pd.columns)):
        # beta
        data_pd['%s mu (mean)' % (beta_label)] = numpy.concatenate((moments['beta']['mean'][beta_idx, :], [0]))
        data_pd['%s mu (std)' % (beta_label)] = numpy.concatenate((moments['beta']['std'][beta_idx, :], [0]))

        logging.info('finished sampling')

        # s = srqt(s2)
        data_devs['%s std (mean)' % (beta_label)] = data_pd.shape[0] * [moments['s']['mean'][beta_idx]]
        data_devs['%s std (std)' % (beta_label)] = data_pd.shape[0] * [moments['s']['std'][beta_idx]]

    # Adjust the last row to account for softmax transformation

    data_pd.iloc[[0, -1]] += 0.01
    #data_pd.drop(data_pd.tail(1).index, inplace=True)

    # Save the data to the outfile
    data_pd.to_csv(output_filename, sep='\t', header=True, index=False)

    # Append the final guide to the file with mean zero



    logging.info('saved output file: %s' % (output_filename))
    print('decryptr: saved file of posterior moments: %s' % (output_filename))

    if outfile_devs != False:
        data_devs.to_csv(outfile_devs, sep='\t', header=True, index=False)
        logging.info('saved standard deviation moment file: %s' % (outfile_devs))
