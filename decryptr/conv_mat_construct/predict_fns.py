import numpy as np
import re
from scipy import sparse
import time
from keras.models import load_model
from Bio import SeqIO
from multiprocessing import Process, Manager
import csv
import pickle
import os
from alive_progress import alive_bar, config_handler
from halo import Halo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def cmat_run(targets_file, cas9_alg, filter_dev, filter_window, spacers, species, reference, uniqueness, ignore_spec, logfile,
             processes, cmat_name):

    if cas9_alg == 'false':
        cas9_alg = 'False'
    if cas9_alg == 'true':
        cas9_alg = 'True'

    if cas9_alg == 'True':
        assert spacers != None, 'specified cas9_alg = True but did not include spacer file argument'

    # Runner function for constructing the convolution matrix.

    # if both guide-alignment and repair prediction are off, run the generalized cmat construction fn

    map_file = None

    if species == None:
        if uniqueness == None:
            print("decryptr: No species or uniqueness file specified. Not accounting for guide specificity.")

    if cas9_alg == 'False':


        make_cmat(targets_file, filter_dev, filter_window, cmat_name)
        print("decryptr: saved convolution matrix")


        if ignore_spec == 'False':

            cur_path = os.path.dirname(os.path.realpath(__file__))

            # check which chromosome we're looking at
            chromosome = np.loadtxt(targets_file, dtype=str)[0]
            targets = np.asarray(np.loadtxt(targets_file, dtype=str)[1:], dtype=int)

            region_start = int(max(np.min(targets) - (filter_window - 1) / 2, 0))
            region_end = int(np.max(targets) + (filter_window - 1) / 2 + 1)


            if species != None:
                assert uniqueness == None, 'you specified a species and a uniqueness file. If you want to use your own uniqueness please' \
                                          'do not specify a species (as it is meant for looking up reference and uniqueness files).'

                ref_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/hg19'


                if os.path.exists(ref_path + "/fastas") == False:
                    print("decryptr: first time aligning to hg19")
                    print("decryptr: downloading hg19 reference")
                    os.system('mkdir ' + str(ref_path) + '/fastas')
                    os.system(
                        'wget --timestamping --show-progress --directory-prefix=' + ref_path + "/fastas" + ' ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/*')
                    print("decryptr: unzipping all hg19 genome .fasta files")
                    os.system('gunzip ' + str(ref_path) + '/fastas/*.fa.gz')


                if os.path.exists(ref_path + "/uniqueness") == False:
                    print("decryptr: downloading mappability track")
                    os.system(
                        'wget --timestamping --show-progress --directory-prefix=' + ref_path + "/uniqueness" + ' http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/wgEncodeDukeMapabilityUniqueness20bp.bigWig')

                #ref_file = curpath + "/hg19/" + chromosome + '.fa'

                # Convert bigWig to .wig

                print("decryptr: getting uniqueness of the region under consideration")
                os.system('chmod 777 ' + cur_path + '/bigWigToWig')
                os.system(cur_path + '/bigWigToWig ' + ref_path + '/uniqueness/wgEncodeDukeMapabilityUniqueness20bp.bigWig ' + ref_path + '/uniqueness/temp.wig' + ' -chrom=' + chromosome + ' -start=' + str(region_start - 10) + ' -end=' + str(region_end + 10))

                map_file = ref_path + '/uniqueness/temp.wig'

            else:
                map_file = uniqueness

    else:

        # check which chromosome we're looking at

        chromosome = np.loadtxt(targets_file, dtype=str)[0]
        targets = np.asarray(np.loadtxt(targets_file, dtype=str)[1:], dtype=int)

        region_start = max(np.min(targets) - 100, 0)
        region_end = np.max(targets) + 100

        # if species is specified look up the proper files. Download them if we don't have them

        cur_path = os.path.dirname(os.path.realpath(__file__))
        ref_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/hg19'
        print(ref_path)

        if os.path.exists(ref_path + "/fastas") == False:
            print("decryptr: first time aligning to hg19")
            print("decryptr: downloading hg19 reference")
            os.system('mkdir ' + str(ref_path) + '/fastas')
            os.system(
                'wget --timestamping --show-progress --directory-prefix=' + ref_path + "/fastas" + ' ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/chromosomes/*')
            print("decryptr: unzipping all hg19 genome .fasta files")
            os.system('gunzip ' + str(ref_path) + '/fastas/*.fa.gz')


        if os.path.exists(ref_path + "/uniqueness") == False:
            print("decryptr: downloading mappability track")
            os.system(
                'wget --timestamping --show-progress --directory-prefix=' + ref_path + "/uniqueness" + ' http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/wgEncodeDukeMapabilityUniqueness20bp.bigWig')

        #ref_file = curpath + "/hg19/" + chromosome + '.fa'

        # Convert bigWig to .wig

        print("decryptr: getting uniqeuness of the region under consideration")
        os.system('chmod 777 ' + cur_path + '/bigWigToWig')
        os.system(cur_path + '/bigWigToWig ' + ref_path + '/uniqueness/wgEncodeDukeMapabilityUniqueness20bp.bigWig ' + ref_path + '/uniqueness/temp.wig' + ' -chrom=' + chromosome + ' -start=' + str(region_start - 10) + ' -end=' + str(region_end + 10))

        map_file = ref_path + '/uniqueness/temp.wig'

        ref_file = ref_path + '/fastas/' + chromosome + '.fa'

        make_cmat_cas9(targets_file, spacers, ref_file, processes, cmat_name)


    if map_file != None:
        if ignore_spec == 'False':

            # Read the mapability .wig file (should be fixed step)

            adjustment_vec = np.zeros(region_end - region_start)
            counter = 0

            with open(map_file, newline='') as csvfile:
                #L = len(csvfile.readlines())

                reader = csv.reader(csvfile, delimiter='\t')
                cur_idx = 0
                threshold = 0.05
                good_flag = False

                # spinner = Halo(text='decryptr: adjusting for sequence specificity', spinner='dots',
                #                color='white', placement='right')
                # spinner.start()

                for row in reader:

                    if row[0][:9] == 'fixedStep':

                        # Check step and span
                        assert row[0][
                               -13:] == 'step=1 span=1', 'step and span of fixedStep .wig file must be one in sequence uniqueness file. Row of the .wig file reads: ' + str(row[0])

                        # Check chromosome
                        file_chrom = row[0].partition("chrom=")[2].partition(" start")[0]
                        assert file_chrom == chromosome, 'there is a chromosome in the uniqueness file (' + str(
                            file_chrom) + ') that is not the chromosome in the target file (' + str(chromosome) + ')'

                        part_start = int(row[0].partition("start=")[2].partition(" step")[0])

                        # check first start to make sure we haven't missed bases

                        if cur_idx == 0:
                            assert part_start <= region_start, 'there are pertubations that target positions before the start of the uniqueness file.'
                        cur_idx = part_start - region_start

                    else:
                        val = float(row[0])
                        if cur_idx > 0:
                            if val > threshold:
                                adjustment_vec[cur_idx] = 1
                            else:
                                counter += 1
                                pass
                        cur_idx += 1

                    if cur_idx >= np.size(adjustment_vec):
                        good_flag = True
                        break

                assert good_flag == True, 'the uniqueness file does not cover the entire region'

                original_cmat = pickle.load(open(cmat_name, "rb"))

                # Check the convolution matrix has same num of columns as entries in adjustment vector
                assert np.size(original_cmat, axis=1) == np.size(adjustment_vec), 'problem with adjustment vector not matching the convolution matrix! this must be a bug.'

                adjustment_vec_2 = adjustment_vec

                for i in range(10, np.size(adjustment_vec) - 10):
                    if adjustment_vec[i] - adjustment_vec[i-1] == 1.0:
                        adjustment_vec_2[i - 10:i] = 0
                    if adjustment_vec[i] - adjustment_vec[i-1] == -1.0:
                        adjustment_vec_2[i:i + 10] = 0

                col_idxs_good = np.argwhere(adjustment_vec_2 == 1).flatten()
                col_idxs_bad = np.argwhere(adjustment_vec_2 == 0).flatten()


                row_idxs = np.argwhere(adjustment_vec_2[targets - region_start] == 1).flatten()

                out_cmat = sparse.lil_matrix(np.shape(original_cmat))
                out_cmat2 = sparse.lil_matrix(np.shape(original_cmat))


                out_cmat[row_idxs, :] = original_cmat.tocsc()[row_idxs, :]
                out_cmat2[:, col_idxs_good] = original_cmat.tocsc()[:, col_idxs_good]

                # out_cmat[:, col_idxs_good] = original_cmat.tocsr()[:, col_idxs_good]
                # out_cmat[:, col_idxs_bad] = original_cmat.tocsr()[:, col_idxs_bad] * 0.1


                #out_cmat[row_idxs, :][:, col_idxs] = original_cmat[row_idxs, :][:, col_idxs]

                pickle.dump(out_cmat, open(cmat_name, "wb"))

                # Determine % what was removed from cmat
                per_rem = (1.0 - np.size(row_idxs)/np.size(original_cmat, axis=0)) * 100

                per_rem_targs = (1.0 - np.size(col_idxs_good)/np.size(original_cmat, axis=1)) * 100

                print("decryptr: adjusted the convolution matrix")


def make_cmat_cas9(target_file, guide_file, reference_file, n_processes, cmat_name):

    num_threads = n_processes

    config_handler.set_global(length=40, spinner='dots_reverse')

    # SCORING TABLE

    dir_path = os.path.dirname(os.path.realpath(__file__))
    f = np.loadtxt(dir_path + '/offtargettable.txt', usecols=range(1, 20))

    # NEED TO LOOK INTO THIS (ALLOW 1 to be the maximum score)
    f[f > 1] = 1
    f_tensor = tensor_from_table(f)

    # trained Convolutional Neural Network
    cnn_model = load_model(dir_path + '/cnn_model.h5', compile=False)

    #########################################
    ###   MAKE CANDIDATE ENCODED ARRAYS   ###
    #########################################

    # Get the target locations from .txt file
    targets = np.loadtxt(target_file, dtype=str)
    chrom = targets[0]
    targets = np.asarray(targets[1:], dtype=int)

    # Calculate region start and end
    region_start = max(np.min(targets) - 100, 0)
    region_end = np.max(targets) + 100

    n_bases = region_end - region_start

    region_sequences = []

    for record in SeqIO.parse(reference_file, "fasta"):
        region_sequences.append(record[region_start:region_end + 1])

    # Find PAM sites in sequences and encode them (could use multiprocessing for this if we need to)

    PAM = 'NGG'

    PAM_sites = []
    encoded_f = []
    encoded_c = []

    for i in range(0, len(region_sequences)):
        PAM_sites.append(find_PAMs_individual(PAM, region_sequences[i]))
        encode_result = encode_sequence(region_sequences[i], PAM_sites[i][0], PAM_sites[i][1])
        encoded_f.append(encode_result[0])
        encoded_c.append(encode_result[1])

    encoded_f = [item for sublist in encoded_f for item in sublist]
    encoded_c = [item for sublist in encoded_c for item in sublist]
    forward_array = np.asarray(encoded_f).transpose()
    complement_array = np.asarray(encoded_c).transpose()

    # Make a list of where the splits are gonna be in flattened conv_mat

    #########################################
    ###      MAKE GUIDE ENCODED ARRAYS    ###
    #########################################

    # Get the guides from the file (clean this up for final product)

    guide_targets = []
    guide_chromosomes = []
    guide_locations = []

    unique_IDs = []
    n_guides = 0
    with open(guide_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row[0] == '':
                break
            else:
                guide_targets.append(str(row[0])[1:])  # < ----- I need to fix this to only use last 19
                unique_IDs.append(n_guides)
                n_guides += 1

    encoded_guides = encode_guide_set(guide_targets)

    #########################################
    ###         OFF-TARGET SCORING        ###
    #########################################


    assert num_threads <= len(encoded_guides), "num_threads must be <= number of guides"

    guides_per_process = int(len(encoded_guides) / num_threads)

    #print("decryptr: using " + str(num_threads) + " cores to make convolution matrix")
    #print("decryptr: there are " + str(len(encoded_guides)) + " guides to align")
    #print("decryptr: allocating ~ " + str(guides_per_process) + " guides to each process")

    list_of_lists_targets = []
    list_of_lists_IDs = []
    s_idx = 0
    for i in range(0, num_threads - 1):
        list_of_lists_targets.append(encoded_guides[s_idx:s_idx + guides_per_process])
        list_of_lists_IDs.append(unique_IDs[s_idx:s_idx + guides_per_process])
        s_idx += guides_per_process
    list_of_lists_targets.append(encoded_guides[s_idx:])
    list_of_lists_IDs.append(unique_IDs[s_idx:])

    # Duplicate mismatch tensor, encodings for the number of threads going to use

    tensor_list = []
    pam_list = []
    encoding_arr_f = []
    encoding_arr_c = []

    for i in range(0, num_threads):
        pam_list.append(PAM_sites)
        tensor_list.append(f_tensor)
        encoding_arr_f.append(forward_array)
        encoding_arr_c.append(complement_array)

    # Begin the processes

    t0 = time.time()

    threshold = 0.99

    print("decryptr: scoring guides")

    if __name__ != "__main__":
        with Manager() as manager:
            master_list = manager.list()
            processes = []
            for k in range(num_threads):
                p = Process(target=score_guides, args=(k, list_of_lists_IDs[k],
                                                       list_of_lists_targets[k], encoding_arr_f[k],
                                                       encoding_arr_c[k], tensor_list[k], threshold,
                                                       master_list))

                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            saved_master_list = [x for x in master_list]

    ################################################
    ###         REPAIR OUTCOME PREDICTION        ###
    ################################################

    t0 = time.time()

    # One-Hot Encode All Sequences

    forward_ohe = []
    complement_ohe = []

    for i in range(0, len(PAM_sites)):
        seq = region_sequences[i]
        for j in range(0, len(PAM_sites[i][0])):
            forward_ohe.append(one_hot_encode(seq, j))
        for j in range(0, len(PAM_sites[i][1])):
            complement_ohe.append(one_hot_encode(seq, j))

    # Predict

    forward_profiles = cnn_model.predict(np.asarray(forward_ohe).reshape(-1, 41, 4))
    complement_profiles = cnn_model.predict(np.asarray(complement_ohe).reshape(-1, 41, 4))

    ####################################################
    ###         CREATE THE CONVOLUTION MATRIX        ###
    ####################################################

    t0 = time.time()

    forward_profiles_list = []
    complement_profiles_list = []
    n_bases_list = []
    n_guides_list = []
    s_idxs_list = []
    e_idxs_list = []
    PAM_sites_list = []

    for i in range(0, num_threads):
        forward_profiles_list.append(forward_profiles)
        complement_profiles_list.append(complement_profiles)
        n_bases_list.append(n_bases)
        n_guides_list.append(n_guides)
        PAM_sites_list.append(PAM_sites)

    print("decryptr: building chunks of convolution matrix from pertubation profiles")

    if __name__ != "__main__":
        with Manager() as manager:
            chunks = manager.list()
            processes = []
            for k in range(num_threads):
                p = Process(target=make_conv_chunk, args=(
                    k, PAM_sites_list[k], forward_profiles_list[k], complement_profiles_list[k],
                    saved_master_list[k],
                    n_bases, n_guides_list[k], chunks))

                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            saved_chunks = [x for x in chunks]

    t0 = time.time()

    spinner = Halo(text='decryptr: combining chunks into one convolution matrix', spinner='dots',
                   color='white', placement='right')
    spinner.start()

    conv_mat = sparse.lil_matrix((n_guides, n_bases))

    for chunk in saved_chunks:
        nonzero = chunk.nonzero()
        conv_mat[nonzero] = chunk[nonzero]

    # see if this works

    for row in range(0, np.size(conv_mat, axis=0)):
        if np.sum(conv_mat[row, :]) != 0:
            conv_mat[row, :] /= np.sum(conv_mat[row, :])

    pickle.dump(conv_mat, open(cmat_name, "wb"))

    spinner.stop()

    print("decryptr: saved the convolution matrix")


def tensor_from_table(f):

    # 4 x 4 x 19 tensor
    # Axis 0 is target nucleotide
    # Axis 1 is reference nucleotide
    # Axis 2 is position in guide
    # 0=A, 1=C, 2=G, 3=T

    mismatch_tensor = np.zeros((4, 4, 19))

    for k in range(0, 19):
        mismatch_tensor[3, :, k] = f[0:4, k]
        mismatch_tensor[2, :, k] = f[4:8, k]
        mismatch_tensor[1, :, k] = f[8:12, k]
        mismatch_tensor[0, :, k] = f[12:16, k]

    return mismatch_tensor


def find_PAMs_individual(PAM, record_object):

    sequence = record_object.seq._data
    rev_comp_sequence = record_object.seq.reverse_complement()._data

    expr = '(?=' + PAM.replace('N', '.') + ')'
    idxs = [m.start() for m in re.finditer(expr, sequence, re.IGNORECASE)]
    rev_idxs = [m.start() for m in re.finditer(expr, rev_comp_sequence, re.IGNORECASE)]

    # Only return idxs that are greater than 19 (must be a better way to do this)

    out_idxs = [s for s in idxs if s >= 19]
    out_idxs = [s for s in out_idxs if s <= len(sequence) - 21]
    out_rev_idxs = [s for s in rev_idxs if s >= 19]
    out_rev_idxs = [s for s in out_rev_idxs if s <= len(sequence) - 21]

    return [out_idxs, out_rev_idxs]


def encode_sequence(record_object, idxs, rev_idxs):

    sequence = record_object.seq._data
    rev_comp_sequence = record_object.seq.reverse_complement()._data

    f_encoded = []
    for f_idx in idxs:
        dn = np.zeros(19, dtype='int')
        t = np.fromstring(sequence[f_idx - 19: f_idx], dtype='B')
        dn[t == ord('A')] = 0
        dn[t == ord('a')] = 0
        dn[t == ord('C')] = 1
        dn[t == ord('c')] = 1
        dn[t == ord('G')] = 2
        dn[t == ord('g')] = 2
        dn[t == ord('T')] = 3
        dn[t == ord('t')] = 3
        dn[t == ord('N')] = 3
        dn[t == ord('n')] = 3

        f_encoded.append(dn)

    c_encoded = []
    for c_idx in rev_idxs:
        dn = np.zeros(19, dtype='int')
        t = np.fromstring(rev_comp_sequence[c_idx - 19: c_idx], dtype='B')
        dn[t == ord('A')] = 0
        dn[t == ord('a')] = 0
        dn[t == ord('C')] = 1
        dn[t == ord('c')] = 1
        dn[t == ord('G')] = 2
        dn[t == ord('g')] = 2
        dn[t == ord('T')] = 3
        dn[t == ord('t')] = 3
        dn[t == ord('N')] = 3
        dn[t == ord('n')] = 3

        c_encoded.append(dn)

    return f_encoded, c_encoded

def encode_guide_set(guide_sequences):

    out_list = []

    for guide in guide_sequences:
        dn = np.zeros(19, dtype='int')
        t = np.fromstring(guide, dtype='B')
        dn[t == ord('A')] = 0
        dn[t == ord('a')] = 0
        dn[t == ord('C')] = 1
        dn[t == ord('c')] = 1
        dn[t == ord('G')] = 2
        dn[t == ord('g')] = 2
        dn[t == ord('T')] = 3
        dn[t == ord('t')] = 3
        dn[t == ord('N')] = 3
        dn[t == ord('n')] = 3

        out_list.append(dn)

    return out_list

def one_hot_encode(reference, pam_index):
    feature_mat = np.zeros((4, 41))

    for k in range(0, 41):
        if reference[pam_index - 20 + k] == 'A':
            feature_mat[0, k] = 1
        elif reference[pam_index - 20 + k] == 'a':
            feature_mat[0, k] = 1
        elif reference[pam_index - 20 + k] == 'T':
            feature_mat[1, k] = 1
        elif reference[pam_index - 20 + k] == 't':
            feature_mat[1, k] = 1
        elif reference[pam_index - 20 + k] == 'G':
            feature_mat[2, k] = 1
        elif reference[pam_index - 20 + k] == 'g':
            feature_mat[2, k] = 1
        elif reference[pam_index - 20 + k] == 'C':
            feature_mat[3, k] = 1
        elif reference[pam_index - 20 + k] == 'c':
            feature_mat[3, k] = 1

    return feature_mat

def score_guides(k, IDs, targets, f_encoding, c_encoding, mismatch_array, thresh, out_list):


    #print('Scoring Process ' + str(k + 1) + ' Started')

    score_list = []

    n_f = np.size(f_encoding, axis=1)
    n_c = np.size(c_encoding, axis=1)
    last_idx_f = np.array([range(0, 19), ] * n_f).transpose()
    last_idx_c = np.array([range(0, 19), ] * n_c).transpose()

    if k != 0:

        for i in range(0, len(targets)):

            forward_scores = np.prod(mismatch_array[np.array([targets[i],] * n_f).transpose(), f_encoding, last_idx_f], axis=0)
            complement_scores = np.prod(mismatch_array[np.array([targets[i], ] * n_c).transpose(), c_encoding, last_idx_c], axis=0)


            forward_scores[forward_scores < thresh] = 0
            complement_scores[complement_scores < thresh] = 0

            sparse_forward = sparse.lil_matrix(forward_scores)
            sparse_complement = sparse.lil_matrix(complement_scores)

            score_list.append([IDs[i], sparse_forward, sparse_complement])

    else:

        L = len(targets)
        with alive_bar(L) as bar:

            for i in range(0, len(targets)):

                forward_scores = np.prod(mismatch_array[np.array([targets[i],] * n_f).transpose(), f_encoding, last_idx_f], axis=0)
                complement_scores = np.prod(mismatch_array[np.array([targets[i], ] * n_c).transpose(), c_encoding, last_idx_c], axis=0)

                forward_scores[forward_scores < thresh] = 0
                complement_scores[complement_scores < thresh] = 0

                sparse_forward = sparse.lil_matrix(forward_scores)
                sparse_complement = sparse.lil_matrix(complement_scores)

                score_list.append([IDs[i], sparse_forward, sparse_complement])

                bar()



    out_list.append(score_list)

def make_conv_chunk(k, PAM_sites, forward_profile, complement_profile, master_list_k, n_bases, n_guides, chunks):

    # print('Conv mat process ' + str(k + 1) + ' Started')

    num_guides = n_guides
    num_bases = n_bases

    out_chunk = sparse.lil_matrix((num_guides, num_bases), dtype=float)

    if k != 0:

        for entry in master_list_k:
            ID = entry[0]
            forward = entry[1]
            complement = entry[2]

            # Forward positions

            n = 0
            for j in range(0, len(PAM_sites)):
                starting_idx = 0
                idxs = forward[0, n:n+len(PAM_sites[j][0])].nonzero()[1]

                if idxs.any():

                    score_and_profile = forward_profile[n + idxs, :] * np.tile(forward[0, n + idxs].toarray().transpose(), 41)
                    position_in_conv_mat = np.asarray(starting_idx + np.asarray(PAM_sites[j][0])[idxs])

                    l_idx = ((position_in_conv_mat)[:, np.newaxis] + np.asarray(range(-20, 21))).flatten()
                    rhs = score_and_profile.flatten()

                    out_chunk[ID, l_idx] = rhs

                n += len(PAM_sites[j][0])

            # Complement positions

            n = 0
            for j in range(0, len(PAM_sites)):
                ending_idx = num_bases
                idxs = complement[0, n:n + len(PAM_sites[j][1])].nonzero()[1]

                if idxs.any():
                    score_and_profile = complement_profile[n + idxs, :] * np.tile(complement[0, n + idxs].toarray().transpose(),
                                                  41)

                    position_in_conv_mat = np.asarray(ending_idx - np.asarray(PAM_sites[j][1])[idxs])
                    out_chunk[ID, (position_in_conv_mat[:, np.newaxis] + np.flip(np.asarray(range(-20, 21)))).flatten()] += score_and_profile.flatten()

                n += len(PAM_sites[j][1])

    if k == 0:

        L = len(master_list_k)
        with alive_bar(L) as bar:

            for entry in master_list_k:
                ID = entry[0]
                forward = entry[1]
                complement = entry[2]

                # Forward positions

                n = 0
                for j in range(0, len(PAM_sites)):
                    starting_idx = 0
                    idxs = forward[0, n:n + len(PAM_sites[j][0])].nonzero()[1]

                    if idxs.any():
                        score_and_profile = forward_profile[n + idxs, :] * np.tile(
                            forward[0, n + idxs].toarray().transpose(), 41)
                        position_in_conv_mat = np.asarray(starting_idx + np.asarray(PAM_sites[j][0])[idxs])

                        l_idx = ((position_in_conv_mat)[:, np.newaxis] + np.asarray(range(-20, 21))).flatten()
                        rhs = score_and_profile.flatten()

                        out_chunk[ID, l_idx] = rhs

                    n += len(PAM_sites[j][0])

                # Complement positions

                n = 0
                for j in range(0, len(PAM_sites)):
                    ending_idx = num_bases
                    idxs = complement[0, n:n + len(PAM_sites[j][1])].nonzero()[1]

                    if idxs.any():
                        score_and_profile = complement_profile[n + idxs, :] * np.tile(
                            complement[0, n + idxs].toarray().transpose(),
                            41)

                        position_in_conv_mat = np.asarray(ending_idx - np.asarray(PAM_sites[j][1])[idxs])
                        out_chunk[ID, (position_in_conv_mat[:, np.newaxis] + np.flip(
                            np.asarray(range(-20, 21)))).flatten()] += score_and_profile.flatten()

                    n += len(PAM_sites[j][1])

                bar()

    chunks.append(out_chunk)



def make_cmat(target_file, filter_dev, window_size, cmat_name):

    # Check and set up all the input arguments
    filter_dev = float(filter_dev)
    window_size = int(window_size)
    assert window_size % 2 != 0, 'Window size bust be an odd integer'

    # Get the target locations from .txt file
    targets = np.loadtxt(target_file, dtype=int, skiprows=1)

    # Calculate region start and end
    region_start = int(max(np.min(targets) - (window_size - 1)/2, 0))
    region_end = int(np.max(targets) + (window_size - 1)/2 + 1)

    # Make the convolution matrix

    N_guides = np.size(targets)
    N_bases = region_end - region_start

    cmat = sparse.lil_matrix((N_guides, N_bases))

    # Make the Gaussian window (for some reason this isn't in my version of scipy.signal)

    window = np.zeros(window_size)
    for i in range(0, window_size):
        window[i] = np.exp(-0.5 * ((i - ((window_size - 1) / 2)) / filter_dev) ** 2)

    window /= np.sum(window)

    # Check to ensure all the guide targets are a filter cutoff within region when we put them in the
    # convolution matrix

    # Should also check for overlapping regions

    for i in range(0, N_guides):

            location = targets[i]
            cmat[i, int(location - region_start - (window_size - 1) / 2): int(
                location - region_start + 1 + (window_size - 1) / 2)] += window


    pickle.dump(cmat, open(cmat_name, "wb"))
