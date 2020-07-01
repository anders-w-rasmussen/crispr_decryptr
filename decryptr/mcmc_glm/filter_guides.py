import numpy as np
import os
import csv


def filter_guides(count_file, target_file, spacers_file, threshold, species):
    chrom = np.loadtxt(target_file, dtype=str)[0]
    targets = np.asarray(np.loadtxt(target_file, dtype=str)[1:], dtype=int)


    region_start = int(np.min(targets))
    region_end = int(np.max(targets))
    threshold = threshold

    if species == 'hg19':
        # check whether the uniqueness file for hg19 exists

        ref_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/hg19'
        if os.path.exists(ref_path+ "/uniqueness") == False:
            print("decryptr: downloading sequence uniqueness information for hg19 from ENCODE")

            os.system('mkdir ' + '/uniqueness')
            os.system('wget --timestamping --show-progress --directory-prefix=' + ref_path+ "/uniqueness" + ' http://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/wgEncodeDukeMapabilityUniqueness20bp.bigWig')

        # convert the uniqueness .bigWig to a fixedStep .wig file

        cur_path = os.path.dirname(os.path.realpath(__file__))

        os.system('chmod 777 ' + cur_path + '/bigWigToWig')
        
        os.system(
            cur_path + '/bigWigToWig ' +  ref_path + '/uniqueness/wgEncodeDukeMapabilityUniqueness20bp.bigWig ' +  ref_path + '/uniqueness/temp.wig' + ' -chrom=' + chrom + ' -start=' + str(
                region_start - 10) + ' -end=' + str(region_end + 10))

        # print(
        #     cur_path + '/bigWigToWig ' + ref_path + '/uniqueness/wgEncodeDukeMapabilityUniqueness20bp.bigWig ' + ref_path + '/uniqueness/temp.wig' + ' -chrom=' + chrom + ' -start=' + str(
        #         region_start - 10) + ' -end=' + str(region_end + 10))


        unique_file = ref_path + '/uniqueness/temp.wig'


    # Read the uniqueness file

    adjustment_vec = np.zeros(region_end - region_start + 1)
    counter = 0

    with open(unique_file, newline='') as csvfile:
        # L = len(csvfile.readlines())

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
                       -13:] == 'step=1 span=1', 'step and span of fixedStep .wig file must be one in sequence uniqueness file. Row of the .wig file reads: ' + str(
                    row[0])

                # Check chromosome
                file_chrom = row[0].partition("chrom=")[2].partition(" start")[0]
                assert file_chrom == chrom, 'there is a chromosome in the uniqueness file (' + str(
                    file_chrom) + ') that is not the chromosome in the target file (' + str(chrom) + ')'

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

    # Now filter targets

    target_relative = targets - region_start

    adjustment_vec_2 = adjustment_vec

    for i in range(10, np.size(adjustment_vec) - 10):
        if adjustment_vec[i] - adjustment_vec[i - 1] == 1.0:
            adjustment_vec_2[i - 10:i] = 0
        if adjustment_vec[i] - adjustment_vec[i - 1] == -1.0:
            adjustment_vec_2[i:i + 10] = 0


    unique_idxs = np.argwhere(adjustment_vec_2[target_relative] == 1).flatten()

    percent_unique = str( (1 - (float(np.size(unique_idxs)) / float(np.size(targets)))) * 100)

    print("decryptr: " + percent_unique[0:4] + "% of the guides were not sufficienty unique")

    targets_out = np.asarray(targets[unique_idxs])


    with open(target_file.split(".")[0] + '_filtered.tsv', 'w', newline='') as f:
        f.write(str(chrom) + '\n')
        for i in range(0, np.size(targets_out)):
            f.write(str(targets_out[i]) + '\n')

    if spacers_file != None:
        spacers = np.loadtxt(spacers_file, dtype=str)
        spacers_out = np.asarray(spacers[unique_idxs])

        with open(spacers_file.split(".")[0] + '_filtered.tsv', 'w', newline='') as f:
            for i in range(0, np.size(targets_out)):
                f.write(str(spacers_out[i]) + '\n')

    with open(count_file.split(".")[0] + '_filtered.tsv', 'w', newline='') as f:
        with open(count_file, 'r') as in_file:
            counter = 0
            for line in in_file:
                if counter == 0:
                    f.write(line)
                    counter += 1
                else:
                    if counter - 1 in unique_idxs:
                        f.write(line)
                    counter += 1

