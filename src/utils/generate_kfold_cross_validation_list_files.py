'''
    Program that can create a k-fold cross validation list files by-patient
'''
import argparse
import json
import os
import pandas as pd
import csv
from random import shuffle
import math


class D(dict):
    def __missing__(self, key):
        self[key] = D()
        return self[key]


def get_case_names(in_file):
    '''
    function that can extract the case 
    hard coded to use the fname[0:5] as case name
    '''
    dat = pd.read_table(in_file, header=None)
    print('Number of samples # = ' + str(len(dat)))
    case_names = []
    for index, row in dat.iterrows():
        fname = os.path.basename(row[0])
        current_case_name = fname[0:5]  # possible bug here, id of patient is [0:5]

        if current_case_name not in case_names:
            case_names.extend([current_case_name])
    print('Unique case names# = ' + str(len(case_names)))
    return case_names


def count_cases_with_ROIname(checkValue, master_list, start_char):
    num_cases = 0
    d = D()
    for each in master_list:
        cur_case_view = os.path.basename(each[0])
        # case_num = cur_case_view[0:5]
        case_num = cur_case_view[0:-12]
        # print(case_num + '\t' + case_num[0])

        if str(case_num[0]) == start_char:
            type_mb = each[1]
            if checkValue != -1:
                if int(type_mb) == checkValue:
                    if case_num in d:
                        d[case_num] += 1
                    else:
                        d[case_num] = 1
                        num_cases += 1
            else:
                if case_num in d:
                    d[case_num] += 1
                else:
                    d[case_num] = 1
                    num_cases += 1
    return num_cases, d


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def divide_data_folds_randomly_write(num_cases, d_in, num_folds, listAll2, file_names):
    """
    listAll2 is a list of lists, each list being:
        ['<complete path to case .dat>', <0 or 1 for malignant or benign>, 0]
    """
    print('num cases = ' + str(num_cases))
    print('len(filenames) = ' + str(len(file_names)))
    x = [i for i in range(num_cases)]
    shuffle(x)
    chunk_size = int(math.ceil(num_cases / num_folds))
    chunks_ = list(chunks(x, chunk_size))
    print('len(chunks_) = ' + str(len(chunks_)) + ', num_folds = ' + str(num_folds))
    # check here
    # if len(chunks_) > num_folds:
    while len(chunks_) > num_folds:
        # merge the last chunk into the last but one chunk
        chunks_[num_folds-1].extend(chunks_[num_folds])
        chunks_.pop(num_folds)
    print('len(chunks_) = ' + str(len(chunks_)) + ', num_folds = ' + str(num_folds))
    # for i in range(num_folds):
    #     print([i, len(chunks_[i])])
    keys_ = list(d_in)
    chunk_id = 0
    lesion_idx = 0
    for count, each_chunk in enumerate(chunks_):
        # print(str(count) + '\t', end="")
        fname = file_names[count]
        print(fname, str(count), str(len(chunks_[count])))
        with open(fname, 'a') as fp:
            for each_case_index in each_chunk:
                case_num = keys_[each_case_index]
                # check if case_num is in the master list file
                # adds every image taken of that patient

                case_list = (x for x in listAll2 if case_num in x[0])
                for agg in case_list:
                    fp.write(agg[0] + '\t' + agg[1] + '\t' + agg[2] + '\n')
                    lesion_idx += 1
        chunk_id += 1
    print('total lesions written = ' + str(lesion_idx))


def split_benign_and_malignant(num_benign_cases, benign_dict,
                               num_malignant_cases, malignant_dict, nfolds, listAll, file_names):
    # create a third dict that only contains the patients that have both malignant and benign tumors
    # because the images have correlation so they must go together - either to the training
    # or to the validation

    # find out which patients are in both dicts
    patients_in_benign = list(benign_dict)
    patients_in_both = [patient for patient in patients_in_benign
                        if patient in malignant_dict and malignant_dict[patient] > 0]

    # construct dict with those patients
    benign_and_malignant_dict = D()
    for patient in patients_in_both:
        benign_and_malignant_dict[patient] = benign_dict[patient]
        benign_and_malignant_dict[patient] += malignant_dict[patient]
    num_benign_and_malignant_cases = len(benign_and_malignant_dict)

    # remove those patients from the others dicts
    # both passed dicts must be copied, in case they are used elsewhere outside
    new_benign_dict = {key: value for key, value in benign_dict.items() if key not in benign_and_malignant_dict}
    benign_dict = new_benign_dict
    num_only_benign_cases = len(benign_dict)

    new_malignant_dict = {key: value for key, value in malignant_dict.items() if
                          key not in benign_and_malignant_dict}
    malignant_dict = new_malignant_dict
    num_only_malignant_cases = len(malignant_dict)

    print(">> [TS] BENIGN >>>>>")
    divide_data_folds_randomly_write(num_only_benign_cases,
                                     benign_dict, nfolds, listAll, file_names)
    print(">> [TS] MALIGNANT >>>>>")
    divide_data_folds_randomly_write(num_only_malignant_cases,
                                     malignant_dict, nfolds, listAll, file_names)
    print(">> [TS] BENIGN AND MALIGNANT >>>>>")
    divide_data_folds_randomly_write(num_benign_and_malignant_cases,
                                     benign_and_malignant_dict, nfolds, listAll, file_names)


def generate_kfolds(args):
    # read the input training list and create dict with case names
    case_names = get_case_names(args.input_train_file)
    # print(case_names)
    listAll = []
    with open(args.input_train_file, 'r') as fp:
        for line in csv.reader(fp, delimiter='\t'):
            listAll.append(line)
        print('[trtr] There are ' + str(len(listAll)) + ' lesions in the master list')
    # # 
    file_names = []
    for nfold in range(args.folds):
        file_names.append(os.path.join(args.output_base_dir, 'exp' + '_f' + str(nfold) + '.lis'))
    print(file_names)
    for start_char in ['5', 'f', 's', 'p', 'm']:
        num_benign_cases, benign_dict = count_cases_with_ROIname(0, listAll, start_char)
        num_malignant_cases, malignant_dict = count_cases_with_ROIname(1, listAll, start_char)
        print(start_char + ': Benign/Malignant = ' + str(num_benign_cases) + '/' + str(num_malignant_cases))

        split_benign_and_malignant(num_benign_cases, benign_dict,
                                    num_malignant_cases, malignant_dict, args.folds, listAll, file_names)
    
    for i in range(args.folds):
        print("\n>> >> fold# " + str(i))
        # set the output directories
        dir_out = os.path.join(args.output_base_dir, 'f' + str(i))
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        # # merge all the training folds except for the ith fold
        # # create this fold's training and test lists
        new_tr_file = os.path.join(dir_out, 'tr.lis')
        new_ts_file = os.path.join(dir_out, 'ts.lis')
        print('[TR] Merging to:' + new_tr_file)
        with open(new_tr_file, 'a') as fp:
            for j in range(args.folds):
                if j == i:
                    continue
                print(file_names[j])
                fp.write(open(file_names[j], 'r').read())
        # # the ith fold will be the test fold
        print('[TS] Copying to: ' + new_ts_file)
        with open(new_ts_file, 'w+') as fp:
            print(file_names[i])
            fp.write(open(file_names[i], 'r').read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate k-fold list files')
    parser.add_argument('-i', '--input_train_file', help='input training list file', required=True)
    parser.add_argument('-o', '--output_base_dir', help='output dir', required=True)
    parser.add_argument('-f', '--folds', type=int, default=4, help='num. of folds')
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.output_base_dir):
        os.makedirs(args.output_base_dir)

    # # save the args
    with open(os.path.join(args.output_base_dir, 'args.json'), 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    # # # ========================================
    generate_kfolds(args)
    print('END.')
