'''
    Script to iteratively unzip the open-AI data repo

    open-AI has 6142 top level directories (-maxdepth 1)

    Example usage:
        python unzip_iteratively_open_AI.py -i /home/ravi.samala/DATA/temp/open_AI/ -o /home/ravi.samala/DATA/temp/open_AI_unzip/ -l /home/ravi.samala/DATA/temp/open_AI_unzip/unzip_log.txt
        python unzip_iteratively_open_AI.py -i /gpfs_projects/ravi.samala/DATA/MIDRC2/open_AI/ -o /gpfs_projects/ravi.samala/DATA/MIDRC2/open_AI_unzip/ -l /gpfs_projects/ravi.samala/DATA/MIDRC2/open_AI_unzip/unzip_log.txt
'''
import os
import argparse
import pydicom
from zipfile import ZipFile, BadZipFile


def search_files(folder, extension):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith(extension):
                matches.append(os.path.join(root, filename))
    return matches


def unzip_iteratively(in_dir, out_dir, log_file):
    '''
        input: root directory
        output: empty output directory
    '''

    zip_files_list = search_files(in_dir, 'zip')
    
    with open(log_file, 'w') as flog:
        for countr, each_zip_file in enumerate(zip_files_list):
            try:
                # # opening the zip file in READ mode
                with ZipFile(each_zip_file, 'r') as zip:
                    print(each_zip_file)
                    cur_out_dir = os.path.dirname(each_zip_file).replace(in_dir, out_dir)

                    if not os.path.exists(cur_out_dir):
                        os.makedirs(cur_out_dir)
                    # # printing all the contents of the zip file
                    # # zip.printdir()

                    # # extracting all the files
                    zip.extractall(cur_out_dir)
                    flog.write(str(countr) + '\tSUCCESS\t' + each_zip_file + '\n')
            except BadZipFile:
                print('UNZIP failed for: ' + each_zip_file)
                flog.write(str(countr) + '\tFAILED\t' + each_zip_file + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='unzip_data_repo')
    parser.add_argument('-i', '--input_root_dir',
                        default='/home/ravi.samala/DATA/temp/open_AI/')
    parser.add_argument('-o', '--output_root_dir',
                        default='/home/ravi.samala/DATA/temp/open_AI_unzip/')
    parser.add_argument('-l', '--log_file',
                        default='/home/ravi.samala/DATA/temp/open_AI_unzip/unzip_log.txt')
    args = parser.parse_args()
    # #
    unzip_iteratively(args.input_root_dir, args.output_root_dir, args.log_file)

