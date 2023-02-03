from argparse import ArgumentParser
import numpy as np
import h5py
import sys
import os

def convert_npz_to_hdf5(in_file, out_folder=None, overwrite=False):
    '''
        Converts .npz files (either array.npz or activations.npz) to hdf5 format.
        ------------------------------------------------------------------------
        input_file : can be either a specific file to be converted or a folder containing multiple files to be converted
        out_folder : folder to save hdf5 files, if not provided, files will be saved in the same folder as the original .npz file
    '''
    # overwrite wraning
    if overwrite:
        print("Overwrite: True")
    # file management
    if os.path.isdir(in_file):
        input_files = [os.path.join(in_file, f) for f in os.listdir(in_file) if f.endswith(".npz")]
    else:
        input_files = [in_file]
    num_files = len(input_files)
    for file_num, file in enumerate(input_files):
        if out_folder is None: # by default save in same folder as input
            out_folder = "/".join(file.split("/")[:-1])
        h5_filepath = os.path.join(out_folder, file.split("/")[-1].replace(".npz", ".hdf5"))
        try:
            f = h5py.File(h5_filepath, 'w-')
        except:
            if overwrite:
                f = h5py.File(h5_filepath, 'w')
            else:
                print(f"File already exists at {h5_filepath} and overwrite = {overwrite}")
                continue
        if file.endswith("_arrays.npz"): # converting an array file
            ftype = 'arrays'
        elif file.endswith("_activations.npz"): # converting an activation map file
            ftype = 'activations'
        else:
            raise Exception(f"Could not determine file structure from file name {file}")
        in_arrs = dict(np.load(file, allow_pickle=True))
        # print()
        total_arrs = len(in_arrs)
        for ii, id in enumerate(in_arrs):
            arr = in_arrs[id]
            if ftype == 'activations':
                grp = f.create_group(name=id) # each triplet has a group, each activation layer is a dataset
                [_, n_maps, _, _] = arr.shape # each array is the shape (num_images, activation_maps, width, height)
                # we want each activation map separated into its own dataset
                for m in range(n_maps):
                    grp.create_dataset(name=f"activation_{m}", data=arr[:,m,:,:])
            elif ftype == 'arrays':
                    f.create_dataset(name=id, data=arr) # each triplet is a dataset
            sys.stdout.write('\r\x1b[K' + f"File: {file_num+1}/{num_files} ({ii+1}/{total_arrs})")
        f.close()
       
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_folder", default=None)
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()
    convert_npz_to_hdf5(in_file=args.input_file, out_folder=args.output_folder, overwrite=args.overwrite)