#!/bin/bash

# IN_FILE can either be a single .npz to be converted or a folder containing multiple .npz files, which it will iterate through
IN_FILE=""

## OVERWRITING:
    # By default, the program will not overwrite existing hdf5 files, instead skipping any file that would overwrite.
    # If it is desired for the files to be overwritten, pass the additional argument --overwrite
## OUTPUT LOCATION:
    # By default, the new hdf5 files will be created in the same folder as the input .npz file.
    # If a different location is desired, a destination folder can be passed with the "-o" or "--output_folder" arguments

python ../src/convert_npz.py -i ${IN_FILE}