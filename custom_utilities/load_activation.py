import numpy as np
import h5py

def load_map(in_file, activation_number):
    """ 
        loads information from an activation file, and collects all images relating to the specified activation number,
        returning a single array where dim 0 represents the number of images.
    """
    f = h5py.File(in_file, 'r')
    total_length = 0
    arrays = []
    for name in f.keys():
        total_length += f[f"{name}/activation_{activation_number}"].shape[0]
        arrays.append(f[f"{name}/activation_{activation_number}"][()])
    print(f"Found {total_length} images")
    out_arr = np.vstack(arrays)
    print("activation map array shape: ", out_arr.shape)
    f.close()
    return out_arr
