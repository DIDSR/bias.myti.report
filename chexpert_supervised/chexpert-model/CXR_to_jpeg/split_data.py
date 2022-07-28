import os
import numpy as np
import pandas as pd

def split_data(input_csv, output_loc):
    df = pd.read_csv(input_csv, index_col=0)
    df['split'] = np.random.randn(df.shape[0], 1)
    msk = np.random.rand(len(df)) <= 0.7
    train = df[msk]
    val = df[~msk]
    
    train.columns = train.columns.str.replace('jpeg file', 'Path')
    train = train.drop(['patient id', 'dicom file', 'split'], axis=1)
    
    val.columns = val.columns.str.replace('jpeg file', 'Path')
    val = val.drop(['patient id', 'dicom file', 'split'], axis=1)
       
    train.to_csv(os.path.join(output_loc, "TCIA_1C_train.csv"))
    val.to_csv(os.path.join(output_loc, "TCIA_1C_valid.csv"))
    

if __name__ == '__main__':
    in_csv = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/TCIA_1C_jpeg_summary.csv"
    out_loc = "/gpfs_projects/alexis.burgon/OUT/2022_CXR/RICORD_1c_training/"
    split_data(in_csv, out_loc)