'''
    Custom data loader for data images in a list
'''
import os
import numpy as np
import struct
import math
import random
import os
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
import pandas as pd
from torchvision import transforms

transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])



def read_jpg(imageName):
    """ 
    Reads jpg image and applies the transforms listed in the file (default=rotation).
    
    Arguments
    ==========
    imageName : str
        The file path of the input image
    
    Returns
    =======
    Tensor
    """
    return transform(Image.open(imageName).convert('RGB'))


class Dataset(BaseDataset):
    """ Class for customized dataset 

    Arguments
    =========
    list_file : str
        The file path to the file containing the input information; to be read into a pandas.DataFrame.
    train_flag : bool
        Training process indicator.
    default_out_class : str
        The name of the column in list_file that indicates the model's output class.
    default_patient_id : str
        The name of the column in list_file that indicates the sample's patient id.
    default_path : str
        The name of the column in list_file that indicates the sample's file path.
    
    """
    def __init__(
            self,
            list_file,
            train_flag=True,
            default_out_class='Yes',
            default_patient_id='patient_id',
            default_path='Path'
    ):
        # #
        self.train_flag = train_flag
        # # read the CSV file with header
        df = pd.read_csv(list_file)
        ids = df[default_patient_id].tolist() # # Patient IDs
        dats = df[default_path].tolist()   # # JPEGs
        labels = df[default_out_class].tolist() # # class label

        self.patient_ids, self.images, self.class_values = ids, dats, labels


    def __getitem__(self, i):
        # # implement data augmentation later
        img_o = read_jpg(self.images[i])
        lbl_o = self.class_values[i]

        return self.patient_ids[i], self.images[i], img_o, int(lbl_o)

    def __len__(self):
        return len(self.images)
