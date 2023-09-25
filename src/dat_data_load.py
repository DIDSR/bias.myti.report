'''
    Custom data loader for dat images in a list
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

# def read_dat_rot(imageName, rot_flag, custom_scale):
#     '''
#         custom function to read dat files, provide ratation-based augmentation and custom scaling
#     '''
#     with open(imageName, "rb") as f:
#         bytes = f.read(4)
#         size = struct.unpack('<HH', bytes)
#         wd = size[0]
#         ht = size[1]
#         widthToRead = int(math.floor((wd + 1) / 2) * 2)
#         heightToRead = int(math.floor((ht + 1) / 2) * 2)
#         argm = "=" + str((widthToRead) - 2) + "h"
#         hd = struct.unpack(argm, (f.read((heightToRead-2) * 2)))
#         argm = "=" + str(widthToRead * heightToRead) + "h"
#         img = list(struct.unpack(argm, (f.read(widthToRead * heightToRead * 2))))

#     npA = np.array(img)
#     npAr1 = np.reshape(npA, (widthToRead, heightToRead), order='C')
#     npAr2 = npAr1.copy()
#     if 1 <= rot_flag <= 3:
#         npAr2 = np.rot90(npAr1, rot_flag)
#     elif rot_flag == 4:
#         npAr2 = np.fliplr(npAr1)
#     elif 5 <= rot_flag <= 7:
#         npAr5 = np.fliplr(npAr1)
#         npAr2 = np.rot90(npAr5, rot_flag-4)
#     npAr3 = np.zeros((3, widthToRead, heightToRead))
#     npAr3[0, :, :] = npAr2[:, :]
#     npAr3[1, :, :] = npAr2[:, :]
#     npAr3[2, :, :] = npAr2[:, :]
#     npAr3 = npAr3.astype(np.float32)
#     # # special
#     if custom_scale:
#         # # custom scale only for BG corrected mammo masses
#         # # adjust this if the task changes
#         npAr3 = (npAr3 - 400)/(2000.0 - 400.0)
#         npAr3[npAr3 < 0] = 0
#         npAr3[npAr3 > 1] = 1
#     # # <<
#     return npAr3


def read_jpg(imageName):
    '''
    function to read jpg image with rotation enabled for 
    data augmentation
    '''
    return transform(Image.open(imageName).convert('RGB'))


class Dataset(BaseDataset):
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
        print(df.columns.values.tolist())
        ids = df[default_patient_id].tolist() # # Patient IDs
        dats = df[default_path].tolist()   # # JPEGs
        labels = df[default_out_class].tolist() # # class label

        c = list(zip(ids, dats, labels))
        # # Randomize the list
        if train_flag:
            random.shuffle(c)
        self.patient_ids, self.images, self.class_values = zip(*c)


    def __getitem__(self, i):
        # # implement data augmentation later
        img_o = read_jpg(self.images[i])
        lbl_o = self.class_values[i]

        return self.patient_ids[i], self.images[i], img_o, int(lbl_o)

    def __len__(self):
        return len(self.images)
