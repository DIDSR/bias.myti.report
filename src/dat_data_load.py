from __future__ import division
import os
import numpy as np
import struct
import math
import random
import os
import sys
from skimage.measure import block_reduce
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
# #
# #

# #
# #
def read_dat_rot(imageName, rot_flag, custom_scale):
    with open(imageName, "rb") as f:
        bytes = f.read(4)
        size = struct.unpack('<HH', bytes)
        wd = size[0]
        ht = size[1]
        widthToRead = int(math.floor((wd + 1) / 2) * 2)
        heightToRead = int(math.floor((ht + 1) / 2) * 2)
        argm = "=" + str((widthToRead) - 2) + "h"
        hd = struct.unpack(argm, (f.read((heightToRead-2) * 2)))
        argm = "=" + str(widthToRead * heightToRead) + "h"
        img = list(struct.unpack(argm, (f.read(widthToRead * heightToRead * 2))))

    npA = np.array(img)
    npAr1 = np.reshape(npA, (widthToRead, heightToRead), order='C')
    npAr2 = npAr1.copy()
    if 1 <= rot_flag <= 3:
        npAr2 = np.rot90(npAr1, rot_flag)
    elif rot_flag == 4:
        npAr2 = np.fliplr(npAr1)
    elif 5 <= rot_flag <= 7:
        npAr5 = np.fliplr(npAr1)
        npAr2 = np.rot90(npAr5, rot_flag-4)
    npAr3 = np.zeros((3, widthToRead, heightToRead))
    npAr3[0, :, :] = npAr2[:, :]
    npAr3[1, :, :] = npAr2[:, :]
    npAr3[2, :, :] = npAr2[:, :]
    npAr3 = npAr3.astype(np.float32)
    # # special
    if custom_scale:
        # # custom scale only for BG corrected mammo masses
        # # adjust this if the task changes
        npAr3 = (npAr3 - 400)/(2000.0 - 400.0)
        npAr3[npAr3 < 0] = 0
        npAr3[npAr3 > 1] = 1
    # # <<
    return npAr3


class Dataset(BaseDataset):
    def __init__(
            self,
            list_file,
            crop_to_224=True,
            train_flag=True,
            custom_scale=False
    ):
        # #
        self.crop_to_224 = crop_to_224
        self.train_flag = train_flag
        self.custom_scale = custom_scale
        lines = open(list_file).readlines()
        info = np.array([t.rstrip().split('\t') for t in lines])
        dats = info[:, 0]
        labels = info[:, 1]

        # # Randomize the list
        c = list(zip(dats, labels))
        if train_flag:
            random.shuffle(c)
        self.images, self.class_values = zip(*c)


    def __getitem__(self, i):
        # #
        if self.train_flag:
            rot_flag = random.randint(0, 7)
            img_o = read_dat_rot(self.images[i], rot_flag, self.custom_scale)
        else:
            img_o = read_dat_rot(self.images[i], 0, self.custom_scale)
        lbl_o = self.class_values[i]
        # #
        if self.crop_to_224:
            if self.train_flag:
                x = random.randint(0, 256 - 224 - 1)
                y = random.randint(0, 256 - 224 - 1)
                img_o = img_o[:, x:x + 224, y:y + 224]
            else:
                img_o = img_o[:, 16:16 + 224, 16:16 + 224]

        return os.path.basename(self.images[i]), img_o, int(lbl_o)

    def __len__(self):
        return len(self.images)


