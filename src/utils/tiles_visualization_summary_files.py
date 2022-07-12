'''
    Program to visualize the images as tile. 
    It is expected that all the images will be same size.

    RKS, 06/06/2022
'''
import os
import csv
import math
import struct
import numpy as np
from array import array
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import argparse
import pydicom
from skimage.measure import block_reduce
import pandas as pd

# # Fixed params
DIV_PIXEL_VALUE = 1
MIN_TEXT_PIXEL_VALUE = 0
MAX_TEXT_PIXEL_VALUE = 16384


class SpriteSheetReader:
    def __init__(self, im, imageName):
        im2 = Image.new("F", (roi_size, roi_size))
        im_i = np.reshape(im, (roi_size,roi_size), order='C')
        im2 = Image.fromarray(im_i)
        #
        # draw = ImageDraw.Draw(im2)
        # # font = ImageFont.truetype("./Arimo-Bold.ttf", 14)
        # font = ImageFont.truetype("./Arimo-Bold.ttf", 8)
        # draw.text((0, 0), os.path.basename(imageName)[5:], (1.0), font=font)
        #
        self.spritesheet = im2
        self.tileSize = roi_size
        self.margin = 1

    def getTile(self, tileX, tileY):
        posX = (self.tileSize * tileX) + (self.margin * (tileX + 1))
        posY = (self.tileSize * tileY) + (self.margin * (tileY + 1))
        box = (posX, posY, posX + self.tileSize, posY + self.tileSize)
        return self.spritesheet.crop(box)

class SpriteSheetWriter:
    def __init__(self, tileSize, spriteSheetSizeW, spriteSheetSizeH):
        self.tileSize = tileSize
        self.spriteSheetSizeW = spriteSheetSizeW
        self.spriteSheetSizeH = spriteSheetSizeH
        self.spritesheet = Image.new("F", (self.spriteSheetSizeW, self.spriteSheetSizeH))
        self.tileX = 0
        self.tileY = 0
        self.margin = 1

    def getCurPos(self):
        self.posX = (self.tileSize * self.tileX) + (self.margin * (self.tileX + 1))
        self.posY = (self.tileSize * self.tileY) + (self.margin * (self.tileY + 1))
        if (self.posX + self.tileSize > self.spriteSheetSizeW):
            self.tileX = 0
            self.tileY = self.tileY + 1
            self.getCurPos()
        if (self.posY + self.tileSize > self.spriteSheetSizeH):
            raise Exception('Image does not fit within spritesheet!')

    def addImage(self, image):
        self.getCurPos()
        destBox = (self.posX, self.posY, self.posX + image.size[0], self.posY + image.size[1])
        self.spritesheet.paste(image, destBox)
        self.tileX = self.tileX + 1

    def show(self):
        self.spritesheet.show()

    def save(self, path_fName):
        self.spritesheet.save(path_fName)


def write_dat_image_to_tile_file_with_title(img, writer, title):
    tile = Image.fromarray(img, mode='I')
    d = ImageDraw.Draw(tile)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 72)
    d.text((0,0), title, fill=MIN_TEXT_PIXEL_VALUE, font=font)
    d.text((0,80), title, fill=MAX_TEXT_PIXEL_VALUE, font=font)
    writer.addImage(tile)


def read_dcm(imageName, ROI_SIZE):
    '''
        Function to read dcm image with options to downsample and 
        fit the size of the tile
    '''
    ds = pydicom.read_file(imageName)
    img = ds.pixel_array
    while img.shape[0] > ROI_SIZE and img.shape[1] > ROI_SIZE:
        img = block_reduce(img, block_size=(2, 2), func=np.mean)
    img = img / float(DIV_PIXEL_VALUE)
    return img.astype(np.int32)


def select_all_images(in_df):
    '''
        By default select all images in the dataframe
    '''
    imgs_lis = []
    for row in in_df['images']:
        imgs_lis += row
    return imgs_lis


def create_files_from_list(args):
    '''
        function that reads a file with list of images 
        along the rows
    '''
    df = pd.read_json(args.in_summary_json, orient='table')
    imgs_list = select_all_images(df)
    print('Num of images in list = ' + str(len(imgs_list)))
    num_splits = int(math.ceil((len(imgs_list)*1)/float(args.max_each_tile)))
    print('There are %d patch files' % len(imgs_list))
    print('Number of splits = %d' % num_splits)
    # # setup the grid
    gridN = np.sqrt(args.max_each_tile)
    gridN = np.ceil(gridN)
    gridN = (gridN * args.roi_size) + gridN + 5

    g_iter = 0
    for i in range(num_splits):
        log_name = os.path.join(os.path.dirname(args.output_tile_name),
                        os.path.basename(args.output_tile_name).rsplit('.', 1)[0] + str(i) + '.log')
        with open(log_name, 'w') as fp:
            writer = SpriteSheetWriter(args.roi_size, int(gridN)*1, int(gridN)*1)
            tiff_name = os.path.join(os.path.dirname(args.output_tile_name),
                        os.path.basename(args.output_tile_name).rsplit('.', 1)[0] + str(i) + '.tiff')
            for j in range(args.max_each_tile):
                if g_iter >= len(imgs_list):
                    break
                fname = imgs_list[g_iter]
                img = read_dcm(fname, args.roi_size)
                write_dat_image_to_tile_file_with_title(img, writer, str(j))
                fp.write('{:d}\t{:s}\n'.format(j, fname))
                g_iter += 1
            writer.save(tiff_name)
            print('Tile {} saved to '.format(i) + tiff_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create tiles from patchlist (.raw format)')
    parser.add_argument('-i', '--in_summary_json', help='input summary json file', required=True)
    parser.add_argument('-o', '--output_tile_name', help='output tile name', required=True)
    parser.add_argument('-m', '--max_each_tile', type=int, help='max patches in each tile', required=True)
    parser.add_argument('-r', '--roi_size', type=int, default=32, help='roi size')
    args = parser.parse_args()

    create_files_from_list(args)
    # # EXAMPLE
    # # python tiles_visualization_summary_files.py -i /gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__MIDRC_RICORD_1C.json -o /gpfs_projects/ravi.samala/OUT/2022_CXR/tiles/RICORD_1C_ -m 30 -r 512

