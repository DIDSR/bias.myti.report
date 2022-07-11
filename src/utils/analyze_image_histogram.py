'''
    Program to analyze histograms of different data repos.
    The input to the program is a summary file in json format.
    Uses curve fitting.
'''
import argparse
import pydicom
import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
from matplotlib import rcParams
# formatting
colors = seaborn.color_palette("Paired")
rcParams['font.family'] = 'monospace'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 14
hfont = {'fontname':'monospace', 'fontweight':'bold', 'fontsize':18}
rcParams['figure.figsize'] = (8,6)


def get_hist(args):
    df = pd.read_json(args.input_summary_file, orient='table')
    num_img_read = 0
    for each_row in df['images']:
        for each_img in each_row:
            if 'ERROR' not in each_img:
                ds = pydicom.read_file(each_img)
                img = ds.pixel_array
                img_np = np.reshape(img, (1, img.shape[0] * img.shape[1]))
                img_np = img_np[0]
                img_np = img_np[img_np != 0]
                n, bins = np.histogram(img_np, bins=range(0, 4096))
                bins1 = (bins[1:] + bins[:-1]) / 2
                z = np.polyfit(bins1, n, 5)
                f = np.poly1d(z)
                # calculate new x's and y's
                x_new = np.linspace(bins1[0], bins1[-1], len(bins))
                y_new = f(x_new)
                plt.plot(x_new, y_new, 'g--', linewidth=2, alpha=0.1)
                num_img_read += 1
            break   # # one image per patient
        # # for debug
        if num_img_read == 1000:
            break
    plt.title('Num. of images = ' + str(num_img_read))
    # plt.ylim([0, 6000])
    plt.savefig(args.output_figure_file, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate distribution of dicom tags')
    parser.add_argument('-i', '--input_summary_file', 
        help='Input summary file in json format', 
        default='/gpfs_projects/ravi.samala/OUT/2022_CXR/summary_table__open_AI.json')
    parser.add_argument('-o', '--output_figure_file', 
        help='Output figure file in png', 
        default='/gpfs_projects/ravi.samala/OUT/2022_CXR/ImageHist_summary_table__open_AI.png')
    args = parser.parse_args()
    # # 
    get_hist(args)
