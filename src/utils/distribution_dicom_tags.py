'''
    Program that generates distribution of selected dicom tags from a data repo

'''
import seaborn
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import os
import argparse
import pydicom
from collections import Counter

# formatting
colors = seaborn.color_palette("Paired")
rcParams['font.family'] = 'monospace'
rcParams['font.weight'] = 'bold'
rcParams['font.size'] = 14
hfont = {'fontname':'monospace', 'fontweight':'bold', 'fontsize':18}
rcParams['figure.figsize'] = (8,6)


def searchthis(location, searchterm):
	lis_paths = []
	for dir_path, dirs, file_names in os.walk(location):
		for file_name in file_names:
			fullpath = os.path.join(dir_path, file_name)
			if searchterm in fullpath:
				lis_paths += [fullpath]
	return lis_paths


def get_dist(in_dir, repo_name, dicom_tags, out_dir):
    dcm_files = searchthis(in_dir, '.dcm')
    print('There are {:d} .dcm files'.format(len(dcm_files)))
    tag_acc = {}
    for each_tag in dicom_tags:
        tag_acc[each_tag] = []
    for each_dcm_file in dcm_files:
        ds = pydicom.read_file(each_dcm_file)
        for each_tag in dicom_tags:
            if each_tag in ds:
                # print(ds[each_tag].name)
                tag_acc[each_tag] = tag_acc[each_tag] + [ds[each_tag].value.strip()]
    # print(tag_acc)
    for each_tag in dicom_tags:
        print(ds[each_tag].name.replace)
        out_fig_path_name = os.path.join(out_dir, repo_name + '__' + ds[each_tag].name.replace(' ', '_') + '.png')
        out_json_path_name = os.path.join(out_dir, repo_name + '__' + ds[each_tag].name.replace(' ', '_') + '.json')
        # pd.Series(tag_acc[each_tag]).value_counts(sort=False).plot(kind='bar')
        hist_pd = pd.Series(tag_acc[each_tag]).value_counts(sort=True)
        hist_pd.plot(kind='bar')
        plt.savefig(out_fig_path_name, dpi=300, bbox_inches="tight")
        print(hist_pd)
        # hist_df = hist_pd.rename_axis('unique_values').to_frame('counts')
        hist_df = hist_pd.rename_axis('unique_values').reset_index(name='counts')
        print(hist_df)
        hist_df.to_json(out_json_path_name, indent=4, orient='table', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate distribution of dicom tags')
    # parser.add_argument('-i', '--input_dir', help='Input dir', default='/home/ravi.samala/DATA/temp/open_AI_unzip/')
    # parser.add_argument('-i', '--input_dir', help='Input dir', default='/gpfs_projects/common_data/TCIA/COVID_19_AR/manifest-1594658036421/COVID-19-AR/')
    # parser.add_argument('-i', '--input_dir', help='Input dir', default='/gpfs_projects/common_data/TCIA/COVID_19_NY_SBU/manifest-1628608914773/COVID-19-NY-SBU/')
    # parser.add_argument('-i', '--input_dir', help='Input dir', default='/gpfs_projects/common_data/MIDRC/Release_1c/manifest-1610656454899/MIDRC-RICORD-1C')
    # parser.add_argument('-i', '--input_dir', help='Input dir', default='/gpfs_projects/ravi.samala/DATA/MIDRC2/open_AI_unzip/')
    parser.add_argument('-i', '--input_dir', help='Input dir', default='/gpfs_projects/ravi.samala/DATA/MIDRC2/open_RI_unzip/')
    parser.add_argument('-r', '--repo_name', help='name of the data repository', default='open_RI')
    parser.add_argument('-d', '--dicom_tags', action='append', help='List of dicom tags to track', default=[(0x0008, 0x1030), (0x0008, 0x0060)])
    parser.add_argument('-o', '--output_dir', action='append', help='Output dir to save figures', default='/gpfs_projects/ravi.samala/OUT/2022_CXR/')
    args = parser.parse_args()
    # # 
    print(args.input_dir)
    get_dist(args.input_dir, args.repo_name, args.dicom_tags, args.output_dir)

