'''
    Program to read the .npz files from the decision boundary output code
    and analyse the dense layer o/p using UMAP analysis

    WARNING: the code contains a lot of hard coded variables
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import umap
import os
from sklearn import manifold
import plotly.express as px
import plotly.graph_objects as px
import random
# #
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
# #
# # -----------------------------------------------------------------------------------------
in_dir = '/home/ravi.samala/temp/RAND_0/CHEXPERT_RESNET_0__step_0/Valid_ResNet18_layer4_0__downsample_1/'
out_dir = '/home/ravi.samala/temp/RAND_0/CHEXPERT_RESNET_0__step_0/Valid_ResNet18_layer4_0__downsample_1/'
# # Fixed params, the program expects the following files in the "in_dir"
samples_csv = 'DB_samples.csv'
orig_sample_npz = 'all_orig__activation_maps.npz'
subgroups_dense_npz = {
                        'MWP': 'MWP_arrays__activations.npz',
                        'MWN': 'MWN_arrays__activations.npz',
                        'MBP': 'MBP_arrays__activations.npz',
                        'MBN': 'MBN_arrays__activations.npz',
                        'FWP': 'FWP_arrays__activations.npz',
                        'FWN': 'FWN_arrays__activations.npz',
                        'FBP': 'FBP_arrays__activations.npz',
                        'FBN': 'FBN_arrays__activations.npz',
                        }
# # other plot related params
vicinal_marker_size = 1
original_marker_size = 1
reduction_technique = 'UMAP' # # options: 
                             # # "UMAP", "LLE" Locally linear embedding, "ISO", "MDS" Multidimensional scaling
                             # # "SE" Spectral embedding, "TSNE"
# # -----------------------------------------------------------------------------------------
pal = sns.color_palette("tab10")
clr_orig = pal.as_hex()
# pal = sns.color_palette("husl")
clr_vici = pal.as_hex()
# #
# #
def color_map_color(lst, cmap_name='Wistia', vmin=0, vmax=1):
    minima = min(lst)
    maxima = max(lst)

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    cmp = ListedColormap(['#CC6677', '#88CCEE'])
    mapper = cm.ScalarMappable(norm=norm, cmap=cmp)
    color = []
    for v in lst:
        color.append(mapper.to_rgba(v))
    return color


def do_embedding(merge_np, reduction_technique):
    embedding = []
    if reduction_technique == "UMAP":
        # # UMAP
        reducer2 = umap.UMAP(n_neighbors=30, min_dist=0.001, n_components=3, metric='cosine', random_state=2023)
        embedding = reducer2.fit_transform(merge_np)
    elif reduction_technique == "LLE":
        # # LLE
        params = {
            "n_neighbors": 10,
            "n_components": 2,
            "eigen_solver": "auto",
        }
        # lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
        # lle_standard = manifold.LocallyLinearEmbedding(method="ltsa", **params)
        # lle_standard = manifold.LocallyLinearEmbedding(method="hessian", **params)
        lle_standard = manifold.LocallyLinearEmbedding(method="modified", modified_tol=0.8, **params)
        embedding = lle_standard.fit_transform(merge_np)
    elif reduction_technique == 'ISO':
        isomap = manifold.Isomap(n_neighbors=10, n_components=2, p=1)
        embedding = isomap.fit_transform(merge_np)
    elif reduction_technique == 'MDS':
        md_scaling = manifold.MDS(n_components=2, max_iter=50, n_init=4)
        embedding = md_scaling.fit_transform(merge_np)
    elif reduction_technique == 'SE':
        spectral = manifold.SpectralEmbedding(n_components=2, n_neighbors=10)
        embedding = spectral.fit_transform(merge_np)
    elif reduction_technique == 'TSNE':
        tsne = manifold.TSNE(n_components=2, perplexity=30, n_iter=250, init="random",)
        embedding = tsne.fit_transform(merge_np.astype(np.float32))
    else:
        print('UNKNOWN option for dimensionality reduction technique')
    return embedding

# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # Read the original samples with subgroup labelled information
df = pd.read_csv(os.path.join(in_dir, samples_csv))
# # Read the activation maps of the original deployed samples
# # This var will contain 'base_paths', 'base_groundtruth' and 'base_activation_maps'
orig_DN_arrays = dict(np.load(os.path.join(in_dir, orig_sample_npz), allow_pickle=True))
# print(orig_DN_arrays['base_paths'].shape)
# print(orig_DN_arrays['base_last_dense_layer'].shape)
# print(orig_DN_arrays['base_groundtruth'].shape)
orig_DN_labels = [int(i[0]) for i in orig_DN_arrays['base_groundtruth']]
orig_marker_size = original_marker_size * np.ones(len(orig_DN_labels))
orig_DN_names = [i[2] for i in orig_DN_arrays['base_groundtruth']]
# print(orig_DN_names)
# # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# #
# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # Iterate over the subgroups, extract the relevant subgroup
# # information from the orig_DN_arrays, read vicinal distribution files
# # merge with the vicinal distribution
# # and perform UMAP analysis
vicinal_subgroups = {}
original_subgroups_base_paths = {}
original_subgroups = {}
# vicinal_subgroups_color = {}
# original_subgroups_color = {}
fig1 = plt.figure()
for idx, each_subgroup_key in enumerate(subgroups_dense_npz):
    print('------------------------------------------------')
    print(each_subgroup_key)
    # # extract the relevant subgroup from the DB_samples
    if each_subgroup_key[0] == 'M':
        first_subgroup_index = 'M'
    elif each_subgroup_key[0] == 'F':
        first_subgroup_index = 'F'
    if each_subgroup_key[1] == 'W':
        second_subgroup_index = 'White'
    elif each_subgroup_key[1] == 'B':
        # second_subgroup_index = 'Black_or_African_American'
        second_subgroup_index = 'Black'
    if each_subgroup_key[2] == 'P':
        third_subgroup_index = 'Yes'
    elif each_subgroup_key[2] == 'N':
        third_subgroup_index = 'No'
    # # extract the original subgroup data
    orig_subgroup_df = df[(df[first_subgroup_index] == 1) & (df[second_subgroup_index] == 1) & (df[third_subgroup_index] == 1)]
    current_subgroup_samples = orig_subgroup_df['Path'].tolist()
    # # select the dense layer feature samples based on the above selected subgroup
    orig_indxs = [i for i, x in enumerate(orig_DN_names) if x in current_subgroup_samples]
    print("There are {} samples in DB_samples.csv".format(len(current_subgroup_samples)))
    print("There are {} samples matching all_orig__last_dense".format(len(orig_indxs)))
    # print(orig_subgroup_df)
    # print(orig_DN_arrays['base_activation_maps'][orig_indxs])
    print(orig_DN_arrays['base_activation_maps'][orig_indxs].shape)
    
    # # select a random activation map per image
    orig_DN_arrays_subgroup = orig_DN_arrays['base_activation_maps'][orig_indxs]
    [m, n, p, q] = orig_DN_arrays_subgroup.shape
    orig_DN_arrays_rand_map = np.zeros((m, 1, p, q))
    for each_img in range(m):
        orig_DN_arrays_rand_map[each_img, 0, :, :] = orig_DN_arrays_subgroup[each_img, random.randint(0, n-1), :, :]
    # # reshape
    print(orig_DN_arrays_rand_map.shape)
    orig_DN_arrays_rand_map = np.squeeze(orig_DN_arrays_rand_map, axis=1)
    print(orig_DN_arrays_rand_map.shape)
    # orig_DN_arrays_rand_map = orig_DN_arrays_rand_map.reshape(*orig_DN_arrays_rand_map.shape[:-2], -1)
    orig_DN_arrays_rand_map = orig_DN_arrays_rand_map.reshape((m, p*q))
    print(orig_DN_arrays_rand_map.shape)
    # #
    # # Read the vicinal distribution
    vici_DN_arrays = dict(np.load(os.path.join(in_dir, subgroups_dense_npz[each_subgroup_key]), allow_pickle=True))
    print('subgroup pkl file')
    print("There are {} vicinal distributions".format(len(vici_DN_arrays)))
    triplet_indx = []
    feat_np = []
    for key, value in vici_DN_arrays.items():
        triplet_indx.append(key)
        if len(feat_np) == 0:
            feat_np = value
        else:
            feat_np = np.vstack((feat_np, value))
    
    # # select a random activation map per image
    [m, n, p, q] = feat_np.shape
    feat_np2 = np.zeros((m, 1, p, q))
    for each_img in range(m):
        feat_np2[each_img, 0, :, :] = feat_np[each_img, random.randint(0, n-1), :, :]
    print(feat_np2.shape)
    feat_np2 = np.squeeze(feat_np2, axis=1)
    print(feat_np2.shape)
    # feat_np2 = feat_np2.reshape(*feat_np2.shape[:-2], -1)
    feat_np2 = feat_np2.reshape((m, p*q))
    print(feat_np2.shape)
    print("(vicinal) Shape of the subgroup dense layer: {}".format(feat_np2.shape))
    merge_np = np.vstack((feat_np2, orig_DN_arrays_rand_map))
    vicinal_subgroups[each_subgroup_key] = feat_np2
    original_subgroups[each_subgroup_key] = orig_DN_arrays_rand_map
    original_subgroups_base_paths[each_subgroup_key] = orig_DN_arrays['base_paths'][orig_indxs]
    print(merge_np.shape)
    print("(merged) Shape of the subgroup dense layer: {}".format(merge_np.shape))

# #-------------------------------------------
# # plot the aggregate
# #-------------------------------------------
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
size_orig_idx = [0]
size_vici_idx = [0]
# size_both_idx = [[0, 0]]
size_both_idx = []
prevA = 0
for i, each_subgroup in enumerate(original_subgroups):
    if i == 0:
        merge_dat_orig = original_subgroups[each_subgroup]
        # merge_original_subgroups_base_paths = original_subgroups_base_paths[each_subgroup]
        merge_dat_vici = vicinal_subgroups[each_subgroup]
        merge_dat_both = original_subgroups[each_subgroup]
        merge_dat_both = np.vstack((merge_dat_both, vicinal_subgroups[each_subgroup]))
        size_both_idx = [[0, merge_dat_orig.shape[0], merge_dat_orig.shape[0], merge_dat_orig.shape[0]+merge_dat_vici.shape[0]]]
        prevA = merge_dat_orig.shape[0] + merge_dat_vici.shape[0]
    else:
        merge_dat_orig = np.vstack((merge_dat_orig, original_subgroups[each_subgroup]))
        # merge_original_subgroups_base_paths = np.hstack((merge_original_subgroups_base_paths, original_subgroups_base_paths[each_subgroup]))
        merge_dat_vici = np.vstack((merge_dat_vici, vicinal_subgroups[each_subgroup]))
        merge_dat_both = np.vstack((merge_dat_both, original_subgroups[each_subgroup]))
        merge_dat_both = np.vstack((merge_dat_both, vicinal_subgroups[each_subgroup]))
        size_both_idx.append([prevA, prevA + original_subgroups[each_subgroup].shape[0], prevA + original_subgroups[each_subgroup].shape[0], prevA + original_subgroups[each_subgroup].shape[0] + vicinal_subgroups[each_subgroup].shape[0]])
        prevA = prevA + original_subgroups[each_subgroup].shape[0] + vicinal_subgroups[each_subgroup].shape[0]
    size_orig_idx.append(merge_dat_orig.shape[0])
    size_vici_idx.append(merge_dat_vici.shape[0])
    # size_both_idx.append([merge_dat_orig.shape[0], merge_dat_vici.shape[0]])
print(size_orig_idx)
print(size_vici_idx)
print(size_both_idx)

# # ORIG
embedding2_orig = do_embedding(merge_dat_orig, reduction_technique)
print(embedding2_orig.shape)
embedding2_vicinal = do_embedding(merge_dat_vici, reduction_technique)
print(embedding2_vicinal.shape)
embedding2_both = do_embedding(merge_dat_both, reduction_technique)
print(embedding2_both.shape)

# # PLOTLY - ORIG
plot=px.Figure(data=[
    px.Scatter3d(x=embedding2_orig[size_orig_idx[0]:size_orig_idx[0+1], 0],y=embedding2_orig[size_orig_idx[0]:size_orig_idx[0+1], 1], z=embedding2_orig[size_orig_idx[0]:size_orig_idx[0+1], 2], mode='markers', marker = dict(size = 3,color = clr_orig[0],opacity = 0.8, symbol='square'), name='MWP'),
    px.Scatter3d(x=embedding2_orig[size_orig_idx[1]:size_orig_idx[1+1], 0],y=embedding2_orig[size_orig_idx[1]:size_orig_idx[1+1], 1], z=embedding2_orig[size_orig_idx[1]:size_orig_idx[1+1], 2], mode='markers', marker = dict(size = 3,color = clr_orig[1],opacity = 0.8, symbol='diamond'), name='MWN'),
    px.Scatter3d(x=embedding2_orig[size_orig_idx[2]:size_orig_idx[2+1], 0],y=embedding2_orig[size_orig_idx[2]:size_orig_idx[2+1], 1], z=embedding2_orig[size_orig_idx[2]:size_orig_idx[2+1], 2], mode='markers', marker = dict(size = 3,color = clr_orig[2],opacity = 0.8, symbol='square'), name='MBP'),
    px.Scatter3d(x=embedding2_orig[size_orig_idx[3]:size_orig_idx[3+1], 0],y=embedding2_orig[size_orig_idx[3]:size_orig_idx[3+1], 1], z=embedding2_orig[size_orig_idx[3]:size_orig_idx[3+1], 2], mode='markers', marker = dict(size = 3,color = clr_orig[3],opacity = 0.8, symbol='diamond'), name='MBN'),
    px.Scatter3d(x=embedding2_orig[size_orig_idx[4]:size_orig_idx[4+1], 0],y=embedding2_orig[size_orig_idx[4]:size_orig_idx[4+1], 1], z=embedding2_orig[size_orig_idx[4]:size_orig_idx[4+1], 2], mode='markers', marker = dict(size = 3,color = clr_orig[4],opacity = 0.8, symbol='square'), name='FWP'),
    px.Scatter3d(x=embedding2_orig[size_orig_idx[5]:size_orig_idx[5+1], 0],y=embedding2_orig[size_orig_idx[5]:size_orig_idx[5+1], 1], z=embedding2_orig[size_orig_idx[5]:size_orig_idx[5+1], 2], mode='markers', marker = dict(size = 3,color = clr_orig[5],opacity = 0.8, symbol='diamond'), name='FWN'),
    px.Scatter3d(x=embedding2_orig[size_orig_idx[6]:size_orig_idx[6+1], 0],y=embedding2_orig[size_orig_idx[6]:size_orig_idx[6+1], 1], z=embedding2_orig[size_orig_idx[6]:size_orig_idx[6+1], 2], mode='markers', marker = dict(size = 3,color = clr_orig[6],opacity = 0.8, symbol='square'), name='FBP'),
    px.Scatter3d(x=embedding2_orig[size_orig_idx[7]:size_orig_idx[7+1], 0],y=embedding2_orig[size_orig_idx[7]:size_orig_idx[7+1], 1], z=embedding2_orig[size_orig_idx[7]:size_orig_idx[7+1], 2], mode='markers', marker = dict(size = 3,color = clr_orig[7],opacity = 0.8, symbol='diamond'), name='FBN'),
    ])
# Add dropdown
plot.update_layout(legend=dict(title_font_family="arial black",font=dict(size=24), itemsizing='constant'),
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(label="All",
                     method="update",
                     args=[{"visible": [True, True, True, True, True, True, True, True]},
                           {"title": "All"}]),
                dict(label="MW (P,N)",
                     method="update",
                     args=[{"visible": [True, True, False, False, False, False, False, False]},
                           {"title": "MW (P,N)",
                            }]),
                dict(label="MB (P,N)",
                     method="update",
                     args=[{"visible": [False, False, True, True, False, False, False, False]},
                           {"title": "MB (P,N)",
                            }]),
                dict(label="FW (P,N)",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, False, False]},
                           {"title": "FW (P,N)",
                            }]),
                dict(label="FB (P,N)",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, True]},
                           {"title": "FB (P,N)",
                            }]),
            ]),
        )
    ])
  
plot.write_html(os.path.join(out_dir, reduction_technique + '_orig_out.html'), full_html=False, include_plotlyjs='cdn')
# plot.show()

# # PLOTLY - VICINAL
plot=px.Figure(data=[
    px.Scatter3d(x=embedding2_vicinal[size_vici_idx[0]:size_vici_idx[0+1], 0],y=embedding2_vicinal[size_vici_idx[0]:size_vici_idx[0+1], 1], z=embedding2_vicinal[size_vici_idx[0]:size_vici_idx[0+1], 2], mode='markers', marker = dict(size = 3,color = clr_vici[0],opacity = 0.8, symbol='square'), name='MWP'),
    px.Scatter3d(x=embedding2_vicinal[size_vici_idx[1]:size_vici_idx[1+1], 0],y=embedding2_vicinal[size_vici_idx[1]:size_vici_idx[1+1], 1], z=embedding2_vicinal[size_vici_idx[1]:size_vici_idx[1+1], 2], mode='markers', marker = dict(size = 3,color = clr_vici[1],opacity = 0.8, symbol='diamond'), name='MWN'),
    px.Scatter3d(x=embedding2_vicinal[size_vici_idx[2]:size_vici_idx[2+1], 0],y=embedding2_vicinal[size_vici_idx[2]:size_vici_idx[2+1], 1], z=embedding2_vicinal[size_vici_idx[2]:size_vici_idx[2+1], 2], mode='markers', marker = dict(size = 3,color = clr_vici[2],opacity = 0.8, symbol='square'), name='MBP'),
    px.Scatter3d(x=embedding2_vicinal[size_vici_idx[3]:size_vici_idx[3+1], 0],y=embedding2_vicinal[size_vici_idx[3]:size_vici_idx[3+1], 1], z=embedding2_vicinal[size_vici_idx[3]:size_vici_idx[3+1], 2], mode='markers', marker = dict(size = 3,color = clr_vici[3],opacity = 0.8, symbol='diamond'), name='MBN'),
    px.Scatter3d(x=embedding2_vicinal[size_vici_idx[4]:size_vici_idx[4+1], 0],y=embedding2_vicinal[size_vici_idx[4]:size_vici_idx[4+1], 1], z=embedding2_vicinal[size_vici_idx[4]:size_vici_idx[4+1], 2], mode='markers', marker = dict(size = 3,color = clr_vici[4],opacity = 0.8, symbol='square'), name='FWP'),
    px.Scatter3d(x=embedding2_vicinal[size_vici_idx[5]:size_vici_idx[5+1], 0],y=embedding2_vicinal[size_vici_idx[5]:size_vici_idx[5+1], 1], z=embedding2_vicinal[size_vici_idx[5]:size_vici_idx[5+1], 2], mode='markers', marker = dict(size = 3,color = clr_vici[5],opacity = 0.8, symbol='diamond'), name='FWN'),
    px.Scatter3d(x=embedding2_vicinal[size_vici_idx[6]:size_vici_idx[6+1], 0],y=embedding2_vicinal[size_vici_idx[6]:size_vici_idx[6+1], 1], z=embedding2_vicinal[size_vici_idx[6]:size_vici_idx[6+1], 2], mode='markers', marker = dict(size = 3,color = clr_vici[6],opacity = 0.8, symbol='square'), name='FBP'),
    px.Scatter3d(x=embedding2_vicinal[size_vici_idx[7]:size_vici_idx[7+1], 0],y=embedding2_vicinal[size_vici_idx[7]:size_vici_idx[7+1], 1], z=embedding2_vicinal[size_vici_idx[7]:size_vici_idx[7+1], 2], mode='markers', marker = dict(size = 3,color = clr_vici[7],opacity = 0.8, symbol='diamond'), name='FBN'),
    ])
# Add dropdown
plot.update_layout(legend=dict(title_font_family="arial black",font=dict(size=24), itemsizing='constant'),
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(label="All",
                     method="update",
                     args=[{"visible": [True, True, True, True, True, True, True, True]},
                           {"title": "All"}]),
                dict(label="MW (P,N)",
                     method="update",
                     args=[{"visible": [True, True, False, False, False, False, False, False]},
                           {"title": "MW (P,N)",
                            }]),
                dict(label="MB (P,N)",
                     method="update",
                     args=[{"visible": [False, False, True, True, False, False, False, False]},
                           {"title": "MB (P,N)",
                            }]),
                dict(label="FW (P,N)",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, False, False]},
                           {"title": "FW (P,N)",
                            }]),
                dict(label="FB (P,N)",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, True, True]},
                           {"title": "FB (P,N)",
                            }]),
            ]),
        )
    ])
  
plot.write_html(os.path.join(out_dir, reduction_technique + '_vici_out.html'), full_html=False, include_plotlyjs='cdn')


# # PLOTLY - BOTH
# # red - ORIG-P, orange - VICI-P
# # green - ORIG-N, blue - VICI-N
plot=px.Figure(data=[
    # # MWP
    px.Scatter3d(x=embedding2_both[size_both_idx[0][0]:size_both_idx[0][1], 0],y=embedding2_both[size_both_idx[0][0]:size_both_idx[0][1], 1], z=embedding2_both[size_both_idx[0][0]:size_both_idx[0][1], 2], mode='markers', marker = dict(size = 5,color = 'red',opacity = 0.8, symbol='square'), name='MWP - ORIG'),
    px.Scatter3d(x=embedding2_both[size_both_idx[0][2]:size_both_idx[0][3], 0],y=embedding2_both[size_both_idx[0][2]:size_both_idx[0][3], 1], z=embedding2_both[size_both_idx[0][2]:size_both_idx[0][3], 2], mode='markers', marker = dict(size = 3,color = 'orange',opacity = 0.5), name='MWP - VICI'),
    # # MWN
    px.Scatter3d(x=embedding2_both[size_both_idx[1][0]:size_both_idx[1][1], 0],y=embedding2_both[size_both_idx[1][0]:size_both_idx[1][1], 1], z=embedding2_both[size_both_idx[1][0]:size_both_idx[1][1], 2], mode='markers', marker = dict(size = 5,color = 'green',opacity = 0.8, symbol='diamond'), name='MWN - ORIG'),
    px.Scatter3d(x=embedding2_both[size_both_idx[1][2]:size_both_idx[1][3], 0],y=embedding2_both[size_both_idx[1][2]:size_both_idx[1][3], 1], z=embedding2_both[size_both_idx[1][2]:size_both_idx[1][3], 2], mode='markers', marker = dict(size = 3,color = 'blue',opacity = 0.3), name='MWN - VICI'),
    # # MBP
    px.Scatter3d(x=embedding2_both[size_both_idx[2][0]:size_both_idx[2][1], 0],y=embedding2_both[size_both_idx[2][0]:size_both_idx[2][1], 1], z=embedding2_both[size_both_idx[2][0]:size_both_idx[2][1], 2], mode='markers', marker = dict(size = 5,color = 'red',opacity = 0.8, symbol='square'), name='MBP - ORIG'),
    px.Scatter3d(x=embedding2_both[size_both_idx[2][2]:size_both_idx[2][3], 0],y=embedding2_both[size_both_idx[2][2]:size_both_idx[2][3], 1], z=embedding2_both[size_both_idx[2][2]:size_both_idx[2][3], 2], mode='markers', marker = dict(size = 3,color = 'orange',opacity = 0.3), name='MBP - VICI'),
    # # MBN
    px.Scatter3d(x=embedding2_both[size_both_idx[3][0]:size_both_idx[3][1], 0],y=embedding2_both[size_both_idx[3][0]:size_both_idx[3][1], 1], z=embedding2_both[size_both_idx[3][0]:size_both_idx[3][1], 2], mode='markers', marker = dict(size = 5,color = 'green',opacity = 0.8, symbol='diamond'), name='MBN - ORIG'),
    px.Scatter3d(x=embedding2_both[size_both_idx[3][2]:size_both_idx[3][3], 0],y=embedding2_both[size_both_idx[3][2]:size_both_idx[3][3], 1], z=embedding2_both[size_both_idx[3][2]:size_both_idx[3][3], 2], mode='markers', marker = dict(size = 3,color = 'blue',opacity = 0.3), name='MBN - VICI'),
    # #
    # # FWP
    px.Scatter3d(x=embedding2_both[size_both_idx[4][0]:size_both_idx[4][1], 0],y=embedding2_both[size_both_idx[4][0]:size_both_idx[4][1], 1], z=embedding2_both[size_both_idx[4][0]:size_both_idx[4][1], 2], mode='markers', marker = dict(size = 5,color = 'red',opacity = 0.8, symbol='square'), name='FWP - ORIG'),
    px.Scatter3d(x=embedding2_both[size_both_idx[4][2]:size_both_idx[4][3], 0],y=embedding2_both[size_both_idx[4][2]:size_both_idx[4][3], 1], z=embedding2_both[size_both_idx[4][2]:size_both_idx[4][3], 2], mode='markers', marker = dict(size = 3,color = 'orange',opacity = 0.3), name='FWP - VICI'),
    # # FWN
    px.Scatter3d(x=embedding2_both[size_both_idx[5][0]:size_both_idx[5][1], 0],y=embedding2_both[size_both_idx[5][0]:size_both_idx[5][1], 1], z=embedding2_both[size_both_idx[5][0]:size_both_idx[5][1], 2], mode='markers', marker = dict(size = 5,color = 'green',opacity = 0.8, symbol='diamond'), name='FWN - ORIG'),
    px.Scatter3d(x=embedding2_both[size_both_idx[5][2]:size_both_idx[5][3], 0],y=embedding2_both[size_both_idx[5][2]:size_both_idx[5][3], 1], z=embedding2_both[size_both_idx[5][2]:size_both_idx[5][3], 2], mode='markers', marker = dict(size = 3,color = 'blue',opacity = 0.3), name='FWN - VICI'),
    # # FBP
    px.Scatter3d(x=embedding2_both[size_both_idx[6][0]:size_both_idx[6][1], 0],y=embedding2_both[size_both_idx[6][0]:size_both_idx[6][1], 1], z=embedding2_both[size_both_idx[6][0]:size_both_idx[6][1], 2], mode='markers', marker = dict(size = 5,color = 'red',opacity = 0.8, symbol='square'), name='FBP - ORIG'),
    px.Scatter3d(x=embedding2_both[size_both_idx[6][2]:size_both_idx[6][3], 0],y=embedding2_both[size_both_idx[6][2]:size_both_idx[6][3], 1], z=embedding2_both[size_both_idx[6][2]:size_both_idx[6][3], 2], mode='markers', marker = dict(size = 3,color = 'orange',opacity = 0.3), name='FBP - VICI'),
    # # FBN
    px.Scatter3d(x=embedding2_both[size_both_idx[7][0]:size_both_idx[7][1], 0],y=embedding2_both[size_both_idx[7][0]:size_both_idx[7][1], 1], z=embedding2_both[size_both_idx[7][0]:size_both_idx[7][1], 2], mode='markers', marker = dict(size = 5,color = 'green',opacity = 0.8, symbol='diamond'), name='FBN - ORIG'),
    px.Scatter3d(x=embedding2_both[size_both_idx[7][2]:size_both_idx[7][3], 0],y=embedding2_both[size_both_idx[7][2]:size_both_idx[7][3], 1], z=embedding2_both[size_both_idx[7][2]:size_both_idx[7][3], 2], mode='markers', marker = dict(size = 3,color = 'blue',opacity = 0.3), name='FBN - VICI'),
    ])
# Add dropdown
plot.update_layout(legend=dict(title_font_family="arial black",font=dict(size=24), itemsizing='constant'),
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(label="All",
                     method="update",
                     args=[{"visible": [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]},
                           {"title": "All"}]),
                dict(label="MW (P,N)",
                     method="update",
                     args=[{"visible": [True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False]},
                           {"title": "MW (P,N)",
                            }]),
                dict(label="MB (P,N)",
                     method="update",
                     args=[{"visible": [False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False]},
                           {"title": "MB (P,N)",
                            }]),
                dict(label="FW (P,N)",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, False, True, True, True, True, False, False, False, False]},
                           {"title": "FW (P,N)",
                            }]),
                dict(label="FB (P,N)",
                     method="update",
                     args=[{"visible": [False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True]},
                           {"title": "FB (P,N)",
                            }]),
            ]),
        )
    ])
  
plot.write_html(os.path.join(out_dir, reduction_technique + '_both_out.html'), full_html=False, include_plotlyjs='cdn')
