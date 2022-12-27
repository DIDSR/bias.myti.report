'''
    Program to read the .npz files from the decision boundary output code
    and analyse the dense layer o/p using UMAP analysis
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
# #
sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
# #
# # -----------------------------------------------------------------------------------------
in_dir = '/home/ravi.samala/temp/RAND_0/CHEXPERT_RESNET_0__step_0/DB_uncertainty_test_bk/'
out_dir = '/home/ravi.samala/temp/RAND_0/CHEXPERT_RESNET_0__step_0/DB_uncertainty_test_bk/'
# # Fixed params, the program expects the following files in the "in_dir"
samples_csv = 'DB_samples.csv'
orig_sample_npz = 'all_orig__last_dense.npz'
# subgroups_dense_npz = {'MWP': 'MWP_arrays__last_dense.npz',
#                         'MWN': 'MWN_arrays__last_dense.npz',
#                         'MBP': 'MBP_arrays__last_dense.npz',
#                         'MBN': 'MBN_arrays__last_dense.npz',
#                         'FWP': 'FWP_arrays__last_dense.npz',
#                         'FWN': 'FWN_arrays__last_dense.npz',
#                         'FBP': 'FBP_arrays__last_dense.npz',
#                         'FBN': 'FBN_arrays__last_dense.npz',
#                         }
subgroups_dense_npz = {
                        # 'MWP': 'MWP_arrays__last_dense.npz',
                        # 'MWN': 'MWN_arrays__last_dense.npz',
                        # 'MBP': 'MBP_arrays__last_dense.npz',
                        # 'MBN': 'MBN_arrays__last_dense.npz',
                        'FWP': 'FWP_arrays__last_dense.npz',
                        'FWN': 'FWN_arrays__last_dense.npz',
                        # 'FBP': 'FBP_arrays__last_dense.npz',
                        # 'FBN': 'FBN_arrays__last_dense.npz',
                        }
# # other plot related params
vicinal_marker_size = 1
original_marker_size = 1
reduction_technique = 'UMAP' # # options: 
                            # # "UMAP", "LLE" Locally linear embedding, "ISO", "MDS" Multidimensional scaling
                            # # "SE" Spectral embedding, "TSNE"
# # -----------------------------------------------------------------------------------------
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
        reducer2 = umap.UMAP(n_neighbors=30, min_dist=0.001, n_components=2, metric='cosine', random_state=2023)
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


# # Read the original samples with subgroup information
df = pd.read_csv(os.path.join(in_dir, samples_csv))
print(df)
# # Read the original deploled samples
orig_DN_arrays = dict(np.load(os.path.join(in_dir, orig_sample_npz), allow_pickle=True))
print(orig_DN_arrays['base_paths'].shape)
print(orig_DN_arrays['base_last_dense_layer'].shape)
print(orig_DN_arrays['base_groundtruth'].shape)
orig_DN_labels = [int(i[0]) for i in orig_DN_arrays['base_groundtruth']]
orig_marker_size = original_marker_size * np.ones(len(orig_DN_labels))
orig_DN_names = [i[2] for i in orig_DN_arrays['base_groundtruth']]
# print(orig_DN_names)
# #
# # Iterate over the subgroups and perform UMAP analysis
vicinal_subgroups = {}
original_subgroups_base_paths = {}
original_subgroups = {}
# vicinal_subgroups_color = {}
# original_subgroups_color = {}
fig1 = plt.figure()
for idx, each_subgroup_key in enumerate(subgroups_dense_npz):
    print(each_subgroup_key)
    # # extract the relevant subgroup from the DB_samples
    if each_subgroup_key[0] == 'M':
        first_subgroup_index = 'M'
    elif each_subgroup_key[0] == 'F':
        first_subgroup_index = 'F'
    if each_subgroup_key[1] == 'W':
        second_subgroup_index = 'White'
    elif each_subgroup_key[1] == 'B':
        second_subgroup_index = 'Black_or_African_American'
    if each_subgroup_key[2] == 'P':
        third_subgroup_index = 'Yes'
    elif each_subgroup_key[2] == 'N':
        third_subgroup_index = 'No'
    subgroup_df = df[(df[first_subgroup_index] == 1) & (df[second_subgroup_index] == 1) & (df[third_subgroup_index] == 1)]
    current_subgroup_samples = subgroup_df['Path'].tolist()
    # # select the dense layer feature samples based on the above selected subgroup
    orig_indxs = [i for i, x in enumerate(orig_DN_names) if x in current_subgroup_samples]
    print(len(current_subgroup_samples))
    print(len(orig_indxs))
    # print(subgroup_df)
    # #
    # # Read the vicinal distribution
    DN_arrays = dict(np.load(os.path.join(in_dir, subgroups_dense_npz[each_subgroup_key]), allow_pickle=True))
    # print(len(DN_arrays))

    triplet_indx = []
    feat_np = []
    for key, value in DN_arrays.items():
        triplet_indx.append(key)
        if len(feat_np) == 0:
            feat_np = value
        else:
            feat_np = np.vstack((feat_np, value))
    
    print(feat_np.shape)
    color1 = 2 * np.ones(feat_np.shape[0])
    size1 = vicinal_marker_size * np.ones(feat_np.shape[0])
    print(color1.shape)

    merge_np = np.vstack((feat_np, orig_DN_arrays['base_last_dense_layer'][orig_indxs]))
    vicinal_subgroups[each_subgroup_key] = feat_np
    original_subgroups[each_subgroup_key] = orig_DN_arrays['base_last_dense_layer'][orig_indxs]
    original_subgroups_base_paths[each_subgroup_key] = orig_DN_arrays['base_paths'][orig_indxs]
    print(merge_np.shape)

    embedding = do_embedding(merge_np, reduction_technique)

    # #
    color2 = np.concatenate((color1, [orig_DN_labels[i] for i in orig_indxs]))
    size2 = np.concatenate((size1, [orig_marker_size[i] for i in orig_indxs]))
    print(embedding.shape)

    # plt.subplot(2,4,idx+1)
    plt.subplot(2,1,idx+1)
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        alpha=0.7,
        c=color_map_color(color2),
        s=size2
        )
    plt.axis('off')
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(each_subgroup_key + ' (' + str(len(orig_indxs)) + ' with ' + str(feat_np.shape[0]) + ')', fontsize=14)
    # break

classes = ['Original', 'Vicinal']
class_colours = ['#CC6677', '#88CCEE']
recs = []
for i in range(0,len(class_colours)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
plt.legend(recs,classes,loc=4)
fig1.tight_layout()
plt.savefig(os.path.join(out_dir, reduction_technique + '.png'), dpi=600)
# plt.show()

# #-------------------------------------------
# # plot the aggregate
# #-------------------------------------------
size_orig_idx = [0]
size_vici_idx = [0]
size_both_idx = [[0, 0]]
for i, each_subgroup in enumerate(original_subgroups):
    if i == 0:
        merge_dat_orig = original_subgroups[each_subgroup]
        merge_original_subgroups_base_paths = original_subgroups_base_paths[each_subgroup]
        merge_dat_vici = vicinal_subgroups[each_subgroup]
        merge_dat_both = original_subgroups[each_subgroup]
        merge_dat_both = np.vstack((merge_dat_both, vicinal_subgroups[each_subgroup]))
    else:
        merge_dat_orig = np.vstack((merge_dat_orig, original_subgroups[each_subgroup]))
        merge_original_subgroups_base_paths = np.hstack((merge_original_subgroups_base_paths, original_subgroups_base_paths[each_subgroup]))
        merge_dat_vici = np.vstack((merge_dat_vici, vicinal_subgroups[each_subgroup]))
        merge_dat_both = np.vstack((merge_dat_both, original_subgroups[each_subgroup]))
        merge_dat_both = np.vstack((merge_dat_both, vicinal_subgroups[each_subgroup]))
    size_orig_idx.append(merge_dat_orig.shape[0])
    size_vici_idx.append(merge_dat_vici.shape[0])
    size_both_idx.append([merge_dat_orig.shape[0], merge_dat_vici.shape[0]])


# # ORIG
embedding2_orig = do_embedding(merge_dat_orig, reduction_technique)
pal = sns.color_palette("tab10")
clr_orig = pal.as_hex()
legend_str = []
colr_arr = []
# #
fig2 = plt.figure()
for i, each_subgroup in enumerate(original_subgroups):
    legend_str.append(each_subgroup)
    if 'N' in each_subgroup: clr_orig[i] = 'g'
    if 'P' in each_subgroup: clr_orig[i] = 'r'
    colr_arr.append(clr_orig[i])
    plt.scatter(
        embedding2_orig[size_orig_idx[i]:size_orig_idx[i+1], 0],
        embedding2_orig[size_orig_idx[i]:size_orig_idx[i+1], 1],
        alpha=0.7,
        c=clr_orig[i],
        s=10
        )
# plt.axis('off')
plt.gca().set_aspect('equal', 'datalim')
recs = []
for i in range(0,len(colr_arr)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=colr_arr[i]))
plt.legend(recs,legend_str,loc=4)
fig2.tight_layout()
plt.savefig(os.path.join(out_dir, reduction_technique + '_orig_agg2.png'), dpi=600)
# plt.show()

print(merge_dat_orig.shape)
print(merge_original_subgroups_base_paths.shape)
# # # =================================================================================
# # # Additional manual analysis
# def extract_point_cloud(x1, x2, y1, y2, in_embed):
#     a_x_indx = np.argwhere((in_embed[:,0]>x1) & (in_embed[:,0]<x2)).tolist()
#     a_y_indx = np.argwhere((in_embed[:,1]>y1) & (in_embed[:,1]<y2)).tolist()
#     a_x_indx = [v[0] for v in a_x_indx]
#     a_y_indx = [v[0] for v in a_y_indx]
#     a_indx = list(set(a_x_indx) & set(a_y_indx))
#     out_embed = in_embed[a_indx, :]
#     return out_embed, a_indx

# print('=====================')
# print(embedding2_orig.shape)
# x1,x2,y1,y2 = 7.5, 14, 0, 4
# clus1, clus1_idx = extract_point_cloud(x1, x2, y1, y2, embedding2_orig)
# print(clus1.shape)
# print(merge_original_subgroups_base_paths[clus1_idx].shape)
# x1,x2,y1,y2 = 7.5, 14, 4, 9
# clus2, clus2_idx = extract_point_cloud(x1, x2, y1, y2, embedding2_orig)
# print(clus2.shape)
# print(merge_original_subgroups_base_paths[clus2_idx].shape)
# # #
# fig10 = plt.figure()
# plt.scatter(
#     clus1[:, 0],
#     clus1[:, 1],
#     alpha=0.7,
#     c='c',
#     s=10
#     )
# plt.scatter(
#     clus2[:, 0],
#     clus2[:, 1],
#     alpha=0.7,
#     c='y',
#     s=10
#     )
# fig2.tight_layout()
# plt.show()
# # =================================================================================

# # VICINAL
embedding2_vici = do_embedding(merge_dat_vici, reduction_technique)
pal = sns.color_palette("Set2")
clr_vici = pal.as_hex()
legend_str = []
colr_arr = []
# #
fig3 = plt.figure()
for i, each_subgroup in enumerate(original_subgroups):
    legend_str.append(each_subgroup)
    if 'P' in each_subgroup: clr_vici[i] = '#ff7f0e'
    if 'N' in each_subgroup: clr_vici[i] = '#17becf'
    colr_arr.append(clr_vici[i])
    plt.scatter(
        embedding2_vici[size_vici_idx[i]:size_vici_idx[i+1], 0],
        embedding2_vici[size_vici_idx[i]:size_vici_idx[i+1], 1],
        alpha=0.5,
        c=clr_vici[i],
        s=10,
        marker='d'
        )
plt.axis('off')
plt.gca().set_aspect('equal', 'datalim')
recs = []
for i in range(0,len(colr_arr)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=colr_arr[i]))
plt.legend(recs,legend_str,loc=4)
fig3.tight_layout()
plt.savefig(os.path.join(out_dir, reduction_technique + '_vici_agg2.png'), dpi=600)
# plt.show()

# # BOTH
embedding2_both = do_embedding(merge_dat_both, reduction_technique)
# pal = sns.color_palette("Set2")
# clr = pal.as_hex()
legend_str = []
colr_arr = []
# #
fig4 = plt.figure()
for i, each_subgroup in enumerate(original_subgroups):
    legend_str.append(each_subgroup + ' (original)')
    colr_arr.append(clr_orig[i])
    plt.scatter(
        embedding2_both[size_both_idx[i][0]:size_both_idx[i+1][0], 0],
        embedding2_both[size_both_idx[i][0]:size_both_idx[i+1][0], 1],
        alpha=1.0,
        c=clr_orig[i],
        s=10
        )
# plt.axis('off')
# plt.gca().set_aspect('equal', 'datalim')
# fig4.tight_layout()
# plt.savefig(os.path.join(out_dir, reduction_technique + '_both_agg2a.png'), dpi=600)
# fig5 = plt.figure()
for i, each_subgroup in enumerate(original_subgroups):
    legend_str.append(each_subgroup + ' (vicinal)')
    colr_arr.append(clr_vici[i])
    plt.scatter(
        embedding2_both[size_both_idx[i][1]:size_both_idx[i+1][1], 0],
        embedding2_both[size_both_idx[i][1]:size_both_idx[i+1][1], 1],
        alpha=0.5,
        c=clr_vici[i],
        s=10,
        marker='d'
        )
# plt.axis('off')
# plt.gca().set_aspect('equal', 'datalim')
recs = []
for i in range(0,len(colr_arr)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=colr_arr[i]))
plt.legend(recs,legend_str,loc=4)
fig4.tight_layout()
plt.savefig(os.path.join(out_dir, reduction_technique + '_both_agg2b.png'), dpi=600)
plt.show()

