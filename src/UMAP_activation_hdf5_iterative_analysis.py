'''
    Program to read the .hdf5 files from the decision boundary output code
    and iteratively UMAP analysis activation maps
'''
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import umap
import seaborn as sns
import plotly
import h5py
from sklearn import manifold

in_dir = '/gpfs_projects/ravi.samala/OUT/2022_CXR/20221227_UMAP_VicinalProximity_analysis/RAND_0/CHEXPERT_RESNET_0__step_0/Valid_n500s100_ResNet18_layer1_0__bn2/'
out_dir = '/gpfs_projects/ravi.samala/OUT/2022_CXR/20221227_UMAP_VicinalProximity_analysis/RAND_0/CHEXPERT_RESNET_0__step_0/Valid_n500s100_ResNet18_layer1_0__bn2/D3N50D05_all/'

# # Fixed params, the program expects the following files in the "in_dir"
samples_csv = 'DB_samples.csv'
orig_sample_npz = 'all_orig__activation_maps.npz'
subgroups_dense_npz = {
                        'MWP': 'MWP_arrays__activations.hdf5',
                        'MWN': 'MWN_arrays__activations.hdf5',
                        'MBP': 'MBP_arrays__activations.hdf5',
                        'MBN': 'MBN_arrays__activations.hdf5',
                        'FWP': 'FWP_arrays__activations.hdf5',
                        'FWN': 'FWN_arrays__activations.hdf5',
                        'FBP': 'FBP_arrays__activations.hdf5',
                        'FBN': 'FBN_arrays__activations.hdf5',
                        }
# # other plot related params
embd_dims = 3
reduction_technique = 'UMAP' # # options: 
                             # # "UMAP", "LLE" Locally linear embedding, "ISO", "MDS" Multidimensional scaling
                             # # "SE" Spectral embedding, "TSNE"
# # -----------------------------------------------------------------------------------------
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
    # print(f"Found {total_length} images")
    out_arr = np.vstack(arrays)
    # print("activation map array shape: ", out_arr.shape)
    f.close()
    return out_arr


def do_embedding(merge_np, reduction_technique, num_dims):
    embedding = []
    if reduction_technique == "UMAP":
        # # UMAP
        # reducer2 = umap.UMAP(n_neighbors=20, min_dist=0.01, n_components=num_dims, metric='euclidean', random_state=2023) # # default
        reducer2 = umap.UMAP(n_neighbors=50, min_dist=0.5, n_components=num_dims, metric='euclidean', random_state=2023)
        embedding = reducer2.fit_transform(merge_np)
    elif reduction_technique == "LLE":
        # # LLE
        params = {
            "n_neighbors": 10,
            "n_components": 2,
            "eigen_solver": "auto",
        }
        lle_standard = manifold.LocallyLinearEmbedding(method="modified", modified_tol=0.8, **params)
        embedding = lle_standard.fit_transform(merge_np)
    elif reduction_technique == 'ISO':
        isomap = manifold.Isomap(n_neighbors=10, n_components=num_dims, p=1)
        embedding = isomap.fit_transform(merge_np)
    elif reduction_technique == 'MDS':
        md_scaling = manifold.MDS(n_components=num_dims, max_iter=50, n_init=4)
        embedding = md_scaling.fit_transform(merge_np)
    elif reduction_technique == 'SE':
        spectral = manifold.SpectralEmbedding(n_components=num_dims, n_neighbors=30)
        embedding = spectral.fit_transform(merge_np)
    elif reduction_technique == 'TSNE':
        tsne = manifold.TSNE(n_components=num_dims, perplexity=20, n_iter=5000, init="pca", metric='euclidean', n_jobs=5)
        embedding = tsne.fit_transform(merge_np.astype(np.float32))
    else:
        print('UNKNOWN option for dimensionality reduction technique')
    return embedding


def plot_3col_nrows_subplot(sbgroup_name, dim, emb1, emb2, emb3, output_dir, out_fname):
    pal = sns.color_palette("tab10")
    clr = pal.as_hex()
    pal = sns.color_palette("Accent")
    clr2 = pal.as_hex()
    # # https://stackoverflow.com/questions/45577255/plotly-plot-multiple-figures-as-subplots
    fichier_html_graphs=open(out_fname,'w')
    fichier_html_graphs.write("<html><head></head><body>"+"\n")
    for act_map_index, act_map_key in enumerate(emb3):   # # iter on activation maps
        # A1 = emb1[act_map_index]
        # A2 = emb2[act_map_index]
        A3 = emb3[act_map_index]
        # indx1 = A1['indx']
        # embd1 = A1['embedding']
        # indx2 = A2['indx']
        # embd2 = A2['embedding']
        indx3 = A3['indx']
        embd3 = A3['embedding']
        # print('There are {} subgroups'.format(len(indx3)))
        # #
        # #
        if dim == 2:
            data = []
            for ii, (each_index, each_sbgroup) in enumerate(zip(indx3, sbgroup_name)):
                data.extend(
                    [
                        go.Scatter(
                            x = embd3[each_index[0]:each_index[1], 0],
                            y = embd3[each_index[0]:each_index[1], 1], 
                            mode ='markers', 
                            marker = dict(size = 5, color = clr[ii], opacity = 0.8, symbol='square'), 
                            name = each_sbgroup + ' - ' + 'ORIG',
                        ),
                        go.Scatter(
                            x=embd3[each_index[2]:each_index[3], 0],
                            y=embd3[each_index[2]:each_index[3], 1], 
                            mode='markers', 
                            marker = dict(size = 3, color = clr2[ii], opacity = 0.5, symbol='diamond'), 
                            name = each_sbgroup + ' - ' + 'VICI',
                        ),
                    ]
                )
        elif dim == 3:
            data = []
            for ii, (each_index, each_sbgroup) in enumerate(zip(indx3, sbgroup_name)):
                data.extend(
                    [
                        go.Scatter3d(
                            x = embd3[each_index[0]:each_index[1], 0],
                            y = embd3[each_index[0]:each_index[1], 1], 
                            z = embd3[each_index[0]:each_index[1], 2], 
                            mode ='markers', 
                            marker = dict(size = 5, color = clr[ii], opacity = 0.8, symbol='square'), 
                            name = each_sbgroup + ' - ' + 'ORIG',
                        ),
                        go.Scatter3d(
                            x = embd3[each_index[2]:each_index[3], 0],
                            y = embd3[each_index[2]:each_index[3], 1], 
                            z = embd3[each_index[2]:each_index[3], 2], 
                            mode ='markers', 
                            marker = dict(size = 3, color = clr2[ii], opacity = 0.5, symbol='diamond'), 
                            name = each_sbgroup + ' - ' + 'VICI',
                        ),
                    ]
                )
        else:
            print('ERROR. Uknown dimensions to plot!!!')
        # print(data)
        layout = go.Layout(
            title=('Activation# '+str(act_map_index)),
            titlefont=dict(
            family='arial black',
            size=20,
            color='#7f7f7f'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(legend=dict(title_font_family="arial black",font=dict(size=18), itemsizing='constant'))
        plotly.offline.plot(fig, filename=os.path.join(output_dir, '_'.join(sbgroup_name) + '_' + str(dim) + 'chart_'+ str(act_map_index)+'.html'), auto_open=False)
        fichier_html_graphs.write("  <object data=\""+ '_'.join(sbgroup_name) + '_' + str(dim) + 'chart_'+ str(act_map_index)+'.html'+"\" width=\"650\" height=\"500\"></object>"+"\n")
    fichier_html_graphs.write("</body></html>")
    print("CHECK {}".format(out_fname))

# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # Read the original samples with subgroup labelled information
df = pd.read_csv(os.path.join(in_dir, samples_csv))
# # Read the activation maps of the original deployed samples
# # This var will contain 'base_paths', 'base_groundtruth' and 'base_activation_maps'
print('reading original activation maps...')
orig_DN_arrays = dict(np.load(os.path.join(in_dir, orig_sample_npz), allow_pickle=True))
orig_DN_labels = [int(i[0]) for i in orig_DN_arrays['base_groundtruth']]
orig_DN_names = [i[2] for i in orig_DN_arrays['base_groundtruth']]
[_, num_acts, _, _] = orig_DN_arrays['base_activation_maps'].shape
print("There are {} activation maps".format(num_acts))
# # 
emb_ORGI = {}
emb_VICI = {} 
emb_BOTH = {}
subgroups_names = []
# # iterative over each activation map, aggregate original and vicinal samples
for i_act in range(num_acts):
    print('>>>>> Act# {}'.format(i_act))
    vici_subgroups = {}
    orig_subgroups = {}
    orig_subgroups_base_paths = {}
    for idx, each_subgroup_key in enumerate(subgroups_dense_npz):
        print('\t------------------------------------------------' + each_subgroup_key)
        subgroups_names += [each_subgroup_key]
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
        # print("There are {} samples in DB_samples.csv".format(len(current_subgroup_samples)))
        print("\tThere are {} samples matching all_orig__activation_maps".format(len(orig_indxs)))
        orig_DN_arrays_subgroup = orig_DN_arrays['base_activation_maps'][orig_indxs]
        # #
        # # select a Nth activation map - ORIG
        [m, _, p, q] = orig_DN_arrays_subgroup.shape
        orig_DN_arrays_Nth_map = np.zeros((m, 1, p, q))
        for each_img in range(m):
            orig_DN_arrays_Nth_map[each_img, 0, :, :] = orig_DN_arrays_subgroup[each_img, i_act, :, :]
        # # reshape
        orig_DN_arrays_Nth_map = np.squeeze(orig_DN_arrays_Nth_map, axis=1)
        orig_DN_arrays_Nth_map = orig_DN_arrays_Nth_map.reshape((m, p*q))
        print("\t(ORIG) Shape of the subgroup activation layer: {}".format(orig_DN_arrays_Nth_map.shape))
        # # Read the vicinal distribution
        print('\treading subgroup hdf5 file...')
        vici_DN_arrays_Nth_map = load_map(os.path.join(in_dir, subgroups_dense_npz[each_subgroup_key]), i_act)
        [m, p, q] = vici_DN_arrays_Nth_map.shape
        vici_DN_arrays_Nth_map = vici_DN_arrays_Nth_map.reshape((m, p*q))
        # vici_DN_arrays_Nth_map = vici_DN_arrays_Nth_map[1:1000, :]  # # for debugging
        print("\t(VICI) Shape of the subgroup activation layer: {}".format(vici_DN_arrays_Nth_map.shape))
        # # collect for each subgroup
        vici_subgroups[each_subgroup_key] = vici_DN_arrays_Nth_map
        orig_subgroups[each_subgroup_key] = orig_DN_arrays_Nth_map
        orig_subgroups_base_paths[each_subgroup_key] = orig_DN_arrays['base_paths'][orig_indxs]
    # # aggregate
    print('\taggregating...')
    size_orig_idx = [0]
    size_vici_idx = [0]
    size_both_idx = []
    prevA = 0
    for i, each_subgroup in enumerate(orig_subgroups):
        if i == 0:
            merge_dat_orig = orig_subgroups[each_subgroup]
            # merge_original_subgroups_base_paths = original_subgroups_base_paths[each_subgroup]
            merge_dat_vici = vici_subgroups[each_subgroup]
            merge_dat_both = np.vstack((orig_subgroups[each_subgroup], vici_subgroups[each_subgroup]))
            size_both_idx = [[0, merge_dat_orig.shape[0], merge_dat_orig.shape[0], merge_dat_orig.shape[0] + merge_dat_vici.shape[0]]]
            prevA = merge_dat_orig.shape[0] + merge_dat_vici.shape[0]
        else:
            merge_dat_orig = np.vstack((merge_dat_orig, orig_subgroups[each_subgroup]))
            # merge_original_subgroups_base_paths = np.hstack((merge_original_subgroups_base_paths, original_subgroups_base_paths[each_subgroup]))
            merge_dat_vici = np.vstack((merge_dat_vici, vici_subgroups[each_subgroup]))
            merge_dat_both = np.vstack((merge_dat_both, orig_subgroups[each_subgroup]))
            merge_dat_both = np.vstack((merge_dat_both, vici_subgroups[each_subgroup]))
            size_both_idx.append([prevA, \
                                    prevA + orig_subgroups[each_subgroup].shape[0], 
                                    prevA + orig_subgroups[each_subgroup].shape[0], 
                                    prevA + orig_subgroups[each_subgroup].shape[0] + vici_subgroups[each_subgroup].shape[0]])
            prevA = prevA + orig_subgroups[each_subgroup].shape[0] + vici_subgroups[each_subgroup].shape[0]
        size_orig_idx.append(merge_dat_orig.shape[0])
        size_vici_idx.append(merge_dat_vici.shape[0])
    # print(size_orig_idx)
    # print(size_vici_idx)
    # print(size_both_idx)
    print('\tembedding...')
    emb_ORGI = []
    emb_VICI= []
    emb_BOTH[i_act] = {'indx': size_both_idx, 'embedding': do_embedding(merge_dat_both, reduction_technique, embd_dims)}
    if i_act == 5:
        break

# # plot subplots for each activation map
print('Plotting...')
plot_3col_nrows_subplot([*subgroups_dense_npz], embd_dims, emb_ORGI, emb_VICI, emb_BOTH, out_dir, \
                        os.path.join(out_dir, '{}_{}{}.html'.format('_'.join([*subgroups_dense_npz]), reduction_technique, embd_dims)))