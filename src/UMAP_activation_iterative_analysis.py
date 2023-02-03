'''
    Program to read the .npz files from the decision boundary output code
    and iteratively UMAP analysis activation maps
'''
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import umap
import seaborn as sns
import plotly
import plotly.offline as py
import plotly.graph_objs as go
# #
# # -----------------------------------------------------------------------------------------
# in_dir = '/home/ravi.samala/temp/RAND_0/CHEXPERT_RESNET_0__step_0/OOD_openR1_allDX_ResNet18_layer3_0__downsample_1/'
# out_dir = '/home/ravi.samala/temp/RAND_0/CHEXPERT_RESNET_0__step_0/OOD_openR1_allDX_ResNet18_layer3_0__downsample_1/'
in_dir = '/gpfs_projects/ravi.samala/OUT/2022_CXR/20221227_UMAP_VicinalProximity_analysis/RAND_0/CHEXPERT_RESNET_0__step_0/Valid_n500s100_ResNet18_layer1_0__bn2/'
out_dir = '/gpfs_projects/ravi.samala/OUT/2022_CXR/20221227_UMAP_VicinalProximity_analysis/RAND_0/RAND_0/CHEXPERT_RESNET_0__step_0/Valid_n500s100_ResNet18_layer1_0__bn2/'

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
embd_dims = 3
reduction_technique = 'UMAP' # # options: 
                             # # "UMAP", "LLE" Locally linear embedding, "ISO", "MDS" Multidimensional scaling
                             # # "SE" Spectral embedding, "TSNE"
# # -----------------------------------------------------------------------------------------
def do_embedding(merge_np, reduction_technique, num_dims):
    embedding = []
    if reduction_technique == "UMAP":
        # # UMAP
        reducer2 = umap.UMAP(n_neighbors=20, min_dist=0.01, n_components=num_dims, metric='euclidean', random_state=2023)
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
        indx3 = A3['indx'][0]
        embd3 = A3['embedding']
        # #
        # color1 = '#00bfff'
        # color2 = '#ff4000'
        # #
        if dim == 2:
            data = [
                # go.Scatter(
                #     x=embd1[indx1[0]:indx1[0+1], 0],
                #     y=embd1[indx1[0]:indx1[0+1], 1], 
                #     mode='markers', 
                #     marker = dict(size = 3,color = clr[i], opacity = 0.8, symbol='square'), 
                #     name=sbgroup_name + ' - ' + 'ORIG',
                # ),
                # go.Scatter(
                #     x=embd2[indx2[0]:indx2[0+1], 0],
                #     y=embd2[indx2[0]:indx2[0+1], 1], 
                #     mode='markers', 
                #     marker = dict(size = 3,color = clr[i], opacity = 0.8, symbol='diamond'), 
                #     name=sbgroup_name + ' - ' + 'VICI',
                # ),
                # # BOTH
                go.Scatter(
                    x=embd3[indx3[0]:indx3[1], 0],
                    y=embd3[indx3[0]:indx3[1], 1], 
                    mode='markers', 
                    marker = dict(size = 5,color = 'red', opacity = 0.8, symbol='square'), 
                    name=sbgroup_name + ' - ' + 'ORIG',
                ),
                go.Scatter(
                    x=embd3[indx3[2]:indx3[3], 0],
                    y=embd3[indx3[2]:indx3[3], 1], 
                    mode='markers', 
                    marker = dict(size = 3,color = 'orange', opacity = 0.3, symbol='diamond'), 
                    name=sbgroup_name + ' - ' + 'VICI',
                ),
            ]
        elif dim == 3:
            data = [
                # go.Scatter3d(
                #     x=embd1[indx1[0]:indx1[0+1], 0],
                #     y=embd1[indx1[0]:indx1[0+1], 1], 
                #     z=embd1[indx1[0]:indx1[0+1], 2], 
                #     mode='markers', 
                #     marker = dict(size = 3,color = clr[i], opacity = 0.8, symbol='square'), 
                #     name=sbgroup_name + ' - ' + 'ORIG',
                # ),
                # go.Scatter3d(
                #     x=embd2[indx2[0]:indx2[0+1], 0],
                #     y=embd2[indx2[0]:indx2[0+1], 1], 
                #     z=embd2[indx2[0]:indx2[0+1], 2], 
                #     mode='markers', 
                #     marker = dict(size = 3,color = clr[i], opacity = 0.8, symbol='diamond'), 
                #     name=sbgroup_name + ' - ' + 'VICI',
                # ),
                # # BOTH
                go.Scatter3d(
                    x=embd3[indx3[0]:indx3[1], 0],
                    y=embd3[indx3[0]:indx3[1], 1], 
                    z=embd3[indx3[0]:indx3[1], 2], 
                    mode='markers', 
                    marker = dict(size = 5,color = 'red', opacity = 0.8, symbol='square'), 
                    name=sbgroup_name + ' - ' + 'ORIG',
                ),
                go.Scatter3d(
                    x=embd3[indx3[2]:indx3[3], 0],
                    y=embd3[indx3[2]:indx3[3], 1], 
                    z=embd3[indx3[2]:indx3[3], 2], 
                    mode='markers', 
                    marker = dict(size = 3,color = 'orange', opacity = 0.3, symbol='diamond'), 
                    name=sbgroup_name + ' - ' + 'VICI',
                ),
            ]
        else:
            print('ERROR. Uknown dimensions to plot!!!')

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
        plotly.offline.plot(fig, filename=os.path.join(output_dir, sbgroup_name + '_' + str(dim) + 'chart_'+ str(act_map_index)+'.html'), auto_open=False)
        fichier_html_graphs.write("  <object data=\""+sbgroup_name + '_' + str(dim) + 'chart_'+ str(act_map_index)+'.html'+"\" width=\"650\" height=\"500\"></object>"+"\n")
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
    # print(orig_DN_arrays['base_activation_maps'][orig_indxs].shape)
    orig_DN_arrays_subgroup = orig_DN_arrays['base_activation_maps'][orig_indxs]
    print(orig_DN_arrays_subgroup.shape)

    # # Read the vicinal distribution
    print('reading subgroup pkl file...')
    vici_DN_arrays = dict(np.load(os.path.join(in_dir, subgroups_dense_npz[each_subgroup_key]), allow_pickle=True))
    print("There are {} vicinal distributions".format(len(vici_DN_arrays)))
    triplet_indx = list(vici_DN_arrays.keys())
    feat_np = np.vstack(list(vici_DN_arrays.values())[:])
    # print(feat_np[0])
    print(feat_np.shape)
    
    # # 
    emb_ORGI = {}
    emb_VICI = {} 
    emb_BOTH = {}
    [_, n, _, _] = orig_DN_arrays_subgroup.shape
    for i_act in range(n):
        print('>>>>> Act# {}'.format(i_act))
        # # select a random activation map per image
        [m, _, p, q] = orig_DN_arrays_subgroup.shape
        orig_DN_arrays_rand_map = np.zeros((m, 1, p, q))
        for each_img in range(m):
            # orig_DN_arrays_rand_map[each_img, 0, :, :] = orig_DN_arrays_subgroup[each_img, random.randint(0, n-1), :, :]
            orig_DN_arrays_rand_map[each_img, 0, :, :] = orig_DN_arrays_subgroup[each_img, i_act, :, :]
        # # reshape
        # print(orig_DN_arrays_rand_map.shape)
        orig_DN_arrays_rand_map = np.squeeze(orig_DN_arrays_rand_map, axis=1)
        # print(orig_DN_arrays_rand_map.shape)
        orig_DN_arrays_rand_map = orig_DN_arrays_rand_map.reshape((m, p*q))
        # print(orig_DN_arrays_rand_map.shape)
        print("(ORIG) Shape of the subgroup dense layer: {}".format(orig_DN_arrays_rand_map.shape))
        # #
        
        # # select a random activation map per image
        [m, _, p, q] = feat_np.shape
        feat_np2 = np.zeros((m, 1, p, q))
        for each_img in range(m):
            # feat_np2[each_img, 0, :, :] = feat_np[each_img, random.randint(0, n-1), :, :]
            feat_np2[each_img, 0, :, :] = feat_np[each_img, i_act, :, :]
        # print(feat_np2.shape)
        feat_np2 = np.squeeze(feat_np2, axis=1)
        # print(feat_np2.shape)
        feat_np2 = feat_np2.reshape((m, p*q))
        # print(feat_np2.shape)
        print("(VICI) Shape of the subgroup dense layer: {}".format(feat_np2.shape))
        merge_np = np.vstack((feat_np2, orig_DN_arrays_rand_map))
        vicinal_subgroups[each_subgroup_key] = feat_np2
        original_subgroups[each_subgroup_key] = orig_DN_arrays_rand_map
        original_subgroups_base_paths[each_subgroup_key] = orig_DN_arrays['base_paths'][orig_indxs]
        # print(merge_np.shape)
        print("(BOTH) Shape of the subgroup dense layer: {}".format(merge_np.shape))

        # #-------------------------------------------
        # # aggregate
        # #-------------------------------------------
        # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
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
                size_both_idx.append([prevA, \
                                        prevA + original_subgroups[each_subgroup].shape[0], 
                                        prevA + original_subgroups[each_subgroup].shape[0], 
                                        prevA + original_subgroups[each_subgroup].shape[0] + vicinal_subgroups[each_subgroup].shape[0]])
                prevA = prevA + original_subgroups[each_subgroup].shape[0] + vicinal_subgroups[each_subgroup].shape[0]
            size_orig_idx.append(merge_dat_orig.shape[0])
            size_vici_idx.append(merge_dat_vici.shape[0])
            # size_both_idx.append([merge_dat_orig.shape[0], merge_dat_vici.shape[0]])
        # print(size_orig_idx)
        # print(size_vici_idx)
        # print(size_both_idx)
        # #
        # # embedding
        print('Embedding...')
        # # # ORIG
        # embedding2_orig = do_embedding(merge_dat_orig, reduction_technique, embd_dims)
        # print(embedding2_orig.shape)
        # embedding2_vicinal = do_embedding(merge_dat_vici, reduction_technique, embd_dims)
        # print(embedding2_vicinal.shape)
        embedding2_both = do_embedding(merge_dat_both, reduction_technique, embd_dims)
        # print(embedding2_both.shape)
        # # append all activation map related embeddings
        # emb_ORGI[i_act] = {'indx': size_orig_idx, 'embedding': embedding2_orig}
        # emb_VICI[i_act] = {'indx': size_vici_idx, 'embedding': embedding2_vicinal}
        emb_BOTH[i_act] = {'indx': size_both_idx, 'embedding': embedding2_both}
        # if i_act == 3:
        #     break
    # # plot subplots for each subgroup
    print('Plotting...')
    plot_3col_nrows_subplot(each_subgroup_key, embd_dims, emb_ORGI, emb_VICI, emb_BOTH, \
        out_dir, os.path.join(out_dir, '{}_{}{}.html'.format(each_subgroup_key, reduction_technique, embd_dims)))
