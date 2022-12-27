from lib2to3.pytree import Base
import torch.utils.data as data
from data.chexpert_dataset import CheXpertDataset
from data.custom_dataset import CustomDataset
from data.base_dataset import BaseDataset
from constants import *
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
# import seaborn as sns
# from shapely.geometry import Polygon, Point
import os
import json
from itertools import product, combinations_with_replacement
from data import get_loader
from random import sample

abbreviation_table = {
    'Female':"F",
    'Male':"M",
    'CR':"C",
    "DX":"D",
    "White":"W",
    'Black':"B",
    "Yes":"P",# positive
    "No":'N'
}

def get_plane(img1, img2, img3):
    a = img2 - img1
    b = img3 - img1
    a = a.to(torch.float)
    b = b.to(torch.float)
    a_norm = torch.dot(a.flatten(), a.flatten()).sqrt()
    a = a / a_norm
    first_coef = torch.dot(a.flatten(), b.flatten())
    b_orthog = b - first_coef * a
    b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()
    b_orthog = b_orthog / b_orthog_norm
    second_coef = torch.dot(b.flatten(), b_orthog.flatten())
    coords = [[0,0], [a_norm,0], [first_coef, second_coef]]
    return a, b_orthog, b, coords


class plane_dataset(BaseDataset):
    '''
    Dataset built of virtual images interpolated from the three input images
    '''
    def __init__(self,
                dataframe, # CXR information dataframe
                img_idxs,
                subgroups,
                prediction_tasks,
                scale=320,
                steps=100,
                randomize=0,
                random_range=(0,255),
                shuffle=0,
                normalize=True,
                shape='rectangle',
                save_images=None):
        # print("CREATING PLANELODAER")
        # print('RANDOMIZE: ', randomize)
        # print("RANDOM RANGE: ", random_range)
        # print("SHUFFLE: ", shuffle)
        self.subgroups = subgroups
        self.steps = steps
        # self.steps = 10
        self.normalize = normalize
        self.return_info_dict = False
        self.study_level = False
        self.prediction_tasks = prediction_tasks
        self.transform = transforms.Compose([transforms.Resize((scale,scale)),transforms.PILToTensor()])
        # BASE IMAGES ====================================
        self.imgs = {ii:{} for ii in range(3)}
        self.img_list = []
        for ii,idx in enumerate(img_idxs):
            row = dataframe.iloc[idx,:]
            self.imgs[ii]['Index'] = idx
            # load image
            self.imgs[ii]['Image'] = self.transform(Image.open(row['Path']).convert('RGB')).to(torch.float)
            self.img_list.append(self.imgs[ii]['Image'])
            label = []
            for key, values in self.subgroups.items():
                for sub in values:
                    if row[sub] == 1:
                        self.imgs[ii][key] = sub
                        label.append(sub)
            self.imgs[ii]['Label'] = "-".join(label)
        self.base_img_mean = torch.mean(torch.stack(self.img_list),[0,2,3], keepdim=True)
        self.base_img_mean = torch.squeeze(self.base_img_mean)
        self.base_img_std = torch.std(torch.stack(self.img_list),[0,2,3], keepdim=True)
        self.base_img_std = torch.squeeze(self.base_img_std)
        # set up normalization for interpolated images
        self.norm = transforms.Normalize(self.base_img_mean, self.base_img_std)

        t_to_PIL = transforms.ToPILImage()
        # random img input settings
        for ii in range(randomize):
            self.imgs[ii]['Image'] = torch.from_numpy(np.random.randint(random_range[0], random_range[1], size=(scale,scale), dtype=np.uint8))
            self.imgs[ii]['Image'] = self.transform(t_to_PIL(self.imgs[ii]['Image']).convert('RGB'))
        # image pixel shuffle
        for ii in range(shuffle):
            # self.imgs[ii]['Image'] = self.imgs[ii]['Image'].numpy()
            np.random.shuffle(self.imgs[ii]['Image'].numpy())
            self.imgs[ii]['Image'] = self.transform(t_to_PIL(self.imgs[ii]['Image']))
        if save_images is not None:
            
            for ii in range(3):
                save_img = t_to_PIL(self.imgs[ii]['Image'])
                save_img.save(os.path.join(save_images,'images', f'img_{ii}_r{randomize}_({random_range})_s{shuffle}.png'))
        # VIRTUAL IMAGES ==================================
        self.vec1, self.vec2, b, self.coords = get_plane(self.imgs[0]['Image'], self.imgs[1]['Image'], self.imgs[2]['Image'])
        # constants
        range_l = 0.1
        range_r = 0.1
        # generate dataset
        self.base_img = self.imgs[0]['Image']
        x_bounds = [coord[0] for coord in self.coords]
        y_bounds = [coord[1] for coord in self.coords]
        self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
        self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]
        len1 = self.bound1[-1] - self.bound1[0]
        len2 = self.bound2[-1] - self.bound2[0]
        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, self.steps)
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, self.steps)
        grid = torch.meshgrid([list1, list2])
        # Different plot shape options
        if shape == 'rectangle':
            self.coefs1 = grid[0].flatten()
            self.coefs2 = grid[1].flatten()
        elif shape == 'triangle':
            from shapely.geometry import Polygon, Point
            poly = Polygon(self.coords)
            xcoefs1 = grid[0].flatten()
            ycoefs1 = grid[1].flatten()
            dat = np.hstack((xcoefs1[:,np.newaxis], ycoefs1[:,np.newaxis]))
            contains = np.vectorize(lambda p: poly.contains(Point(p)), signature="(n)->()")
            flags = contains(np.array(dat))
            indices = [i for i, x in enumerate(flags) if x]
            self.coefs1 = xcoefs1[indices]
            self.coefs2 = ycoefs1[indices]
        elif shape == 'line':
            poly = Polygon(self.coords)
            xcoefs1 = grid[0].flatten()
            ycoefs1 = grid[1].flatten()
            dat = np.hstack((xcoefs1[:,np.newaxis], ycoefs1[:,np.newaxis]))
            contains = np.vectorize(lambda p:Point(p).distance(poly.boundary)<150, signature="(n)->()")
            flags = contains(np.array(dat))
            indices = [i for i, x in enumerate(flags) if x]
            self.coefs1 = xcoefs1[indices]
            self.coefs2 = ycoefs1[indices]
        self.df = self.load_df()

    def load_df(self):
        images = []
        for i in range(self.coefs1.shape[0]):
            images += [self.base_img + self.coefs1[i] * self.vec1 + self.coefs2[i] * self.vec2]
            # normalize values based on base images
            if self.normalize:
                images[i] = self.norm(images[i])
        labels = [np.array([0]*len(self.prediction_tasks))] * len(images)
        self.images = images
        df = pd.DataFrame(list(zip(images, labels)), columns=['Image', 'Label'])

        # images = []
        # coef_df = pd.DataFrame([self.coefs1, self.coefs2], columns=['coefs1', 'coefs2'])
        # coef_df = coef_df.assign(Image = lambda row: (self.base_img + row['coefs1']*self.vec1 + row['coefs2']*self.vec2))
        # if self.normalize:
        #     coef_df['Image'] = coef_df['Image'].apply(lambda x: self.norm(x)) 
        # coef_df['Label'] = [np.array([0]*len(self.prediction_tasks))]
        # self.images = coef_df['Image'].values()
        # df = coef_df[['Image','Label']]
        return df

    def __len__(self):
        return self.coefs1.shape[0]

    def get_image(self, index):
        return self.df.at[index, 'Image'], self.df.at[index,'Label']

    def get_study(self, index=None):
        images = self.df['Image'].tolist()
        labels = self.df['Label'].tolist()
        return images, labels

    def __getitem__(self, index):
        if self.study_level:
            return self.get_study(index)
        else:
            return self.get_image(index)

def get_planeloader(data_args, **kwargs):
    dataset = plane_dataset(**kwargs)
    planeloader = data.DataLoader(dataset=dataset,
                                  batch_size = data_args.batch_size,
                                  shuffle=False,
                                  num_workers=data_args.num_workers)
    return planeloader

def catagorize(row, subgroups):
    label = []
    for s in subgroups:
        if s in row:
            label.append(row[s])
    return "-".join(label)

def decision_region_analysis(predictions,
                             planeloader,
                             classes):
    out_df = pd.DataFrame()
    for c in classes:
        out_df[f"{c} score"] = predictions[c]
    out_df['x'] = planeloader.dataset.coefs1.numpy()
    out_df['y'] = planeloader.dataset.coefs2.numpy()
    return out_df, out_df.to_numpy()


def old_decision_region_analysis(predictions,
                             planeloader,
                             classes,
                             save_loc=None,
                             generate_plot=True,
                             color_dict=None,
                             label_with='class',
                             title_with='index'):   

    # summary_df = pd.DataFrame({"class":classes, 'occurances':[0]*len(classes), 'percent':[0]*len(classes)})
    indiv_classes =list(set(x for c in classes for x in c.split("-")))
    class_dict = {key:None for key in planeloader.dataset.subgroups}
    class_dict['class'] = classes
    class_df = pd.DataFrame(class_dict)
    class_df = class_df.set_index("class")
    for c in classes:
        for key, vals in planeloader.dataset.subgroups.items():
            for v in vals:
                if v in c.split("-"):
                    class_df.at[c,key] = v
    class_df.dropna(how='all', axis=1, inplace=True)
    summary_df = pd.DataFrame(columns = ('Class','Occurances', 'Percent'))
    summary_df = summary_df.set_index("Class")
    for key, vals in planeloader.dataset.subgroups.items():
        outputs = [v for v in vals if v in indiv_classes]
        if len(outputs) == 0:
            continue
        if len(outputs) == 1:
            print("???")
            continue
        # currently only works for binary outputs
        conditions = [predictions[outputs[0]] > predictions[outputs[1]], predictions[outputs[0]] < predictions[outputs[1]]]
        choices = [outputs[0], outputs[1]]
        predictions[key] = np.select(conditions, choices, default='Unknown')
    predictions['Label'] = predictions.apply(lambda row: catagorize(row, planeloader.dataset.subgroups), axis=1)
    # print("PREDICTIONS:")
    # print(predictions)
    predicted_labels = pd.DataFrame({"Label":predictions['Label'].values, 'x':planeloader.dataset.coefs1.numpy(), 'y':planeloader.dataset.coefs2.numpy()})
    # output_array = predicted_labels.to_numpy()
    predicted_classes = predicted_labels['Label'].unique().tolist()
    # TODO: remove hardcoding
    score_dict = {}
    for cls in classes:
        score_dict[f"{cls} Score"] = predictions[cls].values
    score_dict['x'] = planeloader.dataset.coefs1.numpy()
    score_dict['y'] = planeloader.dataset.coefs2.numpy()
    # predicted_scores = pd.DataFrame({"Score":predictions['Yes'].values, 'x':planeloader.dataset.coefs1.numpy(), 'y':planeloader.dataset.coefs2.numpy()})
    predicted_scores = pd.DataFrame(score_dict)
    output_array = predicted_scores.to_numpy()
    # get the number of predictions for each class
    for ii, row in class_df.iterrows():
        temp_df = predictions.copy()
        for col in class_df.columns:
            temp_df = temp_df[temp_df[col] == row[col]]
        summary_df.at[ii, "Occurances"] = len(temp_df)
        summary_df.at[ii, 'Percent'] = (len(temp_df)/len(predictions)) * 100
    if generate_plot: # TODO
        import seaborn as sns
        if save_loc is None:
            print("cannot generate plots without a save location")
        else:
            # set up visual options:
            if color_dict is None:
                col_palette = sns.color_palette("hls", 14)
                col_map = ListedColormap(col_palette.as_hex())
                cmaplist = [col_map(i) for i in range(col_map.N)][:len(classes)]
                col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist,N=len(classes))
                label_color_dict = dict(zip([*range(len(classes))], cmaplist))
                color_dict = {label:label_color_dict[ii] for ii, label in enumerate(predicted_classes)}
            
            fig, ax = plt.subplots(figsize=(8,6))
            ax.tick_params(labelsize=8)
            color_idx = [color_dict[label] for label in predicted_labels['Label'].values]
            # color_idx = [color_dict[label] for label in predicted_classes]
            x = planeloader.dataset.coefs1.numpy()
            y = planeloader.dataset.coefs2.numpy()
            scatter = ax.scatter(x,y,c=color_idx, s=10)
            coords = planeloader.dataset.coords
            img_markers = [".",".","."]
            img_points = []
            for i in range(3):
                # label = planeloader.dataset
                if label_with == 'class':
                    label=planeloader.dataset.imgs[i]['Label']
                elif label_with == 'index':
                    label=planeloader.dataset.imgs[i]['Index']
                elif label_with == 'none':
                    label = ''
                img_points.append(ax.scatter(coords[i][0], coords[i][1], s=10, c='black'))
                plt.text(coords[i][0]-1500, coords[i][1]+1500, label)
            
            patch_list = []
            for ii, label in enumerate(classes): #TODO: update to only classes found
                patch_list.append(mpatches.Patch(color=color_dict[label], label=label))
            
            ax.axis("off")
            plt.legend(handles=patch_list,
                       bbox_to_anchor=(0.5,0),
                       loc='upper center',
                       ncol=len(patch_list),
                       borderaxespad=2)
            # TODO: title plot with indexs
            if title_with == 'index':
                plt.title(f"{planeloader.dataset.imgs[0]['Index']}, {planeloader.dataset.imgs[1]['Index']}, {planeloader.dataset.imgs[2]['Index']}")
            elif title_with == 'class':
                if len(set([planeloader.dataset.imgs[i]['Label'] for i in range(3)])) == 1:
                    # base triplet is one class
                    plt.title(planeloader.dataset.imgs[0]['Label'])
                else:
                    plt.title(f"{planeloader.dataset.imgs[0]['Label']}, {planeloader.dataset.imgs[1]['Label']}, {planeloader.dataset.imgs[2]['Label']}")
            elif title_with == 'both':
                if len(set([planeloader.dataset.imgs[i]['Label'] for i in range(3)])) == 1:
                    # base triplet is one class
                    plt.title(f"{planeloader.dataset.imgs[0]['Label']} ({planeloader.dataset.imgs[0]['Index']}, {planeloader.dataset.imgs[1]['Index']}, {planeloader.dataset.imgs[2]['Index']})")
                else:
                    plt.title(f"{planeloader.dataset.imgs[0]['Label']}, {planeloader.dataset.imgs[1]['Label']}, {planeloader.dataset.imgs[2]['Label']} ({planeloader.dataset.imgs[0]['Index']}, {planeloader.dataset.imgs[1]['Index']}, {planeloader.dataset.imgs[2]['Index']})")
            plt.savefig(save_loc, bbox_inches='tight', dpi=300)
            plt.close(fig)
    return summary_df, output_array


class DecisionBoundaryEvaluator():
    def __init__(self, experiment_name, save_location, predictor, transform_args, data_args, model_args, input_classes, output_classes,overwrite=False, save_last_dense_layer=False, ensemble_mode='avg', model_ckpts=None, **kwargs):
        if overwrite:
            print("\noverwrite is on\n")
        self.input_classes = input_classes
        self.output_classes = output_classes
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(save_location, experiment_name)
        self.save_last_dense_layer = save_last_dense_layer
        self.ensemble_mode = ensemble_mode
        self.model_ckpts = model_ckpts
        if type(predictor) == dict:
            self.ensemble = True
        else:
            self.ensemble = False
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)
        self.sample_fp = os.path.join(self.experiment_dir, "DB_samples.csv")
        if os.path.exists(self.sample_fp):
            sample_df = pd.read_csv(self.sample_fp)
        else:
            original_gt = pd.read_csv(data_args.test_csv)
            loader = get_loader(phase=data_args.phase,
                                data_args=data_args,
                                transform_args=transform_args,
                                is_training=False,
                                return_info_dict=True)
            base_predictions, base_groundtruth, base_paths, base_last_dense_layer = predictor.predict(loader, return_embeddings=True)
            if self.save_last_dense_layer:
                # # save the embeddings
                base_dense_layer_arrays = {'base_paths': base_paths, 'base_last_dense_layer': base_last_dense_layer, 'base_groundtruth': base_groundtruth}
                embedding_npz_fp = os.path.join(self.experiment_dir, "all_orig__last_dense.npz")
                np.savez_compressed(embedding_npz_fp, **base_dense_layer_arrays)
            elif not self.ensemble: # TODO: integrate ensemble and embeddings
                base_predictions, base_groundtruth, base_paths = predictor.predict(loader)
            else:
                b_predictions = []
                for rand, pred in predictor.items():
                    if os.path.exists(self.sample_fp.replace(".csv", f"_{rand}.csv")):
                        b_predictions.append(pd.read_csv(self.sample_fp.replace(".csv", f"_{rand}.csv"), index_col=0))
                    else:
                        bp, base_groundtruth, _ = pred.predict(loader)
                        bp['RAND'] = rand
                        bp.to_csv(self.sample_fp.replace(".csv", f"_{rand}.csv"))
                        b_predictions.append(bp)
                base_predictions = pd.concat(b_predictions, axis=0)
                if self.ensemble_mode == 'avg':
                    base_predictions = base_predictions.groupby("Path").mean()
                    base_predictions = base_predictions.reset_index().rename(columns={'index':'Path'})
            # set up sample dataframe
            sample_df = original_gt[['patient_id', 'Path']].copy()
            for subclasses in self.input_classes.values():
                for ic in subclasses:
                    sample_df[ic] = sample_df.Path.map(original_gt.set_index("Path")[ic])
            for oc in self.output_classes:
                sample_df[f"{oc} score"] = sample_df.Path.map(base_predictions.set_index("Path")[oc])
            sample_df.to_csv(self.sample_fp, index=False)
        self.sample_df = sample_df
        if os.path.exists(os.path.join(self.experiment_dir, 'decision_boundary_args.json')) and overwrite == False:
            # resume trial
            print("resuming trial..")
            self.load()
        else:
            # set up trial
            print("setting up trial..")
            self.trial_setup(**kwargs)
        self.sample_df = sample_df
        for ii, triplet in enumerate(self.triplets):
            if ii not in self.triplet_information:
                ii = str(ii)
            self.run_trial(ii, predictor, data_args, model_args, sample_df)
            self.uncertainty_analysis(self.triplet_information[ii]['triplet id'])
            self.thresholded_analysis(self.triplet_information[ii]['triplet id'])

    def run_trial(self, trip_idx, predictor, data_args, model_args, sample_df):
        if self.triplet_information[trip_idx]['samples'] >= self.max_samples:
            # print(f"{self.triplet_information[trip_idx]['triplet id']} already complete")
            return
        # load triplet information
        img_triplets = pd.read_json(self.triplet_information[trip_idx]['triplet fp'], orient='table')
        if os.path.exists(self.triplet_information[trip_idx]['array fp']):
            DB_arrays = np.load(self.triplet_information[trip_idx]['array fp'], allow_pickle=True)
            DB_arrays = dict(DB_arrays)
            if self.save_last_dense_layer:
                DN_arrays = np.load(self.triplet_information[trip_idx]['array fp'].replace('.npz', '__last_dense.npz'), allow_pickle=True)
                DN_arrays = dict(DN_arrays)
        else:
            DB_arrays = {}
            if self.save_last_dense_layer:
                DN_arrays = {}
        for i in range(self.triplet_information[trip_idx]['samples'], self.max_samples):
            self.sample_df = sample_df
            print(f"{self.triplet_information[trip_idx]['triplet id']}: {i+1}/{self.max_samples}")
            # evaluate for each triplet
            current_sample = self.get_triplet_img_idxs(img_triplets.iloc[i].values)
            for pl_setting in self.planeloader_settings: # TODO: save arrays for multiple trial settings
                # different options (such as random image shuffling, noise replacement, etc.)
                planeloader = get_planeloader(data_args, dataframe=self.sample_df, img_idxs=current_sample, subgroups=self.input_classes, 
                                              prediction_tasks=model_args.tasks, **pl_setting)
                if self.save_last_dense_layer:
                    predictions, groundtruth, last_dense_layer = predictor.predict(planeloader, return_embeddings=True)
                    DN_arrays[f"{i+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}"] = last_dense_layer
                elif not self.ensemble:
                    predictions, groundtruth = predictor.predict(planeloader)
                else: 
                    prediction_list = {}
                    for key, pred in predictor.items():
                        p, groundtruth = pred.predict(planeloader)
                        prediction_list[key] = p
                    predictions = pd.concat(prediction_list.values(), axis=0).reset_index()
                    predictions = predictions.groupby('index').mean()
                for al_setting in self.analysis_settings: #TODO: test with save_last_dense_layer
                    if not self.ensemble:
                        db_results, db_array = decision_region_analysis(predictions, planeloader, self.output_classes)
                        DB_arrays[f"{i+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}"] = db_array
                    else:
                        for key in prediction_list:
                            db_results, db_array = decision_region_analysis(prediction_list[key], planeloader, self.output_classes)
                            DB_arrays[f"{i+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}_{key}"] = db_array
                        # get average prediction Decision region
                        db_results, db_array = decision_region_analysis(predictions, planeloader, self.output_classes)
                        DB_arrays[f"{i+1}__{current_sample[0]}_{current_sample[1]}_{current_sample[2]}_ensemble"] = db_array
            self.triplet_information[trip_idx]['samples'] += 1
            if (i+1) % self.save_every == 0 or not os.path.exists(self.triplet_information[trip_idx]['array fp']): # save progress
                np.savez_compressed(self.triplet_information[trip_idx]['array fp'], **DB_arrays)
                if self.save_last_dense_layer:
                    np.savez_compressed(self.triplet_information[trip_idx]['array fp'].replace('.npz', '__last_dense.npz'), **DN_arrays)
                self.save()
            # return # DEBUG

    def trial_setup(self, planeloader_settings={}, analysis_settings={}, triplets=None, n_samples=10, save_every=1):
        # assign variables as attributes
        self.max_samples = n_samples
        self.save_every = save_every
        # different trial settings
        default_planeloader_settings = {
            'steps':100,
            'shape':'triangle'
        }
        self.planeloader_settings = []
        if planeloader_settings is not None:
            if type(planeloader_settings) is not list:
                planeloader_settings = [planeloader_settings]
            for pl in planeloader_settings:
                for i, j in default_planeloader_settings.items():
                    if i not in pl:
                        pl[i] = j
                self.planeloader_settings.append(pl)
        
        default_analysis_settings = {
            "generate_plot":False
        }
        self.analysis_settings = []
        if analysis_settings is not None:
            if type(analysis_settings) is not list:
                analysis_settings = [analysis_settings]
            for al in analysis_settings:
                for i, j in default_analysis_settings.items():
                    if i not in al:
                        al[i] = j
                self.analysis_settings.append(al)
        if triplets is None:
            self.get_class_triplets()
        else:
            self.triplets = triplets
        self.triplet_information = {}
        for ii, triplet in enumerate(self.triplets):
            self.triplet_information[ii] = {'triplet':triplet,
                                            'samples':0}
        # load in samples from dataframe
            # Note: df columns should contain: patient id, Path, truth values for each of the input classes, and output SCORES for each of the output classes
        # classify each as w/in a subgroup
        self.sample_df['subgroup'] = None
        for isub in self.interaction_subgroups:
            temp_df = self.sample_df.copy()
            for x in isub:
                temp_df = temp_df[temp_df[x] == 1]
            idxs = temp_df.index
            self.sample_df.loc[idxs, 'subgroup'] = "-".join(isub)
        # generate triplets for each subgroup
        # TODO: correct-only / incorrect only restrictions
        for ii, triplet in enumerate(self.triplets): # TODO: non-consistent triplets (?)
            isub = triplet[0]
            sub_id = "".join(isub)
            for i, j in abbreviation_table.items():
                sub_id = sub_id.replace(i,j)
            self.triplet_information[ii]['triplet id'] = sub_id
            sub_trip_fp = os.path.join(self.experiment_dir, f"{sub_id}_triplets.json")
            sub_array_fp = os.path.join(self.experiment_dir, f"{sub_id}_arrays.npz")
            sub_df = self.sample_df[self.sample_df['subgroup'] == "-".join(isub)]
            sub_pids = sub_df['patient_id'].unique()
            sub_pid_triplets = self.get_pid_combos(sub_pids)
            if len(sub_pid_triplets) < self.max_samples:
                raise Exception(f"patient id triplet generation only made {len(sub_pid_triplets)} triplets, cannot generate {self.max_samples} samples")
            sub_img_triplets = self.pid_to_img(sub_pid_triplets, sub_df)
            sub_img_triplets.to_json(sub_trip_fp, orient='table', indent=1)
            self.triplet_information[ii]['triplet fp'] = sub_trip_fp
            self.triplet_information[ii]['array fp'] = sub_array_fp
        self.save()
    
    def get_triplet_img_idxs(self, triplet):
        idxs = []
        for t in triplet:
            idxs.append(self.sample_df[self.sample_df['Path'] == t].index.values[0])
        return idxs

    def pid_to_img(self, pid_triplets, df): # returns triplets of img paths
        img_triplets = []
        for trip in pid_triplets:
            curr_triplet = []
            for pid in trip:
                img_idx = df[df['patient_id'] == pid].sample(1).Path.values[0] # TODO: random state control (?)
                curr_triplet.append(img_idx)
            img_triplets.append(curr_triplet)
        return pd.DataFrame(img_triplets, columns=['img0','img1','img2'])

    def get_pid_combos(self, pids):
        out_pids = []
        # figure out how many times we have to repeat each pid
        n_pid_reuse =int((self.max_samples/(len(pids)/3))+2)
        # get pid combos
        for r in range(n_pid_reuse):
            r_pids = [p for p in pids]
            for n in range(int(len(pids)/3)):
                item = sample(r_pids, 3)
                while set(item) in out_pids: # don't allow the exact same set of patients to be used more than once
                    item = sample(r_pids, 3)
                for i in item:
                    r_pids.remove(i)
                out_pids.append(set(item))
        return [list(p) for p in out_pids]

    def get_class_triplets(self, consistent_only=True): # TODO: inconsistent triplets?
        self.interaction_subgroups = list(product(*self.input_classes.values()))
        if not consistent_only:
            self.triplets = list(combinations_with_replacement(self.interaction_subgroups, 3))
        else:
            self.triplets = [[item]*3 for item in self.interaction_subgroups]

    def save(self):
        '''
        Save experiment attributes to json file
        '''
        with open(os.path.join(self.experiment_dir, 'decision_boundary_args.json'), 'w') as outfile:
            variables = vars(self)
            variables.pop('sample_df')
            json.dump(variables, outfile, indent=2)
    
    def load(self):
        '''
        Load experiemnt from json file
        '''
        with open(os.path.join(self.experiment_dir, 'decision_boundary_args.json'), 'r') as infile:
            input_atts = json.load(infile)
        for key in input_atts:
            setattr(self, key, input_atts[key])

    def plot_sample(self, triplet_id, idx, plot_mode='colorbar'):
        import seaborn as sns
        save_name = os.path.join(self.experiment_dir, 'test_img.png') # TODO
        # get sample
        for ii, trip_dict in self.triplet_information.items():
            if trip_dict['triplet id'] == triplet_id:
                DB_arrays = dict(np.load(trip_dict['array fp'], allow_pickle=True))
        for DB in DB_arrays:
            if int(DB.split("__")[0]) == idx:
                arr = DB_arrays[DB]
                DB_title = DB
        cols = [f"{out} Score" for out in self.output_classes] + ['x','y']
        df = pd.DataFrame(arr, columns=cols)
        if plot_mode == 'threshold': # TODO: threshold and label
            print("WIP")
        elif plot_mode == 'colorbar':
            fig, ax = plt.subplots(figsize=(8,6))
            # currently hardcoded to P/N prediction
            if "N" in triplet_id:
                scatter = ax.scatter(x=df['x'], y=df['y'], c=df['No Score'],cmap=sns.color_palette("viridis_r", as_cmap=True),vmin=0,vmax=1, s=10)
            elif 'P' in triplet_id:
                scatter = ax.scatter(x=df['x'], y=df['y'], c=df['Yes Score'],cmap=sns.color_palette("viridis", as_cmap=True),vmin=0,vmax=1, s=10)
            plt.colorbar(scatter)
            ax.axis("off")
            plt.title(DB_title.split("__")[-1])
            plt.savefig(save_name, dpi=300, bbox_inches='tight')

    def uncertainty_analysis(self, triplet_id):
        save_name = os.path.join(self.experiment_dir, f"{triplet_id}_uncertainty.csv")
        if len(self.output_classes) == 1:
            cls = self.output_classes[0]
        elif "N" in triplet_id:
            cls = "No"
        elif "P" in triplet_id:
            cls = 'Yes'
        else:
            print(f"couldn't get class from {triplet_id}")
        for ii, trip_dict in self.triplet_information.items():
            if trip_dict['triplet id'] == triplet_id:
                DB_arrays = dict(np.load(trip_dict['array fp'], allow_pickle=True))
        df = pd.DataFrame(columns=['subgroup','id','median', 'IQR'])
        for ii, arr in DB_arrays.items():
            cols = [f"{out} Score" for out in self.output_classes] + ['x','y']
            DB_df = pd.DataFrame(arr, columns=cols)
            # get IQR
            Q3 = np.quantile(DB_df[f"{cls} Score"], 0.75)
            Q1 = np.quantile(DB_df[f"{cls} Score"], 0.25)
            df.loc[len(df)] = [triplet_id, ii, DB_df[f"{cls} Score"].median(), Q3-Q1]
        df.to_csv(save_name, index=False)

    def thresholded_analysis(self, triplet_id):
        save_name = os.path.join(self.experiment_dir, f"{triplet_id}_thresholded_summary.csv")
        if len(self.output_classes) == 1:
            cls = self.output_classes[0]
            if 'N' in triplet_id:
                exp_val = 0
            elif 'P' in triplet_id:
                exp_val = 100
        elif "N" in triplet_id:
            cls = "No"
            exp_val = 100
        elif "P" in triplet_id:
            cls = 'Yes'
            exp_val = 100
        else:
            print(f"couldn't get class from {triplet_id}")
        for ii, trip_dict in self.triplet_information.items():
            if trip_dict['triplet id'] == triplet_id:
                DB_arrays = dict(np.load(trip_dict['array fp'], allow_pickle=True))
        df = pd.DataFrame(columns=['subgroup','id', 'difference'])
        for ii, arr in DB_arrays.items():
            cols = [f"{out} Score" for out in self.output_classes] + ['x','y']
            DB_df = pd.DataFrame(arr, columns=cols)
            DB_df[f"{cls} Score"] = DB_df[f"{cls} Score"].round()
            percent = (DB_df[f"{cls} Score"].sum()/len(DB_df)) * 100
            df.loc[len(df)] = [triplet_id, ii, abs(exp_val-percent)]
        df.to_csv(save_name, index=False)
