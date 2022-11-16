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
    predicted_labels = pd.DataFrame({"Label":predictions['Label'].values, 'x':planeloader.dataset.coefs1.numpy(), 'y':planeloader.dataset.coefs2.numpy()})
    output_array = predicted_labels.to_numpy()
    predicted_classes = predicted_labels['Label'].unique().tolist()
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
