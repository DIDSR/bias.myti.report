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
import seaborn as sns

def get_sample_list(subclasses, df):
    output = []
    # select only samples of the particular subclass from the dataframe
    for subclass in subclasses:
        if "F" in subclass:
            sub_df = df[df.Female == 1]
        elif "M" in subclass:
            sub_df = df[df.Male == 1]
        if "CR" in subclass:
            sub_df = sub_df[sub_df.CR == 1]
        elif "DX" in subclass:
            sub_df = sub_df[sub_df.DX == 1]
        # collect the index values
        output.append(list(sub_df.index.values))
    return output
        
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
    def __init__(self, 
                csv_name,
                selection_mode='random',
                samples=None,
                is_training=False, 
                study_level=False, 
                return_info_dict=False, 
                data_args=None,
                steps=100,
                scale=320,
                data_mode='normal',
                model=None,
                save_result_images=False):
        self.csv_name = csv_name
        self.is_training = is_training
        self.study_level = study_level
        self.return_info_dict = return_info_dict
        self.data_args = data_args

        csv_df = pd.read_csv(self.csv_name)
        # # NOTE: currently resizing imgs to 320x320, as they all need to be the exact same dimensions to get plane
        self.transform = transforms.Compose([transforms.Resize((scale,scale)),transforms.PILToTensor()])
        
        # select samples
        sample_idx = []
        if selection_mode == 'index':
            if len(samples) != 3:
                print(f'Index selection requires 3 samples, found {len(samples)}')
                return
            sample_idx = samples
        elif selection_mode == 'class':
            print('=====================')
            print(f'csv_df columns: {csv_df.columns}')
            if len(samples) != 3:
                print(f'Class selection requires 3 samples, found {len(samples)}')
                return
            for sample in samples:
                if 'F' in sample:
                    sub_df = csv_df[csv_df.Female == 1]
                elif 'M' in sample:
                    sub_df = csv_df[csv_df.Male == 1]
                if 'CR' in sample:
                    sub_df = sub_df[sub_df.CR == 1]
                elif 'DX' in sample:
                    sub_df = sub_df[sub_df.DX == 1]
                sample_idx += list(sub_df.sample().index)
        elif selection_mode == 'random':
            print('random')
            sample_idx += list(csv_df.sample(n=3).index)
        else:
            print(f'Unknown selection mode {selection_mode}. Returning')
            return

        print(sample_idx)
        sample_classes = []
        for idx in sample_idx:
            if csv_df.iloc[idx]['Female'] == 1:
                sex = 'Female'
            elif csv_df.iloc[idx]['Male'] == 1:
                sex = 'Male'
            if csv_df.iloc[idx]['CR'] == 1:
                mod = 'CR'
            elif csv_df.iloc[idx]['DX'] == 1:
                mod = 'DX'
            sample_classes += [f'{sex}-{mod}']
        print(sample_classes)
        # load base images ===============
        img_data = ({'path':[csv_df.iloc[i]['Path'] for i in sample_idx],
                    'class':sample_classes})
        imgs = [Image.open(path).convert('RGB') for path in img_data['path']]
        imgs = [self.transform(img).to(torch.float) for img in imgs]
        
        # troubleshooting
        self.base_img_mean = torch.mean(torch.stack(imgs),[0,2,3], keepdim=True)
        self.base_img_mean = torch.squeeze(self.base_img_mean)
        self.base_img_std = torch.std(torch.stack(imgs),[0,2,3], keepdim=True)
        self.base_img_std = torch.squeeze(self.base_img_std)
        # set up normalization for interpolated images
        self.norm = transforms.Normalize(self.base_img_mean, self.base_img_std)
        #self.base_images = imgs
        #self.base_labels = sample_classes
        self.basis = {'images':imgs,
                      'labels':sample_classes,
                      'index':sample_idx,
                      'img_path':img_data['path']}

        # print(img_data['path'])
        # # get plane
        self.vec1, self.vec2, b, self.coords = get_plane(imgs[0], imgs[1], imgs[2])
        # view images
        if save_result_images:
            # save original three images in plot
            fig2, ax2 = plt.subplots(1,3)
            for i in range(3):
                ax2[i].imshow(imgs[i].permute(1,2,0).numpy())
            plt.savefig("/gpfs_projects/alexis.burgon/OUT/2022_CXR/decision_boundaries/base_imgs.png")

        # # set constants
        range_l = 0.1
        range_r = 0.1
        # # create plane dataset
        self.base_img = imgs[0]
        x_bounds = [coord[0] for coord in self.coords]
        y_bounds = [coord[1] for coord in self.coords]

        self.bound1 = [torch.min(torch.tensor(x_bounds)), torch.max(torch.tensor(x_bounds))]
        self.bound2 = [torch.min(torch.tensor(y_bounds)), torch.max(torch.tensor(y_bounds))]
        len1 = self.bound1[-1] - self.bound1[0]
        len2 = self.bound2[-1] - self.bound2[0]

        self.steps = steps

        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, self.steps)
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, self.steps)
        grid = torch.meshgrid([list1, list2])

        self.coefs1 = grid[0].flatten()
        self.coefs2 = grid[1].flatten()
        self.df = self.load_df()

        if save_result_images:
            fig, ax = plt.subplots(self.steps, self.steps)
            for i in range(len(self.df)):
                image = self.df.iloc[i]['Image'].permute(1,2,0).numpy()
                ax[int(i/self.steps), i%self.steps].imshow(image)
                ax[int(i/self.steps), i%self.steps].axis('off')
            plt.savefig("/gpfs_projects/alexis.burgon/OUT/2022_CXR/decision_boundaries/test_images.png")

        
    def load_df(self):
        print("loading df")
        images = []
        for i in range(self.coefs1.shape[0]):
            images += [self.base_img + self.coefs1[i] * self.vec1 + self.coefs2[i] * self.vec2]
            # normalize values based on base images
            images[i] = self.norm(images[i])
                    
        labels = [np.array([0,0,0,0])] * len(images)
        # Unsure of column names needed
        self.images = images
        df = pd.DataFrame(list(zip(images, labels)), columns=['Image', 'Label'])
        return df

    def __len__(self):
        return self.coefs1.shape[0]

    def get_study(self, index=None):
        images = self.df['Image'].tolist()
        labels = self.df['Label'].tolist()
        return images, labels
    
    def get_image(self, index):
        return self.df.iloc[index]['Image'], self.df.iloc[index]['Label']

    def __getitem__(self, index):
        if self.study_level:
            return self.get_study(index)
        else:
            return self.get_image(index)


def get_planeloader(data_args, csv_input,selection_mode, samples, steps, data_mode='normal', model=None, save_result_images=False):
    dataset = plane_dataset(csv_input,
                            selection_mode=selection_mode,
                            samples=samples,
                            steps=steps,
                            data_mode=data_mode,
                            model=model,
                            save_result_images=save_result_images)
    planeloader = data.DataLoader(dataset=dataset,
        batch_size=data_args.batch_size,
        shuffle=False,
        num_workers=data_args.num_workers)
    return planeloader

def imscatter(x, y, image, ax=None, zoom=1):
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x,y)
    artists = []
    ab = AnnotationBbox(im, (x,y), xycoords='data', frameon=False)
    artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x,y]))
    ax.autoscale()
    return artists

# TESTING FUNCTIONALITY
def fudge_data(predictions, planeloader):
    predictions = np.zeros((len(planeloader.dataset),14))
    c1 = round(len(planeloader.dataset)/3)
    c2 = round(2*len(planeloader.dataset)/3)
    predictions[:c1,0] = 1
    predictions[c1:c2,1] = 1
    predictions[c2:,2] = 1
    predictions[1000:2000, 3] = 2
    predictions[6000:7500, 7] = 2
    return predictions
    

def plot_decision_boundaries(predictions,
                            planeloader,
                            labels,
                            synthetic_predictions,
                            classes,
                            plot_mode='max',
                            save_loc="/gpfs_projects/alexis.burgon/OUT/2022_CXR/decision_boundaries/test.png"):
    """
    generates a plot of decision boundaries from predictions on a planeloader dataset
    and saves as a png

    predictions: numpy array of shape (samples, classes)
    planeloader: dataloader of a plane_dataset
    labels: labels of the original three images
    synthetic_predictions: bool, whether or not to fake predictions
    save_loc: save location of output plot
    """
    
    # # plot settings ================================================
    # col_map = cm.get_cmap('tab20')
    col_palette = sns.color_palette("hls", 14)
    color_dict = {"F-CR": "#dd75b0",
                  "F-DX": "#db2b8f",
                  "M-CR": "#759add",
                  "M-DX": "#3973dd"}
    col_palette = sns.color_palette([color_dict['F-CR'],
                                    color_dict['F-DX'],
                                    color_dict['M-CR'],
                                    color_dict['M-DX']])
    fig, ax = plt.subplots(figsize=(8,6))
    #ax.axis('tight')
    ax.tick_params(labelsize=8)

    col_map = ListedColormap(col_palette.as_hex())
    cmaplist = [col_map(i) for i in range(col_map.N)]
    cmaplist = cmaplist[:len(classes)]
    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist,N=len(classes))

    if synthetic_predictions:
        print('Generating synthetic predictions ...')
        predictions = fudge_data(predictions, planeloader)

    preds = torch.tensor(predictions.values)
    summ_df = pd.DataFrame({'class':classes, 'occurances':[0]*len(classes), 'percent':[0]*len(classes)})


    if plot_mode == 'max':
        val = torch.max(preds, dim=1)[0].numpy()
        class_pred = torch.argmax(preds, dim=1).numpy()
    elif plot_mode =='overlap':
        # this is pretty much hardcoded for male-DX, male-CR, female-DX, female-CR
        mod_pred = torch.argmax(preds[:,:2], dim=1).numpy() # CR=0, DX=1
        sex_pred = torch.argmax(preds[:,2:], dim=1).numpy() # F=0, M=1

        class_pred = np.zeros((preds.size()[0], len(classes)))
        for i in range(len(classes)):
            print(f'working on {classes[i]}')
            if 'F' in classes[i]:
                class_pred[:,i] += 1 - sex_pred[:]
            elif 'M' in classes[i]:
                class_pred[:,i] += sex_pred[:]
            if 'CR' in classes[i]:
                class_pred[:,i] +=  1 - mod_pred[:]
            elif 'DX' in classes[i]:
                class_pred[:,i] += mod_pred[:]
        class_pred = np.argmax(class_pred, axis=1)

    classes_found, class_counts = np.unique(class_pred, return_counts=True)

    for i in range(len(classes_found)):
        class_idx = int(classes_found[i])
        summ_df['occurances'].iloc[class_idx] = class_counts[i]
        summ_df['percent'].iloc[class_idx] = (class_counts[i]/np.sum(class_counts))*100

    print(summ_df)

    x = planeloader.dataset.coefs1.numpy()
    y = planeloader.dataset.coefs2.numpy()
    #class_dict = {classes[i]:i for i in range(len(classes))}

    label_color_dict = dict(zip([*range(len(classes))],cmaplist))
    
    color_idx = [label_color_dict[label] for label in class_pred]

    scatter = ax.scatter(x,y,c=color_idx, s=2) # original had alpha=val, but alpha needs to be a scalar and val is a tuple


    coords = planeloader.dataset.coords
    # # add markers for original 3 images
    img_markers = [".",".","."]
    img_points = []
    for i in range(3):
        label = planeloader.dataset.basis['labels'][i]
        img_points.append(ax.scatter(coords[i][0], coords[i][1],s=10, c='black'))
        # place class label with original 3 points (slight offset to increase legibility)
        plt.text(coords[i][0]-1500, coords[i][1]+500, planeloader.dataset.basis['labels'][i])
    #img_legend = plt.legend(img_points, [labels[i] for i in range(len(labels))], bbox_to_anchor=(1.0,1.0), loc='lower left')
    
    patch_list = []
    for i in classes_found:
        patch_list.append(mpatches.Patch(color=cmaplist[i], label=f'{classes[i]}'))
    
    plt.legend(handles=patch_list, 
              bbox_to_anchor=(0.5, 0),
              loc='upper center',
              ncol=len(patch_list),
              borderaxespad=2)
    #plt.gca().add_artist(img_legend)
    plt.title(planeloader.dataset.basis['index'])
    plt.savefig(save_loc, bbox_inches='tight', dpi=300)
    plt.close(fig)
    return




    
