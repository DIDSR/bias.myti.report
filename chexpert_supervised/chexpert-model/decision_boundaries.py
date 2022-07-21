import torch.utils.data as data
from data.chexpert_dataset import CheXpertDataset
from data.custom_dataset import CustomDataset
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


def get_plane(img1, img2, img3):
    a = img2 - img1
    b = img3 - img1
    a = a.to(torch.float)
    b = b.to(torch.float)
    a_norm = torch.dot(a.flatten(), b.flatten())
    a = a / a_norm
    first_coef = torch.dot(a.flatten(), b.flatten())
    b_orthog = b - first_coef * a
    b_orthog_norm = torch.dot(b_orthog.flatten(), b_orthog.flatten()).sqrt()
    b_orthog = b_orthog / b_orthog_norm
    second_coef = torch.dot(b.flatten(), b_orthog.flatten())
    coords = [[0,0], [a_norm,0], [first_coef, second_coef]]
    return a, b_orthog, b, coords

class plane_dataset(data.Dataset):
    def __init__(self, 
                csv_name,
                is_training=False, 
                study_level=False, 
                return_info_dict=False, 
                data_args=None,
                steps=100):
        self.csv_name = csv_name
        self.is_training = is_training
        self.study_level = study_level
        self.return_info_dict = return_info_dict
        self.data_args = data_args

        csv_df = pd.read_csv(self.csv_name)
        self.transform = transforms.Compose([transforms.PILToTensor()])
        # select three images (two pleural effustion negative, one plueral effusion positive)
        img_data = ({
            'path': [csv_df[csv_df['Pleural Effusion'] != 1].iloc[0]['Path'],
            csv_df[csv_df['Lung Lesion'] == 1].iloc[1]['Path'],
            csv_df[csv_df['Atelectasis'] == 1].iloc[0]['Path']],
            'pleural effusion':[0,0,1]
        })
        image_df = pd.DataFrame(data=img_data)
        imgs = [Image.open(path).convert('RGB') for path in img_data['path']]
        imgs = [self.transform(img) for img in imgs]
        self.base_images = imgs
        self.base_labels = [0,1,2]
        # print(img_data['path'])


        # # get plane
        self.vec1, self.vec2, b, self.coords = get_plane(imgs[0], imgs[1], imgs[2])
        # # set constants
        self.resolution = 0.2
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

        # Fixing Problems ========================================================
        self.steps = steps

        list1 = torch.linspace(self.bound1[0] - range_l*len1, self.bound1[1] + range_r*len1, self.steps)
        list2 = torch.linspace(self.bound2[0] - range_l*len2, self.bound2[1] + range_r*len2, self.steps)
        grid = torch.meshgrid([list1, list2])

        self.coefs1 = grid[0].flatten()
        self.coefs2 = grid[1].flatten()
        self.df = self.load_df()

    
    def load_df(self):
        images = []
        for i in range(self.coefs1.shape[0]):
            images += [self.base_img + self.coefs1[i] * self.vec1 + self.coefs2[i] * self.vec2]
            
        # print(f"Generated {len(images)} images")
        labels = [0] * len(images)
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

def get_planeloader(data_args):
    dataset = plane_dataset('/gpfs_projects/ravi.samala/OUT/moco/reorg_chexpert/moving_logs/fine_tune_train_log_small.csv')
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
    

def plot_decision_boundaries(predictions, planeloader, labels, synthetic_predictions,
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
    classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
    'Airspace Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax',
    'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

    summ_df = pd.DataFrame({'class':classes, 'occurances':[0]*len(classes), 'percent':[0]*len(classes)})
    images = planeloader.dataset.base_images

    # # plot settings ================================================
    # col_map = cm.get_cmap('tab20')
    col_palette = sns.color_palette("hls", 14)
    fig, ax = plt.subplots()

    col_map = ListedColormap(col_palette.as_hex())
    cmaplist = [col_map(i) for i in range(col_map.N)]
    cmaplist = cmaplist[:len(classes)]
    col_map = LinearSegmentedColormap.from_list('custom_colormap', cmaplist,N=len(classes))
    
    if synthetic_predictions:
        print('Generating synthetic predictions ...')
        predictions = fudge_data(predictions, planeloader)
        
    preds = torch.from_numpy(predictions)

    val = torch.max(preds, dim=1)[0].numpy()
    class_pred = torch.argmax(preds, dim=1).numpy()
    classes_found, class_counts = np.unique(class_pred, return_counts=True)
    #print(np.unique(class_pred, return_counts=True))
    for i in range(len(classes_found)):
        class_idx = classes_found[i]
        summ_df['occurances'].iloc[class_idx] = class_counts[i]
        summ_df['percent'].iloc[class_idx] = (class_counts[i]/np.sum(class_counts))*100
    print(summ_df)
    
    x = planeloader.dataset.coefs1.numpy()
    y = planeloader.dataset.coefs2.numpy()

    label_color_dict = dict(zip([*range(len(classes))],cmaplist))
    color_idx = [label_color_dict[label] for label in class_pred]

    scatter = ax.scatter(x,y,c=color_idx, s=1.5) # original had alpha=val, but alpha needs to be a scalar and val is a tuple
    coords = planeloader.dataset.coords

    # # add markers for original 3 images
    img_markers = ["^","s","*"]
    img_points = []
    for i in range(3):
        img_points.append(ax.scatter(coords[i][0], coords[i][1], c='black', marker=img_markers[i]))
    img_legend = plt.legend(img_points, [classes[i] for i in labels], bbox_to_anchor=(1.0,1.0), loc='lower left')
    
    patch_list = []
    for i in classes_found:
        patch_list.append(mpatches.Patch(color=cmaplist[i], label=f'{classes[i]}'))
    
    plt.legend(handles=patch_list, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    plt.gca().add_artist(img_legend)
    plt.title('TEST')
    plt.savefig(save_loc, bbox_inches='tight')
    plt.close(fig)
    return




    
