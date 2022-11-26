'''
    Program that implements a simple version of CNN for understanding and debugging purpose
'''
# # Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import lr_scheduler
# # Other
import numpy as np
from sklearn import metrics
# # Other custom
from dat_data_load import Dataset
# # 

# # --------------------------------------------------------------------
# # START HERE
# # --------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
input_tr_lis = '/udsk11/rsamala/Lists/DCNN_lists/FeatEnggSplit/atm3/CADx_Tr_MAM_256x256__322USF_585DM_1032CBIS_1283SFM__T3222.lis'
OUT_PATH = '/nas/unas25/rsamala/2022_MAM_CADx_DCNN/cifar_net.pth'
# # 
train_dataset = Dataset(input_tr_lis, crop_to_224=True, train_flag=True, custom_scale=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
# #
# #
net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 1)
model_ft = net.to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

for epoch in range(50):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, (filenames, inputs, labels) in enumerate(train_loader, 0):
        inputs = inputs.cuda()
        labels = labels.cuda()

        # # zero the parameter gradients
        optimizer.zero_grad()

        # # forward + backward + optimize
        outputs = model_ft(inputs)
        loss = criterion(torch.sigmoid(torch.flatten(outputs)), labels.float())
        loss.backward()
        optimizer.step()

        # # print statistics
        if i % 25 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item() / 2000:.6f}')
print('Finished Training')

# # deployment
type_all = []
scores_all = []
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        fnames, images, labels = data
        # calculate outputs by running images through the network
        outputs = model_ft(images.cuda())
        output_scrs = torch.sigmoid(torch.flatten(outputs))
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.cpu()).sum().item()
        # #
        labl_list = list(labels.cpu().numpy())
        type_all += labl_list
        # scr = list(outputs.cpu().numpy()[:, 1])
        # scr = list(outputs.cpu().numpy())
        scr = list(output_scrs.cpu().numpy())
        scores_all += scr
    # #
    fpr, tpr, thresholds = metrics.roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
    auc_val = metrics.auc(fpr, tpr)
    print('AUC = {}'.format(auc_val))
print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
