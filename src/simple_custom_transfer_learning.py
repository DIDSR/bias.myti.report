'''
    Program that implements a simple version of CNN with custom transfer learning
    for understanding and debugging purpose

    RKS, started Nov 27, 2022. 
    Git is used to track the versions.
'''
# # Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from torch.optim import lr_scheduler
from torchsummaryX import summary
# # Other
import numpy as np
from sklearn import metrics
# # Other custom
from dat_data_load import Dataset
# # 
# # CONSTANTS
resnet18_ordered_layer_names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']

# # --------------------------------------------------------------------
# # START HERE
# # --------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
input_tr_lis = '/nas/unas25/rsamala/2022_MAM_CADx_DCNN/RAND_sampling_experiments/R0/f0/tr.lis'
input_ts_lis = '/nas/unas25/rsamala/2022_MAM_CADx_DCNN/RAND_sampling_experiments/R0/f0/ts.lis'
fine_tuning = 'full'    # # options: 'full' or 'partial'
freeze_upto = 0     # # options will be 0 to number of layers
# # 
def main():
    train_dataset = Dataset(input_tr_lis, crop_to_224=True, train_flag=True, custom_scale=True)
    test_dataset = Dataset(input_ts_lis, crop_to_224=True, train_flag=False, custom_scale=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    num_steps_in_epoch = len(train_loader)
    # #
    # #
    net = models.resnet18(pretrained=True)
    # num_ftrs = net.fc.in_features
    # net.fc = nn.Linear(num_ftrs, 1)
    new_layers = nn.Sequential(nn.Dropout(0.2), nn.Linear(1000, 512), nn.Linear(512, 128), nn.Linear(128, 1))
    net = nn.Sequential(net, new_layers)

    custom_layer_name = resnet18_ordered_layer_names.copy()
    if freeze_upto + 1 >= len(resnet18_ordered_layer_names):
        print('ERROR with choice of freeze_upto')
        return

    first_non_frozen_layer_name = custom_layer_name[freeze_upto + 1]
    # # custom transfer learning >>
    if fine_tuning == 'partial':
        spacer = ''
        start_not_freezing_from_next_layer = False
        # # set the requires_grad to False to all first
        for param in net.parameters():
            param.requires_grad = False
        print('Partial fine tuning selected')
        print('Will freeze upto {} with the layer name of {}'.format(freeze_upto, first_non_frozen_layer_name))
        # # while iterating over the layers, check with the 
        for name, param in net.named_parameters():
            # # check if the names in custom_layer_name is in this name
            if len(custom_layer_name) > 0:
                for ii, each_layer_name in enumerate(custom_layer_name):
                    # # the index in the name.split is important and may change
                    # # based on the pretrained model
                    if each_layer_name == name.split('.')[1]:
                        if each_layer_name == first_non_frozen_layer_name:
                            # # start un-freezing from here onwards
                            start_not_freezing_from_next_layer = True
                        # # found it
                        custom_layer_name.pop(ii)
                        spacer += '\t'
                        break
            if start_not_freezing_from_next_layer:
                param.requires_grad = True
                print('{} {} {}'.format(spacer, 'T', name))
            else:
                print('{} {} {}'.format(spacer, 'F', name))
    elif fine_tuning == 'full':
        print('Full fine tuning selected')
    else:
        print('ERROR. UNKNOWN option for fine_tuning')
        return
    # # <<

    # # debug code to understand how a ROI passes through the network
    x=torch.rand(16,3,224,224)
    print(summary(net, x))
    # return

    model_ft = net.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    my_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=num_steps_in_epoch*5, gamma=0.1)

    for epoch in range(100):  # loop over the dataset multiple times
        avg_loss = 0
        for i, (_, inputs, labels) in enumerate(train_loader, 0):
            inputs = inputs.cuda()
            labels = labels.cuda()

            # # zero the parameter gradients
            optimizer.zero_grad()

            # # forward + backward + optimize
            outputs = model_ft(inputs)
            loss = criterion(torch.sigmoid(torch.flatten(outputs)), labels.float())
            loss.backward()
            optimizer.step()

            # # # print statistics
            avg_loss += loss.item()
            # if i % 25 == 0:
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item() / 2000:.6f}')
        avg_loss = avg_loss / len(train_loader)
        print("> {:d}\t{:1.5f}".format(epoch, avg_loss))
        my_lr_scheduler.step()
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
            scr = list(output_scrs.cpu().numpy())
            scores_all += scr
        # #
        fpr, tpr, _ = metrics.roc_curve(np.array(type_all), np.array(scores_all), pos_label=1)
        auc_val = metrics.auc(fpr, tpr)
        print('AUC = {}'.format(auc_val))
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

if __name__ == '__main__':
    main()
