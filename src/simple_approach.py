'''
    Program that implements a simple version of CNN for understanding and debugging process
'''
# # Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from torch.optim import lr_scheduler
# # Other
from torchsummaryX import summary
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
# # Other custom
from dat_data_load import Dataset
# # 
matplotlib.use('TkAgg')


# functions to show an image
def imshow(img):
    # img = img / 2000 + 0     # unnormalize
    npimg = img.numpy()
    print([np.min(npimg), np.max(npimg)])
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# # Define a simple CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5, 1)
        self.conv3 = nn.Conv2d(16, 32, 5, 1)
        self.fc1 = nn.Linear(32 * 12 * 12, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        # self.fc4 = nn.Linear(64, 2)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x


# # --------------------------------------------------------------------
# # START HERE
# # --------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
input_tr_lis = '/udsk11/rsamala/Lists/DCNN_lists/FeatEnggSplit/atm3/CADx_Tr_MAM_256x256__322USF_585DM_1032CBIS_1283SFM__T3222.lis'
OUT_PATH = '/nas/unas25/rsamala/2022_MAM_CADx_DCNN/cifar_net.pth'
train_dataset = Dataset(input_tr_lis, crop_to_224=True, train_flag=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
# #
# #
# net = Net()
# print(net)
# net.to(device)
# x=torch.rand(16,3,224,224)
# print(summary(net, x.to(device)))
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 1)
model_ft = net.to(device)
# criterion = nn.CrossEntropyLoss()
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
        if i+1 % 25 == 0:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item() / 2000:.6f}')
print('Finished Training')


# torch.save(net.state_dict(), OUT_PATH)

# dataiter = iter(test_loader)
# fnames, images, labels = next(dataiter)

# print('GroundTruth: ', ' '.join(f'{labels[j]:d}' for j in range(4)))
# imshow(torchvision.utils.make_grid(images))

# # load the model
# net = Net()
# net.load_state_dict(torch.load(OUT_PATH))
# outputs = net(images)
# _, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join(f'{predicted[j]:d}' for j in range(4)))

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
        # print(labels)
        # print(predicted)
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
