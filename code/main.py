import os
import numpy as np
import random
import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms

from glob import glob
import sys
from PIL import Image


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def inspect_images_attributes(data_path):
    sets = ['train', 'test']
    print("####################\nChecking attributes:\n####################")
    shapes = {}
    for curr_set in sets:
        images = glob(str(data_path / curr_set / '*/*/*.jpg')) if curr_set is 'train' else glob(str(data_path / curr_set / '*.jpg'))
        images_jpg = [im_path for im_path in images if im_path.endswith('.jpg')]
        images_png = [im_path for im_path in images if im_path.endswith('.png')]
        print("{} dataset contains {} jpg images and {} png images.".format(curr_set, len(images_jpg),
                                                                            len(images_png)))
        for im_name in images:
            im_shape = cv2.imread(str(im_name)).shape
            if im_shape not in shapes:
                shapes[im_shape] = 1
            else:
                shapes[im_shape] += 1

    print("Possible shapes are:\n {}".format(shapes))


def load_data(data_path, val_famillies="F09"):
    all_images = glob(str(data_path / 'train/*/*/*.jpg'))
    train_images = [x for x in all_images if val_famillies not in x]
    num_train_images = len(train_images)
    val_images = [x for x in all_images if val_famillies in x]
    num_val_images = len(val_images)
    train_family_persons_tree = {}
    val_family_persons_tree = {}
    delim = '\\' if sys.platform.startswith('win') else "/"

    for im_path in train_images:
        family_name = im_path.split(delim)[-3]
        person = im_path.split(delim)[-2]
        if family_name not in train_family_persons_tree:
            train_family_persons_tree[family_name] = {}
        if person not in train_family_persons_tree[family_name]:
            train_family_persons_tree[family_name][person] = []
        train_family_persons_tree[family_name][person].append(im_path)

    for im_path in val_images:
        family_name = im_path.split(delim)[-3]
        person = im_path.split(delim)[-2]
        if family_name not in val_family_persons_tree:
            val_family_persons_tree[family_name] = {}
        if person not in val_family_persons_tree[family_name]:
            val_family_persons_tree[family_name][person] = []
        val_family_persons_tree[family_name][person].append(im_path)

    all_relationships = pd.read_csv(str(data_path / "train_relationships.csv"))
    all_relationships = list(zip(all_relationships.p1.values, all_relationships.p2.values))
    train_pairs = [x for x in all_relationships if val_famillies not in x[0]]
    val_pairs = [x for x in all_relationships if val_famillies in x[0]]
    # make sure no need to check x[1]
    print("Total train pairs:", len(train_pairs))
    print("Total val pairs:", len(val_pairs))
    print("Total train images:", num_train_images)
    print("Total val images:", num_val_images)
    print("Dataset size: ", num_val_images + num_train_images)
    print("#########################################\n#########################################")

    return train_family_persons_tree, train_pairs, val_family_persons_tree, val_pairs

    # train_lables = pd.read_csv(data_path / 'train.csv')
    # test_lables = pd.read_csv(data_path / 'test.csv')
    # train_images_paths = [str(data_path / "train_images" / im_path) for im_path in os.listdir(data_path / "train_images")]
    # test_images_paths = [str(data_path / "test_images" / im_path) for im_path in os.listdir(data_path / "test_images")]
    # return train_images_paths, train_lables, test_images_paths, test_lables


def imshow(img, text=None, should_save=False):#for showing the data you loaded to dataloader
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration,loss):# for showing loss value changed with iter
    plt.plot(iteration,loss)
    plt.show()


class trainingDataset(Dataset):
    """
    All datasets are subclasses of torch.utils.data.Dataset i.e, they have __getitem__ and __len__ methods implemented.
    Hence, they can all be passed to a torch.utils.data.DataLoader which can load multiple samples
    parallelly using torch.multiprocessing workers
    """
    def __init__(self, imageFolderDataset, relationships, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.relationships = relationships  # choose either train or val dataset to use
        self.transform = transform
        self.delim = '\\' if sys.platform.startswith('win') else "/"

    def __getitem__(self, index):
        # Returns two images and whether they are related.
        # for each relationship in train_relationships.csv,
        # the first img comes from first row, and the second is either specially chosen related person or randomly chosen non-related person
        img0_info = self.relationships[index][0]
        img0_path = glob(str(self.imageFolderDataset.root / img0_info) + self.delim + "*.jpg")

        try:
            img0_path = random.choice(img0_path)
        except:
            img0_path = random.choice(img0_path)

# is it better to do that in advance?
        cand_relationships = [x for x in self.relationships if
                              x[0] == img0_info or x[1] == img0_info]  # found all candidates related to person in img0
        if cand_relationships == []:  # in case no relationship is mentioned. But it is useless here because I choose the first person line by line.
            relative_label = 0 # no matter what image we feed in, it is not related...
        else:
            relative_label = random.randint(0, 1)    # 50% of sampling a relative

        if relative_label == 1:  # 1 means related, and 0 means non-related.
            img1_info = random.choice(cand_relationships)  # choose the second person from related relationships
            if img1_info[0] != img0_info:
                img1_info = img1_info[0]
            else:
                img1_info = img1_info[1]
            img1_path = glob(str(self.imageFolderDataset.root / img1_info) + self.delim + "*.jpg")  # randomly choose a img of this person

            try:
                img1_path = random.choice(img1_path)
            except:
                img1_path = random.choice(img1_path)

        else:  # 0 means non-related
            randChoose = True  # in case the chosen person is related to first person
            while randChoose:
                img1_path = random.choice(self.imageFolderDataset.imgs)[0]
                img1_info = img1_path.split(self.delim)[-3] + self.delim + img1_path.split(self.delim)[-2]
                randChoose = False
                for x in cand_relationships:  # if so, randomly choose another person
                    if x[0] == img1_info or x[1] == img1_info:
                        randChoose = True
                        break

        img0 = Image.open(img0_path)
        img1 = Image.open(img1_path)

        if self.transform is not None:  # I think the transform is essential if you want to use GPU, because you have to trans data to tensor first.
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, relative_label  # the returned data from dataloader is img=[batch_size,channels,width,length], relative_label=[batch_size,label]

    def __len__(self):
        return len(self.relationships)  # essential for choose the num of data in one epoch


class SiameseNetwork(nn.Module):  # A simple implementation of siamese network, ResNet50 is used, and then connected by three fc layer.
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # self.cnn1 = models.resnet50(pretrained=True)#resnet50 doesn't work, might because pretrained model recognize all faces as the same.
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=.2),
        )
        self.fc1 = nn.Linear(2 * 32 * 100 * 100, 500)
        # self.fc1 = nn.Linear(2*1000, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 2)

    def forward(self, input1, input2):  # did not know how to let two resnet share the same param.
        output1 = self.cnn1(input1)
        output1 = output1.view(output1.size()[0], -1)  # make it suitable for fc layer.
        output2 = self.cnn1(input2)
        output2 = output2.view(output2.size()[0], -1)

        output = torch.cat((output1, output2), 1)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

#########################
### code starts here! ###
#########################

# setting the seed
np.random.seed(43)
NUM_WORKERS = 1

# Hyper params
BATCH_SIZE = 64
NUMBER_EPOCHS = 100
IMG_SIZE = 100

# Load data:
val_families = "F09" # all families starts with this str will be sent to validation set.
data_path = Path('C:\\Users\\Oren Peer\\Documents\\technion\\OneDrive - Technion\\Master\\DL_proj\\data\\faces') if sys.platform.startswith('win') \
    else    Path('/home/oren/PycharmProjects/DL_proj/data/faces/')

train_family_persons_tree, train_pairs, val_family_persons_tree, val_pairs = load_data(data_path)
# inspect_images_attributes(data_path) # All images are jpg, possible shapes:  {(224, 224, 3): 18661}

folder_dataset = dset.ImageFolder(root=data_path / 'train')
# Transforms are common image transformations. They can be chained together using Compose.
trainset = trainingDataset(imageFolderDataset=folder_dataset,
                                        relationships=train_pairs,
                                        transform=transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                                      transforms.ToTensor()]))
trainloader = DataLoader(trainset,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        batch_size=BATCH_SIZE)

valset = trainingDataset(imageFolderDataset=folder_dataset,
                                        relationships=val_pairs,
                                        transform=transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
                                                                      transforms.ToTensor()]))
valloader = DataLoader(valset,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        batch_size=BATCH_SIZE)


# Visualize data in dataloader.
# trainset.__getitem__(1)
# dataiter = iter(trainloader)
# example_batch = next(dataiter)
# concatenated = torch.cat((example_batch[0],example_batch[1]),0)
# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy())

net = SiameseNetwork().cuda()
criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

counter = []
loss_history = []
iteration_number = 0

for epoch in range(0, NUMBER_EPOCHS):
    print("Epoch：", epoch, " start.")
    for i, data in enumerate(trainloader, 0):
        img0, img1, labels = data  # img=tensor[batch_size,channels,width,length], label=tensor[batch_size,label]
        img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda()  # move to GPU
        # print("epoch：", epoch, "No." , i, "th inputs", img0.data.size(), "labels", labels.data.size())
        optimizer.zero_grad()  # clear the calculated grad in previous batch
        outputs = net(img0, img1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:  # show changes of loss value after each 10 batches
            # print("Epoch number {}\n Current loss {}\n".format(epoch,loss.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss.item())

    # test the network after finish each epoch, to have a brief training result.
    correct_val = 0
    total_val = 0
    with torch.no_grad():  # essential for testing!!!!
        for data in valloader:
            img0, img1, labels = data
            img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda()
            outputs = net(img0, img1)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    print('Accuracy of the network on the', total_val, 'val pairs in', val_families, ': %d %%' % (100 * correct_val / total_val))
    show_plot(counter, loss_history)