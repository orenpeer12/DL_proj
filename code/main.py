import numpy as np

from pathlib import Path

import getpass
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms

import TrainDataset
import SiameseNetwork
from utils import *

# setting the seed
np.random.seed(43)
NUM_WORKERS = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Hyper params
BATCH_SIZE = 64
NUMBER_EPOCHS = 100
IMG_SIZE = 100

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load data:
val_families = "F09" # all families starts with this str will be sent to validation set.
if getpass.getuser() == 'nirgreshler':
    data_path = Path('E:\\DL_Course\\FacesInTheWild\\data\\faces') if sys.platform.startswith('win') \
        else Path('../data/faces/')
else:
    data_path = Path('..\\data\\faces') if sys.platform.startswith('win') \
        else Path('../data/faces/')

train_family_persons_tree, train_pairs, val_family_persons_tree, val_pairs = load_data(data_path)
# inspect_images_attributes(data_path) # All images are jpg, possible shapes:  {(224, 224, 3): 18661}

folder_dataset = dset.ImageFolder(root=data_path / 'train')
# Transforms are common image transformations. They can be chained together using Compose.
trainset = TrainDataset(imageFolderDataset=folder_dataset,
                        relationships=train_pairs,
                        transform=image_transforms["train"])
trainloader = DataLoader(trainset,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        batch_size=BATCH_SIZE)

valset = TrainDataset(imageFolderDataset=folder_dataset,
                      relationships=val_pairs,
                      transform=image_transforms["valid"])
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

# net = SiameseNetwork()
net = SiameseNetwork().to(device)  # TODO change to ResNet ?
# criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
criterion = nn.BCELoss()  # use a Classification Cross-Entropy loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # TODO change to Adam ?

counter = []
loss_history = []
iteration_number = 0

for epoch in range(0, NUMBER_EPOCHS):
    print("Epoch：", epoch, " start.")
    for i, data in enumerate(trainloader, 0):
        img0, img1, labels = data  # img=tensor[batch_size,channels,width,length], label=tensor[batch_size,label]
        img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)  # move to GPU
        # print("epoch：", epoch, "No." , i, "th inputs", img0.data.size(), "labels", labels.data.size())
        optimizer.zero_grad()  # clear the calculated grad in previous batch
        outputs = net(img0, img1)
        loss = criterion(outputs.view(-1), labels.float())
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