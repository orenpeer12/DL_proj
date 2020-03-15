import numpy as np
import time
from pathlib import Path
import getpass

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from TrainDataset import *
from SiameseNetwork import *
from utils import *

# setting the seed
np.random.seed(43)
NUM_WORKERS = 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# Hyper params
hyper_params = {
    "lr": 1e-3,
    "BATCH_SIZE": 256,
    "NUMBER_EPOCHS": 100,
    "weight_decay": 1e-4
}
print("Hyper parameters:", hyper_params)

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomRotation(degrees=3),
        # transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Load data:
data_folder = 'train'
val_families = "F09" # all families starts with this str will be sent to validation set.
if getpass.getuser() == 'nirgreshler':
    data_path = Path('E:\\DL_Course\\FacesInTheWild\\data\\faces') if sys.platform.startswith('win') \
        else Path('../data/faces/')
else:
    data_path = Path('..\\data\\faces') if sys.platform.startswith('win') \
        else Path('../data/faces/')

train_family_persons_tree, train_pairs, val_family_persons_tree, val_pairs = load_data(data_path)
# inspect_images_attributes(data_path) # All images are jpg, possible shapes:  {(224, 224, 3): 18661}

folder_dataset = dset.ImageFolder(root=data_path / data_folder)

trainset = TrainDataset(imageFolderDataset=folder_dataset,
                        relationships=train_pairs,
                        transform=image_transforms["train"],
                        familise_trees=train_family_persons_tree)

trainloader = DataLoader(trainset,
                         shuffle=True,
                         num_workers=NUM_WORKERS,
                         batch_size=hyper_params["BATCH_SIZE"])

valset = TrainDataset(imageFolderDataset=folder_dataset,
                      relationships=val_pairs,
                      transform=image_transforms["valid"],
                      familise_trees=val_family_persons_tree)

valloader = DataLoader(valset,
                       shuffle=True,
                       num_workers=NUM_WORKERS,
                       batch_size=hyper_params["BATCH_SIZE"])


# Visualize data in dataloader.
# trainset.__getitem__(1)
# dataiter = iter(trainloader)
# example_batch = next(dataiter)
# concatenated = torch.cat((example_batch[0],example_batch[1]),0)
# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy())

# net = SiameseNetwork()
net = SiameseNetwork().to(device)

criterion = nn.CrossEntropyLoss().to(device)    # use a Classification Cross-Entropy loss
# criterion = nn.BCELoss()
# optimizer = optim.Adam(net.classifier.parameters(), lr=lr, weight_decay=1e-4)
optimizer = optim.Adam(net.parameters(), lr=hyper_params["lr"], weight_decay=hyper_params["weight_decay"])

counter = []
train_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

net.train()
print("Start training!")
for epoch in range(0, hyper_params["NUMBER_EPOCHS"]):
    # initialize epoch meters.
    epoch_start_time = time.time()
    train_loss = 0
    val_loss = 0
    train_acc = 0
    val_acc = 0

    net.train()
    for i, data in enumerate(trainloader, 0):
        img0, img1, labels = data  # img=tensor[batch_size,channels,width,length], label=tensor[batch_size,label]
        img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)  # move to GPU
        optimizer.zero_grad()  # clear the calculated grad in previous batch
        outputs = net(img0, img1)
        # loss = criterion(outputs.view(-1), labels.float())
        loss = criterion(outputs, labels)
        # add bach loss and acc to epoch-loss\acc
        # loss:
        train_loss += loss.item()
        # acc:
        # predicted = torch.round(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        # correct_val += (predicted.long()[0, :] == labels).sum().item()
        train_acc += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()

    epoch_time = time.time() - epoch_start_time
    train_acc /= trainset.__len__()
    train_loss /= trainset.__len__()

    # test the network after finish each epoch, to have a brief training result.
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        net.eval()
        for data in valloader:
            img0, img1, labels = data
            img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda()
            outputs = net(img0, img1)
            # predicted = torch.round(outputs.data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            # correct_val += (predicted.long()[0, :] == labels).sum().item()
            val_acc += (predicted == labels).sum().item()
        net.train()

    val_acc /= valset.__len__()
    val_loss /= valset.__len__()

    train_history["train_loss"].append(train_loss); train_history["val_loss"].append(val_loss)
    train_history["train_acc"].append(train_acc); train_history["val_acc"].append(val_acc)

    train_loss_diff, val_loss_diff, train_acc_diff, val_acc_diff = extract_diff_str(train_history)

    print('Epoch {:3d} done in {:.1f} sec | train_acc: {:.4f} {} | val_acc: {:.4f} {} | train_loss: {:.6f} {} | val_loss: {:.6f} {}'.format(
        epoch, epoch_time,
        train_acc, train_acc_diff, val_acc, val_acc_diff,
        train_loss, train_loss_diff, val_loss, val_loss_diff))

    # show_plot(counter, loss_history)