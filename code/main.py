import numpy as np
import time
from pathlib import Path
import getpass

import torch
import torch.nn as nn
from torch import optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from OurDataset import *
from SiameseNetwork import *
from utils import *
import json
# BB
# setting the seed
# np.random.seed(43)
NUM_WORKERS = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
SAVE_MODELS = False
CREATE_SUBMISSION = True

# Hyper params
hyper_params = {
    # "init_lr": 7e-4,
    "init_lr": 1e-4,
    "BATCH_SIZE": 32,
    "NUMBER_EPOCHS": 300,
    "weight_decay": 0,
    "decay_lr": True,
    "lr_decay_factor": 0.5,
    "lr_patience": 15,  # decay every X epochs without improve
    "min_lr": 1e-6,
    "Comments": "Resnet 50, classifier 64->32->1"
}
print("Hyper parameters:", hyper_params)

model_time = round(time.time())

# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomRotation(degrees=3),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(256),
        transforms.CenterCrop(197),
        transforms.ToTensor(),
        scale_tensor_255,
        transforms.Normalize(mean=[131.0912, 103.8827, 91.4953],
                             std=[1, 1, 1])
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(197),
        transforms.ToTensor(),
        scale_tensor_255,
        transforms.Normalize(mean=[131.0912, 103.8827, 91.4953],
                             std=[1, 1, 1])
    ]),
}
# image_transforms = {"train": None, "valid": None}
# Load data:
# \\data\\faces

val_families = "F09" # all families starts with this str will be sent to validation set.
# root_folder = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root_folder = os.getcwd()

# path to the folder contains all data folders and csv files
data_path = root_folder / 'data' / 'faces'
if SAVE_MODELS:
    os.mkdir(root_folder / 'models' / str(model_time))
    # save hyper parameters to json:
    with open(str(root_folder / 'models' / str(model_time) / "Hyper_Params.json"), 'w') as fp:
        json.dump(hyper_params, fp)

train_family_persons_tree, train_pairs, val_family_persons_tree, val_pairs = load_data(data_path)

folder_dataset = dset.ImageFolder(root=data_path / 'train')

trainloader, valloader = create_datasets(folder_dataset, train_pairs, val_pairs, image_transforms,
                                         train_family_persons_tree, val_family_persons_tree,
                                         hyper_params, NUM_WORKERS)

# Visualize data in dataloader.
# trainset.__getitem__(1)
# dataiter = iter(trainloader)
# example_batch = next(dataiter)
# concatenated = torch.cat((example_batch[0],example_batch[1]),0)
# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy())

net = SiameseNetwork(model_time)
# net = nn.DataParallel(net)
net.to(device)

# criterion = nn.CrossEntropyLoss(reduction='sum').to(device)    # use a Classification Cross-Entropy loss
criterion = nn.BCELoss().to(device)     # try F.BCE...
optimizer = optim.Adam(net.parameters(), lr=hyper_params["init_lr"], weight_decay=hyper_params["weight_decay"])
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=hyper_params['lr_decay_factor'],
    patience=hyper_params['lr_patience'], verbose=1)

train_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0
IMPROVED = False     # save model only if it improves val acc.
curr_lr = hyper_params['init_lr']
print("Start training model {}! init lr: {}".format(model_time, curr_lr))
for epoch in range(0, hyper_params["NUMBER_EPOCHS"]):
    # Decay learning rate
    # if hyper_params['decay_lr'] and (epoch) % hyper_params['lr_decay_rate'] == 0 \
    #         and epoch > 0 and curr_lr > hyper_params['min_lr']:
    #     curr_lr *= hyper_params['lr_decay_factor']
    #     update_lr(optimizer, curr_lr)
    #     print('New lr: {}', curr_lr)

    # initialize epoch meters.
    batch_counter = 0
    epoch_start_time = time.time()
    train_loss = 0
    val_loss = 0
    train_acc = 0
    val_acc = 0

    net.train()  # move the net to train mode
    for i, data in enumerate(trainloader):
        IMPROVED = False
        img0, img1, labels = data  # img=tensor[batch_size,channels,width,length], label=tensor[batch_size,label]
        img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)  # move to GPU
        optimizer.zero_grad()  # clear the calculated grad in previous batch
        outputs = net(img0, img1)
        # loss = criterion(outputs.view(-1), labels.float())
        loss = criterion(outputs, labels.float().view(outputs.shape))
        # add bach loss and acc to epoch-loss\acc
        # loss:
        train_loss += loss.item()
        # acc:
        predicted = torch.round(outputs.data).long().view(-1)   # FOR BCE
        # _, predicted = torch.max(outputs.data, 1) # FOR CR
        train_acc += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()
        batch_counter += 1

    epoch_time = time.time() - epoch_start_time
    train_acc /= (0.01*trainloader.dataset.__len__())
    train_loss /= batch_counter

    # test the network after finish each epoch, to have a brief training result.
    correct_val = 0
    total_val = 0
    batch_counter = 0
    with torch.no_grad():
        net.eval()
        for data in valloader:
            img0, img1, labels = data
            img0, img1, labels = img0.cuda(), img1.cuda(), labels.cuda()
            outputs = net(img0, img1)
            # predicted = torch.round(outputs.data)
            v_loss = criterion(outputs, labels.float().view(outputs.shape))
            val_loss += v_loss.item()
            # _, predicted = torch.max(outputs.data, 1)
            predicted = torch.round(outputs.data).long().view(-1)
            # correct_val += (predicted.long()[0, :] == labels).sum().item()
            val_acc += (predicted == labels).sum().item()
            batch_counter += 1
    val_acc /= (0.01*valloader.dataset.__len__())
    val_loss /= batch_counter

    # LR Scheduler
    lr_scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        IMPROVED = True

    train_history["train_loss"].append(train_loss); train_history["val_loss"].append(val_loss)
    train_history["train_acc"].append(train_acc); train_history["val_acc"].append(val_acc)

    train_loss_diff, val_loss_diff, train_acc_diff, val_acc_diff = extract_diff_str(train_history)

    print('Epoch {:3d} done in {:.1f} sec | train_acc: {:.4f}% {} | val_acc: {:.4f}% {} | train_loss: {:.6f} {} | val_loss: {:.6f} {}; {}'.format(
        epoch+1, epoch_time,
        train_acc, train_acc_diff, val_acc, val_acc_diff,
        train_loss, train_loss_diff, val_loss, val_loss_diff, "(I)" if IMPROVED else ""))
    #
    if SAVE_MODELS:
        if IMPROVED:
            torch.save(net.state_dict(), root_folder / 'models' / str(model_time) / '{}_e{}_vl{:.4f}_va{:.2f}.pt'.format(model_time, epoch, val_loss, val_acc))
    np.save(root_folder / 'curves' / str(model_time), train_history)

    # show_plot(train_history)

if CREATE_SUBMISSION:
    model_name = get_best_model(model_folder=root_folder / 'models' / str(model_time), measure='val_acc')
    create_submission(root_folder=root_folder, model_name=model_name, transform=image_transforms['valid'], net=net)
