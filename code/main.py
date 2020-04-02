# region Imports
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
import os
# endregion

# region Run Settings and Definitions

# np.random.seed(43)
NUM_WORKERS = 4
GPU_ID = 0 if 'nir' in os.getcwd() else 1
# GPU_ID = 0

device = torch.device('cuda: ' + str(GPU_ID) if (torch.cuda.is_available() and not sys.platform.__contains__('win')) else 'cpu')
SAVE_MODELS = False
CREATE_SUBMISSION = True
# for colab:
# root_folder = Path('/root/')
root_folder = Path(os.getcwd())

# set kaggle folder
os.environ["KAGGLE_CONFIG_DIR"] = str(root_folder / '..')
# endregion

# val_sets = ["F06", "F07", "F08", "F09"]
val_sets = ["F09"]
dataset_version = 'data_mod'
# dataset_version = 'data'
# For now, ensembles are different in val-sets.
# region Hyper Parameters
hyper_params = {
    "equal_sampling": 1,  # whether to sample equally from each class in each batch
    "init_lr": 1e-5,
    "BATCH_SIZE": 32,
    "NUMBER_EPOCHS": 100,
    "weight_decay": 1e-5,
    "decay_lr": True,
    "lr_decay_factor": 0.1,
    "lr_patience": 10,  # decay every X epochs without improve
    "es_patience": 20,
    "es_delta": 0.001
}
print("Hyper parameters:", hyper_params)
# endregion
mean = [91.4953, 103.8827, 131.0912]
std = [1, 1, 1]
# region Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomRotation(degrees=3),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(),
        transforms.ToTensor(),
        scale_tensor_255,
        rgb2bgr,
        transforms.Normalize(mean=mean,
                             std=std)
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.ToTensor(),
        scale_tensor_255,
        rgb2bgr,
        transforms.Normalize(mean=mean,
                             std=std)
    ]),
}
# endregion

# region Load Data
# val_families = "F09"  # all families starts with this str will be sent to validation set.

# path to the folder contains all data folders and csv files
data_path = root_folder / dataset_version / 'faces'
folder_dataset = dset.ImageFolder(root=data_path / 'train')

# Visualize data in dataloader.
# trainset.__getitem__(1)
# dataiter = iter(trainloader)
# example_batch = next(dataiter)
# concatenated = torch.cat((example_batch[0],example_batch[1]),0)
# imshow(torchvision.utils.make_grid(concatenated))
# print(example_batch[2].numpy())
# endregion

# region Define Model
model_name = time.strftime('%d.%m.%H.%M.%S')
# ensemble.append({"model name": model_name, "val family:": val_families})
net = SiameseNetwork(model_name)
net.to(device)
# endregion

# region Define Loss and Optimizer
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=hyper_params["init_lr"], weight_decay=hyper_params["weight_decay"])
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=hyper_params['lr_decay_factor'],
    patience=hyper_params['lr_patience'], verbose=1)

# endregion

# THE FOLLOWING COMMENTED CODE IS ONLY FOR SUBMITTING
# model_name = '02.04.09.42.16'
# best_model_name = get_best_model(model_folder=root_folder / 'models' / model_name, measure='val_acc', measure_rank=1)
# create_submission(
#     root_folder=root_folder, model_name=best_model_name, transform=image_transforms['valid'], device=device)
# print('Created submission file', best_model_name.replace('.pt', '.csv'))
# submission_file_path = str(root_folder / 'submissions_files' / best_model_name.replace('.pt', '.csv'))
# # submit file
# os.system('kaggle competitions submit -c recognizing-faces-in-the-wild -f ' + \
#           submission_file_path + ' -m ' + best_model_name.replace('.pt', '.csv') + '_after_load')
# # show submissions
# os.system('kaggle competitions submissions recognizing-faces-in-the-wild')
# exit()

# region Training
train_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc = 0
IMPROVED = False     # save model only if it improves val acc.
curr_lr = hyper_params['init_lr']
print("Start training model {}! init lr: {}".format(model_name, curr_lr))

# region Save Run Definitions (Model and Hyper Parameters)
if SAVE_MODELS:
    # save pre-trained model and our classifier arch.
    hyper_params["feature_extractor"] = net.features._get_name()
    hyper_params["classifier"] = net.classifier.__str__()
    hyper_params["criterion"] = criterion.__str__()
    hyper_params["optimizer"] = optimizer.__str__()
    hyper_params["transforms"] = transforms.__str__()
    os.mkdir(root_folder / 'models' / model_name)
# endregion
# region Save Run Definitions (Model and Hyper Parameters)
if SAVE_MODELS:
    # save pre-trained model and our classifier arch.
    hyper_params["val_families"] = val_sets
    # save hyper parameters to json:
    with open(str(root_folder / 'models' / model_name / "Hyper_Params.json"), 'w') as fp:
        json.dump(hyper_params, fp)
# endregion

for val_famillies in val_sets:
    early_stopping = EarlyStopping(patience=hyper_params["es_patience"], delta=hyper_params["es_delta"], verbose=True)
    train_family_persons_tree, train_pairs, val_family_persons_tree, val_pairs, train_ppl, val_ppl = \
        load_data(data_path, val_famillies=val_famillies)

    trainloader, valloader = create_datasets(folder_dataset, train_pairs, val_pairs, image_transforms,
                                             train_family_persons_tree, val_family_persons_tree,
                                             train_ppl, val_ppl, hyper_params, NUM_WORKERS)

    steps_per_epoch = np.ceil(len(train_pairs) / hyper_params["BATCH_SIZE"]).__int__() if \
        hyper_params["equal_sampling"] else 1
    for epoch in range(0, hyper_params["NUMBER_EPOCHS"]):
        # initialize epoch meters.
        batch_counter = 0
        epoch_start_time = time.time()
        train_loss = 0
        val_loss = 0
        train_acc = 0
        val_acc = 0

        net.train()  # move the net to train mode
        for j in range(steps_per_epoch):
            for i, data in enumerate(trainloader):
                IMPROVED = False
                img0, img1, labels = data
                # print(labels.float().mean())
                img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)  # move to GPU
                optimizer.zero_grad()  # clear the calculated grad in previous batch
                outputs = net(img0, img1)
                loss = criterion(outputs, labels.float().view(outputs.shape))
                # add batch loss and acc to epoch-loss\acc
                # loss:
                train_loss += loss.item()
                # acc:
                predicted = torch.round(outputs.data).long().view(-1)   # FOR BCE
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
                img0, img1, labels = img0.to(device), img1.to(device), labels.to(device)
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
        curr_lr = optimizer.param_groups[0]['lr']

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            IMPROVED = True

        # melt_model(net)

        train_history["train_loss"].append(train_loss)
        train_history["val_loss"].append(val_loss)
        train_history["train_acc"].append(train_acc)
        train_history["val_acc"].append(val_acc)

        train_loss_diff, val_loss_diff, train_acc_diff, val_acc_diff = extract_diff_str(train_history)

        print('Epoch {:3d} done in {:.1f} sec | train_acc: {:.4f}% {} | val_acc: {:.4f}% {} | train_loss: {:.6f} {} | val_loss: {:.6f} {}; {}'.format(
            epoch+1, epoch_time,
            train_acc, train_acc_diff, val_acc, val_acc_diff,
            train_loss, train_loss_diff, val_loss, val_loss_diff, "(I)" if IMPROVED else ""))

        if SAVE_MODELS and IMPROVED:
            torch.save(net.state_dict(), root_folder / 'models' / model_name / '{}_e{}_vl{:.4f}_va{:.2f}_state.pt'.format(model_name, epoch+1, val_loss, val_acc))
            torch.save(net  , root_folder / 'models' / model_name / '{}_e{}_vl{:.4f}_va{:.2f}_model.pt'.format(model_name, epoch+1, val_loss, val_acc))
        np.save(root_folder / 'curves' / model_name, train_history)
        early_stopping(-val_acc, net)

        if early_stopping.early_stop:
            print("Early stopping! onto next val-family!")
            break
# endregion

# # Save ensemble to json...
# if SAVE_MODELS:
#     with open(str(root_folder / 'ensembles' / ensemble[0]) + ".json", 'w') as fp:
#         json.dump(ensemble, fp)

# region Submission
if SAVE_MODELS and CREATE_SUBMISSION:
    best_model_name = get_best_model(model_folder=root_folder / 'models' / model_name, measure='val_acc', measure_rank=1)
    create_submission(
        root_folder=root_folder, model_name=best_model_name, transform=image_transforms['valid'], device=device)
    print('Created submission file', best_model_name.replace('.pt', '.csv'))
    submission_file_path = str(root_folder / 'submissions_files' / best_model_name.replace('.pt', '.csv'))
    # submit file
    os.system('kaggle competitions submit -c recognizing-faces-in-the-wild -f ' +
              submission_file_path + ' -m ' + best_model_name.replace('.pt', '.csv'))
    # show submissions
    os.system('kaggle competitions submissions recognizing-faces-in-the-wild')
# endregion
