from collections import defaultdict

import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from shutil import copyfile
import torch
from torch.utils.data import DataLoader
from OurDataset import *
from SiameseNetwork import *
import torchvision.transforms as transforms

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
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


def load_data(data_path, val_famillies):
    all_images = glob(str(data_path / 'train/*/*/*.jpg'))
    train_images = [x for x in all_images if val_famillies not in x]
    num_train_images = len(train_images)
    val_images = [x for x in all_images if val_famillies in x]
    num_val_images = len(val_images)
    train_family_persons_tree = {}
    val_family_persons_tree = {}
    my_os = 'win' if sys.platform.startswith('win') else "linux"
    delim = '\\' if sys.platform.startswith('win') else "/"

    # train_person_to_images_map = defaultdict(list)
    ppl = [x.split(delim)[-3] + delim + x.split(delim)[-2] for x in all_images]
    train_ppl = list(np.unique([x.split(delim)[-3] + delim + x.split(delim)[-2] for x in train_images]))
    val_ppl = list(np.unique([x.split(delim)[-3] + delim + x.split(delim)[-2] for x in val_images]))

    # for x in train_images:
    #     train_person_to_images_map[x.split(delim)[-3] + delim + x.split(delim)[-2]].append(x)
    # val_person_to_images_map = defaultdict(list)
    # for x in val_images:
    #     val_person_to_images_map[x.split(delim)[-3] + delim + x.split(delim)[-2]].append(x)

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
    if my_os == 'win':
        for idx in range(len(all_relationships['p1'])):
            all_relationships['p1'][idx] = all_relationships['p1'][idx].replace('/', "\\")
            all_relationships['p2'][idx] = all_relationships['p2'][idx].replace('/', "\\")
    all_relationships = list(zip(all_relationships.p1.values,
                             all_relationships.p2.values))
    all_relationships = [x for x in all_relationships if x[0] in ppl and x[1] in ppl]  # filter unused relationships
    train_pairs = [x for x in all_relationships if val_famillies not in x[0]]
    val_pairs = [x for x in all_relationships if val_famillies in x[0]]
    # make sure no need to check x[1]
    print("Validation families: ", val_famillies)
    print("Total train pairs:", len(train_pairs))
    print("Total val pairs:", len(val_pairs))
    print("Total train images:", num_train_images)
    print("Total val images:", num_val_images)
    print("Dataset size: ", num_val_images + num_train_images)
    print("#########################################")

    return train_family_persons_tree, train_pairs, val_family_persons_tree, val_pairs, train_ppl, val_ppl

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


def show_plot(train_history):# for showing loss value changed with iter
    plt.ion()
    if not isinstance(train_history, dict):
        train_history = np.load(train_history, allow_pickle=True).item()
    f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    ax1.plot(train_history['train_loss'])
    ax1.plot(train_history['val_loss'])
    ax1.set_title('Loss curves')
    ax1.legend(['train', 'validation'])
    ax2.plot(train_history['train_acc'])
    ax2.plot(train_history['val_acc'])
    ax2.set_title('Acc. curves')
    ax2.legend(['train', 'validation'])
    plt.show()


def extract_diff_str(train_history):
    epoch = len(train_history['train_loss'])
    train_loss_diff = train_history['train_loss'][-1] - train_history['train_loss'][-2] if epoch > 1 else 0
    val_loss_diff = train_history['val_loss'][-1] - train_history['val_loss'][-2] if epoch > 1 else 0
    train_acc_diff = train_history['train_acc'][-1] - train_history['train_acc'][-2] if epoch > 1 else 0
    val_acc_diff = train_history['val_acc'][-1] - train_history['val_acc'][-2] if epoch > 1 else 0

    train_loss_diff = '(+{:.6f})'.format(train_loss_diff) if train_loss_diff >= 0 else '({:.6f})'.format(train_loss_diff)
    val_loss_diff = '(+{:.6f})'.format(val_loss_diff) if val_loss_diff >= 0 else '({:.6f})'.format(val_loss_diff)
    train_acc_diff = '(+{:.4f})'.format(train_acc_diff) if train_acc_diff >= 0 else '({:.4f})'.format(train_acc_diff)
    val_acc_diff = '(+{:.4f})'.format(val_acc_diff) if val_acc_diff >= 0 else '({:.4f})'.format(val_acc_diff)

    return train_loss_diff, val_loss_diff, train_acc_diff, val_acc_diff


def create_submission(root_folder, model_name, transform, device=None, net=None):
    # gpu or cpu:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the sample submission file for loading pairs, and create the new submission file.
    sample_submission_path = root_folder / 'data' / 'faces' / 'sample_submission.csv'
    dst_submission_path = root_folder / 'submissions_files' / model_name.replace('.pt', '.csv')
    copyfile(sample_submission_path, dst_submission_path)
    df_submit = pd.read_csv(str(dst_submission_path))

    # create the testset and data loader:
    testset = TestDataset(df=df_submit, root_dir=root_folder / 'data' / 'faces' / 'test', transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=8)

    # if needed, load model:
    if net is None:
        model_time = model_name.split('_')[0]
        net = torch.load(root_folder / 'models' / model_time / model_name.replace('.pt', '_model.pt'))
        net.load_state_dict(torch.load(root_folder / 'models' / model_time / model_name.replace('.pt', '_state.pt')))

    # pass testset through model:
    net.eval()
    net.to(device)
    for i, data in enumerate(test_loader, 0):
        row, img0, img1 = data
        row, img0, img1 = row.to(device), img0.to(device), img1.to(device)

        output = net(img0, img1)
        # predicted = torch.round(output.data).long().view(-1)
        predicted = output.data.view(-1)
        # sm = output.softmax(dim=1)
        # _, pred = torch.round(sm, 1)

        for idx, item in enumerate(row.cpu()):
            df_submit.loc[item.item(), 'is_related'] = predicted[idx].item()

    # write submission
    df_submit['is_related'].value_counts()

    res = df_submit.to_csv(dst_submission_path, index=False)


def get_best_model(model_folder, measure='val_acc', measure_rank=1):
    all_models = [m for m in os.listdir(model_folder) if m.endswith('_model.pt')]
    measures = []
    for m in all_models:
        if measure == 'val_acc':
            mod_measure = float(m.split('_')[3].split('.pt')[0].split('va')[-1])
        if measure == 'val_loss':
            mod_measure = float(m.split('_')[2].split('vl')[-1])
        measures.append(mod_measure)
    best_model_idx = np.argsort(measures)[-measure_rank]
    return all_models[best_model_idx].replace('_model', '')


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_train_val(family_name, data_path):

    all_images = glob(str(data_path / 'train/*/*/*.jpg'))
    relationships = pd.read_csv(str(data_path / "train_relationships.csv"))

    # Get val_person_image_map
    val_famillies = family_name
    train_images = [x for x in all_images if val_famillies not in x]
    val_images = [x for x in all_images if val_famillies in x]

    train_person_to_images_map = defaultdict(list)

    ppl = [x.split("/")[-3] + "/" + x.split("/")[-2] for x in all_images]

    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    val_person_to_images_map = defaultdict(list)

    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    # Get the train and val dataset
    #     relationships = pd.read_csv(train_file_path)
    relationships = list(zip(relationships.p1.values, relationships.p2.values))
    relationships = [x for x in relationships if x[0] in ppl and x[1] in ppl]

    train = [x for x in relationships if val_famillies not in x[0]]
    val = [x for x in relationships if val_famillies in x[0]]

    return train, val, train_person_to_images_map, val_person_to_images_map


def scale_tensor_255(tensor, scale=255):
    return tensor.mul(scale)


def rgb2bgr(img):
    img = img[[2, 1, 0], :, :]
    return img


def count_params(net):
    trainable_model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    non_trainable_model_parameters = filter(lambda p: not (p.requires_grad), net.parameters())
    trainable_params = sum([np.prod(p.size()) for p in trainable_model_parameters])
    non_trainable_params = sum([np.prod(p.size()) for p in non_trainable_model_parameters])
    print("Num. of trainable parameters: {:,}, num. of frozen parameters: {:,}, total: {:,}".format(
        trainable_params, non_trainable_params, trainable_params + non_trainable_params))
#
#
# def melt_model(net):
#     melt_ratio = 0.
#     for p in net.features.parameters():
#         if random.uniform(0, 1) > melt_ratio:
#             p.require_grad = True
#     net.to(device)
#     optimizer = optim.Adam(net.parameters(), lr=curr_lr, weight_decay=hyper_params["weight_decay"])
#     count_params(net)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model) # TODO
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter > int(3*self.patience/4):
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model) # TODO
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss