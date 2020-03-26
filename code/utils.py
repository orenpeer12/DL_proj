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


def load_data(data_path, val_famillies="F09"):
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
    print("Total train pairs:", len(train_pairs))
    print("Total val pairs:", len(val_pairs))
    print("Total train images:", num_train_images)
    print("Total val images:", num_val_images)
    print("Dataset size: ", num_val_images + num_train_images)
    print("#########################################")

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


def create_submission(root_folder, model_name, transform, net=None):
    # gpu or cpu:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get the sample submission file for loading pairs, and create the new submission file.
    sampe_submission_path = root_folder / 'data' / 'faces' / 'sample_submission.csv'
    dst_submission_path = root_folder / 'submissions_files' / model_name.replace('.pt', '.csv')
    copyfile(sampe_submission_path, dst_submission_path)
    df_submit = pd.read_csv(str(dst_submission_path))

    # create the testset and data loader:
    testset = TestDataset(df=df_submit, root_dir=root_folder / 'data' / 'faces' / 'test', transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=8)

    # if needed, load model:
    if net is None:
        model_time = model_name.split('_')[0]
        # net = ResNet(model_time).to(device)
        # net = ResNet(ResidualBlock, [4,4,4]).to(device)
        net.load_state_dict(torch.load(root_folder / 'models' / model_time / model_name))

    # pass testset through model:
    net.eval()
    for i, data in enumerate(test_loader, 0):
        row, img0, img1 = data
        row, img0, img1 = row.cuda(), img0.cuda(), img1.cuda()

        output = net(img0, img1)
        sm = output.softmax(dim=1)
        _, pred = torch.max(sm, 1)

        for idx, item in enumerate(row.cpu()):
            df_submit.loc[item.item(), 'is_related'] = pred[idx].item()

    # write submission
    df_submit['is_related'].value_counts()

    res = df_submit.to_csv(dst_submission_path, index=False)


def get_best_model(model_folder, measure='val_acc'):
    all_models = [m for m in os.listdir(model_folder) if m.endswith('.pt')]
    measures = []
    for m in all_models:
        if measure == 'val_acc':
            mod_measure = float(m.split('_')[3].split('.pt')[0].split('va')[-1])
        if measure == 'val_loss':
            mod_measure = float(m.split('_')[2].split('vl')[-1])
        measures.append(mod_measure)
    best_model_idx = np.argmax(measures)
    return all_models[best_model_idx]
#
# root_path = Path('/home/oren/PycharmProjects/DL_proj')
# model_name = "1585131238_e56_vl0.6654_va63.18.pt"
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])])
# create_submission(root_folder=root_path, model_name=model_name, transform=transform)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr