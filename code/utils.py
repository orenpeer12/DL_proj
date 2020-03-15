import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
from glob import glob
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt


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


def show_plot(iteration,loss):# for showing loss value changed with iter
    plt.plot(iteration,loss)
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

def create_submition(results, sampe_submition_path):
    df_submit = pd.read_csv(sampe_submition_path)
    df_submit.is_related = results
    df_submit.to_csv('submission.csv', index=False)