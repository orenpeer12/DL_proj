import numpy as np
import os
import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import torch

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
    print("####################\nChecking attributes:\n####################")
    test_images = os.listdir(data_path / 'test_images')
    train_images = os.listdir(data_path / 'train_images')
    test_images_png = [im for im in test_images if im.endswith('.png')]
    test_images_jpg = [im for im in test_images if im.endswith('.jpg')]
    train_images_png = [im for im in train_images if im.endswith('.png')]
    train_images_jpg = [im for im in train_images if im.endswith('.jpg')]
    print("Train set containes {} jpg images and {} png images.".format(len(train_images_jpg),
                                                                        len(train_images_png)))
    print("Test set containes {} jpg images and {} png images.".format(len(test_images_jpg),
                                                                        len(test_images_png)))
    for images, im_set in zip([train_images, test_images], ['train_images', 'test_images']):
        shapes = set()
        for idx, im_name in enumerate(images):
            if idx%1000 == 0:
                printProgressBar(idx, len(images), 'Finished: ', im_set)
            im_shape = cv2.imread(str(data_path / im_set / im_name)).shape
            if im_shape not in shapes:
                shapes.add(im_shape)
        print("Possible {} shapes in {}:".format(shapes, im_set))
        print(shapes)

def load_data(data_path):
    train_lables = pd.read_csv(data_path / 'train.csv')
    test_lables = pd.read_csv(data_path / 'test.csv')
    train_images_paths = [str(data_path / "train_images" / im_path) for im_path in os.listdir(data_path / "train_images")]
    test_images_paths = [str(data_path / "test_images" / im_path) for im_path in os.listdir(data_path / "test_images")]
    return train_images_paths, train_lables, test_images_paths, test_lables


# Load data:
data_path = Path('/home/oren/PycharmProjects/DL_proj/data/')
# inspect_images_attributes(data_path) # All images are jpg, possible shapes:
train_images_paths, train_lables, test_images_paths, test_lables = load_data(data_path)

X = [cv2.resize(cv2.imread(im_path), (224, 224, )) for im_path in train_images_paths]
# im = cv2.imread(train_images_paths[0])
# im = cv2.resize(im, (224, 224, ))

