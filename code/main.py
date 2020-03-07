import numpy as np
import os
import pandas as pd
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Load data:
data_path = Path('/home/oren/PycharmProjects/DL_proj/data/')
train_lables = pd.read_csv(data_path / 'train.csv')
test_lables = pd.read_csv(data_path / 'test.csv')

test_images = os.listdir(data_path / 'test_images')
# test_images_png = [im for im in test_images if im.endswith('.png')]
# test_images_jpg = [im for im in test_images if im.endswith('.jpg')]
shapes = set()
for idx, im_name in enumerate(test_images):
    if idx%1000 == 0:
        print("procceced {} % of all images".format(100 * idx / len(test_images)))
    im_shape = cv2.imread(str(data_path / 'test_images' / im_name)).shape
    if im_shape not in shapes:
        shapes.add(im_shape)