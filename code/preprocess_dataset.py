import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import sys
from shutil import copyfile

class CLAHE_and_SimpleWB:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        self.wb = cv2.xphoto.createSimpleWB()
        self.wb.setP(0.4)

    def perform(self, img):
        img_wb = self.wb.balanceWhite(img)
        img_lab = cv2.cvtColor(img_wb, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(img_lab)
        res_l = self.clahe.apply(l)
        res = cv2.merge((res_l, a, b))
        res = cv2.cvtColor(res, cv2.COLOR_Lab2BGR)
        return res

plt.ion()

data_path = Path(os.getcwd()) / 'data' / 'faces'
train_imgs_paths = glob(str(data_path / 'train/*/*/*.jpg'))
test_imgs_paths = glob(str(data_path / 'test/*.jpg'))
transformer = CLAHE_and_SimpleWB()

new_dataset_name = 'data_mod'

data_aug_path = Path(str(data_path).replace('data', new_dataset_name))
if os.path.isdir(data_aug_path):
    print('data_aug exist.. delete or change name.')
    exit()
os.mkdir(data_aug_path.parent)
os.mkdir(data_aug_path)

delim = '\\' if sys.platform.startswith('win') else "/"

imgs = []
compare = []
def create_mod_ds(imgs_paths, ds_type, delim,data_aug_path):
    os.mkdir(data_aug_path / ds_type)
    for idx, im_path in enumerate(imgs_paths):
        img = plt.imread(im_path)
        img_t = transformer.perform(img)
        new_img_path = im_path.replace('data', new_dataset_name)
        if ds_type == 'train':
            family, person = new_img_path.split(ds_type)[1].split(delim)[1:3]
            if not os.path.isdir(data_aug_path / ds_type / family):
                os.mkdir(data_aug_path / ds_type / family)
            if not os.path.isdir(data_aug_path / ds_type / family / person):
                os.mkdir(data_aug_path / ds_type / family / person)
        cv2.imwrite(new_img_path, cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB))
        # compare.append(np.concatenate([img, img_t], axis=1))
        if False and idx % 4 == 0 and idx > 0:
            fig = plt.figure()
            plt.show()
            fig.tight_layout()

            ax = []
            for i in range(1, 6):
                ax.append(fig.add_subplot(5, 1, i))
                plt.imshow(compare[i-1])
            plt.pause(.001)
            input("Press [enter] to continue.")
            plt.close('all')
            compare = []

create_mod_ds(imgs_paths=train_imgs_paths, ds_type='train', data_aug_path=data_aug_path, delim=delim)
create_mod_ds(imgs_paths=test_imgs_paths, ds_type='test', data_aug_path=data_aug_path, delim=delim)

copyfile(str(data_path / 'sample_submission.csv'), str(data_aug_path / 'sample_submission.csv'))
copyfile(str(data_path / 'train_relationships.csv'), str(data_aug_path / 'train_relationships.csv'))
