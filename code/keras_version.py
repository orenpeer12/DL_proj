# General Imports:
import numpy as np
import pandas as pd
import os
from pathlib import Path
import glob
import sys

# Keras
from collections import defaultdict
from glob import glob
from random import choice, sample
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Dense, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract
from keras.models import Model
from keras.optimizers import Adam

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace


import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

####
####

root_folder = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'data' / 'faces'
print(os.listdir(root_folder))
val_families = "F09" # all families starts with this str will be sent to validation set.


train_file_path = root_folder / 'train_relationships.csv'
train_folders_path = str(Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / 'data' / 'faces' / 'train')

# val_famillies_list = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09"]
val_famillies = "F09"

all_images = glob(train_folders_path + "/*/*/*.jpg")
relationships = pd.read_csv(train_file_path)

def get_train_val(family_name, relationships=relationships):
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


# def load_dataset(val_families="F09", data_path='/root/faces/data/'):
#     ####
#     all_images = glob(str(data_path / 'train/*/*/*.jpg'))
#     train_images = [x for x in all_images if val_families not in x]
#     num_train_images = len(train_images)
#     val_images = [x for x in all_images if val_families in x]
#     num_val_images = len(val_images)
#     train_family_persons_tree = {}
#     val_family_persons_tree = {}
#     my_os = 'win' if sys.platform.startswith('win') else "linux"
#     delim = '\\' if sys.platform.startswith('win') else "/"
#
#     # train_person_to_images_map = defaultdict(list)
#     ppl = [x.split(delim)[-3] + delim + x.split(delim)[-2] for x in all_images]
#     for im_path in train_images:
#         family_name = im_path.split(delim)[-3]
#         person = im_path.split(delim)[-2]
#         if family_name not in train_family_persons_tree:
#             train_family_persons_tree[family_name] = {}
#         if person not in train_family_persons_tree[family_name]:
#             train_family_persons_tree[family_name][person] = []
#         train_family_persons_tree[family_name][person].append(im_path)
#
#     for im_path in val_images:
#         family_name = im_path.split(delim)[-3]
#         person = im_path.split(delim)[-2]
#         if family_name not in val_family_persons_tree:
#             val_family_persons_tree[family_name] = {}
#         if person not in val_family_persons_tree[family_name]:
#             val_family_persons_tree[family_name][person] = []
#         val_family_persons_tree[family_name][person].append(im_path)
#
#     all_relationships = pd.read_csv(str(data_path / "train_relationships.csv"))
#
#     all_relationships = list(zip(all_relationships.p1.values, all_relationships.p2.values))
#     all_relationships = [x for x in all_relationships if x[0] in ppl and x[1] in ppl]
#
#     train_pairs = [x for x in all_relationships if val_families not in x[0]]
#     val_pairs = [x for x in all_relationships if val_families in x[0]]
#
#     # make sure no need to check x[1]
#     print("Total train pairs:", len(train_pairs))
#     print("Total val pairs:", len(val_pairs))
#     print("Total train images:", num_train_images)
#     print("Total val images:", num_val_images)
#     print("Dataset size: ", num_val_images + num_train_images)
#     print("#########################################")
#
#     return train_pairs, val_pairs, train_family_persons_tree, val_family_persons_tree
####
####
def read_img(path):
    img = image.load_img(path, target_size=(197, 197))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)

def gen(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 2)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1 = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1 = np.array([read_img(x) for x in X1])

        X2 = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2 = np.array([read_img(x) for x in X2])

        yield [X1, X2], labels

####
####
def Model1(num_trainable_layers=3, im_shape=197, lr=1e-5):
    # placeholders for inputs:
    input_1 = Input(shape=(im_shape, im_shape, 3))
    input_2 = Input(shape=(im_shape, im_shape, 3))

    # Load a pre-trained model without the "top" - i.e. the classifier head.
    base_model = VGGFace(model='resnet50', include_top=False)

    for x in base_model.layers[:-3]:
        x.trainable = True
        print(x.name)

    for x in base_model.layers[-3:]:
        x.trainable = False
        print(x.name)

    # extract features using the pre-trained model:
    x1 = base_model(input_1)
    x2 = base_model(input_2)

    # NIR: ????
    x1_max = GlobalMaxPool2D()(x1)
    x1_avg = GlobalAvgPool2D()(x1)

    x2_max = GlobalMaxPool2D()(x2)
    x2_avg = GlobalAvgPool2D()(x2)

    x1 = Concatenate(axis=-1)([x1_max, x1_avg])
    x2 = Concatenate(axis=-1)([x2_max, x2_avg])

    x3 = Subtract()([x1, x2])
    x3 = Multiply()([x3, x3])

    x1_ = Multiply()([x1, x1])
    x2_ = Multiply()([x2, x2])
    x4 = Subtract()([x1_, x2_])
    x = Concatenate(axis=-1)([x4, x3])

    x = Dense(100, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(25, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)

    # This is the keras way of defining a sequential model...
    model = Model([input_1, input_2], out)


    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(lr))
    model.summary()
    return model

####
####

# Load family trees:
# train_pairs, val_pairs, train_family_persons_tree,  val_family_persons_tree = \
#     load_dataset(val_families=val_families, data_path=root_folder)

def train_model1():
    train, val, train_person_to_images_map, val_person_to_images_map = get_train_val(val_families)
    file_path = f"vgg_face_.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=10, verbose=1)
    es = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=20, verbose=1)
    callbacks_list = [checkpoint, reduce_on_plateau, es]

    history = model1.fit_generator(gen(train, train_person_to_images_map, batch_size=16),
                                   use_multiprocessing=False,
                                   validation_data=gen(val, val_person_to_images_map, batch_size=16),
                                   epochs=200, verbose=1,
                                   workers=0, callbacks=callbacks_list,
                                   steps_per_epoch=300, validation_steps=150)
    val_acc_list.append(np.max(history.history['val_acc']))


lr = 1e-5
im_shape = 197
num_trainable_layers = 3
model1 = Model1(num_trainable_layers=num_trainable_layers, im_shape=im_shape, lr=lr)
train_model1()

# n_val_famillies_list = len(val_pairs)
####
####

val_acc_list = []


# def train_model1():
#     train_pairs, val_pairs, train_tree, val_tree = load_dataset(val_families=val_families,
#                                                                                               data_path=root_folder)
#
#     file_path = f"vgg_face_.h5"
#     checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#     reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=10, verbose=1)
#     es = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=20, verbose=1)
#     callbacks_list = [checkpoint, reduce_on_plateau, es]
#
#     history = model1.fit_generator(gen(train_pairs, train_tree, batch_size=16),
#                                    use_multiprocessing=False,
#                                    validation_data=gen(val_pairs, val_tree, batch_size=16),
#                                    epochs=200, verbose=0,
#                                    workers=0, callbacks=callbacks_list,
#                                    steps_per_epoch=300, validation_steps=150)
#     val_acc_list.append(np.max(history.history['val_acc']))
def train_model1():
    train, val, train_person_to_images_map, val_person_to_images_map = get_train_val(val_families)
    file_path = f"vgg_face.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=10, verbose=1)
    es = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=20, verbose=1)
    callbacks_list = [checkpoint, reduce_on_plateau, es]

    history = model1.fit_generator(gen(train, train_person_to_images_map, batch_size=16),
                                   use_multiprocessing=False,
                                   validation_data=gen(val, val_person_to_images_map, batch_size=16),
                                   epochs=200, verbose=0,
                                   workers=0, callbacks=callbacks_list,
                                   steps_per_epoch=300, validation_steps=150)
    val_acc_list.append(np.max(history.history['val_acc']))


train_model1()

