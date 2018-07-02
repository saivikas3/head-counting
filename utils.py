import os
import cv2
import numpy as np

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.utils import np_utils

sd  = 107
from numpy.random import seed
seed(sd)
from tensorflow import set_random_seed
set_random_seed(sd)

def load_data(path,num_classes=4):

    path_background = path + '/background/'
    path_leaf = path + '/leaf/'
    path_ear = path + '/ear/'
    path_flower = path + '/flower/'

    background_images = []
    leaf_images = []
    flower_images = []
    ear_images = []



    for f in os.listdir(path_background):

        img = image.load_img(path_background+'/'+f,target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        background_images.append(x)

    for f in os.listdir(path_leaf):
        img = image.load_img(path_leaf+'/'+f,target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        leaf_images.append(x)

    for f in os.listdir(path_flower):
        img = image.load_img(path_flower+'/'+f,target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        flower_images.append(x)

    for f in os.listdir(path_ear):
        img = image.load_img(path_ear+'/'+f,target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        ear_images.append(x)

    background_n = len(background_images)
    leaf_n = len(leaf_images)
    flower_n = len(flower_images)
    ear_n = len(ear_images)

    backgroundY = np.full(background_n,0)
    leafY = np.full(leaf_n,1)
    earY = np.full(ear_n,2)
    flowerY = np.full(flower_n,3)

    Y = np.concatenate((backgroundY,leafY,earY,flowerY))

    backgroundX = np.asarray(background_images)
    leafX = np.asarray(leaf_images)
    earX = np.asarray(ear_images)
    flowerX = np.asarray(flower_images)

    X = np.concatenate((backgroundX,leafX,earX,flowerX),axis=0)

    Y = np_utils.to_categorical(Y, num_classes)

    return X,Y
