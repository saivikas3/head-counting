# -*- coding: utf-8 -*-
from resnet50 import resnet50_model
from utils import load_data
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import regularizers

from sklearn.metrics import log_loss

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image



if __name__ == '__main__':

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 4
    batch_size = 32
    nb_epoch = 4

    #load train, val and test data
    data_dir = '../train_vanilla'

    train_dir = data_dir + '/train'
    val_dir = data_dir + '/val'
    test_dir = data_dir + '/test'

    X_train, y_train = load_data(train_dir)
    X_val, y_val = load_data(val_dir)
    X_test, y_test = load_data(test_dir)

    print(X_train.shape,y_train.shape)
    print(X_val.shape,y_val.shape)
    print(X_test.shape,y_test.shape)


    # Load our model
    model = resnet50_model(img_rows, img_cols, channel, num_classes, pretrained=False)
