# -*- coding: utf-8 -*-
from resnet50 import resnet50_model
from utils import load_data, load_flower_imgs
from gradcam import compute_gradcam


from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
from keras.utils import np_utils


from sklearn.metrics import log_loss, accuracy_score
import numpy as np

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image



if __name__ == '__main__':

    img_rows, img_cols = 224, 224 # Resolution of inputs
    channel = 3
    num_classes = 4
    batch_size = 32
    nb_epoch = 5

    #load train, val and test data
    data_dir = '../train_vanilla'

    train_dir = data_dir + '/train'
    val_dir = data_dir + '/val'
    test_dir = data_dir + '/test'

    X_train, y_train = load_data(train_dir)
    X_valid, y_valid = load_data(val_dir)
    X_test, y_test = load_data(test_dir)



    # Load our model
    model = resnet50_model(img_rows, img_cols, channel, num_classes, pretrained=False)

    history = model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, y_valid),
              )


    #get statistics
    train_losses = history.history['loss']
    train_accs = history.history['acc']
    val_losses = history.history['val_loss']
    val_accs = history.history['val_acc']

    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.suptitle('Trained from scratch ResNet-50 on Unaugmented 4 class dataset')

    plt.plot(train_losses,color='b',label='Train Loss')
    plt.plot(val_losses,color='g',label='Validation Loss')
    plt.legend()
    plt.title('Loss vs epochs')

    plt.subplot(122)

    plt.plot(train_accs,color='b',label='Train Accuracy')
    plt.plot(val_accs,color='g',label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy vs epochs')

    plt.savefig('history/scratch/fromscratch_resnet_results.png')

    test_score, test_acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)

    print('Final Values after Training: ')
    print('Training Accuracy: ',train_accs[-1])
    print('Validation Accuracy: ',val_accs[-1])
    print('Test Accuracy: ',test_acc)
    print '\n'
    print('Training Loss: ',train_losses[-1])
    print('Validation Loss: ',val_losses[-1])




    #Get Grad CAM outputs
    layer_names = ['res5c_branch2c','res5c_branch2b','res5c_branch2a','res5b_branch2c','res5b_branch2b']

    #get 5 random flower images
    flower_imgs = load_flower_imgs(test_dir,num=5)

    #Compute grad cam on each of them
    index = 1
    for flower_img in flower_imgs:
        plt.subplot(5,6,index)
        plt.axis("off")
        plt.imshow(flower_img)

        index += 1
        for layer_name in layer_names:
            compute_gradcam(model, flower_img, layer_name, cls=3, visualize=True, plot_index=index)
            index += 1

    plt.suptitle('Original Image and Grad CAM output of last five conv layers - Training from scratch')
    plt.show()

    #plot the whole thing on a single plot and save
