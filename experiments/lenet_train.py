
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
img_width, img_height = 140, 140

train_data_dir = '../../train_min/train'
validation_data_dir = '../../train_min/val'

nb_train_samples = 470
nb_validation_samples = 158
#
# nb_train_samples = 652
# nb_validation_samples = 220


epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

def lenet(output_classes=1):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape,name='conv1'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool1'))

    model.add(Conv2D(32, (3, 3),name='conv2'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool2'))

    model.add(Conv2D(64, (3, 3),name='conv3'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),name='maxpool3'))

    model.add(Flatten())
    model.add(Dense(64,name='fc1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_classes,name='softmax'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

model = lenet(output_classes=2)
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)
    #class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)
    #class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


model_json = model.to_json()
with open('lenet_2classes.json','w') as f:
    f.write(model_json)
model.save_weights('first_try_2classes.h5')
