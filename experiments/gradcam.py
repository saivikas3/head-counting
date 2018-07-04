

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import tensorflow as tf
from tensorflow.python.framework import ops

from keras.optimizers import SGD
from keras.models import Model
from keras import backend as K
from keras.models import model_from_json

#from load_patches import load_patch_data, load_single_image


from keras import backend as K

K.set_learning_phase(1) #set learning phase


# ## Define model here

# In[2]:


def build_model():
    """Function returning keras model instance.

    Model can be
     - Trained here
     - Loaded with load_model
     - Loaded from keras.applications
    """
    #return VGG16(include_top=True, weights='imagenet')

    model_json_file = open('fpd_model.json','r')
    model_json = model_json_file.read()
    model_json_file.close()

    model = model_from_json(model_json)
    model.load_weights('fpd_model.h5')

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



H, W = 224, 224 # Input shape, defined by the model (model.input_shape)


# ### Utility functions

# In[3]:


def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = image.load_img(path, target_size=(H, W))
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


# ### Guided Backprop

# In[4]:


def build_guided_model():
    """Function returning modified model.

    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) *                    tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model


def guided_backprop(input_model, images, layer_name):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


# ### GradCAM

# In[5]:


def grad_cam(input_model, image, cls, layer_name):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls] #0 = the only image, cls = the index of the class
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    #grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1)) # global average pooling

    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (H, W), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

def grad_cam_batch(input_model, images, classes, layer_name):
    """GradCAM method for visualizing input saliency.
    Same as grad_cam but processes multiple images in one run."""
    loss = tf.gather_nd(input_model.output, np.dstack([range(images.shape[0]), classes])[0])
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(loss, layer_output)[0]
    gradient_fn = K.function([input_model.input, K.learning_phase()], [layer_output, grads])

    conv_output, grads_val = gradient_fn([images, 0])
    weights = np.mean(grads_val, axis=(1, 2))
    cams = np.einsum('ijkl,il->ijk', conv_output, weights)

    # Process CAMs
    new_cams = np.empty((images.shape[0], H, W))
    for i in range(new_cams.shape[0]):
        cam_i = cams[i] - cams[i].mean()
        cam_i = (cam_i + 1e-10) / (np.linalg.norm(cam_i, 2) + 1e-10)
        new_cams[i] = cv2.resize(cam_i, (H, W), cv2.INTER_LINEAR)
        new_cams[i] = np.maximum(new_cams[i], 0)
        new_cams[i] = new_cams[i] / new_cams[i].max()

    return new_cams


# In[12]:


def compute_saliency(model, guided_model, img_path, layer_name='block5_conv3', cls=-1, visualize=True, save=True):
    """Compute saliency using all three approaches.
        -layer_name: layer to compute gradients;
        -cls: class number to localize (-1 for most probable class).
    """
    #preprocessed_input = load_single_image(img_path)
    #preprocessed_input = preprocessed_input.reshape(1,H,W,3)
    preprocessed_input = load_image(img_path)

    predictions = model.predict(preprocessed_input)
#     top_n = 5
#     top = decode_predictions(predictions, top=top_n)[0]
#     classes = np.argsort(predictions[0])[-top_n:][::-1]
    print('Model prediction:')
#     for c, p in zip(classes, top):
#         print('\t{:15s}\t({})\twith probability {:.3f}'.format(p[1], c, p[2]))
    if cls == -1:
        cls = np.argmax(predictions)
#     class_name = decode_predictions(np.eye(1, 1000, cls))[0][0][1]
#     print("Explanation for '{}'".format(class_name))

    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
    gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]

    if save:
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + load_image(img_path, preprocess=False)) / 2
        cv2.imwrite('gradcam.jpg', np.uint8(jetcam))
        cv2.imwrite('guided_backprop.jpg', deprocess_image(gb[0]))
        cv2.imwrite('guided_gradcam.jpg', deprocess_image(guided_gradcam[0]))

    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(131)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(load_image(img_path, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.subplot(132)
        plt.title('Guided Backprop')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(gb[0]), -1))

        plt.subplot(133)
        plt.title('Guided GradCAM')
        plt.axis('off')
        plt.imshow(np.flip(deprocess_image(guided_gradcam[0]), -1))
        plt.show()

    return gradcam, gb, guided_gradcam



# In[34]:


def only_gradcam(model, img_path, layer_name='block5_conv3', cls=-1, visualize=True, save=True):

    preprocessed_input = load_image(img_path)

    predictions = model.predict(preprocessed_input)
    print('Model prediction:')
    if cls == -1:
        cls = np.argmax(predictions)

    print('Class ID: ',cls)
    gradcam = grad_cam(model, preprocessed_input, cls, layer_name)

    if save:
        jetcam = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
        jetcam = (np.float32(jetcam) + load_image(img_path, preprocess=False)) / 2
        cv2.imwrite(layer_name+'_gradcam.jpg', np.uint8(jetcam))

    if visualize:
        plt.figure(figsize=(15, 10))
        plt.subplot(111)
        plt.title('GradCAM')
        plt.axis('off')
        plt.imshow(load_image(img_path, preprocess=False))
        plt.imshow(gradcam, cmap='jet', alpha=0.5)

        plt.show()

    return gradcam

def compute_gradcam(model, image, layer_name, cls=-1, visualize=True, plot_index=1, num_images=1):

    image = image.reshape(1,140,140,3)
    predictions = model.predict(image)

    if cls == -1:
        cls = np.argmax(predictions)

    gradcam = grad_cam(model, image, cls, layer_name)

    if visualize:
        #plt.figure(figsize=(15, 10))
        plt.subplot(num_images,4,plot_index)
        #plt.title('GradCAM')
        plt.axis('off')
        #plt.imshow(image[0])

        plt.imshow(gradcam, cmap='jet', alpha=0.8)
        plt.title('Layer: '+layer_name)
        plt.xlabel('Class '+str(cls))

def gradcam_show(model, image, layer_name, cls=-1):
    #preprocessed_input = preprocess_input(image)
    image = image.reshape(1,140,140,3)
    predictions = model.predict(image)

    if cls == -1:
        cls = np.argmax(predictions)
    gradcam = grad_cam(model, image, cls, layer_name)


    plt.subplot(121)
    plt.imshow(image[0])

    plt.axis('off')
        #plt.imshow(image[0])
    plt.subplot(122)
    plt.imshow(gradcam, cmap='jet', alpha=0.8)
    plt.title('Class '+str(cls))
    plt.show()
# ## Computing saliency

# In[32]:


# model = build_model()
# guided_model = build_guided_model()
#
#
# # In[38]:
#
#
# gradcam = only_gradcam(model,'../sample_flowers/4.bmp', layer_name='res5c_branch2c',
#                                                cls=-1, visualize=False, save=True)
#
# gradcam = only_gradcam(model,'../sample_flowers/4.bmp', layer_name='res5c_branch2b',
#                                                cls=-1, visualize=False, save=True)
# gradcam = only_gradcam(model,'../sample_flowers/4.bmp', layer_name='res5c_branch2a',
#                                                cls=-1, visualize=False, save=True)
# gradcam = only_gradcam(model,'../sample_flowers/4.bmp', layer_name='res5b_branch2c',
#                                                cls=-1, visualize=False, save=True)
# gradcam = only_gradcam(model,'../sample_flowers/4.bmp', layer_name='res5b_branch2b',
#                                                cls=-1, visualize=False, save=True)
# #gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, '../sample_flowers/2.bmp', layer_name='res5a_branch2c',
# #                                               cls=3, visualize=True, save=True)
#
#
# # # Generating explanations for many images
#
# # Load all images from the folder into one array X
#
# # In[ ]:
