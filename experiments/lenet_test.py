from keras import backend as K
from keras.models import load_model, model_from_json
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from gradcam import compute_gradcam, gradcam_show

img_width, img_height = 140, 140
batch_size = 16

test_dir = '../../original_min/'

#load the model
 # Load our model
model_json_file = open('lenet_2classes.json','r')
model_json = model_json_file.read()
model_json_file.close()

model = model_from_json(model_json)
model.load_weights('first_try_2classes.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size)

test_scores = model.evaluate_generator(test_generator)

print 'Test Loss: ',test_scores[0]
print 'Test accuracy: ', test_scores[1]



test_list = []
batch_index = 0

while batch_index <= test_generator.batch_index:
    data = test_generator.next()
    test_list.append(data[0])
    batch_index = batch_index + 1

# now, data_array is the numeric data of whole images
test_array = np.asarray(test_list)
layer_names = ['conv3','conv2','conv1']

img_count = 0
for batch in test_array:

    for image in batch:
        index = 1
        plt.subplot(1,4,index)
        plt.suptitle('Grad CAM output from LeNet')
        plt.axis('off')
        plt.imshow(image)
        plt.title('Original Image')
        for layer in layer_names:
            index += 1
            compute_gradcam(model, image, layer, cls=-1, visualize=True, plot_index=index)


        img_count += 1
        plt.savefig('gradcam_op/gradcam_'+str(img_count)+'.png')
        print('gradcam_op/gradcam_'+str(img_count)+'.png')
#num_images = 16



# flower_imgs = test_array[0][:num_images]


# index = 1
# for flower_img in flower_imgs:
#     plt.subplot(num_images,4,index)
#     plt.axis("off")
#     plt.imshow(flower_img)
#
#     index += 1
#     for layer_name in layer_names:
#         compute_gradcam(model, flower_img, layer_name, cls=0, visualize=True, plot_index=index, num_images=num_images)
#         index += 1
#
# plt.suptitle('Original Image and Grad CAM output of last five conv layers - Fixed')
# plt.show()
