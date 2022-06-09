from gettext import npgettext
import os
from tkinter import N
from cv2 import imshow
from tensorboard import summary
os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import scipy
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from PIL import Image

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, SpatialDropout2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

data_path = '../Dataset/'
batch_size = 32
image_size = 64

def load_files(path, target_size, scale_factor):    
    image_list = []
    filenames = glob.glob(path)
    filenames.sort()
    for filename in filenames:
        im = Image.open(filename)
        w, h = im.size
        im = im.resize((target_size, target_size))
        im=np.asarray(im) / scale_factor
        image_list.append(im)
    return np.asarray(image_list)

image_list_train = load_files(data_path + '/images/train/*.jpg', image_size, 255.0)
mask_list_train = load_files(data_path + '/masks/train/*.png', image_size, 1.0)
mask_list_train = np.reshape(mask_list_train, (np.shape(mask_list_train) + (1, )))
image_list_test = load_files(data_path + '/images/test/*.jpg', image_size, 255.0)
mask_list_test = load_files(data_path + '/masks/test/*.png', image_size, 1.0)
mask_list_test = np.reshape(mask_list_test, (np.shape(mask_list_test) + (1, )))

# print(np.shape(image_list_train))
# print(np.shape(mask_list_train))
# print(np.shape(image_list_test))
# print(np.shape(mask_list_test))


def mask_to_categorical(im, num_classes):    
    one_hot_map = []
    for i in range(num_classes):
        class_map = tf.reduce_all(tf.equal(im, i), axis=-1)
        one_hot_map.append(class_map)
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)    
    return one_hot_map

def categorical_to_mask(im):
    mask = tf.dtypes.cast(tf.argmax(im, axis=2), 'float32') / 255.0
    return mask

# def random_crop(img_x, img_y, random_crop_size):
#     height, width = img_x.shape[0], img_x.shape[1]
#     dy, dx = random_crop_size
#     # x = np.random.randint(0, width - dx + 1)
#     # y = np.random.randint(0, height - dy + 1)
#     x = 0 #removed random crop as all images were same size
#     y = 0 #removed random crop as all images were same size
#     return img_x[y:(y+dy), x:(x+dx), :], img_y[y:(y+dy), x:(x+dx), :]

def catMap(batches):
    while True:
        batch_x, batch_y = next(batches)
        yield (batch_x, mask_to_categorical(batch_y, 2))


train_datagen = ImageDataGenerator(zoom_range=0, horizontal_flip=True, validation_split = 0.1)
train_image_generator = train_datagen.flow(image_list_train, batch_size = batch_size, seed=1, subset = 'training')
train_mask_generator = train_datagen.flow(mask_list_train, batch_size = batch_size, seed=1, subset = 'training')
train_generator = catMap(zip(train_image_generator, train_mask_generator))
# train_generator = zip(train_image_generator,train_mask_generator)

val_image_generator = train_datagen.flow(image_list_train, batch_size = batch_size, seed=1, subset = 'validation')
val_mask_generator = train_datagen.flow(mask_list_train, batch_size = batch_size, seed=1, subset = 'validation')
val_generator = catMap(zip(val_image_generator, val_mask_generator))
# val_generator = zip(val_image_generator,val_mask_generator)

test_datagen = ImageDataGenerator(zoom_range=0)
test_image_generator = test_datagen.flow(image_list_test, batch_size = batch_size, seed=1)
test_mask_generator = test_datagen.flow(mask_list_test, batch_size = batch_size, seed=1)
test_generator = catMap(zip(test_image_generator, test_mask_generator))

# Visualise the training and test images & masks
# sample = next(train_generator)
# fig = plt.figure(figsize=[5, 4])
# for i,img in enumerate(sample[0]):
#     if (i < 32):
#         ax = fig.add_subplot(8, 8, i*2 + 1)
#         ax.imshow(img, extent=[0, 256, 0, 256])
#         ax = fig.add_subplot(8, 8, i*2 + 2)
#         ax.imshow(categorical_to_mask(sample[1][i,:,:,:]))

test_data, test_gt = next(test_generator)
# fig = plt.figure(figsize=[5, 4])
# for i,img in enumerate(test_data):
#     if (i < 32):
#         ax = fig.add_subplot(8, 8, i*2 + 1)
#         ax.imshow(img, extent=[0, 256, 0, 256])
#         ax = fig.add_subplot(8, 8, i*2 + 2)
#         ax.imshow(categorical_to_mask(test_gt[i,:,:,:]))

# plt.show()

# input, colour images
def model1():
    input_img = Input(shape=(image_size, image_size, 3))

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(2, (1, 1), activation='sigmoid', padding='same')(x)

    segmenter = Model(input_img, decoded)
    return segmenter


def model2():
    input_img = Input(shape=(image_size, image_size, 3))

    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2), padding='same')(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    up1 = UpSampling2D((2, 2))(conv4)
    merge1 = concatenate([conv3,up1], axis = 3)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    up2 = UpSampling2D((2, 2))(conv5)
    merge2 = concatenate([conv2,up2], axis = 3)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge2)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)

    up3 = UpSampling2D((2, 2))(conv6)
    merge3 = concatenate([conv1,up3], axis = 3)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge3)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv7)
    decoded = Conv2D(2, (1, 1), activation='sigmoid', padding='same')(conv7)

    segmenter = Model(input_img, decoded)
    return segmenter

def model3():
    input_img = Input(shape=(image_size, image_size, 3))

    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = Conv2D(16, (3, 3), activation=None, padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = SpatialDropout2D(0.2)(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = SpatialDropout2D(0.2)(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = SpatialDropout2D(0.2)(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D((2, 2), padding='same')(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation=None, padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = SpatialDropout2D(0.2)(conv4)
    conv4 = Activation('relu')(conv4)

    up1 = UpSampling2D((2, 2))(conv4)
    merge1 = concatenate([conv3,up1], axis = 3)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = SpatialDropout2D(0.2)(conv5)
    conv5 = Activation('relu')(conv5)

    up2 = UpSampling2D((2, 2))(conv5)
    merge2 = concatenate([conv2,up2], axis = 3)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge2)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = SpatialDropout2D(0.2)(conv6)
    conv6 = Activation('relu')(conv6)

    up3 = UpSampling2D((2, 2))(conv6)
    merge3 = concatenate([conv1,up3], axis = 3)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge3)
    conv7 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = SpatialDropout2D(0.2)(conv7)
    conv7 = Activation('relu')(conv7)

    decoded = Conv2D(2, (1, 1), activation='sigmoid', padding='same')(conv7)
    segmenter = Model(input_img, decoded)
    return segmenter


def focal_loss(target, output, gamma=2):
    output /= K.sum(output, axis=-1, keepdims=True)
    eps = K.epsilon()
    output = K.clip(output, eps, 1. - eps)
    return -K.sum(K.pow(1. - output, gamma) * target * K.log(output), axis=-1)

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

def create_callbacks():
    scheduler_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto', restore_best_weights=True)
    return [scheduler_callback, early_stopping]

def create_optimiser():
    return tf.keras.optimizers.Adam()
steps_per_epoch = 100
epochs = 30

def visualise(model, test_data):
    pred = model.predict(test_data)
    fig = plt.figure(figsize=[5, 4])
    norm = colors.Normalize()
    cmap = cm.hsv
    background = colors.colorConverter.to_rgba('g')
    weed = colors.colorConverter.to_rgba('r')
    for i,img in enumerate(test_data):
        if (i < 32):
            #Original image
            ax = fig.add_subplot(8, 8, i*2 + 1)
            ax.imshow(img)
            #Image with label mask
            # ax = fig.add_subplot(8, 12, i*3 + 2)
            # maskOverlay = cmap(norm(categorical_to_mask(test_gt[i,:,:,:])))
            # maskOverlay[categorical_to_mask(test_gt[i,:,:,:])<=0,:] = background
            # maskOverlay[categorical_to_mask(test_gt[i,:,:,:])>0,:] = weed
            # ax.imshow(img)
            # ax.imshow(maskOverlay, alpha = 0.3)
            #Image with prediction mask
            ax = fig.add_subplot(8, 8, i*2 + 2)
            predOverlay = cmap(norm(categorical_to_mask(pred[i,:,:,:])))
            predOverlay[categorical_to_mask(pred[i,:,:,:])<=0,:] = background
            predOverlay[categorical_to_mask(pred[i,:,:,:])>0,:] = weed
            ax.imshow(img)
            ax.imshow(predOverlay, alpha = 0.3 )

# model = tf.keras.models.load_model('models/mymodel2-128')
model = model2()
print(model.summary())
model.compile(optimizer=create_optimiser(), loss='binary_crossentropy', metrics='accuracy')
model.fit(train_generator, steps_per_epoch = steps_per_epoch, epochs = epochs, validation_data = val_generator, validation_steps = 10,
              callbacks=create_callbacks())

model.evaluate(test_generator, steps = 100, batch_size = batch_size)
# model.save('models/mymodel2-128/')
visualise(model, test_data)
plt.show()