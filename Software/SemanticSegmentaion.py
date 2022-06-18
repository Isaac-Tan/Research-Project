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
import time
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, SpatialDropout2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

#Variables
dataPath = '../Dataset/'
batchSize = 50
imageSize = 256
stepsPerEpoch = 100
epochs = 15

def loadFiles(path, targetSize, scaleFactor):    
    imageList = []
    filenames = glob.glob(path)
    filenames.sort()
    for filename in filenames:
        im = Image.open(filename)
        im = im.resize((targetSize, targetSize))
        im=np.asarray(im) / scaleFactor
        imageList.append(im)
    return np.asarray(imageList)

imageListTrain = loadFiles(dataPath + '/images/train/*.jpg', imageSize, 255.0)
maskListTrain = loadFiles(dataPath + '/masks/train/*.png', imageSize, 1.0)
maskListTrain = np.reshape(maskListTrain, (np.shape(maskListTrain) + (1, )))
imageListTest = loadFiles(dataPath + '/images/test/*.jpg', imageSize, 255.0)
maskListTest = loadFiles(dataPath + '/masks/test/*.png', imageSize, 1.0)
maskListTest = np.reshape(maskListTest, (np.shape(maskListTest) + (1, )))

def maskToCategorical(im, numClasses):    
    oneHotMap = []
    for i in range(numClasses):
        classMap = tf.reduce_all(tf.equal(im, i), axis=-1)
        oneHotMap.append(classMap)
    oneHotMap = tf.stack(oneHotMap, axis=-1)
    oneHotMap = tf.cast(oneHotMap, tf.float32)    
    return oneHotMap

def categoricalToMask(im):
    mask = tf.dtypes.cast(tf.argmax(im, axis=2), 'float32') / 255.0
    return mask

def catMap(batches):
    while True:
        batchX, batchY = next(batches)
        yield (batchX, maskToCategorical(batchY, 2))


trainDatagen = ImageDataGenerator(zoom_range=0, horizontal_flip=True, validation_split = 0.1)
trainImageGenerator = trainDatagen.flow(imageListTrain, batch_size = batchSize, seed=1, subset = 'training')
trainMaskGenerator = trainDatagen.flow(maskListTrain, batch_size = batchSize, seed=1, subset = 'training')
trainGenerator = catMap(zip(trainImageGenerator, trainMaskGenerator))

validationImageGenerator = trainDatagen.flow(imageListTrain, batch_size = batchSize, seed=1, subset = 'validation')
validationMaskGenerator = trainDatagen.flow(maskListTrain, batch_size = batchSize, seed=1, subset = 'validation')
validationGenerator = catMap(zip(validationImageGenerator, validationMaskGenerator))

testDatagen = ImageDataGenerator(zoom_range=0)
testImageGenerator = testDatagen.flow(imageListTest, batch_size = batchSize, seed=1)
testMaskGenerator = testDatagen.flow(maskListTest, batch_size = batchSize, seed=1)
testGenerator = catMap(zip(testImageGenerator, testMaskGenerator))

def unet():
    inputImage = Input(shape=(imageSize, imageSize, 3))

    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputImage)
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
    pool4 = MaxPooling2D((2, 2), padding='same')(conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up1 = UpSampling2D((2, 2))(conv5)
    merge1 = concatenate([conv4,up1], axis = 3)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge1)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up2 = UpSampling2D((2, 2))(conv6)
    merge2 = concatenate([conv3,up2], axis = 3)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge2)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    up3 = UpSampling2D((2, 2))(conv7)
    merge3 = concatenate([conv2,up3], axis = 3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(merge3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    up4 = UpSampling2D((2, 2))(conv8)
    merge4 = concatenate([conv1,up4], axis = 3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(merge4)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)
    decoded = Conv2D(2, (1, 1), activation='sigmoid', padding='same')(conv9)

    segmenter = Model(inputImage, decoded)
    return segmenter

def scheduler(epoch, rate):
    if epoch < 10:
        return rate
    else:
        return rate * tf.math.exp(-0.1)

def createCallbacks():
    schedulerCallback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto', restore_best_weights=True)
    return [schedulerCallback, earlyStopping]


testData, testGT = next(testGenerator)

def visualise(model, testData):
    start = time.time()
    pred = model.predict(testData)
    elapsed = time.time() - start
    print("Prediction latency:", elapsed)
    fig = plt.figure(figsize=[5, 4])
    norm = colors.Normalize()
    cmap = cm.hsv
    background = colors.colorConverter.to_rgba('g')
    weed = colors.colorConverter.to_rgba('r')
    for i,img in enumerate(testData):
        if (i < 50):
            #Original image
            ax = fig.add_subplot(10, 10, i*2 + 1)
            ax.axis('off')
            ax.imshow(img)
            #Image with prediction mask
            ax = fig.add_subplot(10, 10, i*2 + 2)
            predOverlay = cmap(norm(categoricalToMask(pred[i,:,:,:])))
            predOverlay[categoricalToMask(pred[i,:,:,:])<=0,:] = background
            predOverlay[categoricalToMask(pred[i,:,:,:])>0,:] = weed
            ax.axis('off')
            ax.imshow(img)
            ax.imshow(predOverlay, alpha = 0.3 )

# model = tf.keras.models.load_model('models/unet-64')
model = unet()
print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')
model.fit(trainGenerator, steps_per_epoch = stepsPerEpoch, epochs = epochs, validation_data = validationGenerator, validation_steps = 10,
callbacks=createCallbacks())

model.evaluate(testGenerator, steps = 100, batch_size = batchSize)
model.save('models/unet-256/')
visualise(model, testData)

plt.show()