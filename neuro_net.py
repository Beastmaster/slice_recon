


import os
import sys
import time
import random
import glob
import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
import cv2
import h5py as h5


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Input
from keras.layers.convolutional import * #Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import SGD , Adam 
from keras.layers.merge import *

def neuro_net(weight_path=None):
    initializer = 'glorot_normal'
    
    input1 = Input(shape=(400,400,1),name="xrep")
    input2 = Input(shape=(400,400,1),name="yrep")

    conv1_1 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(input1)
    bn1_1 = BatchNormalization()(conv1_1)
    conv1_2 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    conv1_3 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn1_2)
    bn1_3 = BatchNormalization()(conv1_3)
    
    conv1_12 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(input2)
    bn1_12 = BatchNormalization()(conv1_12)
    conv1_22 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn1_12)
    bn1_22 = BatchNormalization()(conv1_22)
    conv1_32 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn1_22)
    bn1_32 = BatchNormalization()(conv1_32)
    
    concat1 = concatenate([bn1_3,bn1_32])

    conv1_13  = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(concat1)
    bn1_13    = BatchNormalization()(conv1_13)
    conv1_23  = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn1_13)
    bn1_23    = BatchNormalization()(conv1_23)
    conv1_33  = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn1_23)
    bn1_33    = BatchNormalization()(conv1_33)

    ### pool 1
    pool1 = MaxPooling2D((2,2), strides=(2,2))(bn1_33)  # 400

    conv2_1 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(pool1)
    bn2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    
    # pool 2
    pool2 = MaxPooling2D((2,2), strides=(2,2))(bn2_2)  # 200

    conv3_1 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(pool2)
    bn3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn3_1)
    bn3_2 = BatchNormalization()(conv3_2)

    # pool 3
    pool3 = MaxPooling2D((2,2), strides=(2,2))(bn3_2) # 100

    conv4_1 = Conv2D(512, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(pool3)
    bn4_1 = BatchNormalization()(conv4_1)
    
    # up-pool 3   ## 200
    un_pool3 = Conv2DTranspose(256,(2,2), strides=(2,2),padding='same',activation='relu',kernel_initializer=initializer)(bn4_1)


    conv5_1 = Conv2D(256, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(un_pool3)
    bn5_1 = BatchNormalization()(conv5_1)
    conv5_2 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn5_1)
    bn5_2 = BatchNormalization()(conv5_2)
    

    # un_pool 2
    un_pool2 = Conv2DTranspose(128,(2,2), strides=(2,2),padding='same',activation='relu',kernel_initializer=initializer)(bn5_2)

    
    conv6_1 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(un_pool2)
    bn6_1 = BatchNormalization()(conv6_1)
    conv6_2 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn6_1)
    bn6_2 = BatchNormalization()(conv6_2)

    # un_pool 1
    un_pool1 = Conv2DTranspose(64,(2,2), strides=(2,2),padding='same',activation='relu',kernel_initializer=initializer)(bn6_2)

    
    conv7_1 = Conv2D(128, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(un_pool1)
    bn7_1 = BatchNormalization()(conv7_1)
    conv7_2 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn7_1)
    bn7_2 = BatchNormalization()(conv7_2)
    conv7_3 = Conv2D(64, (3, 3),padding='same', activation='relu',kernel_initializer=initializer)(bn7_2)
    bn7_3 = BatchNormalization()(conv7_3)

    # regression
    score = Conv2D(1, (1, 1),padding='same' ,kernel_initializer=initializer)(bn7_3)
    #722

    model = Model(input=[input1,input2],output=score)
    if weight_path:
        model.load_weights(weight_path)
    return model


if __name__ == "__main__":
    neuro_net()
















