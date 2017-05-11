


from neuro_net import *

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Input
from keras.layers.convolutional import * #Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization 
from keras.optimizers import SGD , Adam 
from keras.layers.merge import *
from keras import losses

import os
import sys
import cv2
import numpy as np
import random
from time import gmtime, strftime
import pickle

def load_data():
    data = []
    for id in range(200,500,5):
        ch1 = os.path.join(os.getcwd(),"data/train/x_{}.png".format(id))
        ch2 = os.path.join(os.getcwd(),"data/train/y_{}.png".format(id))
        lab = os.path.join(os.getcwd(),"data/train/im{}.png".format(id))

        def read(name):
            if(not os.path.exists(name)):
                raise ValueError("{} does not eixts".format(name))
            data=cv2.imread(name,cv2.IMREAD_ANYDEPTH)
            return np.float32(data)
        data.append((read(ch1),read(ch2),read(lab)))
    return data



def start_training(model=None):
    model = neuro_net(model)
    sgd = SGD(lr=5e-4, decay=1e-9, momentum=0.9, nesterov=True)
    adam = Adam(lr = 1e-5, decay = 1e-10)
    model.compile(optimizer=sgd, loss=losses.mean_squared_error) #, metrics=['accuracy']

    # load data
    train_data = load_data()
    random.shuffle(train_data)

    for iter in range(50000):
        ii = iter%(len(train_data))
        if ii == 0:
            random.shuffle(train_data)  # shuffle training set
        
        batch = train_data[ii]
        def expand_dimension(bb):
            batch = np.expand_dims(bb,axis=0)
            batch = np.expand_dims(batch,axis=-1)
            return batch
        batch1 = expand_dimension(batch[0])
        batch2 = expand_dimension(batch[1])
        label = expand_dimension(batch[2])
        # print loss and accuracy
        if (iter+1)%50==0:
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()),"---iteration: {}".format(iter))
            model.fit([batch1,batch2],label,batch_size=1, epochs=1,verbose=1)  
        else:
            model.fit([batch1,batch2],label,batch_size=1, epochs=1,verbose=0)  

        # save model
        if (iter+1)%1000==0:
            fpath = 'weights/weight-{}.h5'.format(iter+1)
            model.save(fpath)
            print('model {} saved..'.format(fpath))
        sys.stdout.flush()



def inference(weight,test_batch):
    model = neuro_net(weight)
    tmp = model.predict(test_batch,batch_size=1, verbose = 0)
    return tmp



if __name__ == '__main__':
    if sys.argv[1]=="train":
        start_training()
        if len(sys.argv>2):
            start_training(sys.argv[2])
    else:
        ww = "weights/weight-20000.h5"
        
        name1 = "data/train/x_350.png"
        name2 = "data/train/y_350.png"
        x_rep = cv2.imread(name1,cv2.IMREAD_ANYDEPTH)
        x_rep = np.expand_dims(x_rep,axis=0)
        x_rep = np.expand_dims(x_rep,axis=-1)
        y_rep = cv2.imread(name2,cv2.IMREAD_ANYDEPTH)
        y_rep = np.expand_dims(y_rep,axis=0)
        y_rep = np.expand_dims(y_rep,axis=-1)

        test_batch = [x_rep,y_rep]
        res = inference(ww,test_batch)
        
        cv2.imwrite("test.png",np.squeeze(res))
        
        plt.subplot(121)
        plt.imshow(np.squeeze(res),cmap="gray")
        plt.subplot(122)
        im = cv2.imread("data/train/im350.png",cv2.IMREAD_ANYDEPTH)
        plt.imshow(im,cmap="gray")
        plt.show()





