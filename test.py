

#from train import *


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pickle

def test_load():
    dd = load_data()
    print(len(dd))
    print(len(dd[0]))

    test_data = dd[0]
    print(type(test_data[0]))
    plt.subplot(131)
    plt.imshow(test_data[0],cmap="gray")
    plt.subplot(132)
    plt.imshow(test_data[1],cmap="gray")
    plt.subplot(133)
    plt.imshow(test_data[2],cmap="gray")
    plt.show()

def test_res():
    pkl_file = open('data.pkl', 'rb')
    data1 = pickle.load(pkl_file)
    pkl_file.close()

    
    print(type(data1))
    print(len(data1))
    print(data1[0].shape)
    plt.subplot(121)
    plt.imshow(np.squeeze(data1[0]),cmap="gray")
    plt.subplot(122)
    plt.imshow(np.squeeze(data1[1]),cmap="gray")
    plt.show()
if __name__ == "__main__":
    test_res()




