import matplotlib.pyplot as plt
import sys, pdb, os
import cv2, glob, pickle
from random import shuffle
import numpy as np
import pandas as pd

class Loader():
    def __init__(self,batch_size=25,grey=True):
        self.batch_size = batch_size
        self.shuffle = True
        self.grey = True
        self.dim=[32,32,1]
        self.dim[2] = 1 if self.grey == True else 3
        self.h, self.w, self.c = self.dim
        self.n_channels = self.c

    def load_cifar10(self,label=8):

        file_path = "cifar-10-batches-py/data_batch_1" 
        with open(file_path, mode='rb') as file:
            data = pickle.load(file)
        imgs = data['data']
        labels = np.array(data['labels'])
        imgs = imgs.reshape(-1,3,self.h,self.w)
        imgs = imgs.transpose([0,2,3,1])
        if self.grey == True:
            imgs = imgs.mean(3)[:,:,:,np.newaxis]
            self.dim = imgs.shape[1:]
        imgs = self.img_norm(imgs)
        return imgs[np.where(labels==label)]

    def img_norm(self,imgs,inverse=False):
        sf = 1/127.5
        return imgs*sf  - 1 if inverse == False else (imgs+1)*1/sf

    def data_gen(self):
        data = self.load_cifar10()

        print("Training examples shape = {0}".format(data.shape))
        while True:
            idx = np.random.randint(0,data.shape[0],self.batch_size)
            yield data[idx]

if __name__ == '__main__':
    dgen = Loader(batch_size=20)
    dgenerator = dgen.data_gen()
    data = next(dgenerator)
    data = dgen.img_norm(data,inverse=True).astype(np.uint8)
    for i in range(10):
        cv2.imwrite('eg/img_{0}.jpg'.format(i),data[i])
