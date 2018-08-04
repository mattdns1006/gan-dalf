import matplotlib.pyplot as plt
import sys, pdb, os
import cv2, glob, pickle
from random import shuffle
import numpy as np
import pandas as pd

class Loader():
    def __init__(self,batch_size=20,breed='scottish_deerhound'):
        self.breed = breed
        self.batch_size = batch_size
        self.shuffle = True

    def load_cifar10(self,label=8,grey=True):
        self.dim=(32,32,3)
        self.h, self.w, self.c = self.dim
        self.n_channels = self.c
        path = '../cifar10/cifar-10-batches-bin/data_batch_3.bin'
        data = np.fromfile(path,dtype=np.uint8)
        idx = np.arange(data.size)
        label_idx = idx[::32*32*3+1]
        labels = data[label_idx]
        img_idx = np.setdiff1d(idx,label_idx)
        imgs = data[img_idx]
        imgs = imgs.reshape(-1,3,32,32)
        imgs = imgs.transpose(0,2,3,1)
        if grey == True:
            imgs = imgs.mean(3)[:,:,:,np.newaxis]
            self.dim = imgs.shape[1:]
        imgs = self.img_norm(imgs)
        return imgs[np.where(labels==label)]

    def img_norm(self,imgs,inverse=False):
        sf = 1/127.5
        return imgs*sf  - 1 if inverse == False else (imgs+1)*1/sf

    def data_gen(self):
        data = self.load_cifar10()
        pdb.set_trace()
        print("Training examples shape = {0}".format(data.shape))
        idx = 0
        while True:
            X = np.empty((self.batch_size,*self.dim))
            for i in range(self.batch_size):
                X[i] = data[idx]
                idx += 1
                if idx == data.shape[0] - 1:
                    idx = 0
            yield X

if __name__ == '__main__':
    dgen = Loader(batch_size=20)
    dgenerator = dgen.data_gen()
    data = next(dgenerator)
    data = dgen.img_norm(data,inverse=True).astype(np.uint8)
    for i in range(10):
        cv2.imwrite('eg/{0}.jpg'.format(i),data[i])
