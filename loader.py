import matplotlib.pyplot as plt
import sys, pdb, os
import cv2, glob, pickle
from random import shuffle
import numpy as np
import pandas as pd

class Loader():
    def __init__(self,batch_size=25,grey=False,dataset='cifar10'):
        self.batch_size = batch_size
        self.shuffle = True
        self.grey = grey 
        self.epoch = 0
        if dataset == 'cifar10':
            self.data = self.load_cifar10()
        self.make_dirs()

    def load_cifar10(self,label=8):
        file_path = "cifar-10-batches-py/data_batch_1" 
        with open(file_path, mode='rb') as file:
            data = pickle.load(file)
        imgs = data['data']
        labels = np.array(data['labels'])
        self.dim=[32,32,3]
        self.h, self.w, self.c = self.dim
        imgs = imgs.reshape(-1,3,self.h,self.w)
        imgs = imgs.transpose([0,2,3,1])
        if self.grey == True:
            imgs = imgs.mean(3)[:,:,:,np.newaxis] # naaive greyscaling need to do (0.3 * R) + (0.59 * G) + (0.11 * B) 
            self.dim = imgs.shape[1:]
            self.c = 1
        imgs = self.img_norm(imgs)
        return imgs[np.where(labels==label)] # just take one class at the moment

    def img_norm(self,imgs,inverse=False):
        sf = 1/127.5
        return imgs*sf  - 1 if inverse == False else (imgs+1)*1/sf

    def data_gen(self):
        print("Training examples shape = {0}".format(self.data.shape))
        idx_start = 0
        n = self.data.shape[0]
        while True:
            idx_end = np.min([idx_start+self.batch_size,n])
            indices = np.arange(idx_start,idx_end,1)
            if idx_end == n:
                idx_start = 0
                self.epoch += 1
            else:
                idx_start += self.batch_size

            yield self.data[indices]

    def make_dirs(self):
        dir_path = "samples"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

if __name__ == '__main__':
    dgen = Loader(batch_size=25)
    dgenerator = dgen.data_gen()
    data = next(dgenerator)
    data = dgen.img_norm(data,inverse=True).astype(np.uint8)
    for i in range(25):
        cv2.imwrite('samples/img_{0}.jpg'.format(i),data[i])
