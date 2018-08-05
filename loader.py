import matplotlib.pyplot as plt
import sys, pdb, os
import cv2, glob, pickle
from random import shuffle
from itertools import cycle
import numpy as np
import pandas as pd

class Loader():
    def __init__(self,batch_size=25,grey=False):
        self.batch_size = batch_size
        self.shuffle = True
        self.dim=[32,32,3]
        self.grey = grey 
        self.dim[2] = 1 if grey == 1 else 3
        self.h, self.w, self.c = self.dim
        self.epoch = 0
        self.label = 0
        labels_name = {0:'plane',1:'car',2:'bird',3:'cat',4:'deer',5:'dog',6:'frog',7:'horse',8:'ship',9:'truck'}
        self.label_name = labels_name[self.label]
        self.make_dirs()

    def load_cifar10_batch(self,label=8):
        file_paths = glob.glob("cifar-10-batches-py/data_batch_*")
        file_paths.sort()
        file_paths_cycle = cycle(file_paths)

        while True:
            self.file_path = next(file_paths_cycle)
            with open(self.file_path, mode='rb') as file:
                data = pickle.load(file)
            imgs = data['data']
            labels = np.array(data['labels'])
            imgs = imgs.reshape(-1,3,self.h,self.w)
            imgs = imgs.transpose([0,2,3,1])
            if self.grey == True:
                imgs = imgs.mean(3)[:,:,:,np.newaxis] # naaive greyscaling need to do (0.3 * R) + (0.59 * G) + (0.11 * B) 
                self.dim = imgs.shape[1:]
                self.c = 1
            imgs = self.img_norm(imgs)
            yield imgs[np.where(labels==label)] # just take one class at the moment

    def img_norm(self,imgs,inverse=False):
        sf = 1/127.5
        return imgs*sf  - 1 if inverse == False else ((imgs+1)*1/sf).astype(np.uint8)

    def data_gen(self):
        cifar_batch_gen = self.load_cifar10_batch()
        cifar_batch = next(cifar_batch_gen)
        idx_start = 0
        idx_end = self.batch_size
        total = 0
        while True:
            if idx_end == cifar_batch.shape[0]:
                cifar_batch = next(cifar_batch_gen)
                print("{0} shape = {1}. Epoch = {2}. Total seen for '{3}' = {4}.".format(
                    self.file_path,cifar_batch.shape,self.epoch,self.label_name,total))
                idx_start = 0
                self.epoch += 0.2
            else:
                idx_start += self.batch_size
            idx_end = np.min([idx_start+self.batch_size,cifar_batch.shape[0]])
            indices = np.arange(idx_start,idx_end,1)
            total += indices.size
            yield cifar_batch[indices]

    def make_dirs(self):
        dir_path = "samples"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

if __name__ == '__main__':
    dgen = Loader(batch_size=25)
    dgenerator = dgen.data_gen()
    while dgen.epoch<10:
        data = next(dgenerator)
        
    data = dgen.img_norm(data,inverse=True)
    for i in range(25):
        cv2.imwrite('samples/img_{0}.jpg'.format(i),data[i])
