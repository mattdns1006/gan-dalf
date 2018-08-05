from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Activation, LeakyReLU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys, pdb, cv2, os
from loader import Loader
os.environ["CUDA_VISIBLE_DEVICES"]="1" 

import numpy as np


class GAN():
    def __init__(self,load):

        self.dgen = Loader()
        self.gen = self.dgen.data_gen()
        self.load = load 
        self.batch_size = self.dgen.batch_size
        self.img_shape = (self.dgen.h, self.dgen.w,self.dgen.c)
        self.h = self.img_shape[0]
        self.w = self.img_shape[1]
        self.channels = self.img_shape[2]
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        if self.load == True:
            print('loaded generator')
            self.generator = load_model('gen.h5')
        else:
            self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        ConvT = Conv2DTranspose
        model = Sequential()
        model.add(Dense(512*4*4,input_shape=(self.latent_dim,)))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())
        model.add(Reshape((4,4,512)))

        model.add(ConvT(256,kernel_size=5,strides=2,padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())

        model.add(ConvT(64,kernel_size=5,strides=2,padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())

        model.add(ConvT(1,kernel_size=5,strides=2,padding='same'))
        model.add(Activation('tanh'))

        print(10*'*'+'Generator'+10*'*')
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',input_shape=(self.w,self.h,self.channels,)))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(64,kernel_size=5,strides=2,padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())

        model.add(Conv2D(64,kernel_size=5,strides=2,padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dense(1, activation='sigmoid'))
        print(10*'*'+'Discriminator '+10*'*')
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, sample_interval=50):

        batch_size = self.batch_size
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        def save_imgs(arr,ext):
            arr = self.dgen.img_norm(arr,inverse=True).astype(np.uint8)
            for i in range(arr.shape[0]):
                cv2.imwrite("eg/{0}_{1}.jpg".format(ext,i),arr[i])
            


        for epoch in range(epochs):

            # Select a random batch of images
            imgs = next(self.gen)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #  Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                save_imgs(imgs,'real')
                save_imgs(gen_imgs,'fake_{0}_'.format(epoch))
            if epoch % 1000 == 0:
                print("Saving")
                self.generator.save('gen.h5')
                self.combined.save('model.h5')


if __name__ == '__main__':
    inference = False
    train = True 
    if train == True:
        gan = GAN(load=False)
        gan.train(epochs=30000, sample_interval=200)
    if inference == True:
        gan = GAN(load=True)
        gan.sample_images(-1,n=10)

