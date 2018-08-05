from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import Activation, LeakyReLU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys, pdb, cv2, os
from loader import Loader

os.environ["CUDA_VISIBLE_DEVICES"]="1" 


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

        self.discriminator = self.build_discriminator() # Discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator() # Build the generator

        # Generator(noise) ---> imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False # Combined model we train the generator

        validity = self.discriminator(img) # Discriminator(image) --> valid or fake?

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

        model.add(ConvT(128,kernel_size=5,strides=2,padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())

        model.add(ConvT(self.channels,kernel_size=5,strides=2,padding='same'))
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

    def train(self):

        n_batches = 100000
        sample_interval=50
        for i in range(n_batches):
            epoch = self.dgen.epoch

            imgs = next(self.gen) # generate real images
            batch_size = imgs.shape[0]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)) # random noise as input to GEN
            gen_imgs = self.generator.predict(noise) # generate fake images
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #  Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim)) # random noise input
            g_loss = self.combined.train_on_batch(noise, valid) # train generator by feeding fake images through discriminator which are likely to produce a valid response
            print ("Epoch %d - batch_no %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch,i,
                d_loss[0], 100*d_loss[1], g_loss))

            if i % sample_interval == 0:
                self.save_imgs(imgs,'real')
                self.save_imgs(gen_imgs,'fake_{0}_{1}_'.format(epoch,i))
            if i % 1000 == 0:
                print("Saving")
                self.generator.save('gen.h5')
                self.combined.save('model.h5')

    def save_imgs(self,arr,ext):
        arr = self.dgen.img_norm(arr,inverse=True).astype(np.uint8)
        for i in range(arr.shape[0]):
            cv2.imwrite("samples/{0}_{1}.jpg".format(ext,i),arr[i])




if __name__ == '__main__':
    inference = False
    train = True 
    if train == True:
        gan = GAN(load=False)
        gan.train()

