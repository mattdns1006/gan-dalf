from __future__ import print_function, division
from argparse import ArgumentParser
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
import seaborn as sns
sns.set()



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

        # Distribution params for latent
        self.latent_dim = 20
        self.loc = 0
        self.scale = 1

        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator() # Discriminator
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        if self.load == True:
            print('Loading models')
            self.generator = load_model('gen.h5')
        else:
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
        
        init_feats = 128
        model.add(Dense(init_feats*4*4,input_shape=(self.latent_dim,)))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())
        model.add(Reshape((4,4,init_feats)))

        model.add(ConvT(64,kernel_size=5,strides=2,padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())

        model.add(ConvT(64,kernel_size=5,strides=2,padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())

        model.add(ConvT(48,kernel_size=5,strides=2,padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())

        model.add(ConvT(24,kernel_size=5,strides=1,padding='same'))
        model.add(LeakyReLU(0.2))
        model.add(BatchNormalization())

        model.add(ConvT(self.channels,kernel_size=3,strides=1,padding='same'))
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
        model.add(Dense(128))
        model.add(LeakyReLU(0.2))
        model.add(Dense(1, activation='sigmoid'))
        print(10*'*'+'Discriminator '+10*'*')
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self):

        n_batches = 30000
        sample_interval= 1000
        self.gen_losses = np.zeros(n_batches) 
        self.disc_metrics = np.zeros((n_batches,2))

        for i in range(n_batches):
            epoch = self.dgen.epoch

            imgs = next(self.gen) # generate real images
            batch_size = imgs.shape[0]
            noise = np.random.normal(self.loc, self.scale, (batch_size, self.latent_dim)) # random noise as input to GEN
            gen_imgs = self.generator.predict(noise) # generate fake images
            #valid = np.ones((batch_size, 1))
            #fake = np.zeros((batch_size, 1))
            valid = np.random.uniform(low=0.9,high=0.99,size=(batch_size, 1))
            fake = np.random.uniform(low=0.01,high=0.1,size=(batch_size, 1))

            # Train the discriminator
            disc_metrics_real = self.discriminator.train_on_batch(imgs, valid)
            disc_metrics_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            self.disc_metrics[i] = 0.5 * np.add(disc_metrics_real, disc_metrics_fake)

            #  Train Generator
            noise = np.random.normal(self.loc, self.scale, (batch_size, self.latent_dim)) # random noise input
            self.gen_losses[i] = self.combined.train_on_batch(noise, valid) # train generator by feeding fake images through discriminator which are likely to produce a valid response

            if i > 20 and i % 10 == 0:
                print ("Epoch %d - batch_no %d [D loss: %f, acc.: %.2f%%] [G loss: %f] (Running means (20obs))" % (epoch,i,
                    self.disc_metrics[i-20:i,0].mean(), 100*self.disc_metrics[i-20:i,1].mean(), self.gen_losses[i-20:i].mean()))

            if i % sample_interval == 0 and i > 0:
                self.save_imgs(gen_imgs,'samples/'.format(epoch,i))
                np.save('gen',self.gen_losses)
                np.save('disc',self.disc_metrics)
                #self.save_imgs(imgs,'real')
                self.generator.save('gen.h5')
                #self.combined.save('model.h5')


    def save_imgs(self,arr,directory):
        arr = self.dgen.img_norm(arr,inverse=True).astype(np.uint8)
        for i in range(arr.shape[0]):
            cv2.imwrite("{0}/{1}.jpg".format(directory,i),arr[i])

    def inference(self):
        np.random.seed(1234)
        noise = np.random.normal(self.loc, self.scale, (1, self.latent_dim)) # random noise as input to GEN
        n_images = 25 
        noise = np.tile(noise,n_images).reshape(n_images,self.latent_dim)
        x = np.linspace(-3.0,3.0,n_images)
        noise[:,3] = x
        imgs = self.generator.predict(noise)
        self.save_imgs(imgs,'inference/')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--inference", default=False, action='store_true', help="Generator inference")
    parser.add_argument("--train", default=False, action='store_true', help="Train algorithm?")
    ags = parser.parse_args()
    if ags.train == True:
            gan = GAN(load=False)
            gan.train()
    if ags.inference == True:
        gan = GAN(load=True)
        gan.inference()


