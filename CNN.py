# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 01:55:00 2020

@author: toshi
"""

import os 
os.chdir("C:\\Users\\toshi\\Downloads\\deepLearning\\Deep_Learning_A_Z\\Volume 1 - Supervised Deep Learning\\Part 2 - Convolutional Neural Networks (CNN)\\Section 8 - Building a CNN\\Convolutional_Neural_Networks\\dataset")
#Part1 : building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#initialing the CNN
classifier = Sequential()

#step-1 convolution
classifier.add(Convolution2D(32, 3, 3, input_shape= (64, 64, 3),  activation='relu'))
             
# step 2 :- pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#step 3 :- flattining
classifier.add(Flatten())


#step 4 :- full connection
classifier.add(Dense(output_dim=128,activation = 'relu'))
#hidden layer
classifier.add(Dense(output_dim=1,activation = 'sigmoid'))



#compling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Part2-fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset//training_set', 
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset//test_set',
                                           target_size=(64, 64),
                                            batch_size=32,
                                           class_mode='binary')

classifier.fit_generator(training_set,samples_per_epoch=8000,
                         nb_epoch=25,validation_data=test_set ,nb_val_samples=2000)


#part 3 / making the new single pridiction

























