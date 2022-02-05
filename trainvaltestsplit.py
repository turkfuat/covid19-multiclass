#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from numpy import asarray
import pandas as pd
import collections
from sklearn.utils import shuffle
import shutil
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam


# In[3]:


normal_df = os.listdir('D:\\ozelveriseti/normal')
normal_df = shuffle(normal_df)
train_normal, test_normal = train_test_split(normal_df, test_size=0.3, random_state=101)
val_normal, test_normal = train_test_split(test_normal, test_size=0.5, random_state=101)
print(len(train_normal), len(val_normal), len(test_normal))


# In[4]:


covid_df = os.listdir('D:\\ozelveriseti/covid')
covid_df = shuffle(covid_df)
train_covid, test_covid = train_test_split(covid_df, test_size=0.3, random_state=101)
val_covid, test_covid = train_test_split(test_covid, test_size=0.5, random_state=101)
print(len(train_covid), len(val_covid), len(test_covid))


# In[17]:


lungopacity_df = os.listdir('D:\\ozelveriseti/lungopacity')
lungopacity_df = shuffle(lungopacity_df)
train_lungopacity, test_lungopacity = train_test_split(lungopacity_df, test_size=0.3, random_state=101)
val_lungopacity, testlungopacity = train_test_split(test_lungopacity, test_size=0.5, random_state=101)
print(len(train_lungopacity), len(val_lungopacity), len(test_lungopacity))


# In[25]:


pneumonia_df = os.listdir('D:\\ozelveriseti/pneumonia')
pneumonia_df = shuffle(pneumonia_df)
train_pneumonia, test_pneumonia = train_test_split(pneumonia_df, test_size=0.3, random_state=101)
val_pneumonia, test_pneumonia = train_test_split(test_pneumonia, test_size=0.5, random_state=101)
print(len(train_pneumonia), len(val_pneumonia), len(test_pneumonia))


# In[26]:


data_path = 'D:\\ozelveriseti/'


# In[27]:


train_path = os.path.join(data_path, 'train')
os.mkdir(train_path)
train_normal_path = os.path.join(train_path, 'normal')
os.mkdir(train_normal_path)
train_Viral_Pneumonia_path = os.path.join(train_path, 'pneumonia')
os.mkdir(train_Viral_Pneumonia_path)
train_Covid_path = os.path.join(train_path, 'covid')
os.mkdir(train_Covid_path)
train_opacity_path = os.path.join(train_path, 'lungopacity')
os.mkdir(train_opacity_path)


# In[28]:


# val_path = os.path.join(data_path, 'val')
# os.mkdir(val_path)
# val_normal_path = os.path.join(val_path, 'normal')
# os.mkdir(val_normal_path)
# val_abnormal_path = os.path.join(val_path, 'abnormal')
# os.mkdir(val_abnormal_path)
val_path = os.path.join(data_path, 'val')
os.mkdir(val_path)
val_normal_path = os.path.join(val_path, 'normal')
os.mkdir(val_normal_path)
val_Viral_Pneumonia_path = os.path.join(val_path, 'pneumonia')
os.mkdir(val_Viral_Pneumonia_path)
val_Covid_path = os.path.join(val_path, 'covid')
os.mkdir(val_Covid_path)
val_opacity_path = os.path.join(val_path, 'lungopacity')
os.mkdir(val_opacity_path)


# In[29]:


# test_path = os.path.join(data_path, 'test')
# os.mkdir(test_path)
# test_normal_path = os.path.join(test_path, 'normal')
# os.mkdir(test_normal_path)
# test_abnormal_path = os.path.join(test_path, 'abnormal')
# os.mkdir(test_abnormal_path)
test_path = os.path.join(data_path, 'test')
os.mkdir(test_path)
test_normal_path = os.path.join(test_path, 'normal')
os.mkdir(test_normal_path)
test_Viral_Pneumonia_path = os.path.join(test_path, 'pneumonia')
os.mkdir(test_Viral_Pneumonia_path)
test_Covid_path = os.path.join(test_path, 'covid')
os.mkdir(test_Covid_path)
test_opacity_path = os.path.join(test_path, 'lungopacity')
os.mkdir(test_opacity_path)


# In[1]:


def img_preprocessing(src):
    img = cv.imread(src, 0)
    org_img = img.copy()
    brightest = np.amax(img)
    darkest = np.amin(img)
    T = darkest + 0.9*(brightest - darkest)
    thre_img = cv.threshold(img, T, 255, cv.THRESH_BINARY)
    thre_img = thre_img[1]
    kernel = np.ones((5, 5),np.uint8)
    cleaned = cv.erode(thre_img,kernel,iterations = 5)
    cleaned = cv.dilate(cleaned,kernel,iterations = 5)
    cleaned = cleaned // 255
    img = img * cleaned
    img = org_img - img
    dim = (224, 224)
    img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
    B = cv.bilateralFilter(img, 9, 75, 75)
    R = cv.equalizeHist(img)
    new_img = cv.merge((B, img, R))
    return new_img


# In[5]:



plt.imshow('D:\\ozelveriseti\\pneumonia/Viral Pneumonia-1.png')


# In[6]:


img = img_preprocessing('D:\\ozelveriseti\\pneumonia/Viral Pneumonia-1.png')
plt.imshow(img)


# In[32]:


# normal train data
for img in train_normal:
    src = 'D:\\ozelveriseti\\normal/' + img
    im = img_preprocessing(src)
    dst = train_path + '/' + 'normal/' + img
    cv.imwrite(dst, im)


# In[35]:


#abnormal train data
for img in train_covid:
    src = 'D:\\ozelveriseti\\covid/' + img
    im = img_preprocessing(src)
    dst = train_path + '/' + 'covid/' + img
    cv.imwrite(dst, im)
for img in train_lungopacity:
     src = 'D:\\ozelveriseti\\lungopacity/' + img
     im = img_preprocessing(src)
     dst = train_path + '/' + 'lungopacity/' + img
     cv.imwrite(dst, im)
for img in train_pneumonia:
    src = 'D:\\ozelveriseti\\pneumonia/' + img
    im = img_preprocessing(src)
    dst = train_path + '/' + 'pneumonia/' + img
    cv.imwrite(dst, im) 


# In[41]:


# normal validation data
for img in val_normal:
    src = 'D:\\ozelveriseti\\normal/' + img
    im = img_preprocessing(src)
    dst = val_path + '/' + 'normal/' + img
    cv.imwrite(dst, im)
#abnormal validation data
for img in val_covid:
    src = 'D:\\ozelveriseti\\covid/' + img
    im = img_preprocessing(src)
    dst = val_path + '/' + 'covid/' + img
    cv.imwrite(dst, im)
for img in val_lungopacity:
     src = 'D:\\ozelveriseti\\lungopacity/' + img
     im = img_preprocessing(src)
     dst = val_path + '/' + 'lungopacity/' + img
     cv.imwrite(dst, im)
for img in val_pneumonia:
    src = 'D:\\ozelveriseti\\pneumonia/' + img
    im = img_preprocessing(src)
    dst = val_path + '/' + 'pneumonia/' + img
    cv.imwrite(dst, im) 


# In[42]:


# normal test data
for img in test_normal:
    src = 'D:\\ozelveriseti\\normal/' + img
    im = img_preprocessing(src)
    dst = test_path + '/' + 'normal/' + img
    cv.imwrite(dst, im)
#abnormal test data
for img in test_covid:
    src = 'D:\\ozelveriseti\\covid/' + img
    im = img_preprocessing(src)
    dst = test_path + '/' + 'covid/' + img
    cv.imwrite(dst, im)
for img in test_lungopacity:
     src = 'D:\\ozelveriseti\\lungopacity/' + img
     im = img_preprocessing(src)
     dst = test_path + '/' + 'lungopacity/' + img
     cv.imwrite(dst, im)
for img in test_pneumonia:
    src = 'D:\\ozelveriseti\\pneumonia/' + img
    im = img_preprocessing(src)
    dst = test_path + '/' + 'pneumonia/' + img
    cv.imwrite(dst, im)


# BURASI DATA AUGMENTATION İLE DEVAM EDER (ZORUNLU DEĞİL)

# In[126]:


train_path = 'D:\\ozelveriseti\\train/'
valid_path = 'D:\\ozelveriseti\\val/'
test_path =  'D:\\ozelveriseti\\test/'


# In[127]:


NUM_AUG_IMAGES_WANTED = 11000

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224


# In[129]:


class_list = ['normal','pneumonia', 'Ccovid','lungopacity']

for item in class_list:
    aug_dir = os.path.join(data_path, 'aug_dir')
    os.mkdir(aug_dir)
    img_dir = os.path.join(aug_dir, 'img_dir')
    os.mkdir(img_dir)
    img_class = item
    img_list = os.listdir(train_path + img_class)
    for fname in img_list:
            src = os.path.join(train_path + img_class, fname)
            dst = os.path.join(img_dir, fname)
            shutil.copyfile(src, dst)
    path = aug_dir
    save_path = train_path + img_class

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    batch_size = 50

    aug_datagen = datagen.flow_from_directory(path,save_to_dir=save_path,save_format='png',target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),batch_size=batch_size)
    num_files = len(os.listdir(img_dir))
    num_batches = int(np.ceil((NUM_AUG_IMAGES_WANTED-num_files)/batch_size))

    for i in range(0,num_batches):
        imgs, labels = next(aug_datagen)
    shutil.rmtree(aug_dir)


# In[130]:


# datagen = ImageDataGenerator(rescale=1.0/255, rotation_range=0.18, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen = ImageDataGenerator(rescale=1.0/255)
train_gen = datagen.flow_from_directory(train_path,
                                        target_size=(224,224),
                                        batch_size=10,
                                        class_mode='categorical')                                      
val_gen = datagen.flow_from_directory(valid_path,
                                        target_size=(224,224),
                                        batch_size=10,
                                        class_mode='categorical')
test_gen = datagen.flow_from_directory(test_path,
                                        target_size=(224,224),
                                        batch_size=1,
                                        class_mode='categorical',
                                        shuffle=False)


# In[ ]:




