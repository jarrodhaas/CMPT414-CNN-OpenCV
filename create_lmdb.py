'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import os
import glob
import random
import numpy as np

import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

train_lmdb = '/Users/JRod/Desktop/Jumper/fashion-data/input/train_lmdb'
validation_lmdb = '/Users/JRod/Desktop/Jumper/fashion-data/input/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)

train_path = './fashion-data/train2.txt'

print train_path

with open(train_path) as f:
    train_data = f.read().splitlines()

with open('./fashion-data/test2.txt') as g:
    test_data = g.read().splitlines()

train_data = ['./fashion-data/images/' + img + '.jpg' for img in train_data ]
test_data = ['./fashion-data/images/' + img + '.jpg' for img in test_data ]

#train_data = [img for img in glob.glob("../input/train/*jpg")]
#test_data = [img for img in glob.glob("../input/test1/*jpg")]

#Shuffle train_data
random.shuffle(train_data)

print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx %  6 == 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

        if 'blouse' in img_path:
            label = 0
        elif 'cloak' in img_path:
            label = 1
        elif 'coat' in img_path:
            label = 2
        elif 'jacket' in img_path:
            label = 3
        elif 'long_dress' in img_path:
            label = 4
        elif 'polo_shirt' in img_path:
            label = 5
        elif 'robe' in img_path:
            label = 6
        elif 'shirt' in img_path:
            label = 7
        elif 'short_dress' in img_path:
            label = 8
        elif 'suit' in img_path:
            label = 9
        elif 'sweater' in img_path:
            label = 10
        elif 't_shirt' in img_path:
            label = 11
        elif 'undergarment' in img_path:
            label = 12
        elif 'uniform' in img_path:
            label = 13
        elif 'vest' in img_path:
            label = 14


        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()


print '\nCreating validation_lmdb'

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx % 6 != 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'blouse' in img_path:
            label = 0
        elif 'cloak' in img_path:
            label = 1
        elif 'coat' in img_path:
            label = 2
        elif 'jacket' in img_path:
            label = 3
        elif 'long_dress' in img_path:
            label = 4
        elif 'polo_shirt' in img_path:
            label = 5
        elif 'robe' in img_path:
            label = 6
        elif 'shirt' in img_path:
            label = 7
        elif 'short_dress' in img_path:
            label = 8
        elif 'suit' in img_path:
            label = 9
        elif 'sweater' in img_path:
            label = 10
        elif 't_shirt' in img_path:
            label = 11
        elif 'undergarment' in img_path:
            label = 12
        elif 'uniform' in img_path:
            label = 13
        elif 'vest' in img_path:
            label = 14

        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print '\nFinished processing all images'
