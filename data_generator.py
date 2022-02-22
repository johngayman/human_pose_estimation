import os
import sys

import tensorflow as tf
import numpy as np
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
import matplotlib.pyplot as plt
from gen_tfrecords import parse_tfrecord_fn


class DataGenerator(object):
  def __init__(self, tfrecord_files, num_examples, inres, outres, 
                     num_keypoints, num_epochs, batch_size, is_train = True):
    self.tfrecord_files = tfrecord_files
    self.num_examples = num_examples
    self.inres = inres
    self.outres = outres
    self.num_keypoints = num_keypoints
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.is_train = is_train

## Prepare and process each tfrecord example
  def prepare_sample(self, example):
    # Getting all the necessary data
    img = example['image']
    img_shape = img.get_shape()
    height = example['height']
    width = example['width']
    xcoords = example['xcoords'] 
    ycoords = example['ycoords']

    # Augment the image and xcoords, ycoords, in image's dimension
    aug_img, aug_xcoords, aug_ycoords = self.tf_augment(img, xcoords, ycoords)
    aug_img.set_shape(img_shape)
    # Recalculate x,y into heatmaps space, outres should be(64, 64, nb_kps)
    h_ratio = self.outres[0] / height
    w_ratio = self.outres[0] / width 
    aug_xcoords = aug_xcoords * tf.cast(w_ratio, tf.float32)
    aug_ycoords = aug_ycoords * tf.cast(h_ratio, tf.float32)   

    #heatmap
    heatmap = self.tf_heatmap(aug_xcoords, aug_ycoords)

    #img to (256, 256)
    aug_img = tf.image.resize(aug_img, (self.inres[0], self.inres[1]))

    return  aug_img, heatmap


  ## Generate dataset
  def generateDataset(self):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.TFRecordDataset(self.tfrecord_files, num_parallel_reads = AUTOTUNE)
    ds = ds.repeat(self.num_epochs)
    if self.is_train:
      ds = ds.shuffle(self.batch_size * 100)#for true randomness, buffer is set to the num of examples in training set
    ds = ds.map(parse_tfrecord_fn, num_parallel_calls = AUTOTUNE)
    ds = ds.map(self.prepare_sample, num_parallel_calls = AUTOTUNE)
    ds = ds.batch(self.batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds
  
  ## Generate heatmap
  def np_heatmap(self, xcoords, ycoords):
    assert len(xcoords) == len(ycoords) == self.num_keypoints == self.outres[2] and self.outres[0] == self.outres[1]
    width = self.outres[0]
    height = self.outres[0]
    depth = len(xcoords)

    #np array filled with 0, image should be (h, w, c)
    heatmaps = np.zeros(shape = (height, width, depth), dtype = np.float32)
    for i in range(depth):
      #should use floor or int
      x_index = round(xcoords[i]) # width
      y_index = round(ycoords[i]) # height
      if(0 < x_index < width) and (0 < y_index < height):
        heatmaps[y_index][x_index][i] = 1.0 #entry at keypoint = 1
        heatmaps[:, :, i] = cv2.GaussianBlur(heatmaps[:,:,i],(7, 7), 0) #blurr
        heatmaps[:,:,i] = heatmaps[:,:,i] / heatmaps[:,:,i].max()#normalize
    return heatmaps
  def tf_heatmap(self, xcoords, ycoords):
    return tf.numpy_function(self.np_heatmap, [xcoords, ycoords], tf.float32)

## Augment pipeline
  def np_augment(self, image, xcoords, ycoords):

    # List of tuples [(x1, y1), ...]
    kps = [] 
    idx = [] #to store index of valid kps
    for i in range(len(xcoords)):
      x = xcoords[i]
      y = ycoords[i]
      if 0 < x < image.shape[1] and 0 < y < image.shape[0]:
        kps.append(Keypoint(x = x, y = y))
        idx.append(i)
    kpsoi = KeypointsOnImage(kps, shape = image.shape)

    # Augment
    seed = np.random.randint(1, 2**32-1)
    ia.seed(seed)
    seq = iaa.Sequential([
      iaa.Affine(scale = (0.75, 1.25), rotate = (-30, 30)),
      iaa.Fliplr(0.5),
      ], random_order = False) #cause rotate/scale then flip can be problematic
    aug_img, aug_kps = seq(image = image, keypoints = kpsoi)

    # Turn aug_kps back to x, y lists
    arr = aug_kps.to_xy_array()
    
    temp_xcoords = arr[:, 0]
    temp_ycoords = arr[:, 1]
    aug_xcoords = np.zeros(shape = (self.num_keypoints), dtype = np.float32)
    aug_ycoords = np.zeros(shape = (self.num_keypoints), dtype = np.float32)

    for i, x, y in zip(idx, temp_xcoords, temp_ycoords):
      aug_xcoords[i] = x
      aug_ycoords[i] = y;
    return aug_img, aug_xcoords, aug_ycoords
  def tf_augment(self, img, xcoords, ycoords):
    return tf.numpy_function(self.np_augment, [img, xcoords, ycoords], (tf.float32, tf.float32, tf.float32))





