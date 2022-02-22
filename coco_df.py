import os
import json
import tensorflow as tf
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import argparse

# Constants
ROOT_DIR = "datasets"
TRAIN_IMAGES_DIR = os.path.join(ROOT_DIR, "train2017")
VALID_IMAGES_DIR = os.path.join(ROOT_DIR, "val2017")
TRAIN_IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"
VALID_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"

ANNOTATIONS_DIR = os.path.join(ROOT_DIR, "annotations")
TRAIN_ANNOT_FILE = os.path.join(ANNOTATIONS_DIR, "person_keypoints_train2017.json")
VALID_ANNOT_FILE = os.path.join(ANNOTATIONS_DIR, "person_keypoints_val2017.json")
ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

def prepare(download_images = True):
  if download_images:
    dowload_files(TRAIN_IMAGES_DIR, TRAIN_IMAGES_URL)
    dowload_files(VALID_IMAGES_DIR, VALID_IMAGES_URL)

  dowload_files(ANNOTATIONS_DIR, ANNOTATIONS_URL)


def gen_trainval_df():
  #train
  train_coco = COCO(TRAIN_ANNOT_FILE) 
  train_images_df, train_persons_df = convert_to_df(train_coco)
  train_coco_df = pd.merge(train_images_df, train_persons_df, right_index=True, left_index=True)
  train_coco_df = train_coco_df[(train_coco_df['is_crowd'] == 0) & (train_coco_df['num_keypoints'] > 0)]

  #valid
  valid_coco = COCO(VALID_ANNOT_FILE)
  valid_images_df, valid_persons_df = convert_to_df(valid_coco)
  valid_coco_df = pd.merge(valid_images_df, valid_persons_df, right_index=True, left_index=True)
  valid_coco_df = valid_coco_df[(valid_coco_df['is_crowd'] == 0) & (valid_coco_df['num_keypoints'] > 0)]

  print(f"Only examples that are not crowd and num_keypoints > 0 are chosen !")
  print(f"Length of train df: {len(train_coco_df)}")
  print(f"Length of valid df: {len(valid_coco_df)}")
  return train_coco_df, valid_coco_df

def get_meta(coco):
  # Get all images identifier, the length should be the number of images
  ids = list(coco.imgs.keys())
  for i, img_id in enumerate(ids):
    img_meta = coco.imgs[img_id]
    ann_ids = coco.getAnnIds(imgIds = img_id)
    # retrieve meta data for all persons in the current image
    anns = coco.loadAnns(ann_ids)
    # basic para of an image
    img_file_name = img_meta['file_name']
    w = img_meta['width']
    h = img_meta['height']
    url = img_meta['coco_url']
    

    yield [img_id, img_file_name, w, h, url, anns]

def convert_to_df(coco):
  images_data = []
  persons_data = []

  # iterate over all images
  for img_id, img_file_name, w, h, url, meta in get_meta(coco):
    images_data.append({
        'image_id': int(img_id),
        'coco_url': url,
        'image_path': img_file_name,
        'width': int(w),
        'height': int(h)
        })
    # iterate over all metadata
    for m in meta:
      persons_data.append({
          'ann_id': m['id'], #each example will have a unique id
          'image_id': m['image_id'],
          'is_crowd': m['iscrowd'],
          'bbox': m['bbox'],
          'num_keypoints': m['num_keypoints'],
          'keypoints': m['keypoints'],

      })
  # create dataframes
  images_df = pd.DataFrame(images_data)
  images_df.set_index('image_id', inplace = True) # set imgae_id as 1st row

  persons_df = pd.DataFrame(persons_data)
  persons_df.set_index('image_id', inplace = True)

  return images_df, persons_df

# Download and extract said files, remove the zip afterwards
def dowload_files(files_dir, files_url):
  if not os.path.exists(files_dir):
    zip_file = tf.keras.utils.get_file(
        "file.zip",
        cache_dir=os.path.abspath("."),
        origin=files_url,
        extract=True,
    )
    os.remove(zip_file)
