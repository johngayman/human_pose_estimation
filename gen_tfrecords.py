import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def gen_TFRecords(num_examples_per_record, annot, images_path, output_folder):
  num_tfrecords = len(annot) // num_examples_per_record 
  if len(annot) % num_examples_per_record:
    num_tfrecords += 1 # add one record if there are any remaining samples

  if not os.path.exists("tfrecords/" + output_folder):
    os.makedirs("tfrecords/" + output_folder)  # creating TFRecords output folder
  
  for tfrecord in range(num_tfrecords):
    examples = annot[(tfrecord * num_examples_per_record) : ((tfrecord + 1) * num_examples_per_record)]
    
    with tf.io.TFRecordWriter(
        "tfrecords/" + output_folder + "/file_" + output_folder + "_%.2i-%i.tfrec" % (tfrecord, len(examples))
    ) as writer:
        for index, row in examples.iterrows():
            image_path = images_path + row['image_path']
            image = tf.io.decode_jpeg(tf.io.read_file(image_path))
            tfrecord_example = create_example(image, image_path, row, index)
            writer.write(tfrecord_example.SerializeToString())
  print("TFRecords generated at tfrecords/" + output_folder)

### example
def gen_examples_from_tfrecord(filepath, example_num):
  raw_dataset = tf.data.TFRecordDataset(filepath)
  parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
  for example in parsed_dataset.take(example_num):
      for key in example.keys():
          if key != "image":
              print(f"{key}: {example[key]}")

      xcoords = example["xcoords"].numpy()
      ycoords = example["ycoords"].numpy()
      image = example["image"]
      h, w, c = image.shape
      print(f"Image shape: {image.shape}")
      plt.figure(figsize=(7, 7))
  
      plt.imshow(image)
      plt.scatter(xcoords  , ycoords  , marker = "o") # for heatmap size
      plt.show()  

### For writing
def create_example(image, image_path, example, index):
  ## Bbox
  #0: x left top, 1: y left top, 2: width, 3: height
  bbox_x, bbox_y, bbox_w, bbox_h = [int(i) for i in example['bbox']]

  ## Parse x and y coords
  kps = example['keypoints']
  #recalculate for cropping 
  xcoords = [kps[i] - bbox_x for i in range(len(kps)) if i%3 == 0] 
  ycoords = [kps[i] - bbox_y for i in range(len(kps)) if i%3 == 1]
  #visibility flag
  vis = [kps[i] for i in range(len(kps)) if i%3 == 2]

  ## Annotation id, unique for each example
  ann_id = example['ann_id']

  ## Image id
  image_id = index # since we use image id as index for coco_df

  ## Crop 
  # since we use bbox to crop the bbox width and height will become out new width and height
  image = tf.image.crop_to_bounding_box(image, bbox_y, bbox_x, bbox_h, bbox_w)

  ## Features
  feature = {
        "ann_id": int64_feature(ann_id),
        "image_id": int64_feature(image_id),
        "image": image_feature(image),
        "image_path": bytes_feature(image_path),
        "width": int64_feature(bbox_w),
        "height": int64_feature(bbox_h),
        "xcoords": float_feature_list(xcoords),
        "ycoords": float_feature_list(ycoords),
        "vis": float_feature_list(vis),
    }
  return tf.train.Example(features=tf.train.Features(feature=feature))

### For reading
def parse_tfrecord_fn(example):
    feature_description = {
        "ann_id": tf.io.FixedLenFeature([], tf.int64),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_path": tf.io.FixedLenFeature([], tf.string),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.int64),
        "xcoords": tf.io.VarLenFeature(tf.float32),
        "ycoords": tf.io.VarLenFeature(tf.float32),
        "vis": tf.io.VarLenFeature(tf.float32),
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.image.decode_image(example["image"], channels = 3, dtype = tf.float32, expand_animations =False)
    example["xcoords"] = tf.sparse.to_dense(example["xcoords"])
    example["ycoords"] = tf.sparse.to_dense(example["ycoords"])
    example["vis"] = tf.sparse.to_dense(example["vis"])
    return example

### Helpers
def image_feature(value):
    """Returns a bytes_list from a string / byte."""

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]) #only for jpeg/jpg
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
def int64_feature_list(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))