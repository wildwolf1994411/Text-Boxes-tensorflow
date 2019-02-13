from enum import Enum, IntEnum
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from processing import tf_image
slim = tf.contrib.slim
import pdb
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.
IMAGE_SIZE = (300, 300)

Resize = IntEnum('Resize', ('NONE',                # Nothing!
                            'CENTRAL_CROP',        # Crop (and pad if necessary).
                            'PAD_AND_RESIZE',      # Pad, and resize to output shape.
                            'WARP_RESIZE'))        # Warp resize.

def preprocess_for_train(image, labels, bboxes,
                         out_shape, data_format='NHWC', use_whiten=True,
                         scope='textbox_process_train'):
    with tf.name_scope(scope, 'textbox_process_train', [image, labels, bboxes]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        bboxes = tf_image.clip_bboxes(bboxes)
        dst_image = image
        dst_image, labels, bboxes = tf_image.distorted_bounding_box_crop(image, labels, bboxes)
        dst_image = tf_image.distort_color(dst_image)
        dst_image, bboxes = tf_image.random_flip_left_right(dst_image, bboxes)
        dst_image = tf_image.resize_image(dst_image, out_shape) 
        num = tf.reduce_sum(tf.cast(labels, tf.int32))
        image = dst_image*255.0
        image = tf_image.tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        return image, labels, bboxes, num

def preprocess_for_test(image, out_shape=IMAGE_SIZE, scope='ssd_preprocessing_test'):
    with tf.name_scope(scope):
        image = tf.to_float(image)
        image = tf_image.tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
        image = tf_image.resize_image(image, out_shape)
        bbox_img = tf.constant([0., 0., 1., 1.])
        return image, bbox_img
