# Copyright 2015 The TensorFlow Authors and Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Custom image operations.
Most of the following methods extend TensorFlow image library, and part of
the code is shameless copy-paste of the former!
"""
import tensorflow as tf
import numpy as np
import pdb
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables

# ============================================================
# parameter setting
# ============================================================
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.
EVAL_SIZE = (300, 300)

BBOX_CROP_OVERLAP = 0.1         # Minimum overlap to keep a bbox after cropping.
MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.7, 1.3)  # Distortion ratio during cropping.
EVAL_SIZE = (300, 300)

# =========================================================================== #
# Modification of TensorFlow image routines.
# =========================================================================== #
def _creat_cond(length = 1):
    cond_list = []
    for _ in range(length):
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=None)
        cond = math_ops.less(uniform_random, .5)
        cond_list.append(cond)
    if length == 1:
        return cond
    return cond_list

def _ImageDimensions(image):

    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(3).as_list()
        dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def _Check3DImage(image, require_static=True):

    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' must be three-dimensional.")
    if require_static and not image_shape.is_fully_defined():
        raise ValueError("'image' must be fully defined.")
    if any(x == 0 for x in image_shape):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                         image_shape)
    if not image_shape.is_fully_defined():
        return [check_ops.assert_positive(array_ops.shape(image),
                                          ["all dims of 'image.shape' "
                                           "must be > 0."])]
    else:
        return []

def fix_image_flip_shape(image, result):
    image_shape = image.get_shape()
    if image_shape == tensor_shape.unknown_shape():
        result.set_shape([None, None, None])
    else:
        result.set_shape(image_shape)
    return result


# =========================================================================== #
# Image resize
# =========================================================================== #
def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    with tf.name_scope('resize_image'):
        _, _, channels = _ImageDimensions(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size, method, align_corners)
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image

# =========================================================================== #
# Image random flip
# =========================================================================== #
def random_flip_left_right(image, bboxes, seed=None):
    """Random flip left-right of an image and its bounding boxes.
    """
    def flip_bboxes(bboxes):
        """Flip bounding boxes coordinates.
        """
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                           bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes

    # Random flip. Tensorflow implementation.
    with tf.name_scope('random_flip_left_right'):
        image = ops.convert_to_tensor(image, name='image')
        _Check3DImage(image, require_static=False)
        mirror_cond = _creat_cond()
        # Flip image.
        result = control_flow_ops.cond(mirror_cond,
                                       lambda: array_ops.reverse_v2(image, [1]),
                                       lambda: image)
        # Flip bboxes.
        bboxes = control_flow_ops.cond(mirror_cond,
                                       lambda: flip_bboxes(bboxes),
                                       lambda: bboxes)
        return fix_image_flip_shape(image, result), bboxes


# ===============================================
# distort color
# ===============================================
def distort_color(image, scope=None):
    cond_list = _creat_cond(length = 4)
    with tf.name_scope(scope, 'distort_color', [image]):
        image = control_flow_ops.cond(cond_list[0],
                                       lambda: tf.image.random_brightness(image, max_delta=32. / 255.),
                                       lambda: image)
        image = control_flow_ops.cond(cond_list[1],
                                       lambda: tf.image.random_saturation(image, lower=0.5, upper=1.5),
                                       lambda: image)
        image = control_flow_ops.cond(cond_list[2],
                                       lambda: tf.image.random_hue(image, max_delta=0.2),
                                       lambda: image)
        image = control_flow_ops.cond(cond_list[3],
                                       lambda: tf.image.random_contrast(image, lower=0.5, upper=1.5),
                                       lambda: image)
        return tf.clip_by_value(image, 0.0, 1.0)

# =============================================================
# distorted bounding box
# ============================================================= 
def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.3, 2.0),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                clip_bboxes=True,
                                scope=None):
    cond = _creat_cond()
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        cropped_image.set_shape([None, None, 3])
        bboxes = bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = bboxes_filter_overlap(labels, bboxes,
                                                   threshold=BBOX_CROP_OVERLAP)
        return cropped_image, labels, bboxes

def bboxes_resize(bbox_ref, bboxes, name=None):
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes

def bboxes_filter_overlap(labels, bboxes, threshold=0.1,
                          scope=None):
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),
                                     bboxes)
        mask = scores > threshold
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes

def bboxes_intersection(bbox_ref, bboxes, name=None):
    with tf.name_scope(name, 'bboxes_intersection'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = safe_divide(inter_vol, bboxes_vol, 'intersection')
        return scores

def safe_divide(numerator, denominator, name):
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)

# =========================================
# whiten input image
# =========================================
def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image

# =======================================
# make sure bboxes value will not exceed
# =======================================
def clip_bboxes(bboxes):
    bboxes = tf.minimum(bboxes, 1.0)
    bboxes = tf.maximum(bboxes, 0.0)
    return bboxes
