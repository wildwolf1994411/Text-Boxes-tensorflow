import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import tensorflow.contrib.slim as slim
import pdb

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import sys
# sys.path.append('./')

from nets import textbox, np_methods

from processing import preprocessing

from processing import visualization

gpu_options = tf.GPUOptions(allow_growth=False, per_process_gpu_memory_fraction=0.5)

config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

isess = tf.Session(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

# Evaluation pre-processing: resize to SSD net shape.
image_pre, bbox_img = preprocessing.preprocess_for_test(img_input, net_shape)
image_4d = tf.expand_dims(image_pre, 0)

# Define the txt_box model.
text_box = textbox.TextboxNet()
with slim.arg_scope(text_box.arg_scope(data_format=data_format)):
    localisations, logits, end_points = text_box.net(image_4d, is_training=False)

# Restore SSD model.
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, './tfmodel/svt_model.ckpt-85000')
txt_anchors = text_box.anchors(net_shape)


def process_image(img, select_threshold=0.7 , nms_threshold=0.45, net_shape=(300, 300)):
    # Run txt network.
    _, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, logits, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, txt_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=2)
    
    rbboxes = np_methods.bboxes_clip(rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=100)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


# Test on some demo image and visualize output.
path = './demo/'
image_names = sorted(os.listdir(path))

img = mpimg.imread(path + image_names[0])
rclasses, rscores, rbboxes = process_image(img)

visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

