## create script that download datasets and transform into tf-record
import numpy as np
import scipy.io as sio
import os
import tensorflow as tf
import re
from dataset_utils import int64_feature, float_feature, bytes_feature ,ImageCoder
import cv2
from PIL import Image
import pdb
import xml.etree.ElementTree as ET
data_path = './svt_data/'
tree_path = '/home/qichen/MLproject/TextBoxes/datasets/svt_data/train.xml'
os.chdir(data_path)
tf_filename = 'svt_train.tfrecord'

def parseTree(path):
    dataset = []
    tree = ET.parse(path)
    root = tree.getroot()
    for image in root.findall('image'):
        name = image.find('imageName').text
        rectangles = []
        taggedRectangles = image.find('taggedRectangles')
        resolution = image.find('Resolution')
        shape = [int(resolution.get('y')), int(resolution.get('x')), int(3)]
        for rectangle in taggedRectangles.findall('taggedRectangle'): 
            ymin = float(rectangle.get('y'))
            xmin = float(rectangle.get('x'))
            ymax = float(rectangle.get('height')) + ymin
            xmax = float(rectangle.get('width')) + xmin
            ymin = np.maximum(ymin*1.0/shape[0], 0.0)
            xmin = np.maximum(xmin*1.0/shape[1], 0.0)
            ymax = np.minimum(ymax*1.0/shape[0], 1.0)
            xmax = np.minimum(xmax*1.0/shape[1], 1.0)
            rectangles.append([ymin, xmin, ymax, xmax])
        dataset.append((name, rectangles, shape))
    return dataset

def _convert_to_example(image_data, shape, bbox, label,imname):
	nbbox = np.array(bbox)
	ymin = list(nbbox[:, 0])
	xmin = list(nbbox[:, 1])
	ymax = list(nbbox[:, 2])
	xmax = list(nbbox[:, 3])

	print('shape: {}, height:{}, width:{}'.format(shape,shape[0],shape[1]))
	example = tf.train.Example(features=tf.train.Features(feature={
			'image/height': int64_feature(shape[0]),
			'image/width': int64_feature(shape[1]),
			'image/channels': int64_feature(shape[2]),
			'image/shape': int64_feature(shape),
			'image/object/bbox/xmin': float_feature(xmin),
			'image/object/bbox/xmax': float_feature(xmax),
			'image/object/bbox/ymin': float_feature(ymin),
			'image/object/bbox/ymax': float_feature(ymax),
			'image/object/bbox/label': int64_feature(label),
			'image/format': bytes_feature('jpeg'),
			'image/encoded': bytes_feature(image_data),
			'image/name': bytes_feature(imname),
			}))
	return example

def run():
    dataset = parseTree(tree_path)
    tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
    coder = ImageCoder()
    for data in dataset:
        image_name = data[0]
        bbox       = data[1]
        shape      = data[2]
        label      = [1 for _ in range(len(bbox))] 
        image_path = '/home/qichen/MLproject/TextBoxes/datasets/svt_data/'+image_name
        image_data = tf.gfile.GFile(image_path, 'r').read()
        example    = _convert_to_example(image_data, shape, bbox, label, image_name)
        tfrecord_writer.write(example.SerializeToString())
if __name__ == '__main__':
	run()





