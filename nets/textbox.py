import math
from collections import namedtuple
import pdb
import numpy as np
import tensorflow as tf

from nets import custom_layers
from nets import textbox_function as tb_fun
import tensorflow.contrib.slim as slim

# =========================================================================== #
# Text class definition.
# =========================================================================== #
class TextboxNet(object):
    default_params = tb_fun.default_params
    def __init__(self, params=None):
        self.params = self.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            dropout_keep_prob=0.5,
            scope='vgg_16'):
        """
        Text network definition.
        """
        r = tb_fun.text_net(inputs,
                    feat_layers=self.params.feat_layers,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    scope=scope)
        return r

    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return tb_fun.anchor_boxes(img_shape,
                                  self.params.feat_shapes,
                                  self.params.anchor_ratios,
                                  self.params.scales,
                                  self.params.anchor_sizes,
                                  self.params.anchor_steps,
                                  offset = 0.5,
                                  dtype  = np.float32)

    def bboxes_encode(self, bboxes, anchors, num,
				      scope='text_bboxes_encode'):
	    """Encode labels and bounding boxes.
	    """
	    return tb_fun.tf_text_bboxes_encode(
					    bboxes, anchors, num,
					    match_threshold=self.params.match_threshold,
					    prior_scaling=self.params.prior_scaling,
					    scope=scope)

    def losses(self, logits, localisations,
               glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='txt_losses'):
        """Define the SSD network losses.
        """
        return tb_fun.ssd_losses(logits, localisations,
                          glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)

    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return tb_fun.ssd_arg_scope(weight_decay, data_format=data_format)
