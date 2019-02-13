import math
from collections import namedtuple
import pdb
import numpy as np
import tensorflow as tf
from nets import custom_layers
import tensorflow.contrib.slim as slim

# =========================================================================== #
# Text Boxes parameter default
# =========================================================================== #
SSDParameters = namedtuple('SSDParameters',
                                        ['img_shape',
                                         'feat_layers',
                                         'feat_shapes',
                                         'anchor_ratios',
                                         'normalizations',
                                         'prior_scaling',
                                         'anchor_sizes',
                                         'anchor_steps',
                                         'scales',
                                         'match_threshold'
                                         ])

default_params = SSDParameters(
    img_shape=(300, 300),
    feat_layers=['conv4', 'conv7', 'conv8', 'conv9', 'conv10', 'conv11'],
    feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
    anchor_ratios=[1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    normalizations=[20, -1, -1, -1, -1, -1],
    prior_scaling=[0.1, 0.1, 0.2, 0.2],
    anchor_sizes= [21., 45., 99., 153., 207., 261.], 
    anchor_steps=[8., 16., 32., 64., 100., 300.],
    scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.90],
    match_threshold = 0.5
    )
# =========================================================================== #
# Text Boxes net construct
# =========================================================================== #

def text_net(inputs,
            feat_layers=default_params.feat_layers,
            normalizations=default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            scope='vgg_16'): # checked
    feature_layers = {}
    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=None):
        # Original VGG-16 blocks.
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        feature_layers['conv4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        feature_layers['conv7'] = net

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'conv8'
        with tf.variable_scope(end_point):
	        net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
	        net = custom_layers.pad2d(net, pad=(1, 1))
	        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        feature_layers['conv8'] = net

        end_point = 'conv9'
        with tf.variable_scope(end_point):
	        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
	        net = custom_layers.pad2d(net, pad=(1, 1))
	        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        feature_layers['conv9'] = net

        end_point = 'conv10'
        with tf.variable_scope(end_point):
	        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
	        net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        feature_layers['conv10'] = net

        end_point = 'conv11'
        with tf.variable_scope(end_point):
	        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
	        net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        feature_layers['conv11'] = net
        localisations, logits = text_detect_net(feature_layers, feat_layers)
        return localisations, logits, feature_layers

def text_detect_net(feature_layers, feature_layer_name):
    logits = []
    localisations = []   
    for layer in feature_layer_name:
        with tf.variable_scope(layer + '_box'):
            p, l = text_detect_layer(layer, feature_layers[layer])
        logits.append(p)
        localisations.append(l)   
    return localisations, logits

def text_detect_layer(layer_name, feature_layer):
    if layer_name == 'conv11':
        feature_pred = slim.conv2d(feature_layer, 6*(4+2)*2, [1, 1], activation_fn=None, padding = 'VALID', scope='conv_feat')
    else:
        feature_pred = slim.conv2d(feature_layer, 6*(4+2)*2, [1, 5], activation_fn=None, padding = 'SAME', scope='conv_feat')
    loc_pred_flat, class_pred_flat = tf.split(feature_pred, [6*4*2, 6*2*2], -1)
    loc_pred = tf.reshape(loc_pred_flat, loc_pred_flat.get_shape().as_list()[:-1] + [2, 6, 4])
    class_pred = tf.reshape(class_pred_flat, class_pred_flat.get_shape().as_list()[:-1] + [2 ,6, 2])
    return class_pred, loc_pred
# =========================================================================== #
# Text Boxes anchors
# =========================================================================== #
def textbox_anchor_one_layer(img_shape,
					         feat_size,
					         ratios,
					         scale,
					         sizes,
                             step,
					         offset = 0.5,
					         dtype=np.float32):
    y, x = np.mgrid[0:feat_size[0], 0:feat_size[1]]
    y_offset = y + offset
    y = (y.astype(dtype) + offset) / feat_size[0] # seems don't need steps
    y_offset = (y_offset.astype(dtype) + offset) / feat_size[0]
    x = (x.astype(dtype) + offset) / feat_size[1]
    x_out = np.stack((x, x), -1)
    y_out = np.stack((y, y_offset), -1)
    y_out = np.expand_dims(y_out, axis=-1)
    x_out = np.expand_dims(x_out, axis=-1)
    h = np.zeros((len(ratios), ), dtype=dtype)
    w = np.zeros((len(ratios), ), dtype=dtype)
    for i, r in enumerate(ratios):
        h[i] = sizes / img_shape[0] / math.sqrt(r)
        w[i] = sizes / img_shape[1] * math.sqrt(r)
    return y_out, x_out, h, w

## produce anchor for all layers
def anchor_boxes(img_shape,
                layers_shape,
                anchor_ratios,
                scales,
                anchor_sizes,
                anchor_steps,
                offset=0.5,
                dtype=np.float32):
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = textbox_anchor_one_layer(img_shape, s,
                                                 anchor_ratios,
                                                 scales[i],
                                                 anchor_sizes[i],
                                                 anchor_steps[i],
                                                 offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors

# =========================================================================== #
# ssd arg scope.
# =========================================================================== #

def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc

# =========================================================================== #
# Text Boxes encoder
# =========================================================================== #

def tf_text_bboxes_encode_layer(bboxes,
                               anchors_layer, num, box_detect, idx,
                               matching_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    # Anchors coordinates and volume.
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2. 
    vol_anchors = (xmax - xmin) * (ymax - ymin)
    
    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], yref.shape[2], href.size)
    feat_scores = tf.zeros(shape, dtype=dtype)
    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """
        Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard
    
    def condition(i, feat_scores,box_detect,idx,
							    feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        r = tf.less(i, num)
        return r

    def body(i, feat_scores,box_detect,idx,feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        mask = tf.greater(jaccard, feat_scores)
        mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        fmask = tf.cast(mask, dtype)
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        return [i+1, feat_scores,box_detect,idx,
			            feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.

    i = 0
    [i,feat_scores,box_detect,idx,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
									        [i, feat_scores,box_detect,idx,
									        feat_ymin, feat_xmin,
									        feat_ymax, feat_xmax])
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_localizations, feat_scores, box_detect



def tf_text_bboxes_encode(bboxes,
						 anchors, num,
						 match_threshold=0.5,
						 prior_scaling=[0.1, 0.1, 0.2, 0.2],
						 dtype=tf.float32,
						 scope='text_bboxes_encode'):
	with tf.name_scope('text_bboxes_encode'):
			target_localizations = []
			target_scores = []
			box_detect = tf.zeros((num,),dtype=tf.int32)
			for i, anchors_layer in enumerate(anchors):
					with tf.name_scope('bboxes_encode_block_%i' % i):
							t_loc, t_scores,box_detect = \
									tf_text_bboxes_encode_layer(bboxes,anchors_layer, num, box_detect,i,
																match_threshold,
																prior_scaling, dtype)
							target_localizations.append(t_loc)
							target_scores.append(t_scores)
			return target_localizations, target_scores

# =========================================================================== #
# Text loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
			   glocalisations, gscores,
			   match_threshold,
			   use_hard_neg=False,
			   negative_ratio=3,
			   alpha=1.,
			   label_smoothing=0.,
			   scope=None):
    with tf.name_scope(scope, 'txt_losses'):
        num_classes = 2
        l_cross_pos = []
        l_cross_neg = []
        l_loc = []

        # Flatten out all vectors!
        flogits = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype
        num = tf.ones_like(gscores)
        n = tf.reduce_sum(num)
        # Compute positive matching mask...
        pmask = gscores > match_threshold                   #positive mask
        nmask = gscores <= match_threshold                  #negative mask
        ipmask = tf.cast(pmask, tf.int32)                   #int positive mask
        fpmask = tf.cast(pmask, dtype)                      #float positive mask
        n_pos = tf.reduce_sum(fpmask)                       #calculate all number

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = ipmask)   
                
        l_cross_pos = tf.losses.compute_weighted_loss(loss, fpmask)
        l_cross_pos = tf.identity(l_cross_pos, name = 'l_cross_pos')        

        # Hard negative mining
        fnmask = tf.cast(nmask, dtype)
        loss_neg = tf.where(pmask, loss, tf.zeros_like(fnmask))
        loss_neg_flat = tf.reshape(loss_neg, [-1])
        n_neg = tf.minimum(tf.cast(3*n_pos, tf.int32), tf.cast(n,tf.int32))+1
        val, _ = tf.nn.top_k(loss_neg_flat, k=n_neg)
        minval = val[-1]
        nmask = tf.logical_and(nmask, loss_neg_flat >= minval)
        fnmask = tf.cast(nmask, tf.float32)

        l_cross_neg = tf.losses.compute_weighted_loss(loss, fnmask)
        l_cross_neg = tf.identity(l_cross_neg, name = 'l_cross_neg')

        weights = tf.expand_dims(fpmask, axis=-1)
        l_loc = custom_layers.abs_smooth(localisations - glocalisations)

        l_loc = tf.losses.compute_weighted_loss(l_loc, weights)
        l_loc = tf.identity(l_loc, name = 'l_loc')
        total_loss = tf.add_n([l_loc, l_cross_pos, l_cross_neg], 'total_loss')

        with tf.name_scope('total'):
            # Add to EXTRA LOSSES TF.collection      
            tf.add_to_collection('EXTRA_LOSSES', l_cross_pos)
            tf.add_to_collection('EXTRA_LOSSES', l_cross_neg)
            tf.add_to_collection('EXTRA_LOSSES', l_loc)
            tf.add_to_collection('EXTRA_LOSSES', total_loss)
    return total_loss

