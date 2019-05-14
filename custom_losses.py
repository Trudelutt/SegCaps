# -*- coding: utf-8 -*-
#from __future__ import print_function
'''
Capsules for Object Segmentation (SegCaps)
Original Paper: https://arxiv.org/abs/1804.04241
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file contains the definitions of custom loss functions not present in the default Keras.
'''

import tensorflow as tf
from keras import backend as K
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from os.path import join



def fn1(y_true, y_pred, count):
    np.savez_compressed(join( "debug", "np",  "pred" + str(count) + '.npz'), img=y_pred, mask=y_true)
    count += 1
    return count


#def fn2(a, b):


def dice_soft(y_true, y_pred, loss_type='sorensen', axis=(1,2,3), smooth=1e-5, from_logits=True):
    #tf.enable_eager_execution()
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.
    Parameters
    -----------
    y_pred : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    y_true : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    loss_type : string
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
        If both y_pred and y_true are empty, it makes sure dice is 1.
        If either y_pred or y_true are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``,
        then if smooth is very small, dice close to 0 (even the image values lower than the threshold),
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """
    #shape_y_pred = y_pred.get_shape().as_list()
    #shape_y_pred = K.print_tensor(shape_y_pred, message='\n shape_y_pred = ')

    #axis = tuple(range(1, len(shape_y_pred.as_list())-1))
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    #print("y_pred ", tf.reduce_max(y_pred).eval())
    #y_true = tf.cast(y_true, dtype=tf.float32)
    #smooth = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
    """if not from_logits:
        # transform back to logits
        _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))"""
    #min_pred = tf.reduce_min(y_pred)
    #axis = K.print_tensor(axis, message='\n axis = ')
    #max_pred = tf.reduce_max(y_pred)
    #max_pred = K.print_tensor(max_pred, message='\n max_pred = ')
    #y_pred = tf.clip_by_value(y_pred, min_pred,  max_pred)
    inse_mul = y_pred * y_true
    #min_inse = tf.reduce_min(inse_mul)
    #min_inse = K.print_tensor(min_inse, message='\n min_inse_mul = ')
    #max_inse = tf.reduce_max(inse_mul)
    #inse_mul = K.print_tensor(inse_mul, message='\n mul_inse_mul = ')
    #inse_mul = tf.clip_by_value(inse_mul, min_inse,  max_inse)



    inse = tf.reduce_sum(inse_mul, axis=axis, keepdims=True)
    #inse = K.print_tensor(inse, message='\n inse = ')
    #min_true = tf.reduce_min(y_true)
    #min_true = K.print_tensor(min_true, message='\n min_true = ')
    #max_true = tf.reduce_max(y_true)
    #max_true = K.print_tensor(max_true, message='\n max_true = ')
    #y_true = tf.clip_by_value(y_true, min_true,  max_true)
    #print("inse: ", inse.eval())
    if loss_type == 'jaccard':
        l = tf.reduce_sum(y_pred * y_pred, axis=axis)
        r = tf.reduce_sum(y_true * y_true, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(y_pred, axis=axis, keepdims=True)
        #l = K.print_tensor(l, message='\n l = ')
        #print("l: ", l.eval())
        r = tf.reduce_sum(y_true, axis=axis, keepdims=True)
        #r = K.print_tensor(r, message='\n r = ')
        #print("r: ", r.eval())
    else:
        raise Exception("Unknow loss_type")
    ## old axis=[0,1,2,3]
    # dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # dice = tf.clip_by_value(dice, 0, 1.0-epsilon) # if all empty, dice = 1
    ## new haodong
    #smooth = tf.constant(smooth)
    #smooth= tf.cast(smooth,dtype=tf.float32)
    dice = (2. * inse + smooth) / (l + r + smooth)
    #dice = K.print_tensor(dice, message='\n dice = ')
    #print("dice: ", dice.eval())
    ##
    dice_mean = tf.reduce_mean(dice)
    #dice_mean = K.print_tensor(dice_mean, message='\n dice_mean = ')
    #count = 0
    #np.savez_compressed(join( "debug", "np",  "pred" + str(count) + '.npz'), img=y_pred.numpy(), mask=y_true.numpy())
    #threshold = tf.constant(1)
    #threshold = tf.cast(threshold,dtype=tf.float32)
    #count = tf.cond(dice_mean > threshold, lambda: fn1(y_true, y_pred, count), lambda: count)
        #count += 1
        #np.savez_compressed(join( "debug", "np",  "pred" + str(count) + '.npz'), img=y_pred, mask=y_true)

    return dice_mean


def dice_hard(y_true, y_pred, threshold=0.5, axis=(1,2,3), smooth=1e-5):
    """Non-differentiable Sørensen–Dice coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation i.e. labels are binary.
    The coefficient between 0 to 1, 1 if totally match.
    Parameters
    -----------
    y_pred : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    y_true : tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    threshold : float
        The threshold value to be true.
    axis : list of integer
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator, see ``dice_coe``.
    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`_
    """

    y_pred = tf.cast(y_pred, dtype=tf.float32)
    #threshold = tf.constant(threshold)
    #threshold = tf.cast(threshold,dtype=tf.float32)
    y_pred = tf.cast(tf.greater(y_pred, threshold), dtype=tf.float32)
    #y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_true = tf.cast(tf.greater(y_true, threshold), dtype=tf.float32)
    inse = tf.reduce_sum(y_pred * y_true, axis=axis, keepdims=True)
    #inse = K.print_tensor(inse, message='inse = ')
    l = tf.reduce_sum(y_pred, axis=axis, keepdims=True)
    #y_true = K.print_tensor(l, message='l = ')

    r = tf.reduce_sum(y_true, axis=axis, keepdims=True)
    #r = K.print_tensor(r, message='r = ')
    #smooth = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
    ## old axis=[0,1,2,3]
    # hard_dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # hard_dice = tf.clip_by_value(hard_dice, 0, 1.0-epsilon)
    ## new haodong
    #smooth = tf.constant(smooth)
    #smooth= tf.cast(smooth,dtype=tf.float32)
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    #hard_dice = K.print_tensor(hard_dice, message='hard_dice = ')
    ##
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice


def dice_loss(y_true, y_pred, from_logits=True):
    #y_pred = tf.Print(y_pred, [ tf.reduce_max(y_pred), tf.reduce_min(y_pred), tf.rank(y_pred)], message='y_pred: ')
    #y_true = K.print_tensor(y_true, message='\n y_true = ')
    #y_pred_max =  K.print_tensor(y_pred_max , message='y_pred_max = ')
    dice_mean = dice_soft(y_true, y_pred, from_logits=True)
    #dice_mean = K.print_tensor(dice_mean, message='\n dice_mean_in_loss = ')
    #konst = tf.constant(1.)
    #konst= tf.cast(konst,dtype=tf.float32)
    dice_loss = 1 - dice_mean
    #dice_loss = K.print_tensor(dice_loss, message='\n dice_loss = ')

    return dice_loss

def weighted_binary_crossentropy_loss(pos_weight):
    # pos_weight: A coefficient to use on the positive examples.
    def weighted_binary_crossentropy(target, output, from_logits=False):
        """Binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = tf.convert_to_tensor(1e-7, output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.log(output / (1 - output))

        return tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                       logits=output,
                                                        pos_weight=pos_weight)
    return weighted_binary_crossentropy


def margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0):
    '''
    Args:
        margin: scalar, the margin after subtracting 0.5 from raw_logits.
        downweight: scalar, the factor for negative cost.
    '''

    def _margin_loss(labels, raw_logits):
        """Penalizes deviations from margin for each logit.

        Each wrong logit costs its distance to margin. For negative logits margin is
        0.1 and for positives it is 0.9. First subtract 0.5 from all logits. Now
        margin is 0.4 from each side.

        Args:
        labels: tensor, one hot encoding of ground truth.
        raw_logits: tensor, model predictions in range [0, 1]


        Returns:
        A tensor with cost for each data point of shape [batch_size].
        """
        logits = raw_logits - 0.5
        positive_cost = pos_weight * labels * tf.cast(tf.less(logits, margin),
                                       tf.float32) * tf.pow(logits - margin, 2)
        negative_cost = (1 - labels) * tf.cast(
          tf.greater(logits, -margin), tf.float32) * tf.pow(logits + margin, 2)
        return 0.5 * positive_cost + downweight * 0.5 * negative_cost

    return _margin_loss


def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax

if __name__ == "__main__":
    path_to_np_pred = "debug/np/CT_FFR_Pilot_1_Segmentation_0001_Aorta.npz"
    path_to_np = "np_files/2split_Aorta_net_input_shape(512, 512, 5)_stride1/CT_FFR_Pilot_1_Segmentation_0001_Aorta.npz"
    with np.load(path_to_np) as data:
        val_img = data['img']
        val_mask = data['mask']
    with np.load(path_to_np) as pred_data:
        pred_img = pred_data['img']
    y_true = val_mask
    print(y_true.shape)
    print(np.unique(y_true))
    y_pred = val_img
    print(np.unique(y_pred))
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(y_true.shape[0]):
            print(dice_loss(tf.convert_to_tensor(y_true[i:i+1], dtype=tf.uint8), tf.convert_to_tensor(y_pred[i:i+1], dtype=tf.float32)).eval())
    filled = np.array([
        [[1, 0, 1], [0, 0, 1], [0, 1, 0]],
        [[0, 1, 1], [1, 0, 0], [1, 0, 1]],
        [[1, 1, 0], [1, 1, 1], [0, 0, 0]]
    ])

    ax = make_ax(True)
    ax.voxels(filled, edgecolors='gray')
    plt.show()
