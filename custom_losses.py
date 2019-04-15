# -*- coding: utf-8 -*-
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

def dice_soft(y_true, y_pred, loss_type='sorensen', axis=[1,2,3], smooth=1., from_logits=False):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_hard(y_true, y_pred, threshold=0.5, axis=[1,2,3], smooth=1):
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
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    y_true = tf.cast(y_true > threshold, dtype=tf.float32)
    inse = tf.reduce_sum(tf.multiply(y_pred, y_true), axis=axis)
    l = tf.reduce_sum(y_pred, axis=axis)
    r = tf.reduce_sum(y_true, axis=axis)
    ## old axis=[0,1,2,3]
    # hard_dice = 2 * (inse) / (l + r)
    # epsilon = 1e-5
    # hard_dice = tf.clip_by_value(hard_dice, 0, 1.0-epsilon)
    ## new haodong
    hard_dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    hard_dice = tf.reduce_mean(hard_dice)
    return hard_dice
    """y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)"""


def dice_loss(y_true, y_pred, from_logits=False):
    return 1-dice_soft(y_true, y_pred, from_logits=False)

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
