#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: snippets.py.py
@datetime:2023/1/31 11:50 上午
功能：
"""

from typing import List
import tensorflow as tf
from tensorflow import keras


def shape_list(tensor: tf.Tensor) -> List[int]:
    """
    Deal with dynamic shape in tensorflow cleanly.
    代码来自huggingface transformers

    Args:
        tensor (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def gelu(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
    # tf2.7的实现
    # return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))


custom_objects = {
    "gelu": gelu,
}
keras.utils.get_custom_objects().update(custom_objects)
