#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: layers.py
@datetime:2023/1/31 11:41 上午
功能：TF2.0自定义层
"""

from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from .snippets import shape_list


class PositionLayer(keras.layers.Layer):
    """位置id:从0到length-1
    用法：
    positions = BiasLayer(offset=2)(tf.constant([[2,3,1],[1,0,5]]))
    positions
    输出：
    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[2, 3, 4],
           [2, 3, 4]], dtype=int32)>
    """

    def __init__(self, offset=0, **kwargs):
        super(PositionLayer, self).__init__(**kwargs)
        self.offset = offset

    def call(self, input_ids):
        """

        Args:
            inputs: 主要是能提供batch_size与seq_length，一般是input_ids，shape = (batch_size,seq_length)

        Returns:

        """
        batch_size = shape_list(input_ids)[0]
        seq_length = shape_list(input_ids)[-1]
        positions = tf.tile(
            tf.expand_dims(tf.range(start=0, limit=seq_length, delta=1), 0),
            (batch_size, 1),
        )
        positions = positions + self.offset  # bart的位置id要+2
        return positions

    def get_config(self):
        base_config = super(PositionLayer, self).get_config()
        layer_config = {"offset": self.offset}
        return dict(list(base_config.items()) + list(layer_config.items()))


class SharedEmbeddingLayer(keras.layers.Layer):
    """共享向量层，用于
    1. encoder和decoder获取input token ids的token向量，返回 shape = (batch_size,seq_len,hidden_size)
    2. 最后一层transformers输出层与共享向量层做线性变换，最后计算每个位置最终对应的token。
        一般用于mask language model。返回 shape = (batch_size,seq_len,hidden_size)
    用法：
    layer=SharedEmbeddingLayer(10,8)
    # 向量层
    inputs=tf.constant([[0,3,1],[2,6,4]])
    out1=layer(inputs,mode='embedding')
    print(out1) # shape = (2,3,8)
    # 线性层
    emb=tf.random.uniform((3,2,8))
    out2=layer(emb,mode='linear')
    print(out2) # shape = (3,2,10)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super(SharedEmbeddingLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initializer = initializer

        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=initializer,
            trainable=True,
            name="shared.embedding",
        )

    def call(self, inputs):
        """
        Args:
            inputs: 有两个类型
                - 1.input_ids：类似于BERT的token id输入，shape = (batch_size,seq_len)，整型
                    此时类似于keras.layers.Embedding
                - 2.transformers最后一层的输出，shape = (batch_size,seq_len,hidden_size)，浮点型，
                    此时类似于keras.layers.Dense
        Returns: embedding或线性变换的tensor
        """
        if len(shape_list(inputs)) == 2:
            # embedding向量层 --> shape = (batch_size,seq_len)
            out = tf.gather(self.embeddings, inputs)
        else:
            # linear线性变量 --> shape = (batch_size,seq_len,hidden_size)
            out = tf.matmul(inputs, self.embeddings, transpose_b=True)
        return out

    def get_config(self):
        base_config = super(SharedEmbeddingLayer, self).get_config()
        layer_config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "initializer": self.initializer,
        }
        return dict(list(base_config.items()) + list(layer_config.items()))


class AttentionMaskLayer(keras.layers.Layer):
    """
    创建Attention mask，值为0，表示不可见 ,1，表示可见
    shape = (batch_size,1,seq_len,seq_len)
    用于下一层将padding token忽略，只计算非pad token之间的注意力

    用法：
    attention_mask=tf.constant([[1,1,0],[1,0,0]])
    hidden_states=tf.constant([[0.3,-0.4],[0.1,0.2]])
    attention_mask=AttentionMaskLayer()(attention_mask,hidden_states)
    输出:
    <tf.Tensor: shape=(2, 1, 2, 3), dtype=float32, numpy=
    array([[[[1., 1., 0.],
         [1., 1., 0.]]],


       [[[1., 0., 0.],
         [1., 0., 0.]]]], dtype=float32)>
    """

    def __init__(self, **kwargs):
        super(AttentionMaskLayer, self).__init__(**kwargs)

    def call(self, attention_mask, hidden_states=None):
        """
        Args:
            attention_mask: 由0、1组成的矩阵，shape = (batch_size, src_len)，其中
                - 0：表示pad的token，在softmax时可见
                - 1：表示非pad的token，在softmax时可见
            hidden_states: query tensor，提供目标长度，默认为attention_mask

        Returns: 输出[bsz, 1, tgt_seq_len, src_seq_len]的矩阵
        """
        src_len = shape_list(attention_mask)[1]
        tgt_len = src_len
        if hidden_states is not None:
            tgt_len = shape_list(hidden_states)[1]
        one_cst = tf.constant(1.0)
        mask = tf.cast(attention_mask, dtype=one_cst.dtype)
        attention_mask = tf.tile(mask[:, None, None, :], (1, 1, tgt_len, 1))

        return attention_mask

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法。
    def get_config(self):
        layer_config = super(AttentionMaskLayer, self).get_config()
        return dict(layer_config)


class UniMaskLayer(keras.layers.Layer):
    """
    定义UniLM的Attention Mask（Seq2Seq模型用）
    其中source和target的分区，由segment_ids来表示。
    UniLM: https://arxiv.org/abs/1905.03197
    其中1表示可见，0表示不可见
    用法：
    segment_ids = tf.constant([[0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 1, 1]])
    uni_mask=UniMaskLayer()(segment_ids)
    uni_mask
    输出：
    <tf.Tensor: shape=(3, 4, 4), dtype=float32, numpy=
    array([[[1., 1., 1., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]],

       [[1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]],

       [[1., 1., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]]], dtype=float32)>
    """

    def __init__(self, **kwargs):
        super(UniMaskLayer, self).__init__(**kwargs)

    def call(self, segment_ids):
        """

        Args:
            segment_ids: shape = (batch_size,seq_len),
                用于区分segment，0是第一个文本token，1是第二个文本的token,
                例如[[0,0,0,1],[0,0,1,1]]

        Returns: 返回UniMask

        """
        idxs = keras.backend.cumsum(segment_ids, axis=1)
        mask = idxs[:, None, :] <= idxs[:, :, None]
        mask = K.cast(mask, K.floatx())
        return mask

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法。
    def get_config(self):
        layer_config = super(UniMaskLayer, self).get_config()
        return dict(layer_config)


class CausalMaskLayer(keras.layers.Layer):
    """
    创建causal mask（下三角mask）

    用法：
    inputs=tf.constant([[2,3,1],[5,5,2]])
    causal_mask=CausalMaskLayer()(inputs)
    输出:
    <tf.Tensor: shape=(2, 1, 3, 3), dtype=float32, numpy=
    array([[[[1., 0., 0.],
         [1., 1., 0.],
         [1., 1., 1.]]],


       [[[1., 0., 0.],
         [1., 1., 0.],
         [1., 1., 1.]]]], dtype=float32)>
    """

    def __init__(self, **kwargs):
        super(CausalMaskLayer, self).__init__(**kwargs)

    def call(self, inputs):
        """
        创建causal mask（下三角mask），shape = (batch_size,1,seq_len,seq_len)
        Args:
            inputs: 一个输入tensor, shape = (batch_size,...)

        Returns: causal mask（下三角mask），shape = (batch_size,1,seq_len,seq_len)
        """
        batch_size = shape_list(inputs)[0]
        seq_len = shape_list(inputs)[1]  # 序列长度
        mask = tf.ones((seq_len, seq_len))
        mask_cond = tf.range(seq_len)
        mask = 1.0 - tf.where(
            mask_cond < tf.reshape(mask_cond + 1, (seq_len, 1)), 0.0, mask
        )  # shape = (seq_len,seq_len)
        causal_mask = tf.tile(mask[None, None, :, :], (batch_size, 1, 1, 1))
        return causal_mask

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法。
    def get_config(self):
        layer_config = super(CausalMaskLayer, self).get_config()
        return dict(layer_config)


class MultiHeadAttentionV2(keras.layers.Layer):
    """Multi-headed attention from "Attention Is All You Need"""

    def __init__(
        self,
        embed_dim: int,  # 向量长度，例如bert常用是768
        num_heads: int,  # 多头的数量
        dropout_rate: float = 0.0,
        dense_bias: bool = True,  # 全连接层是否使用bias
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super(MultiHeadAttentionV2, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.dense_bias = dense_bias
        self.kernel_initializer = kernel_initializer

        self.head_dim = embed_dim // num_heads  # 每个头的向量长度
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5  # Q*K点积分除以sqrt(每个头的向量长度)，用于计算attention值

        self.dropout = keras.layers.Dropout(self.dropout_rate, name="dropout")
        # 定义q、k、v、out等dense层
        self.k_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=self.dense_bias,
            kernel_initializer=self.kernel_initializer,
            name="k_proj",
        )
        self.q_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=self.dense_bias,
            kernel_initializer=self.kernel_initializer,
            name="q_proj",
        )
        self.v_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=self.dense_bias,
            kernel_initializer=self.kernel_initializer,
            name="v_proj",
        )
        self.out_proj = keras.layers.Dense(
            self.embed_dim,
            use_bias=self.dense_bias,
            kernel_initializer=self.kernel_initializer,
            name="out_proj",
        )

    def _reshape_into_multi_head(
        self, tensor: tf.Tensor, seq_len: int, batch_size: int
    ):
        return tf.transpose(
            tf.reshape(tensor, (batch_size, seq_len, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )

    def call(
        self,
        hidden_states: tf.Tensor,
        key_states: tf.Tensor = None,
        value_states: tf.Tensor = None,
        attention_mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        attention实现
            - 当self-attention时（常用于encdoer层），hidden_states=key_states=value_states
            - 当cross-attention时（常用于decdoer层），key_states=value_states
            - 当mask-self-attention时（常用于decdoer层），需要传入causal_mask
            其中：
            q_embed_dim=num_heads*head_dim
            k_embed_dim、v_embed_dim可被整除num_heads*head_dim，
            一般情况k_embed_dim=v_embed_dim=q_embed_dim

        Args:
            hidden_states: shape = (batch_size,q_seq_len,q_embed_dim)
            key_states: shape = (batch_size,k_seq_len,q_embed_dim), k_seq_len=v_seq_len
                当k_embed_dim!=q_embed_dim时，会被reshape成(batch_size,k_seq_len2,q_embed_dim)
            value_states: shape = (batch_size,k_seq_len,q_embed_dim), k_seq_len=v_seq_len
                当v_embed_dim!=q_embed_dim时，会被reshape成(batch_size,v_seq_len2,v_embed_dim)
            attention_mask: shape = (batch_size,1,q_seq_len,k_seq_len)，0表示token不可见，1表示可见
                可以是：
                - 普通的attention_mask
                - casual mask(下三角mask): decoder常用的mask，遮住下一个词
                - unilm mask：保留上一句的所有词，遮住下一句话的下一个词
                - None: 无额外的attention_mask

        Returns: attention向量输出attention_output与注意力分数attention_weights
            attention_output: shape = (batch_size,q_seq_len,q_embed_dim)
            attention_weights: shape = (batch_size, num_heads, q_seq_len, k_seq_len)

        """

        batch_size, q_seq_len, embed_dim = shape_list(hidden_states)
        # 1. 计算Q、K、V
        # Q
        query_states = self.q_proj(hidden_states) * self.scaling
        # K
        key_states = self._reshape_into_multi_head(
            self.k_proj(key_states), -1, batch_size
        )
        # V
        value_states = self._reshape_into_multi_head(
            self.v_proj(value_states), -1, batch_size
        )

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        # Q shape = (batch_size*num_heads,q_seq_len,head_dim),
        query_states = tf.reshape(
            self._reshape_into_multi_head(query_states, q_seq_len, batch_size),
            proj_shape,
        )
        # K shape = (batch_size*num_heads,k_seq_len,head_dim), k_seq_len=v_seq_len
        key_states = tf.reshape(key_states, proj_shape)
        # V shape = (batch_size*num_heads,k_seq_len,head_dim)
        value_states = tf.reshape(value_states, proj_shape)

        k_seq_len = shape_list(key_states)[1]  # k_seq_len

        # 2. 计算attention
        # shape = (batch_size*num_heads,q_seq_len,k_seq_len)
        attention_weights = tf.matmul(query_states, key_states, transpose_b=True)

        # 确保当前的token不能看到未来的token
        if attention_mask is not None:
            attention_mask = self._mask_with_negative_inf(
                attention_mask
            )  # 变成无穷大，softmax时成为0
            attention_mask = tf.cast(attention_mask, dtype=attention_weights.dtype)
            attention_weights = (
                tf.reshape(
                    attention_weights,
                    (batch_size, self.num_heads, q_seq_len, k_seq_len),
                )
                + attention_mask
            )

        attention_weights = tf.reshape(
            attention_weights, (batch_size * self.num_heads, q_seq_len, k_seq_len)
        )  # shape = (batch_size*num_heads,q_seq_len,k_seq_len)
        attention_weights = tf.nn.softmax(attention_weights)  # , axis=-1)

        attention_probs = self.dropout(attention_weights)
        # shape = (batch_size*num_heads,q_seq_len,head_dim)
        attention_output = tf.matmul(attention_probs, value_states)

        # 3. out_proj全连接
        attention_output = tf.transpose(
            tf.reshape(
                attention_output, (batch_size, self.num_heads, q_seq_len, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        attention_output = tf.reshape(
            attention_output, (batch_size, q_seq_len, embed_dim)
        )  # shape = (batch_size, seq_len, embed_dim)

        attention_output = self.out_proj(attention_output)
        attention_weights = tf.reshape(
            attention_weights, (batch_size, self.num_heads, q_seq_len, k_seq_len)
        )  # shape = (batch_size, num_heads, q_seq_len, k_seq_len)

        return attention_output, attention_weights

    @staticmethod
    def _mask_with_negative_inf(mask: tf.Tensor):
        """
        mask矩阵变成负无穷值
        Args:
            mask: tf.Tensor, shape = (seq_len,seq_len)
                0: 表示token不可见
                1: 表示token可见
                例如
                [
                    [1, 0, 0],
                    [1, 1, 0],
                    [1, 1, 1],
                ]

        Returns:
            mask 0: 转化成无限小的值，在softmax中变成0.0，从而实现不可见
            mask 1: 转化成0，与attention相加时，不影响可见。
            shape = (batch_size,1,q_seq_len,k_seq_len)，
            例如
                [
                    [0.0, -1e8, -1e8],
                    [0.0, 0.0,  -1e8],
                    [0.0, 0.0,  0.0],
                ]

        """
        mask = (1.0 - mask) * (-1e9)
        return mask

    # 如果要让自定义的Layer通过Functional API 组合成模型时可以序列化，需要自定义get_config方法。
    def get_config(self):
        layer_config = super(MultiHeadAttentionV2, self).get_config()
        layer_config["embed_dim"] = self.embed_dim
        layer_config["num_heads"] = self.num_heads
        layer_config["dropout_rate"] = self.dropout_rate
        layer_config["dense_bias"] = self.dense_bias
        layer_config["kernel_initializer"] = self.kernel_initializer
        return dict(layer_config)


class BiasLayer(keras.layers.Layer):
    """添加bias
    用法：
    bias = BiasLayer()
    t = tf.constant([2, 3])
    print(bias(t))
    t = tf.constant([2.1, 3.2])
    print(bias(t))
    """

    def __init__(self, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BiasLayer, self).build(input_shape)

        self.bias = self.add_weight(
            name="bias", shape=(input_shape[-1],), initializer="zeros", dtype=tf.float32
        )

    def call(self, inputs):
        return tf.cast(self.bias, dtype=inputs.dtype) + inputs


# 在keras的custom_objects添加自定义层，这样keras.models.load_model会自动加载这些层。
custom_objects = {
    "PositionLayer": PositionLayer,
    "SharedEmbeddingLayer": SharedEmbeddingLayer,
    "AttentionMaskLayer": AttentionMaskLayer,
    "UniMaskLayer": UniMaskLayer,
    "CausalMaskLayer": CausalMaskLayer,
    "MultiHeadAttentionV2": MultiHeadAttentionV2,  # 名称+V2，是为了避免与keras自有MultiHeadAttention冲突
    "BiasLayer": BiasLayer,
}
keras.utils.get_custom_objects().update(custom_objects)
