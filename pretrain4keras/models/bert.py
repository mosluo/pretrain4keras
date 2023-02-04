#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: bert.py
@datetime:2023/1/31 11:42 上午
@datetime:2023/1/6 10:14 上午
功能：用keras实现bert，加载google bert参数，精简代码
"""

import json
import tensorflow as tf
from tensorflow import keras
from pretrain4keras.layers import (
    PositionLayer,
    MultiHeadAttentionV2,
    AttentionMaskLayer,
    SharedEmbeddingLayer,
    BiasLayer,
)


def build_embedding_layers(
    inputs,
    shared_embedding_layer,
    initializer=None,
    config=None,
    name_prefix="",
):
    """创建embedding层: token+position+layer_norm+dropout_rate"""
    input_ids, positions, token_type_ids = inputs

    # token向量
    embed_tokens = shared_embedding_layer(input_ids)
    # 位置向量
    embed_positions = keras.layers.Embedding(
        input_dim=config["max_position_embeddings"],
        output_dim=config["hidden_size"],
        embeddings_initializer=initializer,
        name=name_prefix + ".position_embeddings",
    )(positions)
    # segment向量
    embed_token_type_ids = keras.layers.Embedding(
        input_dim=config["type_vocab_size"],
        output_dim=config["hidden_size"],
        embeddings_initializer=initializer,
        name=name_prefix + ".token_type_embeddings",
    )(token_type_ids)

    # 位置向量、token向量、segment向量相加
    x = keras.layers.Add(name=name_prefix + ".add")(
        [embed_tokens, embed_positions, embed_token_type_ids]
    )

    # ln+dropout_rate
    hidden_states = keras.layers.LayerNormalization(
        epsilon=1e-12, name=name_prefix + ".LayerNorm"
    )(x)
    hidden_states = keras.layers.Dropout(
        config["hidden_dropout_prob"], name=name_prefix + ".dropout_rate"
    )(hidden_states)
    return hidden_states


def build_transformer_encoder_layer(
    hidden_states,
    attention_mask,
    initializer=None,
    config=None,
    name_prefix="",
):
    """创建一层transformer_encoder层"""
    embed_dim = config["hidden_size"]
    # 0.各种层定义
    self_attention = MultiHeadAttentionV2(
        embed_dim=embed_dim,
        num_heads=config["num_attention_heads"],
        dropout_rate=config["attention_probs_dropout_prob"],
        kernel_initializer=initializer,
        name=name_prefix + ".attention.self",
    )
    activation_fn = keras.layers.Activation(
        config["hidden_act"], name=name_prefix + ".activation"
    )
    self_attention_layer_norm = keras.layers.LayerNormalization(
        epsilon=1e-12, name=name_prefix + ".attention.output.LayerNorm"
    )

    fc1 = keras.layers.Dense(
        config["intermediate_size"], name=name_prefix + ".intermediate.dense"
    )
    fc2 = keras.layers.Dense(embed_dim, name=name_prefix + ".output.dense")
    final_layer_norm = keras.layers.LayerNormalization(
        epsilon=1e-12, name=name_prefix + ".output.LayerNorm"
    )

    # 1.开始计算
    residual = hidden_states

    # Self Attention
    hidden_states, self_attention_scores = self_attention(
        hidden_states=hidden_states,
        key_states=hidden_states,
        value_states=hidden_states,
        attention_mask=attention_mask,
    )
    hidden_states = keras.layers.Dropout(
        config["hidden_dropout_prob"], name=name_prefix + ".dropout1"
    )(hidden_states)
    hidden_states = keras.layers.Add(name=name_prefix + ".add1")(
        [residual, hidden_states]
    )
    hidden_states = self_attention_layer_norm(hidden_states)

    # Fully Connected
    residual = hidden_states
    hidden_states = activation_fn(fc1(hidden_states))
    hidden_states = keras.layers.Dropout(
        config["attention_probs_dropout_prob"], name=name_prefix + ".activation_dropout"
    )(hidden_states)
    hidden_states = fc2(hidden_states)
    hidden_states = keras.layers.Dropout(
        config["hidden_dropout_prob"], name=name_prefix + ".dropout2"
    )(hidden_states)
    hidden_states = keras.layers.Add(name=name_prefix + ".add2")(
        [residual, hidden_states]
    )
    hidden_states = final_layer_norm(hidden_states)

    return hidden_states, self_attention_scores


def build_pooler_nsp(encoder_hidden_states, initializer, config, name_prefix):
    """创建pooler、nsp层"""
    pooler = keras.layers.Dense(
        config["hidden_size"],
        kernel_initializer=initializer,
        activation="tanh",
        name=name_prefix + ".pooler.dense",
    )(encoder_hidden_states[:, 0])

    nsp = keras.layers.Dense(
        2,
        kernel_initializer=initializer,
        activation="softmax",
        name=name_prefix + ".nsp.dense",
    )(pooler)
    return pooler, nsp


def build_mlm(
    encoder_hidden_states, shared_embedding_layer, initializer, config, name_prefix
):
    """
    创建mlm层 = encoder_hidden_states -> dense -> activation -> ln -> shared_token_embedding -> bias ->softmax
    """
    dense = keras.layers.Dense(
        config["hidden_size"],
        kernel_initializer=initializer,
        activation=config["hidden_act"],
        name=name_prefix + ".dense",
    )(encoder_hidden_states)
    layer_norm = keras.layers.LayerNormalization(
        epsilon=1e-12, name=name_prefix + ".LayerNorm"
    )(dense)
    shared_embedding = shared_embedding_layer(layer_norm)
    bias = BiasLayer(name=name_prefix + ".bias")(shared_embedding)
    softmax = keras.layers.Activation("softmax", name=name_prefix + ".softmax")(bias)
    return softmax


class BertBuilder:
    @staticmethod
    def get_initializer(initializer_range=0.02):
        return keras.initializers.truncated_normal(stddev=initializer_range)

    @staticmethod
    def tokenizer(vocab_file):
        from transformers import BertTokenizer

        return BertTokenizer.from_pretrained(vocab_file)

    @staticmethod
    def read_config_file(config_file):
        """来自google BERT的config文件"""
        with open(config_file) as f:
            config_json = f.read()
        config = json.loads(config_json)

        return config

    @staticmethod
    def get_inputs():
        """创建bert的输入层"""
        input_ids = keras.Input(shape=(None,), name="input_ids", dtype=tf.int32)
        attention_mask = keras.Input(
            shape=(None,), name="attention_mask", dtype=tf.int32
        )
        token_type_ids = keras.Input(
            shape=(None,), name="token_type_ids", dtype=tf.int32
        )
        return input_ids, attention_mask, token_type_ids

    def build_keras_bert_model(self, config):
        """
        创建keras bert model
        """
        initializer = self.get_initializer()
        position_layer = PositionLayer(offset=0, name="positions")
        # 共享token向量层
        shared_embedding_layer = SharedEmbeddingLayer(
            input_dim=config["vocab_size"],
            output_dim=config["hidden_size"],
            initializer=initializer,
            name="bert.embeddings.word_embeddings",
        )

        # 输入
        (input_ids, attention_mask, token_type_ids) = self.get_inputs()

        # 创建encoder
        # 位置id
        encoder_positions = position_layer(input_ids)

        # embedding
        encoder_hidden_states = build_embedding_layers(
            [input_ids, encoder_positions, token_type_ids],
            shared_embedding_layer=shared_embedding_layer,
            initializer=initializer,
            config=config,
            name_prefix="bert.embeddings",
        )

        # attention_mask转化成shape = (batch_size,1,seq_len,seq_len)
        attention_mask_expand = AttentionMaskLayer(name="bert.attention_mask")(
            attention_mask
        )

        # 多层transformer_encoder层
        for i in range(config["num_hidden_layers"]):
            (
                encoder_hidden_states,
                encoder_self_attention_scores,
            ) = build_transformer_encoder_layer(
                encoder_hidden_states,
                attention_mask=attention_mask_expand,
                config=config,
                name_prefix=f"bert.encoder.layer_{i}",
            )

        # pooler = x[:,0] -> dense -> tanh
        # nsp = pooler -> dense -> softmax
        pooler, nsp = build_pooler_nsp(
            encoder_hidden_states, initializer, config, name_prefix="bert"
        )

        # mlm = x -> dense -> ln -> shared_token_embedding -> bias ->softmax
        mlm = build_mlm(
            encoder_hidden_states,
            shared_embedding_layer,
            initializer,
            config,
            name_prefix="bert.mlm",
        )

        bert = keras.Model(
            inputs={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
            outputs={
                "encoder_hidden_states": encoder_hidden_states,
                "mlm": mlm,
                "pooler": pooler,
                "nsp": nsp,
            },
        )
        return bert

    @staticmethod
    def read_google_bert_weights(checkpoint_file):
        """读取google bert的参数，返回字典dict"""
        ck = tf.train.load_checkpoint(checkpoint_file)
        dtype_mapping = ck.get_variable_to_dtype_map()
        tensor_mapping = {}
        for name in sorted(list(dtype_mapping.keys())):
            numpy_tensor = ck.get_tensor(name)
            tensor_mapping[name] = numpy_tensor
        return tensor_mapping

    @staticmethod
    def get_variable_mapping(config):
        """映射到官方BERT权重格式"""
        mapping = {
            "bert.embeddings.word_embeddings": ["bert/embeddings/word_embeddings"],
            "bert.embeddings.token_type_embeddings": [
                "bert/embeddings/token_type_embeddings"
            ],
            "bert.embeddings.position_embeddings": [
                "bert/embeddings/position_embeddings"
            ],
            "bert.embeddings.LayerNorm": [
                "bert/embeddings/LayerNorm/gamma",
                "bert/embeddings/LayerNorm/beta",
            ],
            #         'Embedding-Mapping': [
            #             'bert/encoder/embedding_hidden_mapping_in/kernel',
            #             'bert/encoder/embedding_hidden_mapping_in/bias',
            #         ],
            "bert.pooler.dense": [
                "bert/pooler/dense/kernel",
                "bert/pooler/dense/bias",
            ],
            "bert.nsp.dense": [
                "cls/seq_relationship/output_weights",
                "cls/seq_relationship/output_bias",
            ],
            "bert.mlm.dense": [
                "cls/predictions/transform/dense/kernel",
                "cls/predictions/transform/dense/bias",
            ],
            "bert.mlm.LayerNorm": [
                "cls/predictions/transform/LayerNorm/gamma",
                "cls/predictions/transform/LayerNorm/beta",
            ],
            "bert.mlm.bias": ["cls/predictions/output_bias"],
        }

        for i in range(config["num_hidden_layers"]):
            prefix = "bert/encoder/layer_%d/" % i
            mapping.update(
                {
                    "bert.encoder.layer_%d.attention.self"
                    % i: [
                        prefix + "attention/self/key/kernel",
                        prefix + "attention/self/key/bias",
                        prefix + "attention/self/query/kernel",
                        prefix + "attention/self/query/bias",
                        prefix + "attention/self/value/kernel",
                        prefix + "attention/self/value/bias",
                        prefix + "attention/output/dense/kernel",
                        prefix + "attention/output/dense/bias",
                    ],
                    "bert.encoder.layer_%d.attention.output.LayerNorm"
                    % i: [
                        prefix + "attention/output/LayerNorm/gamma",
                        prefix + "attention/output/LayerNorm/beta",
                    ],
                    "bert.encoder.layer_%d.output.dense"
                    % i: [
                        prefix + "output/dense/kernel",
                        prefix + "output/dense/bias",
                    ],
                    "bert.encoder.layer_%d.intermediate.dense"
                    % i: [
                        prefix + "intermediate/dense/kernel",
                        prefix + "intermediate/dense/bias",
                    ],
                    "bert.encoder.layer_%d.output.LayerNorm"
                    % i: [
                        prefix + "output/LayerNorm/gamma",
                        prefix + "output/LayerNorm/beta",
                    ],
                }
            )

        return mapping

    @staticmethod
    def load_google_bert_weights(keras_bert, tensor_mapping, variable_mapping):
        for layer in keras_bert.layers:
            if layer.name in variable_mapping:
                weights = []
                for weight_name in variable_mapping[layer.name]:
                    if weight_name == "cls/seq_relationship/output_weights":
                        weights.append(tf.transpose(tensor_mapping[weight_name]))
                    else:
                        weights.append(tensor_mapping[weight_name])
                print(layer.name + ":加载参数:" + ",".join(variable_mapping[layer.name]))
                layer.set_weights(weights)
            else:
                print(layer.name + "不加载参数")
        return keras_bert

    def build_bert(self, config_file, checkpoint_file=None, vocab_file=None):
        """创建bert与tokenizer，bert自动加载google bert参数(若有)"""
        config = self.read_config_file(config_file)
        bert = self.build_keras_bert_model(config)
        tokenizer = None
        if checkpoint_file:
            # 加载google bert参数到bert
            tensor_mapping = self.read_google_bert_weights(checkpoint_file)
            variable_mapping = self.get_variable_mapping(config)
            bert = self.load_google_bert_weights(bert, tensor_mapping, variable_mapping)
            del tensor_mapping  # 释放资源
        if vocab_file:
            tokenizer = self.tokenizer(vocab_file)
        return bert, tokenizer, config


if __name__ == "__main__":
    # 0.下载参数，存放于bert_dir下
    # Google原版bert: https://github.com/google-research/bert
    bert_dir = "/Users/mos_luo/project/pretrain_model/bert/chinese_L-12_H-768_A-12/"
    config_file = bert_dir + "bert_config.json"
    checkpoint_file = bert_dir + "bert_model.ckpt"
    vocab_file = bert_dir + "vocab.txt"

    # 1.创建keras bert模型与tokenizer
    keras_bert, tokenizer, config = BertBuilder().build_bert(
        config_file=config_file, checkpoint_file=checkpoint_file, vocab_file=vocab_file
    )
    keras_bert.summary()

    # 2.创建输入样本
    inputs = tokenizer(["语言模型"], return_tensors="tf")
    print(keras_bert(inputs))
