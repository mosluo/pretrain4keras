#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: bart.py
@datetime:2023/1/31 11:42 上午
功能：keras的bart
"""
import json
import pprint
import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras


from pretrain4keras.layers import (
    PositionLayer,
    MultiHeadAttentionV2,
    AttentionMaskLayer,
    SharedEmbeddingLayer,
    CausalMaskLayer,
)


def build_embedding_layers(
    inputs,
    shared_embedding_layer,
    initializer=None,
    config=None,
    name_prefix="",
):
    """创建embedding层: token+position+layer_norm+dropout_rate"""
    tokens, positions = inputs
    # token向量
    embed_tokens = shared_embedding_layer(tokens)
    # 位置向量
    embed_positions = keras.layers.Embedding(
        input_dim=config["max_position_embeddings"] + 2,
        output_dim=config["d_model"],
        embeddings_initializer=initializer,
        name=name_prefix + ".embed_positions",
    )(positions)

    # 位置向量与token向量相加
    x = keras.layers.Add(name=name_prefix + ".add")([embed_tokens, embed_positions])
    # ln+dropout_rate
    hidden_states = keras.layers.LayerNormalization(
        epsilon=1e-5, name=name_prefix + ".layernorm_embedding"
    )(x)
    hidden_states = keras.layers.Dropout(
        config["dropout"], name=name_prefix + ".dropout_rate"
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
    embed_dim = config["d_model"]
    # 0.各种层定义
    self_attention = MultiHeadAttentionV2(
        embed_dim=embed_dim,
        num_heads=config["encoder_attention_heads"],
        dropout_rate=config["attention_dropout"],
        kernel_initializer=initializer,
        name=name_prefix + ".self_attn",
    )
    activation_fn = keras.layers.Activation(
        config["activation_function"], name=name_prefix + ".activation"
    )
    self_attention_layer_norm = keras.layers.LayerNormalization(
        epsilon=1e-5, name=name_prefix + ".self_attn_layer_norm"
    )

    fc1 = keras.layers.Dense(config["encoder_ffn_dim"], name=name_prefix + ".fc1")
    fc2 = keras.layers.Dense(embed_dim, name=name_prefix + ".fc2")
    final_layer_norm = keras.layers.LayerNormalization(
        epsilon=1e-5, name=name_prefix + ".final_layer_norm"
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
        config["dropout"], name=name_prefix + ".dropout1"
    )(hidden_states)
    hidden_states = keras.layers.Add(name=name_prefix + ".add1")(
        [residual, hidden_states]
    )
    hidden_states = self_attention_layer_norm(hidden_states)

    # Fully Connected
    residual = hidden_states
    hidden_states = activation_fn(fc1(hidden_states))
    hidden_states = keras.layers.Dropout(
        config["activation_dropout"], name=name_prefix + ".activation_dropout"
    )(hidden_states)
    hidden_states = fc2(hidden_states)
    hidden_states = keras.layers.Dropout(
        config["dropout"], name=name_prefix + ".dropout2"
    )(hidden_states)
    hidden_states = keras.layers.Add(name=name_prefix + ".add2")(
        [residual, hidden_states]
    )
    hidden_states = final_layer_norm(hidden_states)

    return hidden_states, self_attention_scores


def build_transformer_decoder_layer(
    hidden_states,
    key,
    value,
    cross_attention_mask,
    initializer=None,
    config=None,
    name_prefix="",
):
    """创建一层transformer_decoder层"""
    embed_dim = config["d_model"]
    # 0.各种层定义
    self_attention = MultiHeadAttentionV2(
        embed_dim=embed_dim,
        num_heads=config["decoder_attention_heads"],
        dropout_rate=config["attention_dropout"],
        kernel_initializer=initializer,
        name=name_prefix + ".self_attn",
    )
    activation_fn = keras.layers.Activation(
        config["activation_function"], name=name_prefix + ".activation"
    )
    self_attention_layer_norm = keras.layers.LayerNormalization(
        epsilon=1e-5, name=name_prefix + ".self_attn_layer_norm"
    )
    cross_attention = MultiHeadAttentionV2(
        embed_dim=embed_dim,
        num_heads=config["decoder_attention_heads"],
        dropout_rate=config["attention_dropout"],
        kernel_initializer=initializer,
        name=name_prefix + ".encoder_attn",
    )
    cross_attention_layer_norm = keras.layers.LayerNormalization(
        epsilon=1e-5, name=name_prefix + ".encoder_attn_layer_norm"
    )
    fc1 = keras.layers.Dense(config["decoder_ffn_dim"], name=name_prefix + ".fc1")
    fc2 = keras.layers.Dense(embed_dim, name=name_prefix + ".fc2")
    final_layer_norm = keras.layers.LayerNormalization(
        epsilon=1e-5, name=name_prefix + ".final_layer_norm"
    )

    # 1.开始计算
    residual = hidden_states

    # Self Attention
    causal_mask = CausalMaskLayer(name=name_prefix + ".causal_mask_layer")(
        hidden_states
    )
    hidden_states, self_attention_scores = self_attention(
        hidden_states=hidden_states,
        key_states=hidden_states,
        value_states=hidden_states,
        attention_mask=causal_mask,
    )
    hidden_states = keras.layers.Dropout(
        config["dropout"], name=name_prefix + ".dropout1"
    )(hidden_states)
    hidden_states = keras.layers.Add(name=name_prefix + ".add1")(
        [residual, hidden_states]
    )
    hidden_states = self_attention_layer_norm(hidden_states)

    # Cross-Attention Block
    residual = hidden_states
    hidden_states, cross_attention_scores = cross_attention(
        hidden_states=hidden_states,
        key_states=key,
        value_states=value,
        attention_mask=cross_attention_mask,
    )
    hidden_states = keras.layers.Dropout(
        config["dropout"], name=name_prefix + ".dropout2"
    )(hidden_states)
    hidden_states = keras.layers.Add(name=name_prefix + ".add2")(
        [residual, hidden_states]
    )
    hidden_states = cross_attention_layer_norm(hidden_states)

    # Fully Connected
    residual = hidden_states
    hidden_states = activation_fn(fc1(hidden_states))
    hidden_states = keras.layers.Dropout(
        config["activation_dropout"], name=name_prefix + ".activation_dropout"
    )(hidden_states)
    hidden_states = fc2(hidden_states)
    hidden_states = keras.layers.Dropout(
        config["dropout"], name=name_prefix + ".dropout3"
    )(hidden_states)
    hidden_states = keras.layers.Add(name=name_prefix + ".add3")(
        [residual, hidden_states]
    )
    hidden_states = final_layer_norm(hidden_states)

    return hidden_states, self_attention_scores, cross_attention_scores


class BartBuilder:
    @staticmethod
    def get_initializer(initializer_range=0.02):
        return keras.initializers.truncated_normal(stddev=initializer_range)

    @staticmethod
    def tokenizer(vocab_file, tokenizer_class="BertTokenizer"):
        """
        复旦的中文bart用的是bert的tokenizer，例如fnlp/bart-base-chinese
        如果你用的是非复旦参数，那么要用BartTokenizer
        """
        from transformers import BertTokenizer, BartTokenizer

        if tokenizer_class == "BertTokenizer":
            return BertTokenizer.from_pretrained(vocab_file)
        else:
            return BartTokenizer.from_pretrained(vocab_file)

    @staticmethod
    def read_config_file(config_file):
        """读配置参数，来自huggingface的模型fnlp/bart-base-chinese配置"""
        with open(config_file) as f:
            config_json = f.read()
        config = json.loads(config_json)
        # 将文本true、false改成bool
        for k in config:
            if isinstance(config[k], str):
                if config[k].lower() == "true":
                    config[k] = True
                if config[k].lower() == "false":
                    config[k] = False
        pprint.pprint(config)
        return config

        # config = {
        #     "activation_dropout": 0.1,
        #     "activation_function": "gelu",
        #     "add_bias_logits": False,
        #     "add_final_layer_norm": False,
        #     "architectures": ["BartForConditionalGeneration"],
        #     "attention_dropout": 0.1,
        #     "bos_token_id": 101,
        #     "classif_dropout": 0.1,
        #     "classifier_dropout": 0.0,
        #     "d_model": 768,
        #     "decoder_attention_heads": 12,
        #     "decoder_ffn_dim": 3072,
        #     "decoder_layerdrop": 0.0,
        #     "decoder_layers": 6,
        #     "decoder_start_token_id": 102,
        #     "dropout_rate": 0.1,
        #     "early_stopping": True,
        #     "encoder_attention_heads": 12,
        #     "encoder_ffn_dim": 3072,
        #     "encoder_layerdrop": 0.0,
        #     "encoder_layers": 6,
        #     "eos_token_id": 102,
        #     "forced_eos_token_id": 102,
        #     "gradient_checkpointing": False,
        #     "id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"},
        #     "init_std": 0.02,
        #     "is_encoder_decoder": True,
        #     "label2id": {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
        #     "max_position_embeddings": 1024,
        #     "model_type": "bart",
        #     "no_repeat_ngram_size": 3,
        #     "normalize_before": False,
        #     "normalize_embedding": True,
        #     "num_beams": 4,
        #     "num_hidden_layers": 6,
        #     "pad_token_id": 0,
        #     "scale_embedding": False,
        #     "task_specific_params": {
        #         "summarization": {
        #             "length_penalty": 1.0,
        #             "max_length": 128,
        #             "min_length": 12,
        #             "num_beams": 4,
        #         },
        #         "summarization_cnn": {
        #             "length_penalty": 2.0,
        #             "max_length": 142,
        #             "min_length": 56,
        #             "num_beams": 4,
        #         },
        #         "summarization_xsum": {
        #             "length_penalty": 1.0,
        #             "max_length": 62,
        #             "min_length": 11,
        #             "num_beams": 6,
        #         },
        #     },
        #     "transformers_version": "4.4.1",
        #     "use_cache": True,
        #     "tokenizer_class": "BertTokenizer",
        #     "vocab_size": 51271,
        # }
        #
        # return config

    @staticmethod
    def get_inputs():
        """创建BART的输入层"""
        encoder_tokens = keras.Input(shape=(None,), name="input_ids", dtype=tf.int32)

        encoder_attention_mask = keras.Input(
            shape=(None,),
            name="attention_mask",
            dtype=tf.int32,
        )
        decoder_tokens = keras.Input(
            shape=(None,), name="decoder_input_ids", dtype=tf.int32
        )

        return (
            encoder_tokens,
            encoder_attention_mask,
            decoder_tokens,
        )

    @staticmethod
    def build_encoder(
        inputs, initializer, position_layer, shared_embedding_layer, config
    ):
        """创建keras bart的子模型encoder model，只有encoder层，无decoder层"""
        # 输入
        (encoder_tokens, encoder_attention_mask) = inputs

        encoder_positions = position_layer(encoder_tokens)
        # encoder的输入embedding
        encoder_hidden_states = build_embedding_layers(
            [encoder_tokens, encoder_positions],
            shared_embedding_layer=shared_embedding_layer,
            initializer=initializer,
            config=config,
            name_prefix="model.encoder",
        )

        # attention_mask转化成shape = (batch_size,1,seq_len,seq_len)
        attention_mask = AttentionMaskLayer(name="model.encoder.attention_mask")(
            encoder_attention_mask
        )
        # 多层transformer_encoder层
        for i in range(config["encoder_layers"]):
            (
                encoder_hidden_states,
                encoder_self_attention_scores,
            ) = build_transformer_encoder_layer(
                encoder_hidden_states,
                attention_mask=attention_mask,
                config=config,
                name_prefix=f"model.encoder.layers.{i}",
            )
        return encoder_hidden_states

    @staticmethod
    def build_decoder(
        inputs, initializer, position_layer, shared_embedding_layer, config
    ):
        """创建keras bart的子模型decoder model，只有decoder，无encoder"""
        # 输入
        (encoder_hidden_states, encoder_attention_mask, decoder_tokens) = inputs

        # decoder的输入embedding
        decoder_positions = position_layer(decoder_tokens)
        decoder_hidden_states = build_embedding_layers(
            [decoder_tokens, decoder_positions],
            shared_embedding_layer=shared_embedding_layer,
            initializer=initializer,
            config=config,
            name_prefix="model.decoder",
        )
        cross_attention_mask = AttentionMaskLayer(
            name="model.decoder.cross_attention_mask"
        )(encoder_attention_mask, decoder_hidden_states)
        # 多层transformer_decoder层
        for i in range(config["decoder_layers"]):
            (
                decoder_hidden_states,
                decoder_self_attention_scores,
                decoder_cross_attention_scores,
            ) = build_transformer_decoder_layer(
                hidden_states=decoder_hidden_states,
                key=encoder_hidden_states,
                value=encoder_hidden_states,
                cross_attention_mask=cross_attention_mask,
                initializer=initializer,
                config=config,
                name_prefix=f"model.decoder.layers.{i}",
            )
        lm = shared_embedding_layer(
            decoder_hidden_states  # , mode="linear"
        )  # shape = (batch_size,seq_len,vocab_size)

        return decoder_hidden_states, lm

    def build_keras_bart_model(self, config, mode="encoder_decoder"):
        """
        创建keras bart model，可分别返回encoder、decoder、encdoer_decoder整体模型。
        拆开的原因：在生成文本时，对于同一个句子需要多次infernce才能生成一个完整的句子,
            encoder模型的输出encoder_hidden_states可重复使用，而不需要每次都计算一次。
        Args:
            config: 配置，dict
            mode: 有三种类型
                - encoder: 只返回encoder模型，输出是encoder_hidden_states
                - decoder: 只返回decoder模型，输出是一个字典，包含decoder_hidden_states与lm
                - encoder_decoder: 同时返回encoder与decoder模型，用于fit训练

        Returns:
            返回encoder，或者decoder，或者encoder_decoder

        """
        initializer = self.get_initializer()
        # 位置id层：bart的位置id要加2
        position_layer = PositionLayer(offset=2, name="model.position")
        # 共享token向量层
        shared_embedding_layer = SharedEmbeddingLayer(
            input_dim=config["vocab_size"],
            output_dim=config["d_model"],
            initializer=initializer,
            name="model.shared.token_embedding",
        )

        # 输入
        (encoder_tokens, encoder_attention_mask, decoder_tokens) = self.get_inputs()
        encoder_hidden_states = keras.Input(
            shape=(None, config["d_model"]),
            name="encoder_hidden_states",
            dtype=tf.float32,
        )  # 仅用于decoder

        if mode in ("encoder", "encoder_decoder"):
            # 创建encoder
            encoder_hidden_states = self.build_encoder(
                inputs=[encoder_tokens, encoder_attention_mask],
                initializer=initializer,
                position_layer=position_layer,
                shared_embedding_layer=shared_embedding_layer,
                config=config,
            )

            # 只返回encoder
            if mode in "encoder":
                encoder = keras.Model(
                    inputs={
                        "input_ids": encoder_tokens,
                        "attention_mask": encoder_attention_mask,
                    },
                    outputs=encoder_hidden_states,
                )
                return encoder

        if mode in ("decoder", "encoder_decoder"):
            # 创建decoder
            (decoder_hidden_states, lm) = self.build_decoder(
                inputs=[encoder_hidden_states, encoder_attention_mask, decoder_tokens],
                initializer=initializer,
                position_layer=position_layer,
                shared_embedding_layer=shared_embedding_layer,
                config=config,
            )

            # 只返回decoder
            if mode == "decoder":
                decoder = keras.Model(
                    inputs={
                        "encoder_hidden_states": encoder_hidden_states,
                        "attention_mask": encoder_attention_mask,
                        "decoder_input_ids": decoder_tokens,
                    },
                    outputs={
                        "decoder_hidden_states": decoder_hidden_states,
                        "lm": lm,
                    },
                )
                return decoder

        encoder_decoder = keras.Model(
            inputs={
                "input_ids": encoder_tokens,
                "attention_mask": encoder_attention_mask,
                "decoder_input_ids": decoder_tokens,
            },
            outputs={
                "encoder_hidden_states": encoder_hidden_states,
                "decoder_hidden_states": decoder_hidden_states,
                "lm": lm,
            },
        )
        return encoder_decoder  # 返回完整的模型

    @staticmethod
    def read_pytorch_weights(checkpoint_file):
        """读取pytorch的模型参数，返回字典"""
        state_dict = torch.load(checkpoint_file, map_location="cpu")
        return state_dict

    @staticmethod
    def load_pytorch_weights(model: keras.Model, torch_state_dict):
        def load_embedding_layer_weights(
            tf_layer: keras.layers.Layer, torch_state_dict, torch_tensor_name_prefix
        ):
            np_tensor = torch_state_dict[torch_tensor_name_prefix + ".weight"].numpy()
            print(f"embedding shape={np_tensor.shape}")
            tf_layer.set_weights([np_tensor])

        def load_dense_layer_weights(
            tf_layer: keras.layers.Layer, torch_state_dict, torch_tensor_name_prefix
        ):
            # pt
            # encoder.layers.0.self_attn.k_proj.weight  -->  [768, 768]
            # encoder.layers.0.self_attn.k_proj.bias  -->  [768]
            weight = np.transpose(
                torch_state_dict[torch_tensor_name_prefix + ".weight"].numpy()
            )
            bias = torch_state_dict[torch_tensor_name_prefix + ".bias"].numpy()
            print(f"weight shape={weight.shape}")
            print(f"bias shape={bias.shape}")
            tf_layer.set_weights([weight, bias])

        def load_layer_norm_layer_weights(
            tf_layer: keras.layers.Layer, torch_state_dict, torch_tensor_name_prefix
        ):
            # tf
            # encoder.transformer_encoder_1.final_layer_norm / gamma: 0 --> (768,)
            # encoder.transformer_encoder_1.final_layer_norm / beta: 0 --> (768,)
            # pt
            # encoder.layers.0.final_layer_norm.weight  -->  [768]
            # encoder.layers.0.final_layer_norm.bias  -->  [768]
            gamma = torch_state_dict[torch_tensor_name_prefix + ".weight"].numpy()
            beta = torch_state_dict[torch_tensor_name_prefix + ".bias"].numpy()
            print(f"gamma shape={gamma.shape}")
            print(f"beta shape={beta.shape}")
            tf_layer.set_weights([gamma, beta])

        def load_multi_head_attention_layer_weights(
            tf_layer: keras.layers.Layer, torch_state_dict, torch_tensor_name_prefix
        ):
            # tf
            # encoder.transformer_encoder_0.self_attention / k_proj / kernel: 0 --> (768, 768)
            # encoder.transformer_encoder_0.self_attention / k_proj / bias: 0 --> (768,)
            # encoder.transformer_encoder_0.self_attention / q_proj / kernel: 0 --> (768, 768)
            # encoder.transformer_encoder_0.self_attention / q_proj / bias: 0 --> (768,)
            # encoder.transformer_encoder_0.self_attention / v_proj / kernel: 0 --> (64, 768)
            # encoder.transformer_encoder_0.self_attention / v_proj / bias: 0 --> (768,)
            # encoder.transformer_encoder_0.self_attention / out_proj / kernel: 0 --> (768, 768)
            # encoder.transformer_encoder_0.self_attention / out_proj / bias: 0 --> (768,)
            # pt
            # encoder.layers.1.self_attn.k_proj.weight  -->  [768, 768]
            # encoder.layers.1.self_attn.k_proj.bias  -->  [768]
            # encoder.layers.1.self_attn.v_proj.weight  -->  [768, 768]
            # encoder.layers.1.self_attn.v_proj.bias  -->  [768]
            # encoder.layers.1.self_attn.q_proj.weight  -->  [768, 768]
            # encoder.layers.1.self_attn.q_proj.bias  -->  [768]
            # encoder.layers.1.self_attn.out_proj.weight  -->  [768, 768]
            # encoder.layers.1.self_attn.out_proj.bias  -->  [768]
            tensors = []
            for name in ["k_proj", "q_proj", "v_proj", "out_proj"]:
                weight = np.transpose(
                    torch_state_dict[
                        torch_tensor_name_prefix + "." + name + ".weight"
                    ].numpy()
                )
                bias = torch_state_dict[
                    torch_tensor_name_prefix + "." + name + ".bias"
                ].numpy()
                print(
                    torch_tensor_name_prefix
                    + "."
                    + name
                    + ".weight"
                    + f" shape={weight.shape}"
                )
                print(
                    torch_tensor_name_prefix
                    + "."
                    + name
                    + ".bias"
                    + f"bias shape={bias.shape}"
                )
                tensors.append(weight)
                tensors.append(bias)
            tf_layer.set_weights(tensors)

        """tf模型加载pytorch参数"""
        for layer in model.layers:
            print(f"layer.name={layer.name}")
            if layer.name == "model.shared.token_embedding":
                print(f"layer.name={layer.name}加载load_embedding_layer_weights")
                load_embedding_layer_weights(
                    tf_layer=layer,
                    torch_state_dict=torch_state_dict,
                    torch_tensor_name_prefix="model.shared",
                )
                continue

            if isinstance(layer, keras.layers.Embedding):
                print(f"layer.name={layer.name}加载load_embedding_layer_weights")
                load_embedding_layer_weights(
                    tf_layer=layer,
                    torch_state_dict=torch_state_dict,
                    torch_tensor_name_prefix=layer.name,
                )

            if isinstance(layer, keras.layers.Dense):
                print(f"layer.name={layer.name}加载load_dense_layer_weights")
                load_dense_layer_weights(
                    tf_layer=layer,
                    torch_state_dict=torch_state_dict,
                    torch_tensor_name_prefix=layer.name,
                )

            if isinstance(layer, keras.layers.LayerNormalization):
                print(f"layer.name={layer.name}加载load_layer_norm_layer_weights")
                load_layer_norm_layer_weights(
                    tf_layer=layer,
                    torch_state_dict=torch_state_dict,
                    torch_tensor_name_prefix=layer.name,
                )

            if isinstance(layer, MultiHeadAttentionV2):
                print(
                    f"layer.name={layer.name}加载load_multi_head_attention_layer_weights"
                )
                load_multi_head_attention_layer_weights(
                    tf_layer=layer,
                    torch_state_dict=torch_state_dict,
                    torch_tensor_name_prefix=layer.name,
                )
        print("model参数重置完成！")
        return model

    @staticmethod
    def load_weights_from_keras_bart(keras_bart, bart_sub_model):
        """
        bart的子模型加载keras bart模型的参数。
        子模型的层会自动加载bart相同名字的层参数
        Args:
            keras_bart: 本方法自定义的keras bart模型
            bart_sub_model: encoder模型，或者decoder模型

        Returns: 无返回

        """
        # bart的keras层参数
        bart_weight_dict = {}
        for layer in keras_bart.layers:
            bart_weight_dict[layer.name] = layer.get_weights()

        # 加载keras_bart的参数
        for layer in bart_sub_model.layers:
            if layer.name in bart_weight_dict:
                layer.set_weights(bart_weight_dict[layer.name])
                print(f"{layer.name}加载参数")
            else:
                print(layer.name + "不加载参数")

    def build_bart(self, config_file, checkpoint_file=None, vocab_file=None):
        """创建bart与tokenizer，bart自动加载参数"""
        config = self.read_config_file(config_file)
        keras_bart = self.build_keras_bart_model(config, mode="encoder_decoder")

        if checkpoint_file:
            # 加载参数
            pytorch_state_dict = self.read_pytorch_weights(checkpoint_file)
            for name in pytorch_state_dict:
                print(name, "-->", list(pytorch_state_dict[name].shape))
            keras_bart = self.load_pytorch_weights(keras_bart, pytorch_state_dict)
            del pytorch_state_dict  # 释放资源

        tokenizer = None
        if vocab_file:
            tokenizer = self.tokenizer(vocab_file, config["tokenizer_class"])
        return keras_bart, tokenizer, config


if __name__ == "__main__":
    # 0.手动从fnlp/bart-base-chinese下载文件
    # 从https://huggingface.co/fnlp/bart-base-chinese/tree/4e93f21dca95a07747f434b0f9fe5d49cacc0441下载文件夹的所有文件
    pretrain_dir = "/Users/normansluo/project/pretrain_model/huggingface_transformers/fnlp/bart-base-chinese-v2/"
    checkpoint_file = pretrain_dir + "pytorch_model.bin"
    config_file = pretrain_dir + "config.json"
    vocab_file = pretrain_dir + "vocab.txt"

    # 1.创建keras bart模型
    builder = BartBuilder()
    keras_bart, tokenizer, config = builder.build_bart(
        config_file=config_file, checkpoint_file=checkpoint_file, vocab_file=vocab_file
    )

    # 2.创建输入样本
    inputs = tokenizer(["北京是[MASK]的首都"], return_tensors="tf")
    del inputs["token_type_ids"]
    inputs["decoder_input_ids"] = tf.constant(
        [[102, 101, 6188, 5066, 11009, 4941, 7178, 15134, 23943, 21784]]
    )
    pprint.pprint(inputs)

    # 3.keras bart的输出
    print("=========== keras bart的输出 ============>")
    keras_bart_out = keras_bart(inputs)
    print("keras_bart_out=")
    print(keras_bart_out)
    print(tokenizer.batch_decode(tf.argmax(keras_bart_out["lm"], axis=2).numpy()))
