#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: bart_demo.py
@datetime:2023/1/31 11:40 上午
功能：
"""
import pprint
import tensorflow as tf
import tensorflow.keras as keras
from pretrain4keras.models.bart import BartBuilder

# 0.手动从fnlp/bart-base-chinese下载文件
# 从https://huggingface.co/fnlp/bart-base-chinese/tree/main下载文件夹的所有文件
pretrain_name = "/Users/mos_luo/project/pretrain_model/huggingface_transformers/fnlp/bart-base-chinese-v2"

# 1.创建keras bart模型
builder = BartBuilder()
config = builder.read_config_file()
keras_bart = builder.build_keras_bart_model(config, mode="encoder_decoder")
keras_bart.summary()

# 2.从pytorch bart模型加载参数到keras bart
pt_bin_file = pretrain_name + "/pytorch_model.bin"
pytorch_state_dict = builder.read_pytorch_weights(pt_bin_file)
for name in pytorch_state_dict:
    print(name, "-->", list(pytorch_state_dict[name].shape))
keras_bart = builder.load_pytorch_weights(keras_bart, pytorch_state_dict)

# 3.创建输入样本
tokenizer = builder.tokenizer(pretrain_name)  # 复旦中文bart用bert的tokenizer
inputs = tokenizer(["北京是[MASK]的首都"], return_tensors="tf")
del inputs["token_type_ids"]
inputs["decoder_input_ids"] = tf.constant(
    [[102, 101, 6188, 5066, 11009, 4941, 7178, 15134, 23943, 21784]]
)
pprint.pprint(inputs)

# 4.keras bart的输出
print("=========== keras bart的输出 ============>")
keras_bart_out = keras_bart(inputs)
print("keras_bart_out=")
print(keras_bart_out)
print(tokenizer.batch_decode(tf.argmax(keras_bart_out["lm"], axis=2).numpy()))

# 5.对比transformers的bart输出结果
print("=========== 对比transformers的bart输出结果 ============>")
from transformers import TFBartModel

transformers_bart = TFBartModel.from_pretrained(
    pretrained_model_name_or_path=pretrain_name, from_pt=True
)
transformers_bart_out = transformers_bart(inputs)
print("transformers_bart_out.encoder_last_hidden_state=")
print(transformers_bart_out.encoder_last_hidden_state)
print("transformers_bart_out.last_hidden_state=")
print(transformers_bart_out.last_hidden_state)

# 6.保存
print("=========== 测试保存与加载 =============>")
print("开始保存")
bart_save_dir = "/Users/mos_luo/project/my_model/20220828_keras4bart/bart_save/"
keras_bart.save(bart_save_dir)
print("保存成功")
bart2 = keras.models.load_model(
    bart_save_dir,
    custom_objects={
        "PositionLayer": PositionLayer,
        "MultiHeadAttentionV2": MultiHeadAttentionV2,
        "SharedEmbeddingLayer": SharedEmbeddingLayer,
        "AttentionMaskLayer": AttentionMaskLayer,
        "CausalMaskLayer": CausalMaskLayer,
    },
)
print("加载成功")
print(tokenizer.batch_decode(tf.argmax(bart2(inputs)["lm"], axis=2).numpy()))

# # 7.encoder与decoder拆分
print("=========== encoder与decoder拆分 ================")
# 创建encoder，并加载bart的keras参数
encoder = builder.build_keras_bart_model(config, mode="encoder")
builder.load_weights_from_keras_bart(keras_bart, encoder)

# 创建decoder，并加载bart的keras参数
decoder = builder.build_keras_bart_model(config, mode="decoder")
builder.load_weights_from_keras_bart(keras_bart, decoder)

encoder_hidden_states = encoder(
    {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
)
decoder_out = decoder(
    {
        "attention_mask": inputs["attention_mask"],
        "decoder_input_ids": inputs["decoder_input_ids"],
        "encoder_hidden_states": encoder_hidden_states,
    }
)
print(tokenizer.batch_decode(tf.argmax(decoder_out["lm"], axis=2).numpy()))
