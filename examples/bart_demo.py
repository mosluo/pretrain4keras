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

# 4.对比transformers的bart输出结果
print("=========== 对比transformers的bart输出结果 ============>")
from transformers import TFBartModel

transformers_bart = TFBartModel.from_pretrained(
    pretrained_model_name_or_path=pretrain_dir, from_pt=True
)
transformers_bart_out = transformers_bart(inputs)
print("transformers_bart_out.encoder_last_hidden_state=")
print(transformers_bart_out.encoder_last_hidden_state)
print("transformers_bart_out.last_hidden_state=")
print(transformers_bart_out.last_hidden_state)

# 5.保存
print("=========== 测试保存与加载 =============>")
print("开始保存")
bart_save_dir = "/Users/normansluo/project/my_model/20220828_keras4bart/bart_save/"
keras_bart.save(bart_save_dir)
print("保存成功")
bart2 = keras.models.load_model(bart_save_dir)
print("加载成功")
print(tokenizer.batch_decode(tf.argmax(bart2(inputs)["lm"], axis=2).numpy()))

# 6.encoder与decoder拆分
print("=========== encoder与decoder拆分 ================")
# 6.1创建encoder，并加载bart的keras参数
encoder = builder.build_keras_bart_model(config, mode="encoder")
builder.load_weights_from_keras_bart(keras_bart, encoder)

# 6.2创建decoder，并加载bart的keras参数
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
