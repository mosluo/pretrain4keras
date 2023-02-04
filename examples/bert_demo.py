#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: bert_demo.py
@datetime:2023/1/31 11:40 上午
功能：
"""

import tensorflow as tf
import tensorflow.keras as keras
from pretrain4keras.models.bert import BertBuilder

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
print(tokenizer.batch_decode(tf.argmax(keras_bert(inputs)["mlm"], axis=2).numpy()))

# 4.huggingface bert的输出
print("=========== huggingface bert的输出 ============>")
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = TFBertModel.from_pretrained("bert-base-chinese")
text = "语言模型"
encoded_input = tokenizer(text, return_tensors="tf")
output = model(encoded_input)
print(output)  # last_hidden_states、pooler

# 测试nsp
from transformers import TFBertForNextSentencePrediction

nsp = TFBertForNextSentencePrediction.from_pretrained("bert-base-chinese")(
    encoded_input
)
print(tf.nn.softmax(nsp.logits))

# 5.保存
print("=========== 测试保存与加载 =============>")
print("开始保存")
bert_save_dir = "/Users/mos_luo/project/my_model/20230130_keras4bert/bert_save/"
keras_bert.save(bert_save_dir)
print("保存成功")
bert2 = keras.models.load_model(bert_save_dir)
print("加载成功")
print(keras_bert(inputs))
print(tokenizer.batch_decode(tf.argmax(bert2(inputs)["mlm"], axis=2).numpy()))
