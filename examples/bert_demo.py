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

# 0.手动从bert-base-chinese下载文件
# 从https://huggingface.co/bert-base-chinese/tree/main下载文件夹的所有文件
# pretrain_name = "/Users/mos_luo/project/pretrain_model/huggingface_transformers/bert-base-chinese"

# 1.创建keras bert模型
builder = BertBuilder()
config = builder.read_config_file()
keras_bert = builder.build_keras_bert_model(config)
keras_bert.summary()

# 2.加载参数到keras bert
checkpoint_file = "/Users/mos_luo/project/pretrain_model/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
tensor_mapping = builder.read_google_bert_weights(checkpoint_file)
variable_mapping = builder.get_variable_mapping(config)
keras_bert = builder.load_google_bert_weights(
    keras_bert, tensor_mapping, variable_mapping
)

# 3.创建输入样本
tokenizer = builder.tokenizer()
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
bert2 = keras.models.load_model(
    bert_save_dir,
)
print("加载成功")
print(keras_bert(inputs))
print(tokenizer.batch_decode(tf.argmax(bert2(inputs)["mlm"], axis=2).numpy()))