# README

## 动机
- 用tf.keras (TF2.0+) 的稳定API实现NLP预训练模型，例如BERT、BART等。
- 不做过多的自定义类、方法，力图代码简洁，易懂，保存可扩展性。

## 支持的模型
- BERT
- BART

## 使用例子
### BERT
- bert参数下载：
    - Google原版bert: https://github.com/google-research/bert
- 代码 
```python 
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

# 2.创建输入样本
# tokenizer = builder.tokenizer(vocab_file)
inputs = tokenizer(["语言模型"], return_tensors="tf")
print(keras_bert(inputs))
```
### BART
- BART参数下载
- 代码

## requirements
- python>=3.6
- tensorflow>=2.0.0
- numpy
- transformers=4.25.1
    - 主要是为了提供tokenizer，不是必须的，可以不装。
    - 你也可以用其他的tokenizer实现。


## 更新日志
- 2023.01.15：添加BART
- 2023.01.30：添加BERT