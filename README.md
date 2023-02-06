# README

## 动机
- 用tf.keras (TF2.0+) 的稳定API实现NLP预训练模型，例如BERT、BART等。
- 不做过多的自定义类、方法，力图代码简洁，易懂，易扩展。

## 支持的模型
- BERT
- BART

## 使用例子
### 安装
```
pip install pretrain4keras
```
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
    - 例如从https://huggingface.co/fnlp/bart-base-chinese/tree/4e93f21dca95a07747f434b0f9fe5d49cacc0441下载文件夹的所有文件
    - huggingface transformers的参数与json的可能会发现变动，所以下载时需要指定版本id，例如4e93f21dca95a07747f434b0f9fe5d49cacc0441
- 示例代码
```python
import pprint
import tensorflow as tf
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
```

## requirements
- python>=3.6
- tensorflow>=2.0.0
- numpy
- transformers=4.25.1
    - 主要是为了提供tokenizer，不是必须的，可以不装。
    - 你也可以用其他的tokenizer实现。

## 参考
- https://github.com/bojone/bert4keras
- https://github.com/huggingface/transformers

## 更新日志
- 2023.01.15：添加BART
- 2023.01.30：添加BERT