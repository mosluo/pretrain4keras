# README

## 动机
- 用tf.keras (TF2.0+) 的稳定API实现NLP预训练模型，例如BERT、BART等。
- 不做过多的自定义类、方法，力图代码简洁，易懂，保存可扩展性。

## 支持的模型
- BERT
- BART

## 使用例子
### BERT
- bert参数下载
- 代码 
```python 
from pretrain4keras import bert_builder
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