import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pretrain4keras",
    version="0.0.4",
    author="mos_luo",
    author_email="mos_luo@163.com",
    description="用tf.keras (TF2.0+) 的稳定API实现NLP预训练模型，例如BERT、BART等。",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mosluo/pretrain4keras.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)