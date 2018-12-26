# Keras: 基于 Python 的深度学习库

<img src='https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png', style='max-width: 600px;'>



## 你恰好发现了 Keras。

Keras 是一个用 Python 编写的高级神经网络 API，它能够以 [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), 或者 [Theano](https://github.com/Theano/Theano) 作为后端运行。Keras 的开发重点是支持快速的实验。*能够以最小的时延把你的想法转换为实验结果，是做好研究的关键。*

如果你在以下情况下需要深度学习库，请使用 Keras：

- 允许简单而快速的原型设计（由于用户友好，高度模块化，可扩展性）。
- 同时支持卷积神经网络和循环神经网络，以及两者的组合。
- 在 CPU 和 GPU 上无缝运行。

查看文档，请访问 [Keras.io](https://keras-zh.readthedocs.io/)。

Keras 兼容的 Python 版本: __Python 2.7-3.6__。


------------------


## 指导原则

- __用户友好。__ Keras 是为人类而不是为机器设计的 API。它把用户体验放在首要和中心位置。Keras 遵循减少认知困难的最佳实践：它提供一致且简单的 API，将常见用例所需的用户操作数量降至最低，并且在用户错误时提供清晰和可操作的反馈。

- __模块化。__ 模型被理解为由独立的、完全可配置的模块构成的序列或图。这些模块可以以尽可能少的限制组装在一起。特别是神经网络层、损失函数、优化器、初始化方法、激活函数、正则化方法，它们都是可以结合起来构建新模型的模块。

- __易扩展性。__ 新的模块是很容易添加的（作为新的类和函数），现有的模块已经提供了充足的示例。由于能够轻松地创建可以提高表现力的新模块，Keras 更加适合高级研究。

- __基于 Python 实现。__ Keras 没有特定格式的单独配置文件。模型定义在 Python 代码中，这些代码紧凑，易于调试，并且易于扩展。


------------------


## 快速开始：30 秒上手 Keras

Keras 的核心数据结构是 __model__，一种组织网络层的方式。最简单的模型是 [Sequential 顺序模型](/getting-started/sequential-model-guide)，它由多个网络层线性堆叠。对于更复杂的结构，你应该使用 [Keras 函数式 API](/getting-started/functional-api-guide)，它允许构建任意的神经网络图。

`Sequential` 模型如下所示：

```python
from keras.models import Sequential

model = Sequential()
```

可以简单地使用 `.add()` 来堆叠模型：

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

在完成了模型的构建后, 可以使用 `.compile()` 来配置学习过程：

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

如果需要，你还可以进一步地配置你的优化器。Keras 的核心原则是使事情变得相当简单，同时又允许用户在需要的时候能够进行完全的控制（终极的控制是源代码的易扩展性）。

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

现在，你可以批量地在训练数据上进行迭代了：

```python
# x_train 和 y_train 是 Numpy 数组 -- 就像在 Scikit-Learn API 中一样。
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

或者，你可以手动地将批次的数据提供给模型：

```python
model.train_on_batch(x_batch, y_batch)
```

只需一行代码就能评估模型性能：

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

或者对新的数据生成预测：

```python
classes = model.predict(x_test, batch_size=128)
```

构建一个问答系统，一个图像分类模型，一个神经图灵机，或者其他的任何模型，就是这么的快。深度学习背后的思想很简单，那么它们的实现又何必要那么痛苦呢？

有关 Keras 更深入的教程，请查看：

- [开始使用 Sequential 模型](/getting-started/sequential-model-guide)
- [开始使用函数式 API](/getting-started/functional-api-guide)

在代码仓库的 [examples 目录](https://github.com/keras-team/keras/tree/master/examples)中，你会找到更多高级模型：基于记忆网络的问答系统、基于栈式 LSTM 的文本生成等等。


------------------


## 安装指引

在安装 Keras 之前，请安装以下后端引擎之一：TensorFlow，Theano，或者 CNTK。我们推荐 TensorFlow 后端。

- [TensorFlow 安装指引](https://www.tensorflow.org/install/)。
- [Theano 安装指引](http://deeplearning.net/software/theano/install.html#install)。
- [CNTK 安装指引](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine)。

你也可以考虑安装以下**可选依赖**：

- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (如果你计划在 GPU 上运行 Keras，建议安装)。
- HDF5 和 [h5py](http://docs.h5py.org/en/latest/build.html) (如果你需要将 Keras 模型保存到磁盘，则需要这些)。
- [graphviz](https://graphviz.gitlab.io/download/) 和 [pydot](https://github.com/erocarrera/pydot) (用于[可视化工具](https://keras.io/zh/visualization/)绘制模型图)。

然后你就可以安装 Keras 本身了。有两种方法安装 Keras：

- **使用 PyPI 安装 Keras (推荐)：**

```sh
sudo pip install keras
```

如果你使用 virtualenv 虚拟环境, 你可以避免使用 sudo：

```sh
pip install keras
```

- **或者：使用 GitHub 源码安装 Keras：**

首先，使用 `git` 来克隆 Keras：

```sh
git clone https://github.com/keras-team/keras.git
```

然后，`cd` 到 Keras 目录并且运行安装命令：

```sh
cd keras
sudo python setup.py install
```

------------------


## 配置你的 Keras 后端

默认情况下，Keras 将使用 TensorFlow 作为其张量操作库。请[跟随这些指引](https://keras.io/zh/backend/)来配置其他 Keras 后端。

------------------


## 技术支持

你可以提出问题并参与开发讨论：

- [Keras Google group](https://groups.google.com/forum/#!forum/keras-users)。
- [Keras Slack channel](https://kerasteam.slack.com)。 使用 [这个链接](https://keras-slack-autojoin.herokuapp.com/) 向该频道请求邀请函。
- 或者加入 Keras 深度学习交流群，协助文档的翻译工作，群号为 951623081。

你也可以在 [GitHub issues](https://github.com/keras-team/keras/issues) 中发布**漏洞报告和新功能请求**（仅限于此）。注意请先阅读[规范文档](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md)。


------------------


## 为什么取名为 Keras?

Keras (κέρας) 在希腊语中意为 *号角* 。它来自古希腊和拉丁文学中的一个文学形象，首先出现于 *《奥德赛》* 中， 梦神 (_Oneiroi_, singular _Oneiros_) 从这两类人中分离出来：那些用虚幻的景象欺骗人类，通过象牙之门抵达地球之人，以及那些宣告未来即将到来，通过号角之门抵达之人。 它类似于文字寓意，κέρας (号角) / κραίνω (履行)，以及 ἐλέφας (象牙) / ἐλεφαίρομαι (欺骗)。

Keras 最初是作为 ONEIROS 项目（开放式神经电子智能机器人操作系统）研究工作的一部分而开发的。

>_"Oneiroi 超出了我们的理解 - 谁能确定它们讲述了什么故事？并不是所有人都能找到。那里有两扇门，就是通往短暂的 Oneiroi 的通道；一个是用号角制造的，一个是用象牙制造的。穿过尖锐的象牙的 Oneiroi 是诡计多端的，他们带有一些不会实现的信息； 那些穿过抛光的喇叭出来的人背后具有真理，对于看到他们的人来说是完成的。"_ Homer, Odyssey 19. 562 ff (Shewring translation).

------------------
