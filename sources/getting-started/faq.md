# Keras FAQ: 常见问题解答

- [如何引用 Keras?](#how-should-i-cite-keras)
- [如何在 GPU 上运行 Keras?](#how-can-i-run-keras-on-gpu)
- [如何在多 GPU 上运行 Keras 模型？](#how-can-i-run-a-keras-model-on-multiple-gpus)
- ["sample", "batch", "epoch" 分别是什么？](#what-does-sample-batch-epoch-mean)
- [如何保存 Keras 模型？](#how-can-i-save-a-keras-model)
- [为什么训练集误差比测试集的误差高很多？](#why-is-the-training-loss-much-higher-than-the-testing-loss)
- [如何获取中间层的输出？](#how-can-i-obtain-the-output-of-an-intermediate-layer)
- [如何用 Keras 处理超过内存的数据集？](#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory)
- [在验证集的误差不再下降时，如何中断训练？](#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore)
- [验证集划分是如何计算的？](#how-is-the-validation-split-computed)
- [在训练过程中数据是否会混洗？](#is-the-data-shuffled-during-training)
- [如何在每个 epoch 后记录训练集和验证集的误差和准确率？](#how-can-i-record-the-training-validation-loss-accuracy-at-each-epoch)
- [如何「冻结」网络层？](#how-can-i-freeze-keras-layers)
- [如何使用状态 RNNs (stateful RNNs)?](#how-can-i-use-stateful-rnns)
- [如何从 Sequential 模型中移除一个层？](#how-can-i-remove-a-layer-from-a-sequential-model)
- [如何在 Keras 中使用预训练的模型？](#how-can-i-use-pre-trained-models-in-keras)
- [如何在 Keras 中使用 HDF5 输入？](#how-can-i-use-hdf5-inputs-with-keras)
- [Keras 配置文件保存在哪里？](#where-is-the-keras-configuration-file-stored)
- [如何在 Keras 开发过程中获取可复现的结果？](#how-can-i-obtain-reproducible-results-using-keras-during-development)

---

### 如何引用 Keras?

如果 Keras 有助于您的研究，请在你的出版物中引用它。以下是 BibTeX 条目引用的示例：

```
@misc{chollet2015keras,
  title={Keras},
  author={Chollet, Fran\c{c}ois and others},
  year={2015},
  publisher={GitHub},
  howpublished={\url{https://github.com/keras-team/keras}},
}
```

---

### 如何在 GPU 上运行 Keras?

如果你以 TensorFlow 或 CNTK 后端运行，只要检测到任何可用的GPU，那么代码将自动在GPU上运行。

如果你以 Theano 后端运行，则可以使用以下方法之一：

**方法 1**: 使用 Theano flags。
```bash
THEANO_FLAGS=device=gpu,floatX=float32 python my_keras_script.py
```

"gpu" 可能需要根据你的设备标识符（例如gpu0，gpu1等）进行更改。

**方法 2**: 创建 `.theanorc`: [指导教程](http://deeplearning.net/software/theano/library/config.html)

**方法 3**: 在代码的开头手动设置 `theano.config.device`, `theano.config.floatX`：
```python
import theano
theano.config.device = 'gpu'
theano.config.floatX = 'float32'
```

---

### 如何在多 GPU 上运行 Keras 模型?

我们建议使用 TensorFlow 后端。有两种方法可在多个 GPU 上运行单个模型：数据并行和设备并行。

在大多数情况下，你最需要的是数据并行。

#### 数据并行

数据并行包括在每个设备上复制一次目标模型，并使用每个模型副本处理不同部分的输入数据。Keras 有一个内置的实用函数 `keras.utils.multi_gpu_model`，它可以生成任何模型的数据并行版本，在多达 8 个 GPU 上实现准线性加速。

有关更多信息，请参阅 [multi_gpu_model](/utils/#multi_gpu_model) 的文档。这里是一个简单的例子：

```python
from keras.utils import multi_gpu_model

# 将 `model` 复制到 8 个 GPU 上。
# 假定你的机器有 8 个可用的 GPU。
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# 这个 `fit` 调用将分布在 8 个 GPU 上。
# 由于 batch size 为 256，每个 GPU 将处理 32 个样本。
parallel_model.fit(x, y, epochs=20, batch_size=256)
```

#### 设备并行

设备并行性包括在不同设备上运行同一模型的不同部分。对于具有并行体系结构的模型，例如有两个分支的模型，这种方式很合适。

这种并行可以通过使用 TensorFlow device scopes 来实现。这里是一个简单的例子：

```python
# 模型中共享的 LSTM 用于并行编码两个不同的序列
input_a = keras.Input(shape=(140, 256))
input_b = keras.Input(shape=(140, 256))

shared_lstm = keras.layers.LSTM(64)

# 在一个 GPU 上处理第一个序列
with tf.device_scope('/gpu:0'):
    encoded_a = shared_lstm(tweet_a)
# 在另一个 GPU上 处理下一个序列
with tf.device_scope('/gpu:1'):
    encoded_b = shared_lstm(tweet_b)

# 在 CPU 上连接结果
with tf.device_scope('/cpu:0'):
    merged_vector = keras.layers.concatenate([encoded_a, encoded_b],
                                             axis=-1)
```

---

### "sample", "batch", "epoch" 分别是什么？

为了正确地使用 Keras，以下是必须了解和理解的一些常见定义：

- **Sample**: 样本，数据集中的一个元素，一条数据。
  - *例1:* 在卷积神经网络中，一张图像是一个样本。
  - *例2:* 在语音识别模型中，一段音频是一个样本。
- **Batch**: 批，含有 *N* 个样本的集合。每一个 batch 的样本都是独立并行处理的。在训练时，一个 batch 的结果只会用来更新一次模型。
  - 一个 **batch** 的样本通常比单个输入更接近于总体输入数据的分布，batch 越大就越近似。然而，每个 batch 将花费更长的时间来处理，并且仍然只更新模型一次。在推理（评估/预测）时，建议条件允许的情况下选择一个尽可能大的 batch，（因为较大的 batch 通常评估/预测的速度会更快）。 
- **Epoch**: 轮次，通常被定义为 「在整个数据集上的一轮迭代」，用于训练的不同的阶段，这有利于记录和定期评估。
  - 当在 Keras 模型的 `fit` 方法中使用 `evaluation_data` 或 `evaluation_split` 时，评估将在每个 **epoch** 结束时运行。
  - 在 Keras 中，可以添加专门的用于在 epoch 结束时运行的 [callbacks 回调](https://keras.io/zh/callbacks/)。例如学习率变化和模型检查点（保存）。
 
---

### 如何保存 Keras 模型？

#### 保存/加载整个模型（结构 + 权重 + 优化器状态）

*不建议使用 pickle 或 cPickle 来保存 Keras 模型。*

你可以使用 `model.save(filepath)` 将 Keras 模型保存到单个 HDF5 文件中，该文件将包含：

- 模型的结构，允许重新创建模型
- 模型的权重
- 训练配置项（损失函数，优化器）
- 优化器状态，允许准确地从你上次结束的地方继续训练。

你可以使用 `keras.models.load_model(filepath)` 重新实例化模型。`load_model` 还将负责使用保存的训练配置项来编译模型（除非模型从未编译过）。

例子：

```python
from keras.models import load_model

model.save('my_model.h5')  # 创建 HDF5 文件 'my_model.h5'
del model  # 删除现有模型

# 返回一个编译好的模型
# 与之前那个相同
model = load_model('my_model.h5')
```

#### 只保存/加载 模型的结构

如果您只需要保存模型的结构，而非其权重或训练配置项，则可以执行以下操作：

```python
# 保存为 JSON
json_string = model.to_json()

# 保存为 YAML
yaml_string = model.to_yaml()
```

生成的 JSON/YAML 文件是人类可读的，如果需要还可以手动编辑。

你可以从这些数据建立一个新的模型：

```python
# 从 JSON 重建模型：
from keras.models import model_from_json
model = model_from_json(json_string)

# 从 YAML 重建模型：
from keras.models import model_from_yaml
model = model_from_yaml(yaml_string)
```

#### 只保存/加载 模型的权重

如果您只需要 **模型的权重**，可以使用下面的代码以 HDF5 格式进行保存。

请注意，我们首先需要安装 HDF5 和 Python 库 h5py，它们不包含在 Keras 中。

```python
model.save_weights('my_model_weights.h5')
```

假设你有用于实例化模型的代码，则可以将保存的权重加载到具有相同结构的模型中：

```python
model.load_weights('my_model_weights.h5')
```

如果你需要将权重加载到不同的结构（有一些共同层）的模型中，例如微调或迁移学习，则可以按层的名字来加载权重：


```python
model.load_weights('my_model_weights.h5', by_name=True)
```

例如：

```python
"""
假设原始模型如下所示：
    model = Sequential()
    model.add(Dense(2, input_dim=3, name='dense_1'))
    model.add(Dense(3, name='dense_2'))
    ...
    model.save_weights(fname)
"""

# 新模型
model = Sequential()
model.add(Dense(2, input_dim=3, name='dense_1'))  # 将被加载
model.add(Dense(10, name='new_dense'))  # 将不被加载

# 从第一个模型加载权重；只会影响第一层，dense_1
model.load_weights(fname, by_name=True)
```

#### 处理已保存模型中的自定义层（或其他自定义对象）

如果要加载的模型包含自定义层或其他自定义类或函数，则可以通过 `custom_objects` 参数将它们传递给加载机制：

```python
from keras.models import load_model
# 假设你的模型包含一个 AttentionLayer 类的实例
model = load_model('my_model.h5', custom_objects={'AttentionLayer': AttentionLayer})
```

或者，你可以使用 [自定义对象作用域](https://keras.io/zh/utils/#customobjectscope)：

```python
from keras.utils import CustomObjectScope

with CustomObjectScope({'AttentionLayer': AttentionLayer}):
    model = load_model('my_model.h5')
```

自定义对象的处理与 `load_model`, `model_from_json`, `model_from_yaml` 的工作方式相同：

```python
from keras.models import model_from_json
model = model_from_json(json_string, custom_objects={'AttentionLayer': AttentionLayer})
```

---

### 为什么训练误差比测试误差高很多？

Keras 模型有两种模式：训练和测试。正则化机制，如 Dropout 和 L1/L2 权重正则化，在测试时是关闭的。

此外，训练误差是每批训练数据的平均误差。由于你的模型是随着时间而变化的，一个 epoch 中的第一批数据的误差通常比最后一批的要高。另一方面，测试误差是模型在一个 epoch 训练完后计算的，因而误差较小。

---

### 如何获取中间层的输出？

一个简单的方法是创建一个新的模型来输出你所感兴趣的层：

```python
from keras.models import Model

model = ...  # 创建原始模型

layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(data)
```

或者，你也可以构建一个 Keras 函数，该函数将在给定输入的情况下返回某个层的输出，例如：

```python
from keras import backend as K

# 以 Sequential 模型为例
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[3].output])
layer_output = get_3rd_layer_output([x])[0]
```

同样，你可以直接建立一个 Theano 或 TensorFlow 函数。

注意，如果你的模型在训练和测试阶段有不同的行为（例如，使用 `Dropout`, `BatchNormalization` 等），则需要将学习阶段标志传递给你的函数：

```python
get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

# 测试模式 = 0 时的输出
layer_output = get_3rd_layer_output([x, 0])[0]

# 测试模式 = 1 时的输出
layer_output = get_3rd_layer_output([x, 1])[0]
```

---

### 如何用 Keras 处理超过内存的数据集？

你可以使用 `model.train_on_batch(x，y)` 和 `model.test_on_batch(x，y)` 进行批量训练与测试。请参阅 [模型文档](/models/sequential)。

或者，你可以编写一个生成批处理训练数据的生成器，然后使用 `model.fit_generator(data_generator，steps_per_epoch，epochs)` 方法。

你可以在 [CIFAR10 example](https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py) 中找到实践代码。

---

### 在验证集的误差不再下降时，如何中断训练？

你可以使用 `EarlyStopping` 回调函数：

```python
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
```

更多信息请查看 [callbacks 文档](/callbacks)。

---

### 验证集划分是如何计算的？

如果您将 `model.fit` 中的 `validation_split` 参数设置为 0.1，那么使用的验证数据将是最后 10％ 的数据。如果设置为 0.25，就是最后 25% 的数据。注意，在提取分割验证集之前，数据不会被混洗，因此验证集仅仅是传递的输入中最后一个 x％ 的样本。 

所有 epoch 都使用相同的验证集（在同一个 `fit` 中调用）。

---

### 在训练过程中数据是否会混洗？

是的，如果 `model.fit`中的 `shuffle`参数设置为 True（默认值），则训练数据将在每个 epoch 混洗。

验证集永远不会混洗。

---


### 如何在每个 epoch 后记录训练集和验证集的误差和准确率？

`model.fit` 方法返回一个 `History` 回调，它具有包含连续误差的列表和其他度量的 `history` 属性。

```python
hist = model.fit(x, y, validation_split=0.2)
print(hist.history)
```

---

### 如何「冻结」网络层？

「冻结」一个层意味着将其排除在训练之外，即其权重将永远不会更新。这在微调模型或使用固定的词向量进行文本输入中很有用。

您可以将 `trainable` 参数（布尔值）传递给一个层的构造器，以将该层设置为不可训练的：

```python
frozen_layer = Dense(32, trainable=False)
```

另外，可以在实例化之后将网络层的 `trainable` 属性设置为 True 或 False。为了使之生效，在修改 `trainable` 属性之后，需要在模型上调用 `compile()`。这是一个例子：

```python
x = Input(shape=(32,))
layer = Dense(32)
layer.trainable = False
y = layer(x)

frozen_model = Model(x, y)
# 在下面的模型中，训练期间不会更新层的权重
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
trainable_model = Model(x, y)
# 使用这个模型，训练期间 `layer` 的权重将被更新
# (这也会影响上面的模型，因为它使用了同一个网络层实例)
trainable_model.compile(optimizer='rmsprop', loss='mse')

frozen_model.fit(data, labels)  # 这不会更新 `layer` 的权重
trainable_model.fit(data, labels)  # 这会更新 `layer` 的权重
```

---

### 如何使用有状态 RNN (stateful RNNs)?

使 RNN 具有状态意味着每批样品的状态将被重新用作下一批样品的初始状态。

当使用有状态 RNN 时，假定：

- 所有的批次都有相同数量的样本
- 如果 `x1` 和 `x2` 是连续批次的样本，则 `x2[i]` 是 `x1[i]` 的后续序列，对于每个 `i`。

要在 RNN 中使用状态，你需要:

- 通过将 `batch_size` 参数传递给模型的第一层来显式指定你正在使用的批大小。例如，对于 10 个时间步长的 32 样本的 batch，每个时间步长具有 16 个特征，`batch_size = 32`。 
- 在 RNN 层中设置 `stateful = True`。
- 在调用 `fit()` 时指定 `shuffle = False`。

重置累积状态：

- 使用 `model.reset_states()` 来重置模型中所有层的状态
- 使用 `layer.reset_states()` 来重置指定有状态 RNN 层的状态

例子：

```python

x  # 输入数据，尺寸为 (32, 21, 16)
# 将步长为 10 的序列输送到模型中

model = Sequential()
model.add(LSTM(32, input_shape=(10, 16), batch_size=32, stateful=True))
model.add(Dense(16, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 训练网络，根据给定的前 10 个时间步，来预测第 11 个时间步：
model.train_on_batch(x[:, :10, :], np.reshape(x[:, 10, :], (32, 16)))

# 网络的状态已经改变。我们可以提供后续序列：
model.train_on_batch(x[:, 10:20, :], np.reshape(x[:, 20, :], (32, 16)))

# 重置 LSTM 层的状态：
model.reset_states()

# 另一种重置方法：
model.layers[0].reset_states()
```

请注意，`predict`, `fit`, `train_on_batch`, `predict_classes` 等方法 *全部* 都会更新模型中有状态层的状态。这使你不仅可以进行有状态的训练，还可以进行有状态的预测。

---

### 如何从 Sequential 模型中移除一个层？

你可以通过调用 `.pop()` 来删除 `Sequential` 模型中最后添加的层：

```python
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=784))
model.add(Dense(32, activation='relu'))

print(len(model.layers))  # "2"

model.pop()
print(len(model.layers))  # "1"
```

---

### 如何在 Keras 中使用预训练的模型？

我们提供了以下图像分类模型的代码和预训练的权重：

- Xception
- VGG16
- VGG19
- ResNet50
- Inception v3
- Inception-ResNet v2
- MobileNet v1

它们可以使用 `keras.applications` 模块进行导入：

```python
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet

model = VGG16(weights='imagenet', include_top=True)
```

有关一些简单的用法示例，请参阅 [应用模块的文档](/applications)。

有关如何使用此类预训练的模型进行特征提取或微调的详细示例，请参阅 [此博客文章](http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)。

VGG16 模型也是以下几个 Keras 示例脚本的基础：

- [Style transfer](https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py)
- [Feature visualization](https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py)
- [Deep dream](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py)

---

### 如何在 Keras 中使用 HDF5 输入？

你可以使用 `keras.utils.io_utils` 中的 `HDF5Matrix` 类。有关详细信息，请参阅 [HDF5Matrix文档](/utils/#hdf5matrix)。

你也可以直接使用 HDF5 数据集：

```python
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)
```

---

### Keras 配置文件保存在哪里？

所有 Keras 数据存储的默认目录是：

```bash
$HOME/.keras/
```

注意，Windows 用户应该将 `$HOME` 替换为 `％USERPROFILE％`。如果 Keras 无法创建上述目录（例如，由于权限问题），则使用 `/tmp/.keras/` 作为备份。

Keras配置文件是存储在 `$HOME/.keras/keras.json` 中的 JSON 文件。默认的配置文件如下所示：

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

它包含以下字段：

- 图像处理层和实用程序所使用的默认值图像数据格式（`channel_last` 或 `channels_first`）。
- 用于防止在某些操作中被零除的 `epsilon` 模糊因子。
- 默认浮点数据类型。
- 默认后端。详见 [backend 文档](/backend)。

同样，缓存的数据集文件（如使用 `get_file()` 下载的文件）默认存储在 `$HOME/.keras/datasets/` 中。

---

### 如何在 Keras 开发过程中获取可复现的结果？

在模型的开发过程中，能够在一次次的运行中获得可复现的结果，以确定性能的变化是来自模型还是数据集的变化，或者仅仅是一些新的随机样本点带来的结果，有时候是很有用处的。下面的代码片段提供了一个如何获得可复现结果的例子 - 针对 Python 3 环境的 TensorFlow 后端。

```python
import numpy as np
import tensorflow as tf
import random as rn

# 以下是 Python 3.2.3 以上所必需的，
# 为了使某些基于散列的操作可复现。
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926

import os
os.environ['PYTHONHASHSEED'] = '0'

# 以下是 Numpy 在一个明确的初始状态生成固定随机数字所必需的。

np.random.seed(42)

# 以下是 Python 在一个明确的初始状态生成固定随机数字所必需的。

rn.seed(12345)

# 强制 TensorFlow 使用单线程。
# 多线程是结果不可复现的一个潜在的来源。
# 更多详情，见: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# `tf.set_random_seed()` 将会以 TensorFlow 为后端，
# 在一个明确的初始状态下生成固定随机数字。
# 更多详情，见: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# 剩余代码 ...
```
