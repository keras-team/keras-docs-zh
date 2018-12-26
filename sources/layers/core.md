<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L767)</span>
### Dense

```python
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

就是你常用的的全连接层。

`Dense` 实现以下操作：
`output = activation(dot(input, kernel) + bias)`
其中 `activation` 是按逐个元素计算的激活函数，`kernel`
是由网络层创建的权值矩阵，以及 `bias` 是其创建的偏置向量
(只在 `use_bias` 为 `True` 时才有用)。

- __注意__: 如果该层的输入的秩大于2，那么它首先被展平然后
再计算与 `kernel` 的点乘。

__例__


```python
# 作为 Sequential 模型的第一层
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# 现在模型就会以尺寸为 (*, 16) 的数组作为输入，
# 其输出数组的尺寸为 (*, 32)

# 在第一层之后，你就不再需要指定输入的尺寸了：
model.add(Dense(32))
```

__参数__

- __units__: 正整数，输出空间维度。
- __activation__: 激活函数
(详见 [activations](../activations.md))。
若不指定，则不使用激活函数
(即，「线性」激活: `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
(详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
(see [initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向的的正则化函数
(详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层的输出的正则化函数
(它的 "activation")。
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。

__输入尺寸__

nD 张量，尺寸: `(batch_size, ..., input_dim)`。
最常见的情况是一个尺寸为 `(batch_size, input_dim)`
的 2D 输入。

__输出尺寸__

nD 张量，尺寸: `(batch_size, ..., units)`。
例如，对于尺寸为 `(batch_size, input_dim)` 的 2D 输入，
输出的尺寸为 `(batch_size, units)`。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L276)</span>
### Activation

```python
keras.layers.Activation(activation)
```

将激活函数应用于输出。

__参数__

- __activation__: 要使用的激活函数的名称
(详见: [activations](../activations.md))，
或者选择一个 Theano 或 TensorFlow 操作。

__输入尺寸__

任意尺寸。
当使用此层作为模型中的第一层时，
使用参数 `input_shape`
（整数元组，不包括样本数的轴）。


__输出尺寸__

与输入相同。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L80)</span>
### Dropout

```python
keras.layers.Dropout(rate, noise_shape=None, seed=None)
```

将 Dropout 应用于输入。

Dropout 包括在训练中每次更新时，
将输入单元的按比率随机设置为 0，
这有助于防止过拟合。

__参数__

- __rate__: 在 0 和 1 之间浮动。需要丢弃的输入比例。
- __noise_shape__: 1D 整数张量，
表示将与输入相乘的二进制 dropout 掩层的形状。
例如，如果你的输入尺寸为
`(batch_size, timesteps, features)`，然后
你希望 dropout 掩层在所有时间步都是一样的，
你可以使用 `noise_shape=(batch_size, 1, features)`。
- __seed__: 一个作为随机种子的 Python 整数。

__参考文献__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L461)</span>
### Flatten

```python
keras.layers.Flatten(data_format=None)
```

将输入展平。不影响批量大小。

__参数__

- __data_format__：一个字符串，其值为 `channels_last`（默认值）或者 `channels_first`。它表明输入的维度的顺序。此参数的目的是当模型从一种数据格式切换到另一种数据格式时保留权重顺序。`channels_last` 对应着尺寸为 `(batch, ..., channels)` 的输入，而 `channels_first` 对应着尺寸为 `(batch, channels, ...)` 的输入。默认为 `image_data_format` 的值，你可以在 Keras 的配置文件 `~/.keras/keras.json` 中找到它。如果你从未设置过它，那么它将是 `channels_last`

__例__


```python
model = Sequential()
model.add(Conv2D(64, (3, 3),
                 input_shape=(3, 32, 32), padding='same',))
# 现在：model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# 现在：model.output_shape == (None, 65536)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/engine/input_layer.py#L114)</span>
### Input

```python
keras.engine.input_layer.Input()
```

`Input()` 用于实例化 Keras 张量。

Keras 张量是底层后端(Theano, TensorFlow 或 CNTK)
的张量对象，我们增加了一些特性，使得能够通过了解模型的输入
和输出来构建 Keras 模型。

例如，如果 a, b 和 c 都是 Keras 张量，
那么以下操作是可行的：
`model = Model(input=[a, b], output=c)`

添加的 Keras 属性是：
- __`_keras_shape`__: 通过 Keras端的尺寸推理
进行传播的整数尺寸元组。
- __`_keras_history`__: 应用于张量的最后一层。
整个网络层计算图可以递归地从该层中检索。

__参数__

- __shape__: 一个尺寸元组（整数），不包含批量大小。
例如，`shape=(32,)` 表明期望的输入是按批次的 32 维向量。
- __batch_shape__: 一个尺寸元组（整数），包含批量大小。
例如，`batch_shape=(10, 32)` 表明期望的输入是 10 个 32 维向量。
`batch_shape=(None, 32)` 表明任意批次大小的 32 维向量。
- __name__: 一个可选的层的名称的字符串。
在一个模型中应该是唯一的（不可以重用一个名字两次）。
如未提供，将自动生成。
- __dtype__: 输入所期望的数据类型，字符串表示
(`float32`, `float64`, `int32`...)
- __sparse__: 一个布尔值，指明需要创建的占位符是否是稀疏的。
- __tensor__: 可选的可封装到 `Input` 层的现有张量。
如果设定了，那么这个层将不会创建占位符张量。

__返回__

一个张量。

__例__


```python
# 这是 Keras 中的一个逻辑回归
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L310)</span>
### Reshape

```python
keras.layers.Reshape(target_shape)
```

将输入重新调整为特定的尺寸。

__参数__

- __target_shape__: 目标尺寸。整数元组。
不包含表示批量的轴。

__输入尺寸__

任意，尽管输入尺寸中的所有维度必须是固定的。
当使用此层作为模型中的第一层时，
使用参数 `input_shape`
（整数元组，不包括样本数的轴）。

__输出尺寸__

`(batch_size,) + target_shape`

__例__


```python
# 作为 Sequential 模型的第一层
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# 现在：model.output_shape == (None, 3, 4)
# 注意： `None` 是批表示的维度

# 作为 Sequential 模型的中间层
model.add(Reshape((6, 2)))
# 现在： model.output_shape == (None, 6, 2)

# 还支持使用 `-1` 表示维度的尺寸推断
model.add(Reshape((-1, 2, 2)))
# 现在： model.output_shape == (None, 3, 2, 2)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L409)</span>
### Permute

```python
keras.layers.Permute(dims)
```

根据给定的模式置换输入的维度。

在某些场景下很有用，例如将 RNN 和 CNN 连接在一起。

__例__


```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# 现在： model.output_shape == (None, 64, 10)
# 注意： `None` 是批表示的维度
```

__参数__

- __dims__: 整数元组。置换模式，不包含样本维度。
索引从 1 开始。
例如, `(2, 1)` 置换输入的第一和第二个维度。

__输入尺寸__

任意。当使用此层作为模型中的第一层时，
使用参数 `input_shape`
（整数元组，不包括样本数的轴）。

__输出尺寸__

与输入尺寸相同，但是维度根据指定的模式重新排列。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L523)</span>
### RepeatVector

```python
keras.layers.RepeatVector(n)
```

将输入重复 n 次。

__例__


```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# 现在： model.output_shape == (None, 32)
# 注意： `None` 是批表示的维度

model.add(RepeatVector(3))
# 现在： model.output_shape == (None, 3, 32)
```

__参数__

- __n__: 整数，重复次数。

__输入尺寸__

2D 张量，尺寸为 `(num_samples, features)`。

__输出尺寸__

3D 张量，尺寸为 `(num_samples, n, features)`。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L565)</span>
### Lambda

```python
keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)
```

将任意表达式封装为 `Layer` 对象。

__例__


```python
# 添加一个 x -> x^2 层
model.add(Lambda(lambda x: x ** 2))
```
```python
# 添加一个网络层，返回输入的正数部分
# 与负数部分的反面的连接

def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)

model.add(Lambda(antirectifier,
                 output_shape=antirectifier_output_shape))
```

__参数__

- __function__: 需要封装的函数。
将输入张量作为第一个参数。
- __output_shape__: 预期的函数输出尺寸。
    只在使用 Theano 时有意义。
    可以是元组或者函数。
    如果是元组，它只指定第一个维度；
        样本维度假设与输入相同：
        `output_shape = (input_shape[0], ) + output_shape`
        或者，输入是 `None` 且样本维度也是 `None`：
        `output_shape = (None, ) + output_shape`
        如果是函数，它指定整个尺寸为输入尺寸的一个函数：
        `output_shape = f(input_shape)`
- __arguments__: 可选的需要传递给函数的关键字参数。

__输入尺寸__

任意。当使用此层作为模型中的第一层时，
使用参数 `input_shape`
（整数元组，不包括样本数的轴）。

__输出尺寸__

由 `output_shape` 参数指定
(或者在使用 TensorFlow 时，自动推理得到)。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L911)</span>
### ActivityRegularization

```python
keras.layers.ActivityRegularization(l1=0.0, l2=0.0)
```

网络层，对基于代价函数的输入活动应用一个更新。

__参数__

- __l1__: L1 正则化因子 (正数浮点型)。
- __l2__: L2 正则化因子 (正数浮点型)。

__输入尺寸__

任意。当使用此层作为模型中的第一层时，
使用参数 `input_shape`
（整数元组，不包括样本数的轴）。

__输出尺寸__

与输入相同。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L28)</span>
### Masking

```python
keras.layers.Masking(mask_value=0.0)
```

使用覆盖值覆盖序列，以跳过时间步。

对于输入张量的每一个时间步（张量的第一个维度），
如果所有时间步中输入张量的值与 `mask_value` 相等，
那么这个时间步将在所有下游层被覆盖 (跳过)
（只要它们支持覆盖）。

如果任何下游层不支持覆盖但仍然收到此类输入覆盖信息，会引发异常。

__例__


考虑将要喂入一个 LSTM 层的 Numpy 矩阵 `x`，
尺寸为 `(samples, timesteps, features)`。
你想要覆盖时间步 #3 和 #5，因为你缺乏这几个
时间步的数据。你可以：

- 设置 `x[:, 3, :] = 0.` 以及 `x[:, 5, :] = 0.`
- 在 LSTM 层之前，插入一个 `mask_value=0` 的 `Masking` 层：

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```
---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L140)</span>
### SpatialDropout1D

```python
keras.layers.SpatialDropout1D(rate)
```

Dropout 的 Spatial 1D 版本

此版本的功能与 Dropout 相同，但它会丢弃整个 1D 的特征图而不是丢弃单个元素。如果特征图中相邻的帧是强相关的（通常是靠前的卷积层中的情况），那么常规的 dropout 将无法使激活正则化，且导致有效的学习速率降低。在这种情况下，SpatialDropout1D 将有助于提高特征图之间的独立性，应该使用它来代替 Dropout。

__参数__

- __rate__: 0 到 1 之间的浮点数。需要丢弃的输入比例。

__输入尺寸__

3D 张量，尺寸为：`(samples, timesteps, channels)`

__输出尺寸__

与输入相同。

__参考文献__

- [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L177)</span>
### SpatialDropout2D

```python
keras.layers.SpatialDropout2D(rate, data_format=None)
```

Dropout 的 Spatial 2D 版本

此版本的功能与 Dropout 相同，但它会丢弃整个 2D 的特征图而不是丢弃单个元素。如果特征图中相邻的像素是强相关的（通常是靠前的卷积层中的情况），那么常规的 dropout 将无法使激活正则化，且导致有效的学习速率降低。在这种情况下，SpatialDropout2D 将有助于提高特征图之间的独立性，应该使用它来代替 dropout。

__参数__

- __rate__: 0 到 1 之间的浮点数。需要丢弃的输入比例。
- __data_format__：`channels_first` 或者 `channels_last`。在 `channels_first`  模式中，通道维度（即深度）位于索引 1，在 `channels_last` 模式中，通道维度位于索引 3。默认为 `image_data_format` 的值，你可以在 Keras 的配置文件 `~/.keras/keras.json` 中找到它。如果你从未设置过它，那么它将是 `channels_last`

__输入尺寸__

4D 张量，如果 data_format＝`channels_first`，尺寸为 `(samples, channels, rows, cols)`，如果 data_format＝`channels_last`，尺寸为 `(samples, rows, cols, channels)`

__输出尺寸__

与输入相同。

__参考文献__

- [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L227)</span>
### SpatialDropout3D

```python
keras.layers.SpatialDropout3D(rate, data_format=None)
```

Dropout 的 Spatial 3D 版本

此版本的功能与 Dropout 相同，但它会丢弃整个 3D 的特征图而不是丢弃单个元素。如果特征图中相邻的体素是强相关的（通常是靠前的卷积层中的情况），那么常规的 dropout 将无法使激活正则化，且导致有效的学习速率降低。在这种情况下，SpatialDropout3D 将有助于提高特征图之间的独立性，应该使用它来代替 dropout。

__参数__

- __rate__: 0 到 1 之间的浮点数。需要丢弃的输入比例。
- __data_format__：`channels_first` 或者 `channels_last`。在 `channels_first`  模式中，通道维度（即深度）位于索引 1，在 `channels_last` 模式中，通道维度位于索引 4。默认为 `image_data_format` 的值，你可以在 Keras 的配置文件 `~/.keras/keras.json` 中找到它。如果你从未设置过它，那么它将是 `channels_last`

__输入尺寸__

5D 张量，如果 data_format＝`channels_first`，尺寸为 `(samples, channels, dim1, dim2, dim3)`，如果 data_format＝`channels_last`，尺寸为 `(samples, dim1, dim2, dim3, channels)`

__输出尺寸__

与输入相同。

__参考文献__

- [Efficient Object Localization Using Convolutional Networks](https://arxiv.org/abs/1411.4280)
