<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/local.py#L19)</span>
### LocallyConnected1D

```python
keras.layers.LocallyConnected1D(filters, kernel_size, strides=1, padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1D 输入的局部连接层。

`LocallyConnected1D` 层与 `Conv1D` 层的工作方式相同，除了权值不共享外，
也就是说，在输入的每个不同部分应用不同的一组过滤器。

__例子__

```python
# 将长度为 3 的非共享权重 1D 卷积应用于
# 具有 10 个时间步长的序列，并使用 64个 输出滤波器
model = Sequential()
model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
# 现在 model.output_shape == (None, 8, 64)
# 在上面再添加一个新的 conv1d
model.add(LocallyConnected1D(32, 3))
# 现在 model.output_shape == (None, 6, 32)
```

__参数__

- __filters__: 整数，输出空间的维度
（即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者单个整数表示的元组或列表，
指明 1D 卷积窗口的长度。
- __strides__: 一个整数，或者单个整数表示的元组或列表，
指明卷积的步长。
指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: 当前仅支持 `"valid"` (大小写敏感)。
`"same"` 可能会在未来支持。
- __activation__: 要使用的激活函数
(详见 [activations](../activations.md))。
如果你不指定，则不使用激活函数
(即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
(详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
(详见 [initializers](../initializers.md))。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。

__输入尺寸__

3D 张量，尺寸为： `(batch_size, steps, input_dim)`。

__输出尺寸__

3D 张量 ，尺寸为：`(batch_size, new_steps, filters)`，
`steps` 值可能因填充或步长而改变。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/local.py#L182)</span>
### LocallyConnected2D

```python
keras.layers.LocallyConnected2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2D 输入的局部连接层。

`LocallyConnected2D` 层与 `Conv2D` 层的工作方式相同，除了权值不共享外，
也就是说，在输入的每个不同部分应用不同的一组过滤器。

__例子__

```python
# 在 32x32 图像上应用 3x3 非共享权值和64个输出过滤器的卷积
# 数据格式 `data_format="channels_last"`：
model = Sequential()
model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
# 现在 model.output_shape == (None, 30, 30, 64)
# 注意这一层的参数数量为 (30*30)*(3*3*3*64) + (30*30)*64

# 在上面再加一个 3x3 非共享权值和 32 个输出滤波器的卷积：
model.add(LocallyConnected2D(32, (3, 3)))
# 现在 model.output_shape == (None, 28, 28, 32)
```

__参数__

- __filters__: 整数，输出空间的维度
（即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 2 个整数表示的元组或列表，
指明 2D 卷积窗口的宽度和高度。
可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 2 个整数表示的元组或列表，
指明卷积沿宽度和高度方向的步长。
可以是一个整数，为所有空间维度指定相同的值。
- __padding__: 当前仅支持 `"valid"` (大小写敏感)。
`"same"` 可能会在未来支持。
- __data_format__: 字符串，
`channels_last` (默认) 或 `channels_first` 之一。
输入中维度的顺序。
`channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
`channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
找到的 `image_data_format` 值。
如果你从未设置它，将使用 "channels_last"。
- __activation__: 要使用的激活函数
(详见 [activations](../activations.md))。
如果你不指定，则不使用激活函数
(即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器
(详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
(详见 [initializers](../initializers.md))。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。

__输入尺寸__

4D 张量，尺寸为：
`(samples, channels, rows, cols)`，如果 data_format='channels_first'；
或者 4D 张量，尺寸为：
`(samples, rows, cols, channels)`，如果 data_format='channels_last'。

__输出尺寸__

4D 张量，尺寸为：
`(samples, filters, new_rows, new_cols)`，如果 data_format='channels_first'；
或者 4D 张量，尺寸为：
`(samples, new_rows, new_cols, filters)`，如果 data_format='channels_last'。
`rows` 和 `cols` 的值可能因填充而改变。
