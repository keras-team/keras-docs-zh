<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L241)</span>
### Conv1D

```python
keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1D 卷积层 (例如时序卷积)。

该层创建了一个卷积核，该卷积核以
单个空间（或时间）维上的层输入进行卷积，
以生成输出张量。
如果 `use_bias` 为 True，
则会创建一个偏置向量并将其添加到输出中。
最后，如果 `activation` 
不是 `None`，它也会应用于输出。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数（整数元组或 `None`），例如，
`(10, 128)` 表示 10 个 128 维的向量组成的向量序列，
`(None, 128)` 表示 128 维的向量组成的变长序列。

__参数__

- __filters__: 整数，输出空间的维度
    （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者单个整数表示的元组或列表，
    指明 1D 卷积窗口的长度。
- __strides__: 一个整数，或者单个整数表示的元组或列表，
    指明卷积的步长。
    指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"`, `"causal"` 或 `"same"` 之一 (大小写敏感)
    `"valid"` 表示「不填充」。
    `"same"` 表示填充输入以使输出具有与原始输入相同的长度。
    `"causal"` 表示因果（膨胀）卷积，
    例如，`output[t]` 不依赖于 `input[t+1:]`，
    在模型不应违反时间顺序的时间数据建模时非常有用。
    详见 [WaveNet: A Generative Model for Raw Audio, section 2.1](https://arxiv.org/abs/1609.03499)。
- __data_format__: 字符串,
    `"channels_last"` (默认) 或 `"channels_first"` 之一。输入的各个维度顺序。
    `"channels_last"` 对应输入尺寸为 `(batch, steps, channels)`
    (Keras 中时序数据的默认格式)
    而 `"channels_first"` 对应输入尺寸为 `(batch, channels, steps)`。
- __dilation_rate__: 一个整数，或者单个整数表示的元组或列表，指定用于膨胀卷积的膨胀率。
    当前，指定任何 `dilation_rate` 值 != 1 与指定 stride 值 != 1 两者不兼容。
- __activation__: 要使用的激活函数
    (详见 [activations](../activations.md))。
    如未指定，则不使用激活函数
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

3D 张量 ，尺寸为 `(batch_size, steps, input_dim)`。

__输出尺寸__

3D 张量，尺寸为 `(batch_size, new_steps, filters)`。
由于填充或窗口按步长滑动，`steps` 值可能已更改。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L367)</span>
### Conv2D

```python
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2D 卷积层 (例如对图像的空间卷积)。

该层创建了一个卷积核，
该卷积核对层输入进行卷积，
以生成输出张量。
如果 `use_bias` 为 True，
则会创建一个偏置向量并将其添加到输出中。
最后，如果 `activation` 
不是 `None`，它也会应用于输出。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数
（整数元组，不包含样本表示的轴），例如，
`input_shape=(128, 128, 3)` 表示 128x128 RGB 图像，
在 `data_format="channels_last"` 时。

__参数__

- __filters__: 整数，输出空间的维度
    （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 2 个整数表示的元组或列表，
    指明 2D 卷积窗口的宽度和高度。
    可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 2 个整数表示的元组或列表，
    指明卷积沿宽度和高度方向的步长。
    可以是一个整数，为所有空间维度指定相同的值。
    指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __data_format__: 字符串，
    `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
    `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
    `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
    它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
    找到的 `image_data_format` 值。
    如果你从未设置它，将使用 `channels_last`。
- __dilation_rate__: 一个整数或 2 个整数的元组或列表，
    指定膨胀卷积的膨胀率。
    可以是一个整数，为所有空间维度指定相同的值。
    当前，指定任何 `dilation_rate` 值 != 1 与
    指定 stride 值 != 1 两者不兼容。
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

- 如果 data_format='channels_first'，
输入 4D 张量，尺寸为 `(samples, channels, rows, cols)`。
- 如果 data_format='channels_last'，
输入 4D 张量，尺寸为 `(samples, rows, cols, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
输出 4D 张量，尺寸为 `(samples, filters, new_rows, new_cols)`。
- 如果 data_format='channels_last'，
输出 4D 张量，尺寸为 `(samples, new_rows, new_cols, filters)`。

由于填充的原因， `rows` 和 `cols` 值可能已更改。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1420)</span>
### SeparableConv1D

```python
keras.layers.SeparableConv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

深度方向的可分离 1D 卷积。

可分离的卷积的操作包括，首先执行深度方向的空间卷积
（分别作用于每个输入通道），紧接一个将所得输出通道
混合在一起的逐点卷积。`depth_multiplier` 参数控
制深度步骤中每个输入通道生成多少个输出通道。

直观地说，可分离的卷积可以理解为一种将卷积核分解成
两个较小的卷积核的方法，或者作为 Inception 块的
一个极端版本。

__参数__

- __filters__: 整数，输出空间的维度
    （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者单个整数表示的元组或列表，
    指明 1D 卷积窗口的长度。
- __strides__: 一个整数，或者单个整数表示的元组或列表，
    指明卷积的步长。
    指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __data_format__: 字符串，
    `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
    `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
    `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
    它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
    找到的 `image_data_format` 值。
    如果你从未设置它，将使用「channels_last」。
- __dilation_rate__: 一个整数，或者单个整数表示的元组或列表，
    为使用扩张（空洞）卷积指明扩张率。
    目前，指定任何 `dilation_rate` 值 != 1 与指定任何 `stride` 值 != 1 两者不兼容。
- __depth_multiplier__: 每个输入通道的深度方向卷积输出通道的数量。
    深度方向卷积输出通道的总数将等于 `filterss_in * depth_multiplier`。
- __activation__: 要使用的激活函数
    (详见 [activations](../activations.md))。
    如果你不指定，则不使用激活函数
    (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __depthwise_initializer__: 运用到深度方向的核矩阵的初始化器
    (详见 [initializers](../initializers.md))。
- __pointwise_initializer__: 运用到逐点核矩阵的初始化器
    (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
    (详见 [initializers](../initializers.md))。
- __depthwise_regularizer__: 运用到深度方向的核矩阵的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __pointwise_regularizer__: 运用到逐点核矩阵的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __depthwise_constraint__: 运用到深度方向的核矩阵的约束函数
    (详见 [constraints](../constraints.md))。
- __pointwise_constraint__: 运用到逐点核矩阵的约束函数
    (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
    (详见 [constraints](../constraints.md))。

__输入尺寸__

- 如果 data_format='channels_first'，
输入 3D 张量，尺寸为 `(batch, channels, steps)`。
- 如果 data_format='channels_last'，
输入 3D 张量，尺寸为 `(batch, steps, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
输出 3D 张量，尺寸为 `(batch, filters, new_steps)`。
- 如果 data_format='channels_last'，
输出 3D 张量，尺寸为 `(batch, new_steps, filters)`。

由于填充的原因， `new_steps` 值可能已更改。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1552)</span>
### SeparableConv2D

```python
keras.layers.SeparableConv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

深度方向的可分离 2D 卷积。

可分离的卷积的操作包括，首先执行深度方向的空间卷积
（分别作用于每个输入通道），紧接一个将所得输出通道
混合在一起的逐点卷积。`depth_multiplier` 参数控
制深度步骤中每个输入通道生成多少个输出通道。

直观地说，可分离的卷积可以理解为一种将卷积核分解成
两个较小的卷积核的方法，或者作为 Inception 块的
一个极端版本。

__参数__

- __filters__: 整数，输出空间的维度
    （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 2 个整数表示的元组或列表，
    指明 2D 卷积窗口的高度和宽度。
    可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 2 个整数表示的元组或列表，
    指明卷积沿高度和宽度方向的步长。
    可以是一个整数，为所有空间维度指定相同的值。
    指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __data_format__: 字符串，
    `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
    `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
    `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
    它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
    找到的 `image_data_format` 值。
    如果你从未设置它，将使用「channels_last」。
- __dilation_rate__: 一个整数，或者 2 个整数表示的元组或列表，
    为使用扩张（空洞）卷积指明扩张率。
    目前，指定任何 `dilation_rate` 值 != 1 与指定任何 `stride` 值 != 1 两者不兼容。
- __depth_multiplier__: 每个输入通道的深度方向卷积输出通道的数量。
    深度方向卷积输出通道的总数将等于 `filterss_in * depth_multiplier`。
- __activation__: 要使用的激活函数
    (详见 [activations](../activations.md))。
    如果你不指定，则不使用激活函数
    (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __depthwise_initializer__: 运用到深度方向的核矩阵的初始化器
    详见 [initializers](../initializers.md))。
- __pointwise_initializer__: 运用到逐点核矩阵的初始化器
    (详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
    (详见 [initializers](../initializers.md))。
- __depthwise_regularizer__: 运用到深度方向的核矩阵的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __pointwise_regularizer__: 运用到逐点核矩阵的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __depthwise_constraint__: 运用到深度方向的核矩阵的约束函数
    (详见 [constraints](../constraints.md))。
- __pointwise_constraint__: 运用到逐点核矩阵的约束函数
    (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
    (详见 [constraints](../constraints.md))。

__输入尺寸__

- 如果 data_format='channels_first'，
输入 4D 张量，尺寸为 `(batch, channels, rows, cols)`。
- 如果 data_format='channels_last'，
输入 4D 张量，尺寸为 `(batch, rows, cols, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
输出 4D 张量，尺寸为 `(batch, filters, new_rows, new_cols)`。
- 如果 data_format='channels_last'，
输出 4D 张量，尺寸为 `(batch, new_rows, new_cols, filters)`。

由于填充的原因， `rows` 和 `cols` 值可能已更改。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1693)</span>
### DepthwiseConv2D

```python
keras.layers.DepthwiseConv2D(kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, bias_constraint=None)
```

深度可分离 2D 卷积。

深度可分离卷积包括仅执行深度空间卷积中的第一步（其分别作用于每个输入通道）。
`depth_multiplier` 参数控制深度步骤中每个输入通道生成多少个输出通道。

__Arguments__

- __kernel_size__: 一个整数，或者 2 个整数表示的元组或列表，
    指明 2D 卷积窗口的高度和宽度。
    可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 2 个整数表示的元组或列表，
    指明卷积沿高度和宽度方向的步长。
    可以是一个整数，为所有空间维度指定相同的值。
    指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __depth_multiplier__: 每个输入通道的深度方向卷积输出通道的数量。
    深度方向卷积输出通道的总数将等于 `filterss_in * depth_multiplier`。
- __data_format__: 字符串，
    `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
    `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
    `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
    它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
    找到的 `image_data_format` 值。
    如果你从未设置它，将使用「channels_last」。
- __activation__: 要使用的激活函数
    (详见 [activations](../activations.md))。
    如果你不指定，则不使用激活函数
    (即线性激活： `a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __depthwise_initializer__: 运用到深度方向的核矩阵的初始化器
    详见 [initializers](../initializers.md))。
- __bias_initializer__: 偏置向量的初始化器
    (详见 [initializers](../initializers.md))。
- __depthwise_regularizer__: 运用到深度方向的核矩阵的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
    (详见 [regularizer](../regularizers.md))。
- __depthwise_constraint__: 运用到深度方向的核矩阵的约束函数
    (详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
    (详见 [constraints](../constraints.md))。

__输入尺寸__

- 如果 data_format='channels_first'，
输入 4D 张量，尺寸为 `(batch, channels, rows, cols)`。
- 如果 data_format='channels_last'，
输入 4D 张量，尺寸为 `(batch, rows, cols, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
输出 4D 张量，尺寸为 `(batch, filters, new_rows, new_cols)`。
- 如果 data_format='channels_last'，
输出 4D 张量，尺寸为 `(batch, new_rows, new_cols, filters)`。

由于填充的原因， `rows` 和 `cols` 值可能已更改。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L627)</span>
### Conv2DTranspose

```python
keras.layers.Conv2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

转置卷积层 (有时被成为反卷积)。

对转置卷积的需求一般来自希望使用
与正常卷积相反方向的变换，
即，将具有卷积输出尺寸的东西
转换为具有卷积输入尺寸的东西，
同时保持与所述卷积相容的连通性模式。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数
（整数元组，不包含样本表示的轴），例如，
`input_shape=(128, 128, 3)` 表示 128x128 RGB 图像，
在 `data_format="channels_last"` 时。

__参数__

- __filters__: 整数，输出空间的维度
    （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 2 个整数表示的元组或列表，
    指明 2D 卷积窗口的高度和宽度。
    可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 2 个整数表示的元组或列表，
    指明卷积沿高度和宽度方向的步长。
    可以是一个整数，为所有空间维度指定相同的值。
    指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __output_padding__: 一个整数，或者 2 个整数表示的元组或列表，
    指定沿输出张量的高度和宽度的填充量。
    可以是单个整数，以指定所有空间维度的相同值。
    沿给定维度的输出填充量必须低于沿同一维度的步长。
    如果设置为 `None` (默认), 输出尺寸将自动推理出来。
- __data_format__: 字符串，
    `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
    `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`，
    `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。
    它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
    找到的 `image_data_format` 值。
    如果你从未设置它，将使用 "channels_last"。
- __dilation_rate__: 一个整数或 2 个整数的元组或列表，
    指定膨胀卷积的膨胀率。
    可以是一个整数，为所有空间维度指定相同的值。
    当前，指定任何 `dilation_rate` 值 != 1 与
    指定 stride 值 != 1 两者不兼容。
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

- 如果 data_format='channels_first'，
输入 4D 张量，尺寸为 `(batch, channels, rows, cols)`。
- 如果 data_format='channels_last'，
输入 4D 张量，尺寸为 `(batch, rows, cols, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
输出 4D 张量，尺寸为 `(batch, filters, new_rows, new_cols)`。
- 如果 data_format='channels_last'，
输出 4D 张量，尺寸为 `(batch, new_rows, new_cols, filters)`。

由于填充的原因， `rows` 和 `cols` 值可能已更改。

如果指定了 `output_padding`:

```python
new_rows = ((rows - 1) * strides[0] + kernel_size[0]
            - 2 * padding[0] + output_padding[0])
new_cols = ((cols - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
```

__参考文献__

- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L498)</span>
### Conv3D

```python
keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

3D 卷积层 (例如立体空间卷积)。

该层创建了一个卷积核，
该卷积核对层输入进行卷积，
以生成输出张量。
如果 `use_bias` 为 True，
则会创建一个偏置向量并将其添加到输出中。
最后，如果 `activation` 
不是 `None`，它也会应用于输出。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数
（整数元组，不包含样本表示的轴），例如，
`input_shape=(128, 128, 128, 1)` 表示 128x128x128 的单通道立体，
在 `data_format="channels_last"` 时。

__参数__

- __filters__: 整数，输出空间的维度
    （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 3 个整数表示的元组或列表，
    指明 3D 卷积窗口的深度、高度和宽度。
    可以是一个整数，为所有空间维度指定相同的值。
- __strides__: 一个整数，或者 3 个整数表示的元组或列表，
    指明卷积沿每一个空间维度的步长。
    可以是一个整数，为所有空间维度指定相同的步长值。
    指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __data_format__: 字符串，
    `channels_last` (默认) 或 `channels_first` 之一，
    表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`，
    `channels_first` 对应输入尺寸为 
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`。
    它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
    找到的 `image_data_format` 值。
    如果你从未设置它，将使用 "channels_last"。
- __dilation_rate__: 一个整数或 3 个整数的元组或列表，
    指定膨胀卷积的膨胀率。
    可以是一个整数，为所有空间维度指定相同的值。
    当前，指定任何 `dilation_rate` 值 != 1 与
    指定 stride 值 != 1 两者不兼容。
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

- 如果 data_format='channels_first'，
输入 5D 张量，尺寸为 `(samples, channels, conv_dim1, conv_dim2, conv_dim3)`。
- 如果 data_format='channels_last'，
输入 5D 张量，尺寸为 `(samples, conv_dim1, conv_dim2, conv_dim3, channels)`。

__输出尺寸__

- 如果 data_format='channels_first'，
输出 5D 张量，尺寸为 `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)`。
- 如果 data_format='channels_last'，
输出 5D 张量，尺寸为 `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)`。

由于填充的原因， `new_conv_dim1`, `new_conv_dim2` 和 `new_conv_dim3` 值可能已更改。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L900)</span>
### Conv3DTranspose

```python
keras.layers.Conv3DTranspose(filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None, data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

转置卷积层 (有时被成为反卷积)。

对转置卷积的需求一般来自希望使用
与正常卷积相反方向的变换，
即，将具有卷积输出尺寸的东西
转换为具有卷积输入尺寸的东西，
同时保持与所述卷积相容的连通性模式。

当使用该层作为模型第一层时，需要提供 `input_shape` 参数
（整数元组，不包含样本表示的轴），例如，
`input_shape=(128, 128, 128, 3)` 表示尺寸 128x128x128 的 3 通道立体，
在 `data_format="channels_last"` 时。

__参数__

- __filters__: 整数，输出空间的维度 
    （即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 3 个整数表示的元组或列表，
    指明 3D 卷积窗口的深度、高度和宽度。
    可以是一个整数，为所有空间维度指定相同的值。 
- __strides__: 一个整数，或者 3 个整数表示的元组或列表，
    指明沿深度、高度和宽度方向的步长。
    可以是一个整数，为所有空间维度指定相同的值。
    指定任何 `stride` 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` (大小写敏感)。
- __output_padding__: 一个整数，或者 3 个整数表示的元组或列表，
    指定沿输出张量的高度和宽度的填充量。
    可以是单个整数，以指定所有空间维度的相同值。
    沿给定维度的输出填充量必须低于沿同一维度的步长。
    如果设置为 `None` (默认), 输出尺寸将自动推理出来。
- __data_format__: 字符串，
    `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。
    `channels_last` 对应输入尺寸为 `(batch, depth, height, width, channels)`，
    `channels_first` 对应输入尺寸为 `(batch, channels, depth, height, width)`。
    它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
    找到的 `image_data_format` 值。
    如果你从未设置它，将使用「channels_last」。
- __dilation_rate__: 一个整数或 3 个整数的元组或列表，
    指定膨胀卷积的膨胀率。
    可以是一个整数，为所有空间维度指定相同的值。
    当前，指定任何 `dilation_rate` 值 != 1 与
    指定 stride 值 != 1 两者不兼容。
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

如果 data_format='channels_first'， 输入 5D 张量，尺寸为
`(batch, channels, depth, rows, cols)`，
如果 data_format='channels_last'， 输入 5D 张量，尺寸为
`(batch, depth, rows, cols, channels)`。

__Output shape__

如果 data_format='channels_first'， 输出 5D 张量，尺寸为
`(batch, filters, new_depth, new_rows, new_cols)`，
如果 data_format='channels_last'， 输出 5D 张量，尺寸为
`(batch, new_depth, new_rows, new_cols, filters)`。

`depth` 和 `rows` 和 `cols` 可能因为填充而改变。
如果指定了 `output_padding`：

```python
new_depth = ((depth - 1) * strides[0] + kernel_size[0]
             - 2 * padding[0] + output_padding[0])
new_rows = ((rows - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
new_cols = ((cols - 1) * strides[2] + kernel_size[2]
            - 2 * padding[2] + output_padding[2])
```

__参考文献__

- [A guide to convolution arithmetic for deep learning]
(https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks]
(http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2375)</span>
### Cropping1D

```python
keras.layers.Cropping1D(cropping=(1, 1))
```

1D 输入的裁剪层（例如时间序列）。

它沿着时间维度（第 1 个轴）裁剪。

__参数__

- __cropping__: 整数或整数元组（长度为 2）。
    在裁剪维度（第 1 个轴）的开始和结束位置
    应该裁剪多少个单位。
    如果只提供了一个整数，那么这两个位置将使用
    相同的值。

__输入尺寸__

3D 张量，尺寸为 `(batch, axis_to_crop, features)`。

__输出尺寸__

3D 张量，尺寸为 `(batch, cropped_axis, features)`。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2407)</span>
### Cropping2D

```python
keras.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```

2D 输入的裁剪层（例如图像）。

它沿着空间维度裁剪，即宽度和高度。

__参数__

- __cropping__: 整数，或 2 个整数的元组，或 2 个整数的 2 个元组。
    - 如果为整数： 将对宽度和高度应用相同的对称裁剪。
    - 如果为 2 个整数的元组：
        解释为对高度和宽度的两个不同的对称裁剪值：
        `(symmetric_height_crop, symmetric_width_crop)`。
    - 如果为 2 个整数的 2 个元组：
        解释为 `((top_crop, bottom_crop), (left_crop, right_crop))`。
- __data_format__: 字符串，
`channels_last` (默认) 或 `channels_first` 之一，
表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
`(batch, height, width, channels)`，
`channels_first` 对应输入尺寸为 
`(batch, channels, height, width)`。
它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
找到的 `image_data_format` 值。
如果你从未设置它，将使用 "channels_last"。


__输出尺寸__

- 如果 data_format='channels_last'，
输出 4D 张量，尺寸为 `(batch, rows, cols, channels)`。
- 如果 data_format='channels_first'，
输出 4D 张量，尺寸为 `(batch, channels, rows, cols)`。


由于填充的原因， `rows` 和 `cols` 值可能已更改。

__输入尺寸__

- 如果 `data_format` 为 `"channels_last"`，
输入 4D 张量，尺寸为 `(batch, cropped_rows, cropped_cols, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
输入 4D 张量，尺寸为 `(batch, channels, cropped_rows, cropped_cols)`。

__输出尺寸__

- 如果 `data_format` 为 `"channels_last"`，
输出 4D 张量，尺寸为 `(batch, cropped_rows, cropped_cols, channels)`
- 如果 `data_format` 为 `"channels_first"`，
输出 4D 张量，尺寸为 `(batch, channels, cropped_rows, cropped_cols)`。

__例子__


```python
# 裁剪输入的 2D 图像或特征图
model = Sequential()
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))
# 现在 model.output_shape == (None, 24, 20, 3)
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Cropping2D(cropping=((2, 2), (2, 2))))
# 现在 model.output_shape == (None, 20, 16. 64)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2490)</span>
### Cropping3D

```python
keras.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)
```

3D 数据的裁剪层（例如空间或时空）。

__参数__

- __cropping__: 整数，或 3 个整数的元组，或 2 个整数的 3 个元组。
    - 如果为整数： 将对深度、高度和宽度应用相同的对称裁剪。
    - 如果为 3 个整数的元组：
        解释为对深度、高度和高度的 3 个不同的对称裁剪值：
        `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`。
    - 如果为 2 个整数的 3 个元组：
        解释为 `((left_dim1_crop, right_dim1_crop), (left_dim2_crop, right_dim2_crop), (left_dim3_crop, right_dim3_crop))`。
- __data_format__: 字符串，
`channels_last` (默认) 或 `channels_first` 之一，
表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`，
`channels_first` 对应输入尺寸为 
`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`。
它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
找到的 `image_data_format` 值。
如果你从未设置它，将使用 "channels_last"。

__输入尺寸__

5D 张量，尺寸为：

- 如果 `data_format` 为 `"channels_last"`: 
`(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis, depth)`
- 如果 `data_format` 为 `"channels_first"`: 
`(batch, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)`

__输出尺寸__

5D 张量，尺寸为：

- 如果 `data_format` 为 `"channels_last"`: 
`(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis, depth)`
- 如果 `data_format` 为 `"channels_first"`: 
`(batch, depth, first_cropped_axis, second_cropped_axis, third_cropped_axis)`。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1943)</span>
### UpSampling1D

```python
keras.layers.UpSampling1D(size=2)
```

1D 输入的上采样层。

沿着时间轴重复每个时间步 `size` 次。

__参数__

- __size__: 整数。上采样因子。

__输入尺寸__

3D 张量，尺寸为 `(batch, steps, features)`。

__输出尺寸__

3D 张量，尺寸为 `(batch, upsampled_steps, features)`。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L1973)</span>
### UpSampling2D

```python
keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')
```

2D 输入的上采样层。

沿着数据的行和列分别重复 `size[0]` 和 `size[1]` 次。

__参数__

- __size__: 整数，或 2 个整数的元组。
行和列的上采样因子。
- __data_format__: 字符串，
`channels_last` (默认) 或 `channels_first` 之一，
表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
`(batch, height, width, channels)`，
`channels_first` 对应输入尺寸为 
`(batch, channels, height, width)`。
它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
找到的 `image_data_format` 值。
如果你从未设置它，将使用 "channels_last"。
- __interpolation__: 字符串，`nearest` 或 `bilinear` 之一。
    注意 CNTK 暂不支持 `bilinear` upscaling，
    以及对于 Theano，只可以使用 `size=(2, 2)`。

__输入尺寸__

- 如果 `data_format` 为 `"channels_last"`，
输入 4D 张量，尺寸为 
`(batch, rows, cols, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
输入 4D 张量，尺寸为 
`(batch, channels, rows, cols)`。

__输出尺寸__

- 如果 `data_format` 为 `"channels_last"`，
输出 4D 张量，尺寸为 
`(batch, upsampled_rows, upsampled_cols, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
输出 4D 张量，尺寸为 
`(batch, channels, upsampled_rows, upsampled_cols)`。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2031)</span>
### UpSampling3D

```python
keras.layers.UpSampling3D(size=(2, 2, 2), data_format=None)
```

3D 输入的上采样层。

沿着数据的第 1、2、3 维度分别重复 
`size[0]`、`size[1]` 和 `size[2]` 次。

__参数__

- __size__: 整数，或 3 个整数的元组。
dim1, dim2 和 dim3 的上采样因子。
- __data_format__: 字符串，
`channels_last` (默认) 或 `channels_first` 之一，
表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`，
`channels_first` 对应输入尺寸为 
`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`。
它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
找到的 `image_data_format` 值。
如果你从未设置它，将使用 "channels_last"。

__输入尺寸__

- 如果 `data_format` 为 `"channels_last"`，
输入 5D 张量，尺寸为 
`(batch, dim1, dim2, dim3, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
输入 5D 张量，尺寸为 
`(batch, channels, dim1, dim2, dim3)`。

__输出尺寸__

- 如果 `data_format` 为 `"channels_last"`，
输出 5D 张量，尺寸为 
`(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
输出 5D 张量，尺寸为 
`(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2123)</span>
### ZeroPadding1D

```python
keras.layers.ZeroPadding1D(padding=1)
```

1D 输入的零填充层（例如，时间序列）。

__参数__

- __padding__: 整数，或长度为 2 的整数元组，或字典。
    - 如果为整数：
        在填充维度（第一个轴）的开始和结束处添加多少个零。
    - 如果是长度为 2 的整数元组：
        在填充维度的开始和结尾处添加多少个零 (`(left_pad, right_pad)`)。

__输入尺寸__

3D 张量，尺寸为 `(batch, axis_to_pad, features)`。

__输出尺寸__

3D 张量，尺寸为 `(batch, padded_axis, features)`。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2158)</span>
### ZeroPadding2D

```python
keras.layers.ZeroPadding2D(padding=(1, 1), data_format=None)
```

2D 输入的零填充层（例如图像）。

该图层可以在图像张量的顶部、底部、左侧和右侧添加零表示的行和列。

__参数__

- __padding__: 整数，或 2 个整数的元组，或 2 个整数的 2 个元组。
    - 如果为整数：将对宽度和高度运用相同的对称填充。
    - 如果为 2 个整数的元组：
    - 如果为整数：: 解释为高度和高度的 2 个不同的对称裁剪值：
        `(symmetric_height_pad, symmetric_width_pad)`。
    - 如果为 2 个整数的 2 个元组：
        解释为 `((top_pad, bottom_pad), (left_pad, right_pad))`。
- __data_format__: 字符串，
`channels_last` (默认) 或 `channels_first` 之一，
表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
`(batch, height, width, channels)`，
`channels_first` 对应输入尺寸为 
`(batch, channels, height, width)`。
它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
找到的 `image_data_format` 值。
如果你从未设置它，将使用 "channels_last"。

__输入尺寸__

- 如果 `data_format` 为 `"channels_last"`，
输入 4D 张量，尺寸为 
`(batch, rows, cols, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
输入 4D 张量，尺寸为 
`(batch, channels, rows, cols)`。

__输出尺寸__

- 如果 `data_format` 为 `"channels_last"`，
输出 4D 张量，尺寸为 
`(batch, padded_rows, padded_cols, channels)`。
- 如果 `data_format` 为 `"channels_first"`，
输出 4D 张量，尺寸为 
`(batch, channels, padded_rows, padded_cols)`。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py#L2234)</span>
### ZeroPadding3D

```python
keras.layers.ZeroPadding3D(padding=(1, 1, 1), data_format=None)
```

3D 数据的零填充层(空间或时空)。

__参数__

- __padding__: 整数，或 3 个整数的元组，或 2 个整数的 3 个元组。
    - 如果为整数：将对深度、高度和宽度运用相同的对称填充。
    - 如果为 3 个整数的元组：
        解释为深度、高度和宽度的三个不同的对称填充值：
        `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
    - 如果为 2 个整数的 3 个元组：解释为
        `((left_dim1_pad, right_dim1_pad), (left_dim2_pad, right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
- __data_format__: 字符串，
`channels_last` (默认) 或 `channels_first` 之一，
表示输入中维度的顺序。`channels_last` 对应输入尺寸为 
`(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`，
`channels_first` 对应输入尺寸为 
`(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`。
它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
找到的 `image_data_format` 值。
如果你从未设置它，将使用 "channels_last"。

__输入尺寸__

5D 张量，尺寸为：

- 如果 `data_format` 为 `"channels_last"`: 
`(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad, depth)`。
- 如果 `data_format` 为 `"channels_first"`: 
`(batch, depth, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`。

__输出尺寸__

5D 张量，尺寸为：

- 如果 `data_format` 为 `"channels_last"`: 
`(batch, first_padded_axis, second_padded_axis, third_axis_to_pad, depth)`。
- 如果 `data_format` 为 `"channels_first"`:
`(batch, depth, first_padded_axis, second_padded_axis, third_axis_to_pad)`。
