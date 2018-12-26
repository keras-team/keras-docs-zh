<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L237)</span>
### RNN

```python
keras.layers.RNN(cell, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

循环神经网络层基类。

__参数__

- __cell__: 一个 RNN 单元实例。RNN 单元是一个具有以下几项的类：
    - 一个 `call(input_at_t, states_at_t)` 方法，
      它返回 `(output_at_t, states_at_t_plus_1)`。
      单元的调用方法也可以采引入可选参数 `constants`，
      详见下面的小节「关于给 RNN 传递外部常量的说明」。
    - 一个 `state_size` 属性。这可以是单个整数（单个状态），
      在这种情况下，它是循环层状态的大小（应该与单元输出的大小相同）。
      这也可以是整数表示的列表/元组（每个状态一个大小）。
    - 一个 `output_size` 属性。 这可以是单个整数或者是一个 TensorShape，
      它表示输出的尺寸。出于向后兼容的原因，如果此属性对于当前单元不可用，
      则该值将由 `state_size` 的第一个元素推断。

    `cell` 也可能是 RNN 单元实例的列表，在这种情况下，RNN 的单元将堆叠在另一个单元上，实现高效的堆叠 RNN。

- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __go_backwards__: 布尔值 (默认 False)。
如果为 True，则向后处理输入序列并返回相反的序列。
- __stateful__: 布尔值 (默认 False)。
如果为 True，则批次中索引 i 处的每个样品的最后状态将用作下一批次中索引 i 样品的初始状态。
- __unroll__: 布尔值 (默认 False)。
如果为 True，则网络将展开，否则将使用符号循环。
展开可以加速 RNN，但它往往会占用更多的内存。
展开只适用于短序列。
- __input_dim__: 输入的维度（整数）。
将此层用作模型中的第一层时，此参数（或者，关键字参数 `input_shape`）是必需的。
- __input_length__: 输入序列的长度，在恒定时指定。
如果你要在上游连接 `Flatten` 和 `Dense` 层，
则需要此参数（如果没有它，无法计算全连接输出的尺寸）。
请注意，如果循环神经网络层不是模型中的第一层，
则需要在第一层的层级指定输入长度（例如，通过 `input_shape` 参数）。

__输入尺寸__

3D 张量，尺寸为 `(batch_size, timesteps, input_dim)`。

__输出尺寸__

- 如果 `return_state`：返回张量列表。
第一个张量为输出。剩余的张量为最后的状态，
每个张量的尺寸为 `(batch_size, units)`。
- 如果 `return_sequences`：返回 3D 张量，
尺寸为 `(batch_size, timesteps, units)`。
- 否则，返回尺寸为 `(batch_size, units)` 的 2D 张量。

__Masking__

该层支持以可变数量的时间步对输入数据进行 masking。
要将 masking 引入你的数据，请使用 [Embedding](embeddings.md) 层，
并将 `mask_zero` 参数设置为 `True`。

__关于在 RNN 中使用「状态（statefulness）」的说明__

你可以将 RNN 层设置为 `stateful`（有状态的），
这意味着针对一个批次的样本计算的状态将被重新用作下一批样本的初始状态。
这假定在不同连续批次的样品之间有一对一的映射。

为了使状态有效：

- 在层构造器中指定 `stateful=True`。
- 为你的模型指定一个固定的批次大小，
如果是顺序模型，为你的模型的第一层传递一个 `batch_input_shape=(...)` 参数。
- 为你的模型指定一个固定的批次大小，
如果是顺序模型，为你的模型的第一层传递一个 `batch_input_shape=(...)`。
如果是带有 1 个或多个 Input 层的函数式模型，为你的模型的所有第一层传递一个 `batch_shape=(...)`。
这是你的输入的预期尺寸，*包括批量维度*。
它应该是整数的元组，例如 `(32, 10, 100)`。
- 在调用 `fit()` 是指定 `shuffle=False`。

要重置模型的状态，请在特定图层或整个模型上调用 `.reset_states()`。

__关于指定 RNN 初始状态的说明__

您可以通过使用关键字参数 `initial_state` 调用它们来符号化地指定 RNN 层的初始状态。
`initial_state` 的值应该是表示 RNN 层初始状态的张量或张量列表。

您可以通过调用带有关键字参数 `states` 的 `reset_states` 方法来数字化地指定 RNN 层的初始状态。
`states` 的值应该是一个代表 RNN 层初始状态的 Numpy 数组或者 Numpy 数组列表。

__关于给 RNN 传递外部常量的说明__

你可以使用 `RNN.__call__`（以及 `RNN.call`）的 `constants` 关键字参数将「外部」常量传递给单元。
这要求 `cell.call` 方法接受相同的关键字参数 `constants`。
这些常数可用于调节附加静态输入（不随时间变化）上的单元转换，也可用于注意力机制。

__例子__


```python
# 首先，让我们定义一个 RNN 单元，作为网络层子类。

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = K.dot(inputs, self.kernel)
        output = h + K.dot(prev_output, self.recurrent_kernel)
        return output, [output]

# 让我们在 RNN 层使用这个单元：

cell = MinimalRNNCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# 以下是如何使用单元格构建堆叠的 RNN的方法：

cells = [MinimalRNNCell(32), MinimalRNNCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L944)</span>
### SimpleRNN

```python
keras.layers.SimpleRNN(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

全连接的 RNN，其输出将被反馈到输入。

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
(详见 [activations](../activations.md))。
默认：双曲正切（`tanh`）。
如果传入 `None`，则不使用激活函数
(即 线性激活：`a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
用于输入的线性转换
(详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
的初始化器，用于循环层状态的线性转换
(详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
(详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于循环层状态的线性转换。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __go_backwards__: 布尔值 (默认 False)。
如果为 True，则向后处理输入序列并返回相反的序列。
- __stateful__: 布尔值 (默认 False)。
如果为 True，则批次中索引 i 处的每个样品
的最后状态将用作下一批次中索引 i 样品的初始状态。
- __unroll__: 布尔值 (默认 False)。
如果为 True，则网络将展开，否则将使用符号循环。
展开可以加速 RNN，但它往往会占用更多的内存。
展开只适用于短序列。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1482)</span>
### GRU

```python
keras.layers.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)
```

门限循环单元网络（Gated Recurrent Unit） - Cho et al. 2014.

有两种变体。默认的是基于 1406.1078v3 的实现，同时在矩阵乘法之前将复位门应用于隐藏状态。
另一种则是基于 1406.1078v1 的实现，它包括顺序倒置的操作。

第二种变体与 CuDNNGRU(GPU-only) 兼容并且允许在 CPU 上进行推理。
因此它对于 `kernel` 和 `recurrent_kernel` 有可分离偏置。
使用 `'reset_after'=True` 和 `recurrent_activation='sigmoid'` 。

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
(详见 [activations](../activations.md))。
默认：双曲正切 (`tanh`)。
如果传入 `None`，则不使用激活函数
(即 线性激活：`a(x) = x`)。
- __recurrent_activation__: 用于循环时间步的激活函数
(详见 [activations](../activations.md))。
默认：分段线性近似 sigmoid (`hard_sigmoid`)。
如果传入 None，则不使用激活函数
(即 线性激活：`a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
用于输入的线性转换
(详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
的初始化器，用于循环层状态的线性转换
(详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
(详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于循环层状态的线性转换。
- __implementation__: 实现模式，1 或 2。
模式 1 将把它的操作结构化为更多的小的点积和加法操作，
而模式 2 将把它们分批到更少，更大的操作中。
这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __go_backwards__: 布尔值 (默认 False)。
如果为 True，则向后处理输入序列并返回相反的序列。
- __stateful__: 布尔值 (默认 False)。
如果为 True，则批次中索引 i 处的每个样品的最后状态
将用作下一批次中索引 i 样品的初始状态。
- __unroll__: 布尔值 (默认 False)。
如果为 True，则网络将展开，否则将使用符号循环。
展开可以加速 RNN，但它往往会占用更多的内存。
展开只适用于短序列。
- __reset_after__: 
- GRU 公约 (是否在矩阵乘法之前或者之后使用重置门)。
False =「之前」(默认)，Ture =「之后」( CuDNN 兼容)。

__参考文献__

- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078)
- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L2034)</span>
### LSTM

```python
keras.layers.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
```

长短期记忆网络层（Long Short-Term Memory） - Hochreiter 1997.

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
(详见 [activations](../activations.md))。
如果传入 `None`，则不使用激活函数
(即 线性激活：`a(x) = x`)。
- __recurrent_activation__: 用于循环时间步的激活函数
(详见 [activations](../activations.md))。
默认：分段线性近似 sigmoid (`hard_sigmoid`)。
如果传入 `None`，则不使用激活函数
(即 线性激活：`a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
用于输入的线性转换
(详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
的初始化器，用于循环层状态的线性转换
(详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
(详见[initializers](../initializers.md)).
- __unit_forget_bias__: 布尔值。
如果为 True，初始化时，将忘记门的偏置加 1。
将其设置为 True 同时还会强制 `bias_initializer="zeros"`。
这个建议来自 [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于循环层状态的线性转换。
- __implementation__: 实现模式，1 或 2。
模式 1 将把它的操作结构化为更多的小的点积和加法操作，
而模式 2 将把它们分批到更少，更大的操作中。
这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __go_backwards__: 布尔值 (默认 False)。
如果为 True，则向后处理输入序列并返回相反的序列。
- __stateful__: 布尔值 (默认 False)。
如果为 True，则批次中索引 i 处的每个样品的最后状态
将用作下一批次中索引 i 样品的初始状态。
- __unroll__: 布尔值 (默认 False)。
如果为 True，则网络将展开，否则将使用符号循环。
展开可以加速 RNN，但它往往会占用更多的内存。
展开只适用于短序列。

__参考文献__

- [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (original 1997 paper)
- [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
- [Supervised sequence labeling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/convolutional_recurrent.py#L788)</span>
### ConvLSTM2D

```python
keras.layers.ConvLSTM2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)
```

卷积 LSTM。

它类似于 LSTM 层，但输入变换和循环变换都是卷积的。

__参数__

- __filters__: 整数，输出空间的维度
（即卷积中滤波器的输出数量）。
- __kernel_size__: 一个整数，或者 n 个整数表示的元组或列表，
指明卷积窗口的维度。
- __strides__: 一个整数，或者 n 个整数表示的元组或列表，
指明卷积的步长。
指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
- __padding__: `"valid"` 或 `"same"` 之一 (大小写敏感)。
- __data_format__: 字符串，
`channels_last` (默认) 或 `channels_first` 之一。
输入中维度的顺序。
`channels_last` 对应输入尺寸为 `(batch, time, ..., channels)`，
`channels_first` 对应输入尺寸为 `(batch, time, channels, ...)`。
它默认为从 Keras 配置文件 `~/.keras/keras.json` 中
找到的 `image_data_format` 值。
如果你从未设置它，将使用 `"channels_last"`。
- __dilation_rate__: 一个整数，或 n 个整数的元组/列表，指定用于膨胀卷积的膨胀率。
目前，指定任何 `dilation_rate` 值 != 1 与指定 stride 值 != 1 两者不兼容。
- __activation__: 要使用的激活函数
(详见 [activations](../activations.md))。
如果传入 None，则不使用激活函数
(即 线性激活：`a(x) = x`)。
- __recurrent_activation__: 用于循环时间步的激活函数
(详见 [activations](../activations.md))。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
用于输入的线性转换
(详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
的初始化器，用于循环层状态的线性转换
(详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
(详见[initializers](../initializers.md)).
- __unit_forget_bias__: 布尔值。
如果为 True，初始化时，将忘记门的偏置加 1。
将其设置为 True 同时还会强制 `bias_initializer="zeros"`。
这个建议来自 [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __go_backwards__: 布尔值 (默认 False)。
如果为 True，则向后处理输入序列并返回相反的序列。
- __stateful__: 布尔值 (默认 False)。
如果为 True，则批次中索引 i 处的每个样品的最后状态
将用作下一批次中索引 i 样品的初始状态。
- __dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于循环层状态的线性转换。

__输入尺寸__

- 如果 data_format='channels_first'，
输入 5D 张量，尺寸为：
`(samples,time, channels, rows, cols)`。
- 如果 data_format='channels_last'，
输入 5D 张量，尺寸为：
`(samples,time, rows, cols, channels)`。

__输出尺寸__

- 如果 `return_sequences`，
    - 如果 data_format='channels_first'，返回 5D 张量，尺寸为：`(samples, time, filters, output_row, output_col)`。
    - 如果 data_format='channels_last'，返回 5D 张量，尺寸为：`(samples, time, output_row, output_col, filters)`。
- 否则，
    - 如果 data_format ='channels_first'，返回 4D 张量，尺寸为：`(samples, filters, output_row, output_col)`。
    - 如果 data_format='channels_last'，返回 4D 张量，尺寸为：`(samples, output_row, output_col, filters)`。

o_row 和 o_col 取决于 filter 和 padding 的尺寸。

__异常__

- __ValueError__: 无效的构造参数。

__参考文献__

- [Convolutional LSTM Network: A Machine Learning Approach for
Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)。
当前的实现不包括单元输出的反馈回路。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L779)</span>
### SimpleRNNCell

```python
keras.layers.SimpleRNNCell(units, activation='tanh', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

SimpleRNN 的单元类。

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
(详见 [activations](../activations.md))。
默认：双曲正切 (`tanh`)。
如果传入 `None`，则不使用激活函数
(即 线性激活：`a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
用于输入的线性转换
(详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
的初始化器，用于循环层状态的线性转换
(详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
(详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于循环层状态的线性转换。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1163)</span>
### GRUCell

```python
keras.layers.GRUCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, reset_after=False)
```

GRU 层的单元类。

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
(详见 [activations](../activations.md))。
默认：双曲正切 (`tanh`)。
如果传入 `None`，则不使用激活函数
(即 线性激活：`a(x) = x`)。
- __recurrent_activation__: 用于循环时间步的激活函数
(详见 [activations](../activations.md))。
默认：分段线性近似 sigmoid (`hard_sigmoid`)。
如果传入 `None`，则不使用激活函数
(即 线性激活：`a(x) = x`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
用于输入的线性转换
(详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
的初始化器，用于循环层状态的线性转换
(详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
(详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于循环层状态的线性转换。
- __implementation__: 实现模式，1 或 2。
模式 1 将把它的操作结构化为更多的小的点积和加法操作，
而模式 2 将把它们分批到更少，更大的操作中。
这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。
- __reset_after__: 
- GRU 公约 (是否在矩阵乘法之前或者之后使用重置门)。
False = "before" (默认)，Ture = "after" ( CuDNN 兼容)。
- __reset_after__: GRU convention (whether to apply reset gate after or
before matrix multiplication). False = "before" (default),
True = "after" (CuDNN compatible).


----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py#L1756)</span>
### LSTMCell

```python
keras.layers.LSTMCell(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1)
```

LSTM 层的单元类。

__参数__

- __units__: 正整数，输出空间的维度。
- __activation__: 要使用的激活函数
(详见 [activations](../activations.md))。
默认：双曲正切（`tanh`）。
如果传入 `None`，则不使用激活函数
(即 线性激活：`a(x) = x`)。
- __recurrent_activation__: 用于循环时间步的激活函数
(详见 [activations](../activations.md))。
默认：分段线性近似 sigmoid (`hard_sigmoid`)。
- __use_bias__: 布尔值，该层是否使用偏置向量。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
用于输入的线性转换
(详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
的初始化器，用于循环层状态的线性转换
(详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
(详见[initializers](../initializers.md)).
- __unit_forget_bias__: 布尔值。
如果为 True，初始化时，将忘记门的偏置加 1。
将其设置为 True 同时还会强制 `bias_initializer="zeros"`。
这个建议来自 [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)。
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。
- __dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于输入的线性转换。
- __recurrent_dropout__: 在 0 和 1 之间的浮点数。
单元的丢弃比例，用于循环层状态的线性转换。
- __implementation__: 实现模式，1 或 2。
模式 1 将把它的操作结构化为更多的小的点积和加法操作，
而模式 2 将把它们分批到更少，更大的操作中。
这些模式在不同的硬件和不同的应用中具有不同的性能配置文件。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L135)</span>
### CuDNNGRU

```python
keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

由 [CuDNN](https://developer.nvidia.com/cudnn) 支持的快速 GRU 实现。

只能以 TensorFlow 后端运行在 GPU 上。

__参数__

- __units__: 正整数，输出空间的维度。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
用于输入的线性转换
(详见 [initializers](../initializers.md))。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
的初始化器，用于循环层状态的线性转换
(详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
(详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: Regularizer function applied to
the output of the layer (its "activation").
(see [regularizer](../regularizers.md)).
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __stateful__: 布尔值 (默认 False)。
如果为 True，则批次中索引 i 处的每个样品的最后状态
将用作下一批次中索引 i 样品的初始状态。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/cudnn_recurrent.py#L328)</span>
### CuDNNLSTM

```python
keras.layers.CuDNNLSTM(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
```

由 [CuDNN](https://developer.nvidia.com/cudnn) 支持的快速 LSTM 实现。

只能以 TensorFlow 后端运行在 GPU 上。

__参数__

- __units__: 正整数，输出空间的维度。
- __kernel_initializer__: `kernel` 权值矩阵的初始化器，
用于输入的线性转换
(详见 [initializers](../initializers.md))。
- __unit_forget_bias__: 布尔值。
如果为 True，初始化时，将忘记门的偏置加 1。
将其设置为 True 同时还会强制 `bias_initializer="zeros"`。
这个建议来自 [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)。
- __recurrent_initializer__: `recurrent_kernel` 权值矩阵
的初始化器，用于循环层状态的线性转换
(详见 [initializers](../initializers.md))。
- __bias_initializer__:偏置向量的初始化器
(详见[initializers](../initializers.md)).
- __kernel_regularizer__: 运用到 `kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __recurrent_regularizer__: 运用到 `recurrent_kernel` 权值矩阵的正则化函数
(详见 [regularizer](../regularizers.md))。
- __bias_regularizer__: 运用到偏置向量的正则化函数
(详见 [regularizer](../regularizers.md))。
- __activity_regularizer__: 运用到层输出（它的激活值）的正则化函数
(详见 [regularizer](../regularizers.md))。
- __kernel_constraint__: 运用到 `kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __recurrent_constraint__: 运用到 `recurrent_kernel` 权值矩阵的约束函数
(详见 [constraints](../constraints.md))。
- __bias_constraint__: 运用到偏置向量的约束函数
(详见 [constraints](../constraints.md))。
- __return_sequences__: 布尔值。是返回输出序列中的最后一个输出，还是全部序列。
- __return_state__: 布尔值。除了输出之外是否返回最后一个状态。
- __stateful__: 布尔值 (默认 False)。
如果为 True，则批次中索引 i 处的每个样品的最后状态
将用作下一批次中索引 i 样品的初始状态。
