<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L743)</span>
### Dense

```python
keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

就是普通的全连接层。

`Dense` 实现以下操作：
`output = activation(dot(input, kernel) + bias)`
其中 `activation` 是按逐个元素计算的激活函数，`kernel` 
是由网络层创建的权值矩阵，以及 `bias` 是其创建的偏置向量
(只在 `use_bias` 为 `True` 时才有用)。

- __注意__: 如果该层的输入的秩大于2，那么它首先被展平然后
再计算与 `kernel` 的点乘。

__例子__


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
(即， “线性”激活: `a(x) = x`)。
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

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L280)</span>
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
（整数元组，不包括样本数的轴）


__输出尺寸__

与输入相同

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L78)</span>
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

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L465)</span>
### Flatten

```python
keras.layers.Flatten()
```

将输入展平。不影响批量大小。

__例__


```python
model = Sequential()
model.add(Conv2D(64, 3, 3,
                 border_mode='same',
                 input_shape=(3, 32, 32)))
# 现在：model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# 现在：model.output_shape == (None, 65536)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/engine/topology.py#L1391)</span>
### Input

```python
keras.engine.topology.Input()
```

`Input()` 用于实例化 Keras 张量。

Keras 张量是底层后端(Theano, TensorFlow or CNTK)
的张量对象，我们增加了一些特性，使得能够通过了解模型的输入
和输出来构建Keras模型。

例如，如果 a, b 和 c 都是 Keras 张量，
那么以下操作是可行的：
`model = Model(input=[a, b], output=c)`

添加的 Keras 属性是：
- __`_keras_shape`__: 通过 Keras端的尺寸推理
进行传播的整数尺寸元组。
- __`_keras_history`__: 应用于张量的最后一层。
整个网络层计算图可以递归地从该层中检索。

__参数__

- __shape__: 一个尺寸元组（整数），不包含批量大小。A shape tuple (integer), not including the batch size.
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

__Returns__

一个张量。

__例__


```python
# 这是 Keras 中的一个逻辑回归
x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)
model = Model(x, y)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L314)</span>
### Reshape

```python
keras.layers.Reshape(target_shape)
```

Reshapes an output to a certain shape.

__Arguments__

- __target_shape__: target shape. Tuple of integers.
Does not include the batch axis.

__Input shape__

Arbitrary, although all dimensions in the input shaped must be fixed.
Use the keyword argument `input_shape`
(tuple of integers, does not include the batch axis)
when using this layer as the first layer in a model.

__Output shape__

`(batch_size,) + target_shape`

__Example__


```python
# as first layer in a Sequential model
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12,)))
# now: model.output_shape == (None, 3, 4)
# note: `None` is the batch dimension

# as intermediate layer in a Sequential model
model.add(Reshape((6, 2)))
# now: model.output_shape == (None, 6, 2)

# also supports shape inference using `-1` as dimension
model.add(Reshape((-1, 2, 2)))
# now: model.output_shape == (None, 3, 2, 2)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L413)</span>
### Permute

```python
keras.layers.Permute(dims)
```

Permutes the dimensions of the input according to a given pattern.

Useful for e.g. connecting RNNs and convnets together.

__Example__


```python
model = Sequential()
model.add(Permute((2, 1), input_shape=(10, 64)))
# now: model.output_shape == (None, 64, 10)
# note: `None` is the batch dimension
```

__Arguments__

- __dims__: Tuple of integers. Permutation pattern, does not include the
samples dimension. Indexing starts at 1.
For instance, `(2, 1)` permutes the first and second dimension
of the input.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same as the input shape, but with the dimensions re-ordered according
to the specified pattern.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L500)</span>
### RepeatVector

```python
keras.layers.RepeatVector(n)
```

Repeats the input n times.

__Example__


```python
model = Sequential()
model.add(Dense(32, input_dim=32))
# now: model.output_shape == (None, 32)
# note: `None` is the batch dimension

model.add(RepeatVector(3))
# now: model.output_shape == (None, 3, 32)
```

__Arguments__

- __n__: integer, repetition factor.

__Input shape__

2D tensor of shape `(num_samples, features)`.

__Output shape__

3D tensor of shape `(num_samples, n, features)`.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L542)</span>
### Lambda

```python
keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None)
```

Wraps arbitrary expression as a `Layer` object.

__Examples__


```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
```
```python
# add a layer that returns the concatenation
# of the positive part of the input and
# the opposite of the negative part

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

__Arguments__

- __function__: The function to be evaluated.
Takes input tensor as first argument.
- __output_shape__: Expected output shape from function.
Only relevant when using Theano.
Can be a tuple or function.
If a tuple, it only specifies the first dimension onward;
sample dimension is assumed either the same as the input:
`output_shape = (input_shape[0], ) + output_shape`
or, the input is `None` and
the sample dimension is also `None`:
`output_shape = (None, ) + output_shape`
If a function, it specifies the entire shape as a function of the
input shape: `output_shape = f(input_shape)`
- __arguments__: optional dictionary of keyword arguments to be passed
to the function.

__Input shape__

Arbitrary. Use the keyword argument input_shape
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Specified by `output_shape` argument
(or auto-inferred when using TensorFlow).

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L886)</span>
### ActivityRegularization

```python
keras.layers.ActivityRegularization(l1=0.0, l2=0.0)
```

Layer that applies an update to the cost function based input activity.

__Arguments__

- __l1__: L1 regularization factor (positive float).
- __l2__: L2 regularization factor (positive float).

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as input.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L28)</span>
### Masking

```python
keras.layers.Masking(mask_value=0.0)
```

Masks a sequence by using a mask value to skip timesteps.

For each timestep in the input tensor (dimension #1 in the tensor),
if all values in the input tensor at that timestep
are equal to `mask_value`, then the timestep will be masked (skipped)
in all downstream layers (as long as they support masking).

If any downstream layer does not support masking yet receives such
an input mask, an exception will be raised.

__Example__


Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
to be fed to an LSTM layer.
You want to mask timestep #3 and #5 because you lack data for
these timesteps. You can:

- set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
- insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

```python
model = Sequential()
model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
model.add(LSTM(32))
```
