## 初始化器的用法


初始化定义了设置 Keras 各层权重随机初始值的方法。

用来将初始化器传入 Keras 层的参数名取决于具体的层。通常关键字为 `kernel_initializer` 和 `bias_initializer`:

```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

## 可用的初始化器

下面这些是可用的内置初始化器，是 `keras.initializers` 模块的一部分: 

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L14)</span>
### Initializer

```python
keras.initializers.Initializer()
```

初始化器基类：所有初始化器继承这个类。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L33)</span>
### Zeros

```python
keras.initializers.Zeros()
```

将张量初始值设为 0 的初始化器。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L41)</span>
### Ones

```python
keras.initializers.Ones()
```

将张量初始值设为 1 的初始化器。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L49)</span>
### Constant

```python
keras.initializers.Constant(value=0)
```

将张量初始值设为一个常数的初始化器。

__参数__

- __value__: 浮点数，生成的张量的值。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L66)</span>
### RandomNormal

```python
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
```

按照正态分布生成随机张量的初始化器。

__参数__

- __mean__: 一个 Python 标量或者一个标量张量。要生成的随机值的平均数。
- __stddev__: 一个 Python 标量或者一个标量张量。要生成的随机值的标准差。
- __seed__: 一个 Python 整数。用于设置随机数种子。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L94)</span>
### RandomUniform

```python
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```

按照均匀分布生成随机张量的初始化器。

__参数__

- __minval__: 一个 Python 标量或者一个标量张量。要生成的随机值的范围下限。
- __maxval__: 一个 Python 标量或者一个标量张量。要生成的随机值的范围下限。默认为浮点类型的 1。
- __seed__: 一个 Python 整数。用于设置随机数种子。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L122)</span>
### TruncatedNormal

```python
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```

按照截尾正态分布生成随机张量的初始化器。

生成的随机值与 `RandomNormal` 生成的类似，但是在距离平均值两个标准差之外的随机值将被丢弃并重新生成。这是用来生成神经网络权重和滤波器的推荐初始化器。

__Arguments__

- __mean__: 一个 Python 标量或者一个标量张量。要生成的随机值的平均数。
- __stddev__: 一个 Python 标量或者一个标量张量。要生成的随机值的标准差。
- __seed__: 一个 Python 整数。用于设置随机数种子。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L155)</span>
### VarianceScaling

```python
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```

初始化器能够根据权值的尺寸调整其规模。

使用 `distribution="normal"` 时，样本是从一个以 0 为中心的截断正态分布中抽取的，`stddev = sqrt(scale / n)`，其中 n 是：

- 权值张量中输入单元的数量，如果 mode = "fan_in"。
- 输出单元的数量，如果 mode = "fan_out"。
- 输入和输出单位数量的平均数，如果 mode = "fan_avg"。

使用 `distribution="uniform"` 时，样本是从 [-limit，limit] 内的均匀分布中抽取的，其中 `limit = sqrt(3 * scale / n)`。

__参数__

- __scale__: 缩放因子（正浮点数）。
- __mode__: "fan_in", "fan_out", "fan_avg" 之一。
- __distribution__: 使用的随机分布。"normal", "uniform" 之一。
- __seed__: 一个 Python 整数。作为随机发生器的种子。

__异常__

- __ValueError__: 如果 "scale", mode" 或 "distribution" 参数无效。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L228)</span>
### Orthogonal

```python
keras.initializers.Orthogonal(gain=1.0, seed=None)
```

生成一个随机正交矩阵的初始化器。

__参数__

- __gain__: 适用于正交矩阵的乘法因子。
- __seed__: 一个 Python 整数。作为随机发生器的种子。

__参考文献__

Saxe et al., http://arxiv.org/abs/1312.6120

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L265)</span>
### Identity

```python
keras.initializers.Identity(gain=1.0)
```

生成单位矩阵的初始化器。

仅用于 2D 方阵。

__参数__

- __gain__: 适用于单位矩阵的乘法因子。

----

### lecun_uniform


```python
keras.initializers.lecun_uniform(seed=None)
```


LeCun 均匀初始化器。

它从 [-limit，limit] 中的均匀分布中抽取样本，
其中 `limit` 是 `sqrt(3 / fan_in)`，
`fan_in` 是权值张量中的输入单位的数量。

__参数__

- __seed__: 一个 Python 整数。作为随机发生器的种子。

__返回__

一个初始化器。

__参考文献__

[LeCun 98, Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

----

### glorot_normal


```python
keras.initializers.glorot_normal(seed=None)
```


Glorot 正态分布初始化器，也称为 Xavier 正态分布初始化器。

它从以 0 为中心，标准差为 `stddev = sqrt(2 / (fan_in + fan_out))` 的截断正态分布中抽取样本，
其中 `fan_in` 是权值张量中的输入单位的数量，
`fan_out` 是权值张量中的输出单位的数量。

__参数__

- __seed__: 一个 Python 整数。作为随机发生器的种子。

__返回__

一个初始化器。

__参考文献__

[Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

----

### glorot_uniform


```python
keras.initializers.glorot_uniform(seed=None)
```


Glorot 均匀分布初始化器，也称为 Xavier 均匀分布初始化器。

它从 [-limit，limit] 中的均匀分布中抽取样本，
其中 `limit` 是 `sqrt(6 / (fan_in + fan_out))`，
`fan_in` 是权值张量中的输入单位的数量，
`fan_out` 是权值张量中的输出单位的数量。

__参数__

- __seed__: 一个 Python 整数。作为随机发生器的种子。

__返回__

一个初始化器。

__参考文献__

[Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

----

### he_normal


```python
keras.initializers.he_normal(seed=None)
```


He 正态分布初始化器。

它从以 0 为中心，标准差为 `stddev = sqrt(2 / fan_in)` 的截断正态分布中抽取样本，
其中 `fan_in` 是权值张量中的输入单位的数量，

__参数__

- __seed__: 一个 Python 整数。作为随机发生器的种子。

__返回__

一个初始化器。

__参考文献__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/abs/1502.01852)

----

### lecun_normal


```python
keras.initializers.lecun_normal(seed=None)
```


LeCun 正态分布初始化器。

它从以 0 为中心，标准差为 `stddev = sqrt(1 / fan_in)` 的截断正态分布中抽取样本，
其中 `fan_in` 是权值张量中的输入单位的数量。

__参数__

- __seed__: 一个 Python 整数。作为随机发生器的种子。

__返回__

一个初始化器。

__参考文献__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

----

### he_uniform


```python
keras.initializers.he_uniform(seed=None)
```


He 均匀方差缩放初始化器。

它从 [-limit，limit] 中的均匀分布中抽取样本，
其中 `limit` 是 `sqrt(6 / fan_in)`，
其中 `fan_in` 是权值张量中的输入单位的数量。

__参数__

- __seed__: 一个 Python 整数。作为随机发生器的种子。

__返回__

一个初始化器。

__参考文献__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/abs/1502.01852)


一个初始化器可以作为一个字符串传递（必须匹配上面的一个可用的初始化器），或者作为一个可调用函数传递：

```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# 同样有效;将使用默认参数。
model.add(Dense(64, kernel_initializer='random_normal'))
```


## 使用自定义初始化器

如果传递一个自定义的可调用函数，那么它必须使用参数 `shape`（需要初始化的变量的尺寸）和 `dtype`（数据类型）：


```python
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```
