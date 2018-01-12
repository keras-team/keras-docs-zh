## 初始化器的用法


初始化定义了设置Keras各层权重随机初始值的方法。

用来将初始化器传入keras层的参数名取决于具体的层。通常关键字为`kernel_initializer` and `bias_initializer`:

```python
model.add(Dense(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

## 可用的初始化器

下面这些是可用的内置初始化器，是`keras.initializers`模块的一部分:

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

将张量初始值设为0的初始化器。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L41)</span>
### Ones

```python
keras.initializers.Ones()
```

将张量初始值设为1的初始化器。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L49)</span>
### Constant

```python
keras.initializers.Constant(value=0)
```

将张量初始值设为一个常数的初始化器。

__参数__

- __value__: 浮点数; 生成的张量的值。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L66)</span>
### RandomNormal

```python
keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
```

按照正态分布生成随机张量的初始化器。

__参数__

- __mean__: 一个python标量或者一个标量张量。要生成的随机值的平均数。
- __stddev__: 一个python标量或者一个标量张量。要生成的随机值的标准差。
- __seed__: 一个Python整数。用于设置随机数种子。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L94)</span>
### RandomUniform

```python
keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```

按照均匀分布生成随机张量的初始化器。

__参数__

- __minval__: 一个python标量或者一个标量张量。要生成的随机值的范围下限。
- __maxval__: 一个python标量或者一个标量张量。要生成的随机值的范围下限。默认为浮点类型的1。
- __seed__: 一个Python整数。用于设置随机数种子。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L122)</span>
### TruncatedNormal

```python
keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```

按照截尾正态分布生成随机张量的初始化器。

生成的随机值与`RandomNormal`生成的类似，但是在距离平均值两个标准差之外的随机值将被丢弃并重新生成。这是用来生成神经网络权重和滤波器的推荐初始化器。

__Arguments__

- __mean__: 一个python标量或者一个标量张量。要生成的随机值的平均数。
- __stddev__: 一个python标量或者一个标量张量。要生成的随机值的标准差。
- __seed__: 一个Python整数。用于设置随机数种子。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L155)</span>
### VarianceScaling

```python
keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```

Initializer capable of adapting its scale to the shape of weights.

With `distribution="normal"`, samples are drawn from a truncated normal
distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

- number of input units in the weight tensor, if mode = "fan_in"
- number of output units, if mode = "fan_out"
- average of the numbers of input and output units, if mode = "fan_avg"

With `distribution="uniform"`,
samples are drawn from a uniform distribution
within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

__Arguments__

- __scale__: Scaling factor (positive float).
- __mode__: One of "fan_in", "fan_out", "fan_avg".
- __distribution__: Random distribution to use. One of "normal", "uniform".
- __seed__: A Python integer. Used to seed the random generator.

__Raises__

- __ValueError__: In case of an invalid value for the "scale", mode" or
"distribution" arguments.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L228)</span>
### Orthogonal

```python
keras.initializers.Orthogonal(gain=1.0, seed=None)
```

Initializer that generates a random orthogonal matrix.

__Arguments__

- __gain__: Multiplicative factor to apply to the orthogonal matrix.
- __seed__: A Python integer. Used to seed the random generator.

__References__

Saxe et al., http://arxiv.org/abs/1312.6120

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/initializers.py#L265)</span>
### Identity

```python
keras.initializers.Identity(gain=1.0)
```

Initializer that generates the identity matrix.

Only use for square 2D matrices.

__Arguments__

- __gain__: Multiplicative factor to apply to the identity matrix.

----

### lecun_uniform


```python
lecun_uniform(seed=None)
```


LeCun uniform initializer.

It draws samples from a uniform distribution within [-limit, limit]
where `limit` is `sqrt(3 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

LeCun 98, Efficient Backprop,
- __http__://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

----

### glorot_normal


```python
glorot_normal(seed=None)
```


Glorot normal initializer, also called Xavier normal initializer.

It draws samples from a truncated normal distribution centered on 0
with `stddev = sqrt(2 / (fan_in + fan_out))`
where `fan_in` is the number of input units in the weight tensor
and `fan_out` is the number of output units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

Glorot & Bengio, AISTATS 2010
- __http__://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

----

### glorot_uniform


```python
glorot_uniform(seed=None)
```


Glorot uniform initializer, also called Xavier uniform initializer.

It draws samples from a uniform distribution within [-limit, limit]
where `limit` is `sqrt(6 / (fan_in + fan_out))`
where `fan_in` is the number of input units in the weight tensor
and `fan_out` is the number of output units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

Glorot & Bengio, AISTATS 2010
- __http__://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

----

### he_normal


```python
he_normal(seed=None)
```


He normal initializer.

It draws samples from a truncated normal distribution centered on 0
with `stddev = sqrt(2 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

He et al., http://arxiv.org/abs/1502.01852

----

### lecun_normal


```python
lecun_normal(seed=None)
```


LeCun normal initializer.

It draws samples from a truncated normal distribution centered on 0
with `stddev = sqrt(1 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)

----

### he_uniform


```python
he_uniform(seed=None)
```


He uniform variance scaling initializer.

It draws samples from a uniform distribution within [-limit, limit]
where `limit` is `sqrt(6 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

He et al., http://arxiv.org/abs/1502.01852



An initializer may be passed as a string (must match one of the available initializers above), or as a callable:

```python
from keras import initializers

model.add(Dense(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Dense(64, kernel_initializer='random_normal'))
```


## Using custom initializers

If passing a custom callable, then it must take the argument `shape` (shape of the variable to initialize) and `dtype` (dtype of generated values):

```python
from keras import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Dense(64, kernel_initializer=my_init))
```
