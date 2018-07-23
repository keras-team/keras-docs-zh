
## 激活函数的用法

激活函数可以通过设置单独的激活层实现，也可以在构造层对象时通过传递`activation`参数实现


```python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
```

等价于

```python
model.add(Dense(64, activation='tanh'))
```

你也可以通过传递一个逐元素运算的Theano/TensorFlow/CNTK函数来作为激活函数：


```python
from keras import backend as K

model.add(Dense(64, activation=K.tanh))
model.add(Activation(K.tanh))
```

## 预定义激活函数

### softmax


```python
softmax(x, axis=-1)
```


Softmax 激活函数。

__参数__

- __x__：张量。
- __axis__：整数，代表softmax所作用的维度。

__返回__

softmax变换后的张量。

__异常__

- __ValueError__： In case `dim(x) == 1`。

----

### elu

```python
elu(x, alpha=1.0)
```

指数线性单元。

__参数__

- __x__：张量。
- __alpha__：一个标量，表示负数部分的斜率。

__返回__

线性指数激活：如果 `x > 0`，返回值为 `x`；如果 `x < 0` 返回值为 `alpha * (exp(x)-1)`

__参考文献__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289)

----

### selu


```python
selu(x)
```

可伸缩的指数线性单元（SELU）。

SELU 等同于：`scale * elu(x, alpha)`，其中 alpha 和 scale 是预定义的常量。只要正确初始化权重（参见 `lecun_normal` 初始化方法）并且输入的数量「足够大」（参见参考文献获得更多信息），选择合适的 alpha 和 scale 的值，就可以在两个连续层之间保留输入的均值和方差。

__参数__

- __x__: 一个用来用于计算激活函数的张量或变量。

__返回__

可伸缩的指数线性激活：`scale * elu(x, alpha)`。

__注意__

- 与「lecun_normal」初始化方法一起使用。
- 与 dropout 的变种「AlphaDropout」一起使用。

__参考文献__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

----

### softplus


```python
softplus(x)
```

Softplus 激活函数。

__参数__

- __x__: 张量。

__返回__

Softplus 激活：`log(exp(x) + 1)`。

----

### softsign


```python
softsign(x)
```

Softsign 激活函数。

__参数__

- __x__: 张量。

__返回__

Softsign 激活：`x / (abs(x) + 1)`。

----

### relu


```python
relu(x, alpha=0.0, max_value=None)
```

线性修正单元。

__参数__

- __x__: 张量。
- __alpha__：负数部分的斜率。默认为 0。
- __max_value__：输出的最大值

__返回__

线性修正单元激活：如果 `x > 0`，返回值为 `x`；如果 `x < 0`，返回值为 `alpha * x`。如果定义了 max_value，则结果将截断为此值。

----

### tanh

```python
tanh(x)
```

双曲正切激活函数。

----

### sigmoid


```python
sigmoid(x)
```

Sigmoid 激活函数。

----

### hard_sigmoid


```python
hard_sigmoid(x)
```

Hard sigmoid 激活函数。

计算速度比 sigmoid 激活函数更快。

__参数__

- __x__: 张量。

__返回__

Hard sigmoid 激活：

- 如果 `x < -2.5`，返回 0。
- 如果 `x > 2.5`，返回 1。
- 如果 `-2.5 <= x <= 2.5`，返回 `0.2 * x + 0.5`。

----

### linear


```python
linear(x)
```

线性激活函数（即不做任何改变）


## 高级激活函数

对于Theano/TensorFlow/CNTK不能表达的复杂激活函数，如含有可学习参数的激活函数，可通过[高级激活函数](layers/advanced-activations.md)实现，可以在 `keras.layers.advanced_activations` 模块中找到。 这些高级激活函数包括 `PReLU` 和 `LeakyReLU`
