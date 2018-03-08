
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


Softmax 激活函数.

__Arguments__

x : 张量.
- __axis__: 整数, 代表softmax所作用的维度

__Returns__

softmax变换后的张量.

__Raises__

- __ValueError__: In case `dim(x) == 1`.

----

### elu


```python
elu(x, alpha=1.0)
```

----

### selu


```python
selu(x)
```


可伸缩的指数线性单元 (Klambauer et al., 2017)。

__Arguments__

- __x__: 一个用来用于计算激活函数的张量或变量。

__Returns__

与`x`具有相同类型及形状的张量。

__Note__

- 与 "lecun_normal" 初始化方法一起使用。
- 与 dropout 的变种 "AlphaDropout" 一起使用。

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)

----

### softplus


```python
softplus(x)
```

----

### softsign


```python
softsign(x)
```

----

### relu


```python
relu(x, alpha=0.0, max_value=None)
```

----

### tanh


```python
tanh(x)
```

----

### sigmoid


```python
sigmoid(x)
```

----

### hard_sigmoid


```python
hard_sigmoid(x)
```

----

### linear


```python
linear(x)
```


## 高级激活函数

对于Theano/TensorFlow/CNTK不能表达的复杂激活函数，如含有可学习参数的激活函数，可通过[高级激活函数](layers/advanced-activations.md)实现，如PReLU，LeakyReLU等
