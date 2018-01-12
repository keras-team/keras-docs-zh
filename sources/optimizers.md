
## 优化器的用法

优化器(optimizer)是编译Keras模型的所需的两个参数之一：
```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

你可以先实例化一个优化器对象，然后将它传入`model.compile()`，像上述示例中一样，
或者你可以通过名称来调用优化器。在后一种情况下，将使用优化器的默认参数。


```python
# 传入优化器名称: 默认参数将被采用
model.compile(loss='mean_squared_error', optimizer='sgd')
```

---

## Keras优化器的公共参数


参数`clipnorm`和`clipvalue`能在所有的优化器中使用，用于控制梯度裁剪（Gradient Clipping）：

```python
from keras import optimizers

# 所有参数梯度将被裁剪，让其l2范数最大为1：g * 1 / max(1, l2_norm)
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```

```python
from keras import optimizers

# 所有参数d 梯度将被裁剪到数值范围内：
# 最大值0.5
# 最小值-0.5
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L135)</span>
### SGD

```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```

随机梯度下降优化器

包含扩展功能的支持：
- 动量（momentum）优化,
- 学习率衰减（每次参数更新后）
- Nestrov动量(NAG)优化

__参数__

- __lr__: float >= 0. 学习率
- __momentum__: float >= 0. 参数，用于加速SGD在相关方向上前进，并抑制震荡
- __decay__: float >= 0. 每次参数更新后学习率衰减值.
- __nesterov__: boolean. 是否使用Nesterov动量.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L198)</span>
### RMSprop

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
```

RMSProp优化器.

建议使用优化器的默认参数
（除了学习率lr，它可以被自由调节）


这个优化器通常是训练循环神经网络RNN的不错选择。

__参数__

- __lr__: float >= 0. 学习率.
- __rho__: float >= 0. RMSProp梯度平方的移动均值的衰减率.
- __epsilon__: float >= 0. 模糊因子. 若为 `None`, 默认为 `K.epsilon()`.
- __decay__: float >= 0. 每次参数更新后学习率衰减值.

__引用__

- [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L265)</span>
### Adagrad

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
```

Adagrad优化器.

建议使用优化器的默认参数。

__参数__

- __lr__: float >= 0. 学习率.
- __epsilon__: float >= 0. 若为 `None`, 默认为 `K.epsilon()`.
- __decay__: float >= 0. 每次参数更新后学习率衰减值.

__引用__

- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L324)</span>
### Adadelta

```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
```

Adagrad优化器.

建议使用优化器的默认参数。

__参数__

- __lr__: float >= 0. 学习率，建议保留默认值.
- __rho__: float >= 0. Adadelta梯度平方移动均值的衰减率
- __epsilon__: float >= 0. 模糊因子. 若为 `None`, 默认为 `K.epsilon()`.
- __decay__: float >= 0. 每次参数更新后学习率衰减值.

__引用__

- [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L397)</span>
### Adam

```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
```

Adam优化器.

默认参数遵循原论文中提供的值。


__参数__

- __lr__: float >= 0. 学习率.
- __beta_1__: float, 0 < beta < 1. 通常接近于 1.
- __beta_2__: float, 0 < beta < 1. 通常接近于 1.
- __epsilon__: float >= 0. 模糊因子. 若为 `None`, 默认为 `K.epsilon()`.
- __decay__: float >= 0. 每次参数更新后学习率衰减值.
- __amsgrad__: boolean. 是否应用此算法的AMSGrad变种，来自论文"On the Convergence of Adam and
Beyond".

__引用__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
- [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L486)</span>
### Adamax

```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
```

Adamax优化器，来自Adam论文的第七小节.

它是Adam算法基于无穷范数（infinity norm）的变种。
默认参数遵循论文中提供的值。

__参数__

- __lr__: float >= 0. 学习率.
- __beta_1/beta_2__: floats, 0 < beta < 1. 通常接近于 1.
- __epsilon__: float >= 0. 模糊因子. 若为 `None`, 默认为 `K.epsilon()`.
- __decay__: float >= 0. 每次参数更新后学习率衰减值.

__引用__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L563)</span>
### Nadam

```python
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
```

Nesterov版本Adam优化器.

正像Adam本质上是RMSProp与动量momentum的结合，
Nadam是采用Nesterov momentum版本的Adam优化器。

默认参数遵循论文中提供的值。
建议使用优化器的默认参数。


__参数__

- __lr__: float >= 0. 学习率.
- __beta_1/beta_2__: floats, 0 < beta < 1. 通常接近于 1.
- __epsilon__: float >= 0. 模糊因子. 若为 `None`, 默认为 `K.epsilon()`.

__引用__

- [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
- [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L649)</span>
### TFOptimizer

```python
keras.optimizers.TFOptimizer(optimizer)
```

原生Tensorlfow优化器的包装类（wrapper class）。
