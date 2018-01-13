
## 优化器的使用

优化器是编译 Keras 模型所需的两个参数之一：

```python
from keras import optimizers

model = Sequential()
model.add(Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Activation('tanh'))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
```

我们可以在将优化器传递给 `model.compile()` 之前初始化，正如上面代码所示，或者你可以直接通过名字调用。在后面的例子中，默认的优化器参数将会被使用。

```python
# pass optimizer by name: default parameters will be used
model.compile(loss='mean_squared_error', optimizer='sgd')
```

---

## 所有 Keras 优化器共同的参数

参数 `clipnorm` 和 `clipvalue` 可以被所有的优化器使用来控制梯度的剪裁：

```python
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```

```python
from keras import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L135)</span>
### 随机梯度下降 SGD

```python
keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
```

随机梯度下降优化器.

包含对 momentum, 学习率下降 learning rate decay, 和 Nesterov momentum 的支持.

__参数__

- __lr__: float >= 0. 学习率.
- __momentum__: float >= 0. 在相关方向加速 SGD 和减弱震荡的参数.
- __decay__: float >= 0.学习率每次更新的下降量.
- __nesterov__: boolean. 是否应用 Nesterov momentum.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L198)</span>
### RMSprop

```python
keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
```

RMSProp 优化器.

推荐使用这个优化器的默认设置的参数（除了学习率 learning rate 可以自由调整）.

该优化器通常是适合循环神经网络的选择.

__参数__

- __lr__: float >= 0. 学习率.
- __rho__: float >= 0.
- __epsilon__: float >= 0. 模糊因子（Fuzz factor）. 如果设置为 `None`, 默认就是 `K.epsilon()`.
- __decay__: float >= 0. Learning rate decay over each update.

__References__

- [rmsprop: Divide the gradient by a running average of its recent magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L265)</span>
### Adagrad

```python
keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
```

Adagrad 优化器.

推荐使用该优化器的默认设置的参数.

__参数__

- __lr__: float >= 0. 学习率.
- __epsilon__: float >= 0. 如果设置为 `None`, 默认是 `K.epsilon()`.
- __decay__: float >= 0. 每次更新学习率的下降量.

__引用__

- [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L324)</span>
### Adadelta

```python
keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
```

Adadelta 优化器.

推荐使用该优化器的默认设置的参数.

__参数__

- __lr__: float >= 0. 学习率. 推荐使用该参数的默认设置.
- __rho__: float >= 0.
- __epsilon__: float >= 0. 模糊因子. 如果设置为 `None`, 默认是 `K.epsilon()`.
- __decay__: float >= 0. Learning rate decay over each update.

__引用__

- [Adadelta - an adaptive learning rate method](http://arxiv.org/abs/1212.5701)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L397)</span>
### Adam

```python
keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
```

Adam 优化器.

遵照原始论文中提供的默认参数.

__Arguments__

- __lr__: float >= 0. 学习率.
- __beta_1__: float, 0 < beta < 1. 一般接近于 1.
- __beta_2__: float, 0 < beta < 1. 一般接近于 1.
- __epsilon__: float >= 0. 模糊因子. 如果设置为 `None`, 那么默认为 `K.epsilon()`.
- __decay__: float >= 0. 每次更新学习率的下降量.
- __amsgrad__: boolean. 是否应用来自论文 On the Convergence of Adam and Beyond 的 AMSGrad 变体.

__引用__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
- [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L486)</span>
### Adamax

```python
keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
```

来自 Adam 论文第 7 节的 Adamax 优化器.

基于无穷范数的 Adam 变体. 默认参数遵照论文中给出的设定.

__参数__

- __lr__: float >= 0. 学习率.
- __beta_1/beta_2__: floats, 0 < beta < 1. 一般接近 1.
- __epsilon__: float >= 0. 模糊因子. 如果设为 `None`, 则默认是 `K.epsilon()`.
- __decay__: float >= 0. 每次更新学习率的下降量.

__References__

- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L563)</span>
### Nadam

```python
keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
```

Nesterov Adam 优化器.

如同 Adam 本质上是采用 momentum 的 RMSProp.
Nadam 是采用 Nesterov momentum 的 Adam RMSprop.

默认参数遵照论文中给出的设定.
推荐将该优化器的参数设置为默认的值.

__参数__

- __lr__: float >= 0. 学习率.
- __beta_1/beta_2__: floats, 0 < beta < 1. 一般接近于 1.
- __epsilon__: float >= 0. 模糊因子. 如果设置为 `None`, 则默认是 `K.epsilon()`.

__引用__

- [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
- [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L649)</span>
### TFOptimizer

```python
keras.optimizers.TFOptimizer(optimizer)
```

原生 TensorFlow 优化器的封装类.



