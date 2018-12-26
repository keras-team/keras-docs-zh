<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L19)</span>
### LeakyReLU

```python
keras.layers.LeakyReLU(alpha=0.3)
```

带泄漏的 ReLU。

当神经元未激活时，它仍允许赋予一个很小的梯度：
`f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`.

__输入尺寸__

可以是任意的。如果将该层作为模型的第一层，
则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __alpha__: float >= 0。负斜率系数。

__参考文献__

- [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L59)</span>
### PReLU

```python
keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```

参数化的 ReLU。

形式：
`f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`,
其中 `alpha` 是一个可学习的数组，尺寸与 x 相同。

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层，
则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __alpha_initializer__: 权重的初始化函数。
- __alpha_regularizer__: 权重的正则化方法。
- __alpha_constraint__: 权重的约束。
- __shared_axes__: 激活函数共享可学习参数的轴。
例如，如果输入特征图来自输出形状为 `(batch, height, width, channels)`
的 2D 卷积层，而且你希望跨空间共享参数，以便每个滤波器只有一组参数，
可设置 `shared_axes=[1, 2]`。

__参考文献__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L153)</span>
### ELU

```python
keras.layers.ELU(alpha=1.0)
```

指数线性单元。

形式：
`f(x) =  alpha * (exp(x) - 1.) for x < 0`,
`f(x) = x for x >= 0`.

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层，
则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __alpha__: 负因子的尺度。

__参考文献__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289v1)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L193)</span>
### ThresholdedReLU

```python
keras.layers.ThresholdedReLU(theta=1.0)
```

带阈值的修正线性单元。

形式：
`f(x) = x for x > theta`,
`f(x) = 0 otherwise`.

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层，
则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __theta__: float >= 0。激活的阈值位。

__参考文献__

- [Zero-Bias Autoencoders and the Benefits of Co-Adapting Features](http://arxiv.org/abs/1402.3337)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L233)</span>
### Softmax

```python
keras.layers.Softmax(axis=-1)
```

Softmax 激活函数。

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层，
则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __axis__: 整数，应用 softmax 标准化的轴。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/advanced_activations.py#L265)</span>
### ReLU

```python
keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
```

ReLU 激活函数。

使用默认值时，它返回逐个元素的 `max(x，0)`。

否则：

- 如果 `x >= max_value`，返回 `f(x) = max_value`，
- 如果 `threshold <= x < max_value`，返回 `f(x) = x`,
- 否则，返回 `f(x) = negative_slope * (x - threshold)`。

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层，
则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参数__

- __max_value__: 浮点数，最大的输出值。
- __negative_slope__: float >= 0. 负斜率系数。
- __threshold__: float。"thresholded activation" 的阈值。
