<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L14)</span>
### GaussianNoise

```python
keras.layers.GaussianNoise(stddev)
```

应用以 0 为中心的加性高斯噪声。

这对缓解过拟合很有用
（你可以将其视为随机数据增强的一种形式）。
高斯噪声（GS）是对真实输入的腐蚀过程的自然选择。

由于它是一个正则化层，因此它只在训练时才被激活。

__参数__

- __stddev__: float，噪声分布的标准差。

__输入尺寸__

可以是任意的。
如果将该层作为模型的第一层，则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L58)</span>
### GaussianDropout

```python
keras.layers.GaussianDropout(rate)
```

应用以 1 为中心的 乘性高斯噪声。

由于它是一个正则化层，因此它只在训练时才被激活。

__参数__

- __rate__: float，丢弃概率（与 `Dropout` 相同）。
这个乘性噪声的标准差为 `sqrt(rate / (1 - rate))`。

__输入尺寸__

可以是任意的。
如果将该层作为模型的第一层，则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参考文献__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L106)</span>
### AlphaDropout

```python
keras.layers.AlphaDropout(rate, noise_shape=None, seed=None)
```

将 Alpha Dropout 应用到输入。

Alpha Dropout 是一种 `Dropout`，
它保持输入的平均值和方差与原来的值不变，
以确保即使在 dropout 后也能实现自我归一化。
通过随机将激活设置为负饱和值，
Alpha Dropout 非常适合按比例缩放的指数线性单元（SELU）。

__参数__

- __rate__: float，丢弃概率（与 `Dropout` 相同）。
这个乘性噪声的标准差为 `sqrt(rate / (1 - rate))`。
- __seed__: 用作随机种子的 Python 整数。

__输入尺寸__

可以是任意的。
如果将该层作为模型的第一层，则需要指定 `input_shape` 参数
（整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参考文献__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
