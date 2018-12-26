<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/normalization.py#L16)</span>
### BatchNormalization

```python
keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)
```

批量标准化层 (Ioffe and Szegedy, 2014)。

在每一个批次的数据中标准化前一层的激活项，
即，应用一个维持激活项平均值接近 0，标准差接近 1 的转换。

__参数__

- __axis__: 整数，需要标准化的轴
（通常是特征轴）。
例如，在 `data_format="channels_first"` 的 `Conv2D` 层之后，
在 `BatchNormalization` 中设置 `axis=1`。
- __momentum__: 移动均值和移动方差的动量。
- __epsilon__: 增加到方差的小的浮点数，以避免除以零。
- __center__: 如果为 True，把 `beta` 的偏移量加到标准化的张量上。
如果为 False， `beta` 被忽略。
- __scale__: 如果为 True，乘以 `gamma`。
如果为 False，`gamma` 不使用。
当下一层为线性层（或者例如 `nn.relu`），
这可以被禁用，因为缩放将由下一层完成。
- __beta_initializer__: beta 权重的初始化方法。
- __gamma_initializer__: gamma 权重的初始化方法。
- __moving_mean_initializer__: 移动均值的初始化方法。
- __moving_variance_initializer__: 移动方差的初始化方法。
- __beta_regularizer__: 可选的 beta 权重的正则化方法。
- __gamma_regularizer__: 可选的 gamma 权重的正则化方法。
- __beta_constraint__: 可选的 beta 权重的约束方法。
- __gamma_constraint__: 可选的 gamma 权重的约束方法。

__输入尺寸__

可以是任意的。如果将这一层作为模型的第一层， 则需要指定 `input_shape` 参数 （整数元组，不包含样本数量的维度）。

__输出尺寸__

与输入相同。

__参考文献__

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
