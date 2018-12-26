## 约束项的使用

`constraints` 模块的函数允许在优化期间对网络参数设置约束（例如非负性）。

约束是以层为对象进行的。具体的 API 因层而异，但 `Dense`，`Conv1D`，`Conv2D` 和 `Conv3D` 这些层具有统一的 API。

约束层开放 2 个关键字参数：

- `kernel_constraint` 用于主权重矩阵。
- `bias_constraint` 用于偏置。

```python
from keras.constraints import max_norm
model.add(Dense(64, kernel_constraint=max_norm(2.)))
```

## 可用的约束


<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/constraints.py#L22)</span>
### MaxNorm

```python
keras.constraints.MaxNorm(max_value=2, axis=0)
```

MaxNorm 最大范数权值约束。

映射到每个隐藏单元的权值的约束，使其具有小于或等于期望值的范数。

__参数__

- __m__: 输入权值的最大范数。
- __axis__: 整数，需要计算权值范数的轴。
    例如，在 `Dense` 层中权值矩阵的尺寸为 `(input_dim, output_dim)`，
    设置 `axis` 为 `0` 以约束每个长度为 `(input_dim,)` 的权值向量。
    在 `Conv2D` 层（`data_format="channels_last"`）中，权值张量的尺寸为
    `(rows, cols, input_depth, output_depth)`，设置 `axis` 为 `[0, 1, 2]` 
    以越是每个尺寸为 `(rows, cols, input_depth)` 的滤波器张量的权值。

__参考文献__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting]
(http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/constraints.py#L62)</span>
### NonNeg

```python
keras.constraints.NonNeg()
```

权重非负的约束。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/constraints.py#L71)</span>
### UnitNorm

```python
keras.constraints.UnitNorm(axis=0)
```

映射到每个隐藏单元的权值的约束，使其具有单位范数。

__参数__

- __axis__: 整数，需要计算权值范数的轴。
    例如，在 `Dense` 层中权值矩阵的尺寸为 `(input_dim, output_dim)`，
    设置 `axis` 为 `0` 以约束每个长度为 `(input_dim,)` 的权值向量。
    在 `Conv2D` 层（`data_format="channels_last"`）中，权值张量的尺寸为
    `(rows, cols, input_depth, output_depth)`，设置 `axis` 为 `[0, 1, 2]` 
    以越是每个尺寸为 `(rows, cols, input_depth)` 的滤波器张量的权值。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/constraints.py#L100)</span>
### MinMaxNorm

```python
keras.constraints.MinMaxNorm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)
```

MinMaxNorm 最小/最大范数权值约束。

映射到每个隐藏单元的权值的约束，使其范数在上下界之间。

__参数__

- __min_value__: 输入权值的最小范数。
- __max_value__: 输入权值的最大范数。
- __rate__: 强制执行约束的比例：权值将被重新调整为
    `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`。
    实际上，这意味着 rate = 1.0 代表严格执行约束，而 rate <1.0 意味着权值
    将在每一步重新调整以缓慢移动到所需间隔内的值。
- __axis__: 整数，需要计算权值范数的轴。
    例如，在 `Dense` 层中权值矩阵的尺寸为 `(input_dim, output_dim)`，
    设置 `axis` 为 `0` 以约束每个长度为 `(input_dim,)` 的权值向量。
    在 `Conv2D` 层（`data_format="channels_last"`）中，权值张量的尺寸为
    `(rows, cols, input_depth, output_depth)`，设置 `axis` 为 `[0, 1, 2]` 
    以越是每个尺寸为 `(rows, cols, input_depth)` 的滤波器张量的权值。


---