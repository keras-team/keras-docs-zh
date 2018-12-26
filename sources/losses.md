
## 损失函数的使用

损失函数（或称目标函数、优化评分函数）是编译模型时所需的两个参数之一：

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

```python
from keras import losses

model.compile(loss=losses.mean_squared_error, optimizer='sgd')
```

你可以传递一个现有的损失函数名，或者一个 TensorFlow/Theano 符号函数。
该符号函数为每个数据点返回一个标量，有以下两个参数:

- __y_true__: 真实标签。TensorFlow/Theano 张量。
- __y_pred__: 预测值。TensorFlow/Theano 张量，其 shape 与 y_true 相同。

实际的优化目标是所有数据点的输出数组的平均值。

有关这些函数的几个例子，请查看 [losses source](https://github.com/keras-team/keras/blob/master/keras/losses.py)。

## 可用损失函数

### mean_squared_error


```python
mean_squared_error(y_true, y_pred)
```

----

### mean_absolute_error


```python
mean_absolute_error(y_true, y_pred)
```

----

### mean_absolute_percentage_error


```python
mean_absolute_percentage_error(y_true, y_pred)
```

----

### mean_squared_logarithmic_error


```python
mean_squared_logarithmic_error(y_true, y_pred)
```

----

### squared_hinge


```python
squared_hinge(y_true, y_pred)
```

----

### hinge


```python
hinge(y_true, y_pred)
```

----

### categorical_hinge


```python
categorical_hinge(y_true, y_pred)
```

----

### logcosh


```python
logcosh(y_true, y_pred)
```

预测误差的双曲余弦的对数。

对于小的 `x`，`log(cosh(x))` 近似等于 `(x ** 2) / 2`。对于大的 `x`，近似于 `abs(x) - log(2)`。这表示 'logcosh' 与均方误差大致相同，但是不会受到偶尔疯狂的错误预测的强烈影响。

__参数__

- __y_true__: 目标真实值的张量。
- __y_pred__: 目标预测值的张量。

__返回__

每个样本都有一个标量损失的张量。

----

### categorical_crossentropy


```python
categorical_crossentropy(y_true, y_pred)
```

----

### sparse_categorical_crossentropy


```python
sparse_categorical_crossentropy(y_true, y_pred)
```

----

### binary_crossentropy


```python
binary_crossentropy(y_true, y_pred)
```

----

### kullback_leibler_divergence


```python
kullback_leibler_divergence(y_true, y_pred)
```

----

### poisson


```python
poisson(y_true, y_pred)
```

----

### cosine_proximity


```python
cosine_proximity(y_true, y_pred)
```


----

**注意**: 当使用 `categorical_crossentropy` 损失时，你的目标值应该是分类格式 (即，如果你有 10 个类，每个样本的目标值应该是一个 10 维的向量，这个向量除了表示类别的那个索引为 1，其他均为 0)。 为了将 *整数目标值* 转换为 *分类目标值*，你可以使用 Keras 实用函数 `to_categorical`：

```python
from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
```
