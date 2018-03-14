# Keras 后端

## 什么是 「后端」？

Keras 是一个模型级库，为开发深度学习模型提供了高层次的构建模块。它不处理诸如张量乘积和卷积等低级操作。相反，它依赖于一个专门的、优化的张量操作库来完成这个操作，它可以作为 Keras 的「后端引擎」。相比单独地选择一个张量库，而将 Keras 的实现与该库相关联，Keras 以模块方式处理这个问题，并且可以将几个不同的后端引擎无缝嵌入到 Keras 中。

目前，Keras 有三个后端实现可用: **TensorFlow** 后端，**Theano** 后端，**CNTK** 后端。

- [TensorFlow](http://www.tensorflow.org/) 是由 Google 开发的一个开源符号级张量操作框架。
- [Theano](http://deeplearning.net/software/theano/) 是由蒙特利尔大学的 LISA Lab 开发的一个开源符号级张量操作框架。
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/) 是由微软开发的一个深度学习开源工具包。

将来，我们可能会添加更多后端选项。

----

## 从一个后端切换到另一个后端

如果您至少运行过一次 Keras，您将在以下位置找到 Keras 配置文件：

`$HOME/.keras/keras.json`

如果它不在那里，你可以创建它。

**Windows用户注意事项：** 请将 `$HOME` 修改为 `%USERPROFILE%`。

默认的配置文件如下所示：

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

只需将字段 `backend` 更改为 `theano`，`tensorflow` 或 `cntk`，Keras 将在下次运行 Keras 代码时使用新的配置。

你也可以定义环境变量 ``KERAS_BACKEND``，这会覆盖配置文件中定义的内容：

```bash
KERAS_BACKEND=tensorflow python -c "from keras import backend"
Using TensorFlow backend.
```

----

## keras.json 详细配置


The `keras.json` 配置文件包含以下设置：

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

您可以通过编辑 `$ HOME / .keras / keras.json` 来更改这些设置。

* `image_data_format`: 字符串，`"channels_last"` 或者 `"channels_first"`。它指定了 Keras 将遵循的数据格式约定。(`keras.backend.image_data_format()` 返回它。)
  - 对于 2D 数据 (例如图像)，`"channels_last"` 假定为 `(rows, cols, channels)`，而 `"channels_first"` 假定为 `(channels, rows, cols)`。
  - 对于 3D 数据， `"channels_last"` 假定为 `(conv_dim1, conv_dim2, conv_dim3, channels)`，而 `"channels_first"` 假定为 `(channels, conv_dim1, conv_dim2, conv_dim3)`。
* `epsilon`: 浮点数，用于避免在某些操作中被零除的数字模糊常量。
* `floatx`: 字符串，`"float16"`, `"float32"`, 或 `"float64"`。默认浮点精度。
* `backend`: 字符串， `"tensorflow"`, `"theano"`, 或 `"cntk"`。

----

## 使用抽象 Keras 后端编写新代码

如果你希望你编写的 Keras 模块与 Theano (`th`) 和 TensorFlow (`tf`) 兼容，则必须通过抽象 Keras 后端 API 来编写它们。以下是一个介绍。

您可以通过以下方式导入后端模块：
```python
from keras import backend as K
```

下面的代码实例化一个输入占位符。它等价于 `tf.placeholder()` 或 `th.tensor.matrix()`, `th.tensor.tensor3()`, 等等。

```python
inputs = K.placeholder(shape=(2, 4, 5))
# 同样可以：
inputs = K.placeholder(shape=(None, 4, 5))
# 同样可以：
inputs = K.placeholder(ndim=3)
```

下面的代码实例化一个变量。它等价于 `tf.Variable()` 或 `th.shared()`。

```python
import numpy as np
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# 全 0 变量：
var = K.zeros(shape=(3, 4, 5))
# 全 1 变量：
var = K.ones(shape=(3, 4, 5))
```

你需要的大多数张量操作都可以像在 TensorFlow 或 Theano 中那样完成：

```python
# 使用随机数初始化张量
b = K.random_uniform_variable(shape=(3, 4), low=0, high=1) # Uniform distribution
c = K.random_normal_variable(shape=(3, 4), mean=0, scale=1) # Gaussian distribution
d = K.random_normal_variable(shape=(3, 4), mean=0, scale=1)

# 张量运算
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=1)
a = K.softmax(b)
a = K.concatenate([b, c], axis=-1)
# 等等
```

----

## 后端函数


### epsilon


```python
keras.backend.epsilon()
```


返回数字表达式中使用的模糊因子的值。

__返回__

一个浮点数。

__例子__

```python
>>> keras.backend.epsilon()
1e-07
```

----

### set_epsilon


```python
keras.backend.set_epsilon(e)
```


设置数字表达式中使用的模糊因子的值。

__参数__

- __e__: 浮点数。新的 epsilon 值。

__例子__

```python
>>> from keras import backend as K
>>> K.epsilon()
1e-07
>>> K.set_epsilon(1e-05)
>>> K.epsilon()
1e-05
```

----

### floatx


```python
keras.backend.floatx()
```


以字符串形式返回默认的浮点类型。
(例如，'float16', 'float32', 'float64')。

__返回__

字符串，当前默认的浮点类型。

__例子__

```python
>>> keras.backend.floatx()
'float32'
```

----

### set_floatx


```python
keras.backend.set_floatx(floatx)
```


设置默认的浮点类型。

__参数__

- __floatx__: 字符串，'float16', 'float32', 或 'float64'。

__例子__

```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> K.set_floatx('float16')
>>> K.floatx()
'float16'
```

----

### cast_to_floatx


```python
keras.backend.cast_to_floatx(x)
```


将 Numpy 数组转换为默认的 Keras 浮点类型

__参数__

- __x__: Numpy 数组。

__返回__

相同的 Numpy 数组，转换为它的新类型。

__例子__

```python
>>> from keras import backend as K
>>> K.floatx()
'float32'
>>> arr = numpy.array([1.0, 2.0], dtype='float64')
>>> arr.dtype
dtype('float64')
>>> new_arr = K.cast_to_floatx(arr)
>>> new_arr
array([ 1.,  2.], dtype=float32)
>>> new_arr.dtype
dtype('float32')
```

----

### image_data_format


```python
keras.backend.image_data_format()
```


返回默认图像数据格式约定 ('channels_first' 或 'channels_last')。

__返回__

一个字符串，`'channels_first'` 或 `'channels_last'`

__例子__

```python
>>> keras.backend.image_data_format()
'channels_first'
```

----

### set_image_data_format


```python
keras.backend.set_image_data_format(data_format)
```


设置数据格式约定的值。

__参数__

- __data_format__: 字符串。`'channels_first'` 或 `'channels_last'`。

__例子__

```python
>>> from keras import backend as K
>>> K.image_data_format()
'channels_first'
>>> K.set_image_data_format('channels_last')
>>> K.image_data_format()
'channels_last'
```

----

### get_uid


```python
keras.backend.get_uid(prefix='')
```


获取默认计算图的 uid。

__参数__

- __prefix__: 图的可选前缀。

__返回__

图的唯一标识符。

----

### reset_uids


```python
keras.backend.reset_uids()
```

重置图的标识符。

----

### clear_session


```python
keras.backend.clear_session()
```


销毁当前的 TF 图并创建一个新图。

有用于避免旧模型/网络层混乱。

----

### manual_variable_initialization


```python
keras.backend.manual_variable_initialization(value)
```


设置变量手动初始化的标志。

这个布尔标志决定了变量是否应该在实例化时初始化（默认），
或者用户是否应该自己处理初始化
（例如通过 `tf.initialize_all_variables()`）。

__参数__

- __value__: Python 布尔值。

----

### learning_phase


```python
keras.backend.learning_phase()
```


返回学习阶段的标志。

学习阶段标志是一个布尔张量（0 = test，1 = train），
它作为输入传递给任何的 Keras 函数，以在训练和测试
时执行不同的行为操作。

__返回__

学习阶段 (标量整数张量或 python 整数)。

----

### set_learning_phase


```python
keras.backend.set_learning_phase(value)
```


将学习阶段设置为固定值。

__参数__

- __value__: 学习阶段的值，0 或 1（整数）。

__异常__

- __ValueError__: 如果 `value` 既不是 `0` 也不是 `1`。

----

### is_sparse


```python
keras.backend.is_sparse(tensor)
```

判断张量是否是稀疏张量。

__参数__

- __tensor__: 一个张量实例。

__返回__

布尔值。

__例子__

```python
>>> from keras import backend as K
>>> a = K.placeholder((2, 2), sparse=False)
>>> print(K.is_sparse(a))
False
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
```

----

### to_dense


```python
keras.backend.to_dense(tensor)
```


将稀疏张量转换为稠密张量并返回。

__参数__

- __tensor__: 张量实例（可能稀疏）。

__返回__

一个稠密张量。

__例子__

```python
>>> from keras import backend as K
>>> b = K.placeholder((2, 2), sparse=True)
>>> print(K.is_sparse(b))
True
>>> c = K.to_dense(b)
>>> print(K.is_sparse(c))
False
```

----

### variable


```python
keras.backend.variable(value, dtype=None, name=None, constraint=None)
```


实例化一个变量并返回它。

__参数__

- __value__: Numpy 数组，张量的初始值。
- __dtype__: 张量类型。
- __name__: 张量的可选名称字符串。
- __constraint__: 在优化器更新后应用于变量的可选投影函数。

__返回__

变量实例（包含 Keras 元数据）

__例子__

```python
>>> from keras import backend as K
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val, dtype='float64', name='example_var')
>>> K.dtype(kvar)
'float64'
>>> print(kvar)
example_var
>>> K.eval(kvar)
array([[ 1.,  2.],
       [ 3.,  4.]])
```

----

### constant


```python
keras.backend.constant(value, dtype=None, shape=None, name=None)
```


创建一个常数张量。

__参数__

- __value__: 一个常数值（或列表）
- __dtype__: 结果张量的元素类型。
- __shape__: 可选的结果张量的尺寸。
- __name__: 可选的张量的名称。

__返回__

一个常数张量。

----

### is_keras_tensor


```python
keras.backend.is_keras_tensor(x)
```


判断 `x` 是否是 Keras 张量

「Keras张量」是由 Keras 层（`Layer`类）或 `Input` 返回的张量。

__参数__

- __x__: 候选张量。

__返回__

布尔值：参数是否是 Keras 张量。

__异常__

- __ValueError__: 如果 `x` 不是一个符号张量。

__例子__

```python
>>> from keras import backend as K
>>> from keras.layers import Input, Dense
>>> np_var = numpy.array([1, 2])
>>> K.is_keras_tensor(np_var) # A numpy array is not a symbolic tensor.
ValueError
>>> k_var = tf.placeholder('float32', shape=(1,1))
>>> K.is_keras_tensor(k_var) # A variable indirectly created outside of keras is not a Keras tensor.
False
>>> keras_var = K.variable(np_var)
>>> K.is_keras_tensor(keras_var)  # A variable created with the keras backend is not a Keras tensor.
False
>>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
>>> K.is_keras_tensor(keras_placeholder)  # A placeholder is not a Keras tensor.
False
>>> keras_input = Input([10])
>>> K.is_keras_tensor(keras_input) # An Input is a Keras tensor.
True
>>> keras_layer_output = Dense(10)(keras_input)
>>> K.is_keras_tensor(keras_layer_output) # Any Keras layer output is a Keras tensor.
True
```

----

### placeholder


```python
keras.backend.placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None)
```


实例化一个占位符张量并返回它。

__参数__

- __shape__: 占位符尺寸
(整数元组，可能包含 `None` 项)。
- __ndim__: 张量的轴数。
{`shape`, `ndim`} 至少一个需要被指定。
如果两个都被指定，那么使用 `shape`。
- __dtype__: 占位符类型。
- __sparse__: 布尔值，占位符是否应该有一个稀疏类型。
- __name__: 可选的占位符的名称字符串。

__返回__

张量实例（包括 Keras 元数据）。

__例子__

```python
>>> from keras import backend as K
>>> input_ph = K.placeholder(shape=(2, 4, 5))
>>> input_ph._keras_shape
(2, 4, 5)
>>> input_ph
<tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
```

----

### is_placeholder


```python
keras.backend.is_placeholder(x)
```


判断 `x` 是否是占位符。

__参数__

- __x__: 候选占位符。

__返回__

布尔值。

----

### shape


```python
keras.backend.shape(x)
```


返回张量或变量的符号尺寸。

__参数__

- __x__: 张量或变量。

__返回__

符号尺寸（它本身就是张量）。

__例子__

```python
# TensorFlow 例子
>>> from keras import backend as K
>>> tf_session = K.get_session()
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> inputs = keras.backend.placeholder(shape=(2, 4, 5))
>>> K.shape(kvar)
<tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
>>> K.shape(inputs)
<tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
# 要得到整数尺寸 (相反，你可以使用 K.int_shape(x))
>>> K.shape(kvar).eval(session=tf_session)
array([2, 2], dtype=int32)
>>> K.shape(inputs).eval(session=tf_session)
array([2, 4, 5], dtype=int32)
```

----

### int_shape


```python
keras.backend.int_shape(x)
```


返回张量或变量的尺寸，作为 int 或 None 项的元组。

__参数__

- __x__: 张量或变量。

__返回__

整数元组（或 None 项）。

__例子__

```python
>>> from keras import backend as K
>>> inputs = K.placeholder(shape=(2, 4, 5))
>>> K.int_shape(inputs)
(2, 4, 5)
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.int_shape(kvar)
(2, 2)
```

----

### ndim


```python
keras.backend.ndim(x)
```


以整数形式返回张量中的轴数。

__参数__

- __x__: 张量或变量。

__返回__

Integer (scalar), number of axes.

__例子__

```python
>>> from keras import backend as K
>>> inputs = K.placeholder(shape=(2, 4, 5))
>>> val = np.array([[1, 2], [3, 4]])
>>> kvar = K.variable(value=val)
>>> K.ndim(inputs)
3
>>> K.ndim(kvar)
2
```

----

### dtype


```python
keras.backend.dtype(x)
```


以字符串形式返回 Keras 张量或变量的 dtype。

__参数__

- __x__: 张量或变量。

__返回__

字符串，`x` 的 dtype。

__例子__

```python
>>> from keras import backend as K
>>> K.dtype(K.placeholder(shape=(2,4,5)))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
'float32'
>>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
'float64'
# Keras 变量
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
>>> K.dtype(kvar)
'float32_ref'
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.dtype(kvar)
'float32_ref'
```

----

### eval


```python
keras.backend.eval(x)
```


估计一个变量的值。

__参数__

- __x__: 变量。

__返回__

Numpy 数组。

__例子__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
>>> K.eval(kvar)
array([[ 1.,  2.],
       [ 3.,  4.]], dtype=float32)
```

----

### zeros


```python
keras.backend.zeros(shape, dtype=None, name=None)
```


实例化一个全零变量并返回它。

__参数__

- __shape__: 整数元组，返回的Keras变量的尺寸。
- __dtype__: 字符串，返回的 Keras 变量的数据类型。
- __name__: 字符串，返回的 Keras 变量的名称。

__返回__

一个变量（包括 Keras 元数据），用 `0.0` 填充。
请注意，如果 `shape` 是符号化的，我们不能返回一个变量，
而会返回一个动态尺寸的张量。

__例子__

```python
>>> from keras import backend as K
>>> kvar = K.zeros((3,4))
>>> K.eval(kvar)
array([[ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.]], dtype=float32)
```

----

### ones


```python
keras.backend.ones(shape, dtype=None, name=None)
```


实例化一个全一变量并返回它。

__参数__

- __shape__: 整数元组，返回的Keras变量的尺寸。
- __dtype__: 字符串，返回的 Keras 变量的数据类型。
- __name__: 字符串，返回的 Keras 变量的名称。

__返回__

一个 Keras 变量，用 `1.0` 填充。
请注意，如果 `shape` 是符号化的，我们不能返回一个变量，
而会返回一个动态尺寸的张量。

__例子__

```python
>>> from keras import backend as K
>>> kvar = K.ones((3,4))
>>> K.eval(kvar)
array([[ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.]], dtype=float32)
```

----

### eye


```python
keras.backend.eye(size, dtype=None, name=None)
```


实例化一个单位矩阵并返回它。

__参数__

- __size__: 整数，行/列的数目。
- __dtype__: 字符串，返回的 Keras 变量的数据类型。
- __name__: 字符串，返回的 Keras 变量的名称。

__返回__

Keras 变量，一个单位矩阵。

__例子__

```python
>>> from keras import backend as K
>>> kvar = K.eye(3)
>>> K.eval(kvar)
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]], dtype=float32)
```


----

### zeros_like


```python
keras.backend.zeros_like(x, dtype=None, name=None)
```


实例化与另一个张量相同尺寸的全零变量。

__参数__

- __x__: Keras 变量或 Keras 张量。
- __dtype__: 字符串，返回的 Keras 变量的类型。
如果为 None，则使用 x 的类型。
- __name__: 字符串，所创建的变量的名称。

__返回__

一个 Keras 变量，其形状为 x，用零填充。

__例子__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_zeros = K.zeros_like(kvar)
>>> K.eval(kvar_zeros)
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
```

----

### ones_like


```python
keras.backend.ones_like(x, dtype=None, name=None)
```


实例化与另一个张量相同形状的全一变量。

__参数__

- __x__: Keras 变量或 Keras 张量。
- __dtype__: 字符串，返回的 Keras 变量的类型。
如果为 None，则使用 x 的类型。
- __name__: 字符串，所创建的变量的名称。

__返回__

一个 Keras 变量，其形状为 x，用一填充。

__例子__

```python
>>> from keras import backend as K
>>> kvar = K.variable(np.random.random((2,3)))
>>> kvar_ones = K.ones_like(kvar)
>>> K.eval(kvar_ones)
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]], dtype=float32)
```

----

### identity


```python
keras.backend.identity(x, name=None)
```


返回与输入张量相同内容的张量。

__参数__

- __x__: 输入张量。
- __name__: 字符串，所创建的变量的名称。

__返回__

一个相同尺寸、类型和内容的张量。

----

### random_uniform_variable


```python
keras.backend.random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None)
```


Instantiates a variable with values drawn from a uniform distribution.

__参数__

- __shape__: Tuple of integers, shape of returned Keras variable.
- __low__: Float, lower boundary of the output interval.
- __high__: Float, upper boundary of the output interval.
- __dtype__: String, dtype of returned Keras variable.
- __name__: String, name of returned Keras variable.
- __seed__: Integer, random seed.

__返回__

A Keras variable, filled with drawn samples.

__例子__

```python
# TensorFlow example
>>> kvar = K.random_uniform_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
>>> K.eval(kvar)
array([[ 0.10940075,  0.10047495,  0.476143  ],
       [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
```

----

### random_normal_variable


```python
keras.backend.random_normal_variable(shape, mean, scale, dtype=None, name=None, seed=None)
```


使用从正态分布中抽取的值实例化一个变量。

__参数__

- __shape__: 整数元组，返回的Keras变量的尺寸。
- __mean__: 浮点型，正态分布平均值。
- __scale__: 浮点型，正态分布标准差。
- __dtype__: 字符串，返回的Keras变量的 dtype。
- __name__: 字符串，返回的Keras变量的名称。
- __seed__: 整数，随机种子。

__返回__

一个 Keras 变量，以抽取的样本填充。

__例子__

```python
# TensorFlow 示例
>>> kvar = K.random_normal_variable((2,3), 0, 1)
>>> kvar
<tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
>>> K.eval(kvar)
array([[ 1.19591331,  0.68685907, -0.63814116],
       [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
```

----

### count_params


```python
keras.backend.count_params(x)
```


返回 Keras 变量或张量中的静态元素数。

__参数__

- __x__: Keras 变量或张量。

__返回__

整数，`x` 中的元素数量，即，数组中静态维度的乘积。

__例子__

```python
>>> kvar = K.zeros((2,3))
>>> K.count_params(kvar)
6
>>> K.eval(kvar)
array([[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]], dtype=float32)
```

----

### cast


```python
keras.backend.cast(x, dtype)
```


将张量转换到不同的 dtype 并返回。

你可以转换一个 Keras 变量，但它仍然返回一个 Keras 张量。

__参数__

- __x__: Keras 张量（或变量）。
- __dtype__: 字符串， (`'float16'`, `'float32'` 或 `'float64'`)。

__返回__

Keras 张量，类型为 `dtype`。

__例子__

```python
>>> from keras import backend as K
>>> input = K.placeholder((2, 3), dtype='float32')
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
# It doesn't work in-place as below.
>>> K.cast(input, dtype='float16')
<tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
>>> input
<tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
# you need to assign it.
>>> input = K.cast(input, dtype='float16')
>>> input
<tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
```

----

### update


```python
keras.backend.update(x, new_x)
```


将 `x` 的值更新为 `new_x`。

__参数__

- __x__: 一个 `Variable`。
- __new_x__: 一个与 `x` 尺寸相同的张量。

__返回__

更新后的变量 `x`。

----

### update_add


```python
keras.backend.update_add(x, increment)
```


通过增加 `increment` 来更新 `x` 的值。

__参数__

- __x__: 一个 `Variable`。
- __increment__: 与 `x` 形状相同的张量。

__返回__

更新后的变量 `x`。

----

### update_sub


```python
keras.backend.update_sub(x, decrement)
```


通过减 `decrement` 来更新 `x` 的值。

__参数__

- __x__: 一个 `Variable`。
- __decrement__: 与 `x` 形状相同的张量。

__返回__

更新后的变量 `x`。

----

### moving_average_update


```python
keras.backend.moving_average_update(x, value, momentum)
```


计算变量的移动平均值。

__参数__

- __x__: 一个 `Variable`。
- __value__: 与 `x` 形状相同的张量。
- __momentum__: 移动平均动量。

__返回__

更新变量的操作。

----

### dot


```python
keras.backend.dot(x, y)
```


将 2 个张量（和/或变量）相乘并返回一个*张量*。

当试图将 nD 张量与 nD 张量相乘时，
它会重现 Theano 行为。
(例如 `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

__参数__

- __x__: 张量或变量。
- __y__: 张量或变量。

__返回__

一个张量，`x` 和 `y` 的点积。

__例子__

```python
# 张量之间的点积
>>> x = K.placeholder(shape=(2, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
```

```python
# 张量之间的点积
>>> x = K.placeholder(shape=(32, 28, 3))
>>> y = K.placeholder(shape=(3, 4))
>>> xy = K.dot(x, y)
>>> xy
<tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
```

```python
# 类 Theano 行为的例子
>>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
>>> y = K.ones((4, 3, 5))
>>> xy = K.dot(x, y)
>>> K.int_shape(xy)
(2, 4, 5)
```

----

### batch_dot


```python
keras.backend.batch_dot(x, y, axes=None)
```


批量化的点积。

当 `x` 和 `y` 是批量数据时，
`batch_dot` 用于计算 `x` 和 `y` 的点积，
即尺寸为 `(batch_size, :)`。

`batch_dot` 产生一个比输入尺寸更小的张量或变量。 
如果维数减少到 1，我们使用 `expand_dims` 来确保 ndim 至少为 2。

__参数__

- __x__: `ndim >= 2` 的 Keras 张量或变量。
- __y__: `ndim >= 2` 的 Keras 张量或变量。
- __axes__: 表示目标维度的整数或列表。
`axes[0]` 和 `axes[1]` 的长度必须相同。

__返回__

一个尺寸等于 `x` 的尺寸（减去总和的维度）和 `y` 的尺寸（减去批次维度和总和的维度）的连接的张量。
如果最后的秩为 1，我们将它重新转换为 `(batch_size, 1)`。

__例子__

假设 `x = [[1, 2], [3, 4]]` 和 `y = [[5, 6], [7, 8]]`，
`batch_dot(x, y, axes=1) = [[17], [53]]` 是 `x.dot(y.T)` 的主对角线，
尽管我们不需要计算非对角元素。

尺寸推断：
让 `x` 的尺寸为 `(100, 20)`，以及 `y` 的尺寸为 `(100, 30, 20)`。
如果 `axes` 是 (1, 2)，要找出结果张量的尺寸，
循环 `x` 和 `y` 的尺寸的每一个维度。

* `x.shape[0]` : 100 : 附加到输出形状，
* `x.shape[1]` : 20 : 不附加到输出形状，
`x` 的第一个维度已经被加和了 (`dot_axes[0]` = 1)。
* `y.shape[0]` : 100 : 不附加到输出形状，总是忽略 `y` 的第一维
* `y.shape[1]` : 30 : 附加到输出形状，
* `y.shape[2]` : 20 : 不附加到输出形状，
`y` 的第二个维度已经被加和了 (`dot_axes[0]` = 2)。
`output_shape` = `(100, 30)`

```python
>>> x_batch = K.ones(shape=(32, 20, 1))
>>> y_batch = K.ones(shape=(32, 30, 20))
>>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
>>> K.int_shape(xy_batch_dot)
(32, 1, 30)
```

----

### transpose


```python
keras.backend.transpose(x)
```


将张量转置并返回。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

__例子__

```python
>>> var = K.variable([[1, 2, 3], [4, 5, 6]])
>>> K.eval(var)
array([[ 1.,  2.,  3.],
       [ 4.,  5.,  6.]], dtype=float32)
>>> var_transposed = K.transpose(var)
>>> K.eval(var_transposed)
array([[ 1.,  4.],
       [ 2.,  5.],
       [ 3.,  6.]], dtype=float32)
```

```python
>>> inputs = K.placeholder((2, 3))
>>> inputs
<tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
>>> input_transposed = K.transpose(inputs)
>>> input_transposed
<tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

```

----

### gather


```python
keras.backend.gather(reference, indices)
```

在张量 `reference` 中检索索引 `indices` 的元素。

__参数__

- __reference__: 一个张量。
- __indices__: 索引的整数张量。

__返回__

与 `reference` 类型相同的张量。

----

### max


```python
keras.backend.max(x, axis=None, keepdims=False)
```


张量中的最大值。

__参数__

- __x__: 张量或变量。
- __axis__: 一个整数，需要在哪个轴寻找最大值。
- __keepdims__: 布尔值，是否保留原尺寸。
如果 `keepdims` 为 `False`，则张量的秩减 1。
如果 `keepdims` 为 `True`，缩小的维度保留为长度 1。

__返回__

`x` 中最大值的张量。

----

### min


```python
keras.backend.min(x, axis=None, keepdims=False)
```


张量中的最小值。

__参数__

- __x__: 张量或变量。
- __axis__: 一个整数，需要在哪个轴寻找最大值。
- __keepdims__: 布尔值，是否保留原尺寸。
如果 `keepdims` 为 `False`，则张量的秩减 1。
如果 `keepdims` 为 `True`，缩小的维度保留为长度 1。

__返回__

`x` 中最小值的张量。

----

### sum


```python
keras.backend.sum(x, axis=None, keepdims=False)
```

计算张量在某一指定轴的和。

__参数__

- __x__: 张量或变量。
- __axis__: 一个整数，需要加和的轴。
- __keepdims__: 布尔值，是否保留原尺寸。
如果 `keepdims` 为 `False`，则张量的秩减 1。
如果 `keepdims` 为 `True`，缩小的维度保留为长度 1。

__返回__

`x` 的和的张量。

----

### prod


```python
keras.backend.prod(x, axis=None, keepdims=False)
```

在某一指定轴，计算张量中的值的乘积。

__参数__

- __x__: 张量或变量。
- __axis__: 一个整数需要计算乘积的轴。
- __keepdims__: 布尔值，是否保留原尺寸。
如果 `keepdims` 为 `False`，则张量的秩减 1。
如果 `keepdims` 为 `True`，缩小的维度保留为长度 1。

__返回__

`x` 的元素的乘积的张量。

----

### cumsum


```python
keras.backend.cumsum(x, axis=0)
```

在某一指定轴，计算张量中的值的累加和。


__参数__

- __x__: 张量或变量。
- __axis__: 一个整数，需要加和的轴。

__返回__

`x` 在 `axis` 轴的累加和的张量。

----

### cumprod


```python
keras.backend.cumprod(x, axis=0)
```

在某一指定轴，计算张量中的值的累积乘积。

__参数__

- __x__: 张量或变量。
- __axis__: 一个整数，需要计算乘积的轴。

__返回__

`x` 在 `axis` 轴的累乘的张量。

----

### var


```python
keras.backend.var(x, axis=None, keepdims=False)
```

张量在某一指定轴的方差。

__参数__

- __x__: 张量或变量。
- __axis__: 一个整数，要计算方差的轴。
- __keepdims__: 布尔值，是否保留原尺寸。
如果 `keepdims` 为 `False`，则张量的秩减 1。
如果 `keepdims` 为 `True`，缩小的维度保留为长度 1。

__返回__

`x` 元素的方差的张量。

----

### std


```python
keras.backend.std(x, axis=None, keepdims=False)
```


张量在某一指定轴的标准差。

__参数__

- __x__: 张量或变量。
- __axis__: 一个整数，要计算标准差的轴。
- __keepdims__: 布尔值，是否保留原尺寸。
如果 `keepdims` 为 `False`，则张量的秩减 1。
如果 `keepdims` 为 `True`，缩小的维度保留为长度 1。

__返回__

`x` 元素的标准差的张量。

----

### mean


```python
keras.backend.mean(x, axis=None, keepdims=False)
```


张量在某一指定轴的均值。

__参数__

- __x__: A tensor or variable.
- __axis__: 整数或列表。需要计算均值的轴。
- __keepdims__: 布尔值，是否保留原尺寸。
如果 `keepdims` 为 `False`，则 `axis` 中每一项的张量秩减 1。
如果 `keepdims` 为 `True`，则缩小的维度保留为长度 1。

__返回__

`x` 元素的均值的张量。

----

### any


```python
keras.backend.any(x, axis=None, keepdims=False)
```

reduction

按位归约（逻辑 OR）。

__参数__

- __x__: 张量或变量。
- __axis__: 执行归约操作的轴。
- __keepdims__: 是否放弃或广播归约的轴。

__返回__

一个 uint8 张量 (0s 和 1s)。

----

### all


```python
keras.backend.all(x, axis=None, keepdims=False)
```


按位归约（逻辑 AND）。

__参数__

- __x__: 张量或变量。
- __axis__: 执行归约操作的轴。
- __keepdims__: 是否放弃或广播归约的轴。

__返回__

一个 uint8 张量 (0s 和 1s)。

----

### argmax


```python
keras.backend.argmax(x, axis=-1)
```

返回指定轴的最大值的索引。

__参数__

- __x__: 张量或变量。
- __axis__: 执行归约操作的轴。

__返回__

一个张量。

----

### argmin


```python
keras.backend.argmin(x, axis=-1)
```


返回指定轴的最小值的索引。

__参数__

- __x__: 张量或变量。
- __axis__: 执行归约操作的轴。

__返回__

一个张量。

----

### square


```python
keras.backend.square(x)
```


元素级的平方操作。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### abs


```python
keras.backend.abs(x)
```


元素级的绝对值操作。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### sqrt


```python
keras.backend.sqrt(x)
```


元素级的平方根操作。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### exp


```python
keras.backend.exp(x)
```


元素级的指数运算操作。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### log


```python
keras.backend.log(x)
```


元素级的对数运算操作。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### logsumexp


```python
keras.backend.logsumexp(x, axis=None, keepdims=False)
```


计算 log(sum(exp(张量在某一轴的元素)))。

这个函数在数值上比 log(sum(exp(x))) 更稳定。
它避免了求大输入的指数造成的上溢，以及求小输入的对数造成的下溢。

__参数__

- __x__: 张量或变量。
- __axis__: 一个整数，需要归约的轴。
- __keepdims__: 布尔值，是否保留原尺寸。
如果 `keepdims` 为 `False`，则张量的秩减 1。
如果 `keepdims` 为 `True`，缩小的维度保留为长度 1。

__返回__

归约后的张量。

----

### round


```python
keras.backend.round(x)
```


元素级地四舍五入到最接近的整数。

在平局的情况下，使用的舍入模式是「偶数的一半」。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### sign


```python
keras.backend.sign(x)
```


元素级的符号运算。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### pow


```python
keras.backend.pow(x, a)
```


元素级的指数运算操作。

__参数__

- __x__: 张量或变量。
- __a__: Python 整数。

__返回__

一个张量。

----

### clip


```python
keras.backend.clip(x, min_value, max_value)
```

元素级裁剪。

__参数__

- __x__: 张量或变量。
- __min_value__: Python 浮点或整数。
- __max_value__: Python 浮点或整数。

__返回__

一个张量。

----

### equal


```python
keras.backend.equal(x, y)
```


逐个元素对比两个张量的相等情况。

__参数__

- __x__: 张量或变量。
- __y__: 张量或变量。

__返回__

一个布尔张量。

----

### not_equal


```python
keras.backend.not_equal(x, y)
```

逐个元素对比两个张量的不相等情况。

__参数__

- __x__: 张量或变量。
- __y__: 张量或变量。

__返回__

一个布尔张量。

----

### greater


```python
keras.backend.greater(x, y)
```


逐个元素比对 (x > y) 的真值。

__参数__

- __x__: 张量或变量。
- __y__: 张量或变量。

__返回__

一个布尔张量。

----

### greater_equal


```python
keras.backend.greater_equal(x, y)
```


逐个元素比对 (x >= y) 的真值。

__参数__

- __x__: 张量或变量。
- __y__: 张量或变量。

__返回__

一个布尔张量。

----

### less


```python
keras.backend.less(x, y)
```


逐个元素比对 (x < y) 的真值。

__参数__

- __x__: 张量或变量。
- __y__: 张量或变量。

__返回__

一个布尔张量。

----

### less_equal


```python
keras.backend.less_equal(x, y)
```


逐个元素比对 (x <= y) 的真值。

__参数__

- __x__: 张量或变量。
- __y__: 张量或变量。

__返回__

一个布尔张量。

----

### maximum


```python
keras.backend.maximum(x, y)
```


逐个元素比对两个张量的最大值。

__参数__

- __x__: 张量或变量。
- __y__: 张量或变量。

__返回__

一个张量。

### minimum


```python
keras.backend.minimum(x, y)
```


逐个元素比对两个张量的最小值。

__参数__

- __x__: 张量或变量。
- __y__: 张量或变量。

__返回__

一个张量。

----

### sin


```python
keras.backend.sin(x)
```


逐个元素计算 x 的 sin 值。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### cos


```python
keras.backend.cos(x)
```


逐个元素计算 x 的 cos 值。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### normalize_batch_in_training


```python
keras.backend.normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=0.001)
```


Computes mean and std for batch then apply batch_normalization on batch.

__参数__

- __x__: Input tensor or variable.
- __gamma__: Tensor by which to scale the input.
- __beta__: Tensor with which to center the input.
- __reduction_axes__: iterable of integers,
axes over which to normalize.
- __epsilon__: Fuzz factor.

__返回__

A tuple length of 3, `(normalized_tensor, mean, variance)`.

----

### batch_normalization


```python
keras.backend.batch_normalization(x, mean, var, beta, gamma, epsilon=0.001)
```


Applies batch normalization on x given mean, var, beta and gamma.

I.e. returns:
`output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

__参数__

- __x__: Input tensor or variable.
- __mean__: Mean of batch.
- __var__: Variance of batch.
- __beta__: Tensor with which to center the input.
- __gamma__: Tensor by which to scale the input.
- __epsilon__: Fuzz factor.

__返回__

A tensor.

----

### concatenate


```python
keras.backend.concatenate(tensors, axis=-1)
```


Concatenates a list of tensors alongside the specified axis.

__参数__

- __tensors__: list of tensors to concatenate.
- __axis__: concatenation axis.

__返回__

A tensor.

----

### reshape


```python
keras.backend.reshape(x, shape)
```


Reshapes a tensor to the specified shape.

__参数__

- __x__: Tensor or variable.
- __shape__: Target shape tuple.

__返回__

A tensor.

----

### permute_dimensions


```python
keras.backend.permute_dimensions(x, pattern)
```


Permutes axes in a tensor.

__参数__

- __x__: Tensor or variable.
- __pattern__: A tuple of
dimension indices, e.g. `(0, 2, 1)`.

__返回__

A tensor.

----

### resize_images


```python
keras.backend.resize_images(x, height_factor, width_factor, data_format)
```


Resizes the images contained in a 4D tensor.

__参数__

- __x__: Tensor or variable to resize.
- __height_factor__: Positive integer.
- __width_factor__: Positive integer.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__返回__

A tensor.

__异常__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.

----

### resize_volumes


```python
keras.backend.resize_volumes(x, depth_factor, height_factor, width_factor, data_format)
```


Resizes the volume contained in a 5D tensor.

__参数__

- __x__: Tensor or variable to resize.
- __depth_factor__: Positive integer.
- __height_factor__: Positive integer.
- __width_factor__: Positive integer.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__返回__

A tensor.

__异常__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.

----

### repeat_elements


```python
keras.backend.repeat_elements(x, rep, axis)
```


Repeats the elements of a tensor along an axis, like `np.repeat`.

If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
will have shape `(s1, s2 * rep, s3)`.

__参数__

- __x__: Tensor or variable.
- __rep__: Python integer, number of times to repeat.
- __axis__: Axis along which to repeat.

__返回__

A tensor.

----

### repeat


```python
keras.backend.repeat(x, n)
```


Repeats a 2D tensor.

if `x` has shape (samples, dim) and `n` is `2`,
the output will have shape `(samples, 2, dim)`.

__参数__

- __x__: Tensor or variable.
- __n__: Python integer, number of times to repeat.

__返回__

A tensor.

----

### arange


```python
keras.backend.arange(start, stop=None, step=1, dtype='int32')
```


Creates a 1D tensor containing a sequence of integers.

The function arguments use the same convention as
Theano's arange: if only one argument is provided,
it is in fact the "stop" argument.

The default type of the returned tensor is `'int32'` to
match TensorFlow's default.

__参数__

- __start__: Start value.
- __stop__: Stop value.
- __step__: Difference between two successive values.
- __dtype__: Integer dtype to use.

__返回__

An integer tensor.


----

### tile


```python
keras.backend.tile(x, n)
```


Creates a tensor by tiling `x` by `n`.

__参数__

- __x__: A tensor or variable
- __n__: A list of integer. The length must be the same as the number of
dimensions in `x`.

__返回__

A tiled tensor.

----

### flatten


```python
keras.backend.flatten(x)
```


Flatten a tensor.

__参数__

- __x__: A tensor or variable.

__返回__

A tensor, reshaped into 1-D

----

### batch_flatten


```python
keras.backend.batch_flatten(x)
```


Turn a nD tensor into a 2D tensor with same 0th dimension.

In other words, it flattens each data samples of a batch.

__参数__

- __x__: A tensor or variable.

__返回__

A tensor.

----

### expand_dims


```python
keras.backend.expand_dims(x, axis=-1)
```


Adds a 1-sized dimension at index "axis".

__参数__

- __x__: A tensor or variable.
- __axis__: Position where to add a new axis.

__返回__

A tensor with expanded dimensions.

----

### squeeze


```python
keras.backend.squeeze(x, axis)
```


Removes a 1-dimension from the tensor at index "axis".

__参数__

- __x__: A tensor or variable.
- __axis__: Axis to drop.

__返回__

A tensor with the same data as `x` but reduced dimensions.

----

### temporal_padding


```python
keras.backend.temporal_padding(x, padding=(1, 1))
```


Pads the middle dimension of a 3D tensor.

__参数__

- __x__: Tensor or variable.
- __padding__: Tuple of 2 integers, how many zeros to
add at the start and end of dim 1.

__返回__

A padded 3D tensor.

----

### spatial_2d_padding


```python
keras.backend.spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None)
```


Pads the 2nd and 3rd dimensions of a 4D tensor.

__参数__

- __x__: Tensor or variable.
- __padding__: Tuple of 2 tuples, padding pattern.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__返回__

A padded 4D tensor.

__异常__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.

----

### spatial_3d_padding


```python
keras.backend.spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None)
```


Pads 5D tensor with zeros along the depth, height, width dimensions.

Pads these dimensions with respectively
"padding[0]", "padding[1]" and "padding[2]" zeros left and right.

For 'channels_last' data_format,
the 2nd, 3rd and 4th dimension will be padded.
For 'channels_first' data_format,
the 3rd, 4th and 5th dimension will be padded.

__参数__

- __x__: Tensor or variable.
- __padding__: Tuple of 3 tuples, padding pattern.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__返回__

A padded 5D tensor.

__异常__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.


----

### stack


```python
keras.backend.stack(x, axis=0)
```


Stacks a list of rank `R` tensors into a rank `R+1` tensor.

__参数__

- __x__: List of tensors.
- __axis__: Axis along which to perform stacking.

__返回__

A tensor.

----

### one_hot


```python
keras.backend.one_hot(indices, num_classes)
```


Computes the one-hot representation of an integer tensor.

__参数__

- __indices__: nD integer tensor of shape
`(batch_size, dim1, dim2, ... dim(n-1))`
- __num_classes__: Integer, number of classes to consider.

__返回__

(n + 1)D one hot representation of the input
with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`

----

### reverse


```python
keras.backend.reverse(x, axes)
```


Reverse a tensor along the specified axes.

__参数__

- __x__: Tensor to reverse.
- __axes__: Integer or iterable of integers.
Axes to reverse.

__返回__

A tensor.

----

### get_value


```python
keras.backend.get_value(x)
```


Returns the value of a variable.

__参数__

- __x__: input variable.

__返回__

A Numpy array.

----

### batch_get_value


```python
keras.backend.batch_get_value(ops)
```


Returns the value of more than one tensor variable.

__参数__

- __ops__: list of ops to run.

__返回__

A list of Numpy arrays.

----

### set_value


```python
keras.backend.set_value(x, value)
```


Sets the value of a variable, from a Numpy array.

__参数__

- __x__: Tensor to set to a new value.
- __value__: Value to set the tensor to, as a Numpy array
(of the same shape).

----

### batch_set_value


```python
keras.backend.batch_set_value(tuples)
```


Sets the values of many tensor variables at once.

__参数__

- __tuples__: a list of tuples `(tensor, value)`.
`value` should be a Numpy array.

----

### print_tensor


```python
keras.backend.print_tensor(x, message='')
```


Prints `message` and the tensor value when evaluated.

Note that `print_tensor` returns a new tensor identical to `x`
which should be used in the following code. Otherwise the
print operation is not taken into account during evaluation.

__例子__

```python
>>> x = K.print_tensor(x, message="x is: ")
```

__参数__

- __x__: Tensor to print.
- __message__: Message to print jointly with the tensor.

__返回__

The same tensor `x`, unchanged.

----

### function


```python
keras.backend.function(inputs, outputs, updates=None)
```


Instantiates a Keras function.

__参数__

- __inputs__: List of placeholder tensors.
- __outputs__: List of output tensors.
- __updates__: List of update ops.
- __**kwargs__: Passed to `tf.Session.run`.

__返回__

Output values as Numpy arrays.

__异常__

- __ValueError__: if invalid kwargs are passed in.

----

### gradients


```python
keras.backend.gradients(loss, variables)
```


Returns the gradients of `variables` w.r.t. `loss`.

__参数__

- __loss__: Scalar tensor to minimize.
- __variables__: List of variables.

__返回__

A gradients tensor.

----

### stop_gradient


```python
keras.backend.stop_gradient(variables)
```


Returns `variables` but with zero gradient w.r.t. every other variable.

__参数__

- __variables__: tensor or list of tensors to consider constant with respect
to any other variable.

__返回__

A single tensor or a list of tensors (depending on the passed argument)
that has constant gradient with respect to any other variable.

----

### rnn


```python
keras.backend.rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


Iterates over the time dimension of a tensor.

__参数__

- __step_function__: RNN step function.
- __Parameters__:
- __inputs__: tensor with shape `(samples, ...)` (no time dimension),
representing input for the batch of samples at a certain
time step.
- __states__: list of tensors.
- __返回__:
- __outputs__: tensor with shape `(samples, output_dim)`
(no time dimension).
- __new_states__: list of tensors, same length and shapes
as 'states'. The first state in the list must be the
output tensor at the previous timestep.
- __inputs__: tensor of temporal data of shape `(samples, time, ...)`
(at least 3D).
- __initial_states__: tensor with shape (samples, output_dim)
(no time dimension),
containing the initial values for the states used in
the step function.
- __go_backwards__: boolean. If True, do the iteration over the time
dimension in reverse order and return the reversed sequence.
- __mask__: binary tensor with shape `(samples, time, 1)`,
with a zero for every element that is masked.
- __constants__: a list of constant values passed at each step.
- __unroll__: whether to unroll the RNN or to use a symbolic loop (`while_loop` or `scan` depending on backend).
- __input_length__: not relevant in the TensorFlow implementation.
Must be specified if using unrolling with Theano.

__返回__

A tuple, `(last_output, outputs, new_states)`.

- __last_output__: the latest output of the rnn, of shape `(samples, ...)`
- __outputs__: tensor with shape `(samples, time, ...)` where each
entry `outputs[s, t]` is the output of the step function
at time `t` for sample `s`.
- __new_states__: list of tensors, latest states returned by
the step function, of shape `(samples, ...)`.

__异常__

- __ValueError__: if input dimension is less than 3.
- __ValueError__: if `unroll` is `True` but input timestep is not a fixed number.
- __ValueError__: if `mask` is provided (not `None`) but states is not provided
(`len(states)` == 0).

----

### switch


```python
keras.backend.switch(condition, then_expression, else_expression)
```


Switches between two operations depending on a scalar value.

Note that both `then_expression` and `else_expression`
should be symbolic tensors of the *same shape*.

__参数__

- __condition__: tensor (`int` or `bool`).
- __then_expression__: either a tensor, or a callable that returns a tensor.
- __else_expression__: either a tensor, or a callable that returns a tensor.

__返回__

The selected tensor.

__异常__

- __ValueError__: If rank of `condition` is greater than rank of expressions.

----

### in_train_phase


```python
keras.backend.in_train_phase(x, alt, training=None)
```


Selects `x` in train phase, and `alt` otherwise.

Note that `alt` should have the *same shape* as `x`.

__参数__

- __x__: What to return in train phase
(tensor or callable that returns a tensor).
- __alt__: What to return otherwise
(tensor or callable that returns a tensor).
- __training__: Optional scalar tensor
(or Python boolean, or Python integer)
specifying the learning phase.

__返回__

Either `x` or `alt` based on the `training` flag.
the `training` flag defaults to `K.learning_phase()`.

----

### in_test_phase


```python
keras.backend.in_test_phase(x, alt, training=None)
```


Selects `x` in test phase, and `alt` otherwise.

Note that `alt` should have the *same shape* as `x`.

__参数__

- __x__: What to return in test phase
(tensor or callable that returns a tensor).
- __alt__: What to return otherwise
(tensor or callable that returns a tensor).
- __training__: Optional scalar tensor
(or Python boolean, or Python integer)
specifying the learning phase.

__返回__

Either `x` or `alt` based on `K.learning_phase`.

----

### relu


```python
keras.backend.relu(x, alpha=0.0, max_value=None)
```


Rectified linear unit.

With default values, it returns element-wise `max(x, 0)`.

__参数__

- __x__: A tensor or variable.
- __alpha__: A scalar, slope of negative section (default=`0.`).
- __max_value__: Saturation threshold.

__返回__

A tensor.

----

### elu


```python
keras.backend.elu(x, alpha=1.0)
```


Exponential linear unit.

__参数__

- __x__: A tensor or variable to compute the activation function for.
- __alpha__: A scalar, slope of negative section.

__返回__

A tensor.

----

### softmax


```python
keras.backend.softmax(x)
```


Softmax of a tensor.

__参数__

- __x__: A tensor or variable.

__返回__

A tensor.

----

### softplus


```python
keras.backend.softplus(x)
```


Softplus of a tensor.

__参数__

- __x__: A tensor or variable.

__返回__

A tensor.

----

### softsign


```python
keras.backend.softsign(x)
```


Softsign of a tensor.

__参数__

- __x__: A tensor or variable.

__返回__

A tensor.

----

### categorical_crossentropy


```python
keras.backend.categorical_crossentropy(target, output, from_logits=False)
```


Categorical crossentropy between an output tensor and a target tensor.

__参数__

- __target__: A tensor of the same shape as `output`.
- __output__: A tensor resulting from a softmax
(unless `from_logits` is True, in which
case `output` is expected to be the logits).
- __from_logits__: Boolean, whether `output` is the
result of a softmax, or is a tensor of logits.

__返回__

Output tensor.

----

### sparse_categorical_crossentropy


```python
keras.backend.sparse_categorical_crossentropy(target, output, from_logits=False)
```


Categorical crossentropy with integer targets.

__参数__

- __target__: An integer tensor.
- __output__: A tensor resulting from a softmax
(unless `from_logits` is True, in which
case `output` is expected to be the logits).
- __from_logits__: Boolean, whether `output` is the
result of a softmax, or is a tensor of logits.

__返回__

Output tensor.

----

### binary_crossentropy


```python
keras.backend.binary_crossentropy(target, output, from_logits=False)
```


Binary crossentropy between an output tensor and a target tensor.

__参数__

- __target__: A tensor with the same shape as `output`.
- __output__: A tensor.
- __from_logits__: Whether `output` is expected to be a logits tensor.
By default, we consider that `output`
encodes a probability distribution.

__返回__

A tensor.

----

### sigmoid


```python
keras.backend.sigmoid(x)
```


Element-wise sigmoid.

__参数__

- __x__: A tensor or variable.

__返回__

A tensor.

----

### hard_sigmoid


```python
keras.backend.hard_sigmoid(x)
```


Segment-wise linear approximation of sigmoid.

Faster than sigmoid.
Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

__参数__

- __x__: A tensor or variable.

__返回__

A tensor.

----

### tanh


```python
keras.backend.tanh(x)
```


Element-wise tanh.

__参数__

- __x__: A tensor or variable.

__返回__

A tensor.

----

### dropout


```python
keras.backend.dropout(x, level, noise_shape=None, seed=None)
```


Sets entries in `x` to zero at random, while scaling the entire tensor.

__参数__

- __x__: tensor
- __level__: fraction of the entries in the tensor
that will be set to 0.
- __noise_shape__: shape for randomly generated keep/drop flags,
must be broadcastable to the shape of `x`
- __seed__: random seed to ensure determinism.

__返回__

A tensor.

----

### l2_normalize


```python
keras.backend.l2_normalize(x, axis=None)
```


Normalizes a tensor wrt the L2 norm alongside the specified axis.

__参数__

- __x__: Tensor or variable.
- __axis__: axis along which to perform normalization.

__返回__

A tensor.

----

### in_top_k


```python
keras.backend.in_top_k(predictions, targets, k)
```


Returns whether the `targets` are in the top `k` `predictions`.

__参数__

- __predictions__: A tensor of shape `(batch_size, classes)` and type `float32`.
- __targets__: A 1D tensor of length `batch_size` and type `int32` or `int64`.
- __k__: An `int`, number of top elements to consider.

__返回__

A 1D tensor of length `batch_size` and type `bool`.
`output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
values of `predictions[i]`.

----

### conv1d


```python
keras.backend.conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


1D convolution.

__参数__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __strides__: stride integer.
- __padding__: string, `"same"`, `"causal"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: integer dilate rate.

__返回__

A tensor, result of 1D convolution.

__异常__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### conv2d


```python
keras.backend.conv2d(x, kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2D convolution.

__参数__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __strides__: strides tuple.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
Whether to use Theano or TensorFlow/CNTK data format
for inputs/kernels/outputs.
- __dilation_rate__: tuple of 2 integers.

__返回__

A tensor, result of 2D convolution.

__异常__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### conv2d_transpose


```python
keras.backend.conv2d_transpose(x, kernel, output_shape, strides=(1, 1), padding='valid', data_format=None)
```


2D deconvolution (i.e. transposed convolution).

__参数__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __output_shape__: 1D int tensor for the output shape.
- __strides__: strides tuple.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
Whether to use Theano or TensorFlow/CNTK data format
for inputs/kernels/outputs.

__返回__

A tensor, result of transposed 2D convolution.

__异常__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### separable_conv1d


```python
keras.backend.separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


1D convolution with separable filters.

__参数__

- __x__: input tensor
- __depthwise_kernel__: convolution kernel for the depthwise convolution.
- __pointwise_kernel__: kernel for the 1x1 convolution.
- __strides__: stride integer.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: integer dilation rate.

__返回__

Output tensor.

__异常__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### separable_conv2d


```python
keras.backend.separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2D convolution with separable filters.

__参数__

- __x__: input tensor
- __depthwise_kernel__: convolution kernel for the depthwise convolution.
- __pointwise_kernel__: kernel for the 1x1 convolution.
- __strides__: strides tuple (length 2).
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: tuple of integers,
dilation rates for the separable convolution.

__返回__

Output tensor.

__异常__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### depthwise_conv2d


```python
keras.backend.depthwise_conv2d(x, depthwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2D convolution with separable filters.

__参数__

- __x__: input tensor
- __depthwise_kernel__: convolution kernel for the depthwise convolution.
- __strides__: strides tuple (length 2).
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __dilation_rate__: tuple of integers,
dilation rates for the separable convolution.

__返回__

Output tensor.

__异常__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### conv3d


```python
keras.backend.conv3d(x, kernel, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1))
```


3D convolution.

__参数__

- __x__: Tensor or variable.
- __kernel__: kernel tensor.
- __strides__: strides tuple.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
Whether to use Theano or TensorFlow/CNTK data format
for inputs/kernels/outputs.
- __dilation_rate__: tuple of 3 integers.

__返回__

A tensor, result of 3D convolution.

__异常__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### conv3d_transpose


```python
keras.backend.conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1), padding='valid', data_format=None)
```


3D deconvolution (i.e. transposed convolution).

__参数__

- __x__: input tensor.
- __kernel__: kernel tensor.
- __output_shape__: 1D int tensor for the output shape.
- __strides__: strides tuple.
- __padding__: string, "same" or "valid".
- __data_format__: string, `"channels_last"` or `"channels_first"`.
Whether to use Theano or TensorFlow/CNTK data format
for inputs/kernels/outputs.

__返回__

A tensor, result of transposed 3D convolution.

__异常__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### pool2d


```python
keras.backend.pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max')
```


2D Pooling.

__参数__

- __x__: Tensor or variable.
- __pool_size__: tuple of 2 integers.
- __strides__: tuple of 2 integers.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __pool_mode__: string, `"max"` or `"avg"`.

__返回__

A tensor, result of 2D pooling.

__异常__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.
- __ValueError__: if `pool_mode` is neither `"max"` or `"avg"`.

----

### pool3d


```python
keras.backend.pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max')
```


3D Pooling.

__参数__

- __x__: Tensor or variable.
- __pool_size__: tuple of 3 integers.
- __strides__: tuple of 3 integers.
- __padding__: string, `"same"` or `"valid"`.
- __data_format__: string, `"channels_last"` or `"channels_first"`.
- __pool_mode__: string, `"max"` or `"avg"`.

__返回__

A tensor, result of 3D pooling.

__异常__

- __ValueError__: if `data_format` is neither `"channels_last"` or `"channels_first"`.
- __ValueError__: if `pool_mode` is neither `"max"` or `"avg"`.

----

### bias_add


```python
keras.backend.bias_add(x, bias, data_format=None)
```


Adds a bias vector to a tensor.

__参数__

- __x__: Tensor or variable.
- __bias__: Bias tensor to add.
- __data_format__: string, `"channels_last"` or `"channels_first"`.

__返回__

Output tensor.

__异常__

- __ValueError__: In one of the two cases below:
1. invalid `data_format` argument.
2. invalid bias shape.
the bias should be either a vector or
a tensor with ndim(x) - 1 dimension

----

### random_normal


```python
keras.backend.random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


Returns a tensor with normal distribution of values.

__参数__

- __shape__: A tuple of integers, the shape of tensor to create.
- __mean__: A float, mean of the normal distribution to draw samples.
- __stddev__: A float, standard deviation of the normal distribution
to draw samples.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__返回__

A tensor.

----

### random_uniform


```python
keras.backend.random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None)
```


Returns a tensor with uniform distribution of values.

__参数__

- __shape__: A tuple of integers, the shape of tensor to create.
- __minval__: A float, lower boundary of the uniform distribution
to draw samples.
- __maxval__: A float, upper boundary of the uniform distribution
to draw samples.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__返回__

A tensor.

----

### random_binomial


```python
keras.backend.random_binomial(shape, p=0.0, dtype=None, seed=None)
```


Returns a tensor with random binomial distribution of values.

__参数__

- __shape__: A tuple of integers, the shape of tensor to create.
- __p__: A float, `0. <= p <= 1`, probability of binomial distribution.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__返回__

A tensor.

----

### truncated_normal


```python
keras.backend.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


Returns a tensor with truncated random normal distribution of values.

The generated values follow a normal distribution
with specified mean and standard deviation,
except that values whose magnitude is more than
two standard deviations from the mean are dropped and re-picked.

__参数__

- __shape__: A tuple of integers, the shape of tensor to create.
- __mean__: Mean of the values.
- __stddev__: Standard deviation of the values.
- __dtype__: String, dtype of returned tensor.
- __seed__: Integer, random seed.

__返回__

A tensor.

----

### ctc_label_dense_to_sparse


```python
keras.backend.ctc_label_dense_to_sparse(labels, label_lengths)
```


Converts CTC labels from dense to sparse.

__参数__

- __labels__: dense CTC labels.
- __label_lengths__: length of the labels.

__返回__

A sparse tensor representation of the labels.

----

### ctc_batch_cost


```python
keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
```


Runs CTC loss algorithm on each batch element.

__参数__

- __y_true__: tensor `(samples, max_string_length)`
containing the truth labels.
- __y_pred__: tensor `(samples, time_steps, num_categories)`
containing the prediction, or output of the softmax.
- __input_length__: tensor `(samples, 1)` containing the sequence length for
each batch item in `y_pred`.
- __label_length__: tensor `(samples, 1)` containing the sequence length for
each batch item in `y_true`.

__返回__

Tensor with shape (samples,1) containing the
CTC loss of each element.

----

### ctc_decode


```python
keras.backend.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
```


Decodes the output of a softmax.

Can use either greedy search (also known as best path)
or a constrained dictionary search.

__参数__

- __y_pred__: tensor `(samples, time_steps, num_categories)`
containing the prediction, or output of the softmax.
- __input_length__: tensor `(samples, )` containing the sequence length for
each batch item in `y_pred`.
- __greedy__: perform much faster best-path search if `true`.
This does not use a dictionary.
- __beam_width__: if `greedy` is `false`: a beam search decoder will be used
with a beam of this width.
- __top_paths__: if `greedy` is `false`,
how many of the most probable paths will be returned.

__返回__

- __Tuple__:
- __List__: if `greedy` is `true`, returns a list of one element that
contains the decoded sequence.
If `false`, returns the `top_paths` most probable
decoded sequences.
- __Important__: blank labels are returned as `-1`.
Tensor `(top_paths, )` that contains
the log probability of each decoded sequence.

----

### map_fn


```python
keras.backend.map_fn(fn, elems, name=None, dtype=None)
```


Map the function fn over the elements elems and return the outputs.

__参数__

- __fn__: Callable that will be called upon each element in elems
- __elems__: tensor
- __name__: A string name for the map node in the graph
- __dtype__: Output data type.

__返回__

Tensor with dtype `dtype`.

----

### foldl


```python
keras.backend.foldl(fn, elems, initializer=None, name=None)
```


Reduce elems using fn to combine them from left to right.

__参数__

- __fn__: Callable that will be called upon each element in elems and an
accumulator, for instance `lambda acc, x: acc + x`
- __elems__: tensor
- __initializer__: The first value used (`elems[0]` in case of None)
- __name__: A string name for the foldl node in the graph

__返回__

Tensor with same type and shape as `initializer`.

----

### foldr


```python
keras.backend.foldr(fn, elems, initializer=None, name=None)
```


Reduce elems using fn to combine them from right to left.

__参数__

- __fn__: Callable that will be called upon each element in elems and an
accumulator, for instance `lambda acc, x: acc + x`
- __elems__: tensor
- __initializer__: The first value used (`elems[-1]` in case of None)
- __name__: A string name for the foldr node in the graph

__返回__

Tensor with same type and shape as `initializer`.

----

### local_conv1d


```python
keras.backend.local_conv1d(inputs, kernel, kernel_size, strides, data_format=None)
```


Apply 1D conv with un-shared weights.

__参数__

- __inputs__: 3D tensor with shape: (batch_size, steps, input_dim)
- __kernel__: the unshared weight for convolution,
with shape (output_length, feature_dim, filters)
- __kernel_size__: a tuple of a single integer,
specifying the length of the 1D convolution window
- __strides__: a tuple of a single integer,
specifying the stride length of the convolution
- __data_format__: the data format, channels_first or channels_last

__返回__

the tensor after 1d conv with un-shared weights, with shape (batch_size, output_length, filters)

__异常__

- __ValueError__: if `data_format` is neither `channels_last` or `channels_first`.

----

### local_conv2d


```python
keras.backend.local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None)
```


Apply 2D conv with un-shared weights.

__参数__

- __inputs__: 4D tensor with shape:
(batch_size, filters, new_rows, new_cols)
if data_format='channels_first'
or 4D tensor with shape:
(batch_size, new_rows, new_cols, filters)
if data_format='channels_last'.
- __kernel__: the unshared weight for convolution,
with shape (output_items, feature_dim, filters)
- __kernel_size__: a tuple of 2 integers, specifying the
width and height of the 2D convolution window.
- __strides__: a tuple of 2 integers, specifying the strides
of the convolution along the width and height.
- __output_shape__: a tuple with (output_row, output_col)
- __data_format__: the data format, channels_first or channels_last

__返回__

A 4d tensor with shape:
(batch_size, filters, new_rows, new_cols)
if data_format='channels_first'
or 4D tensor with shape:
(batch_size, new_rows, new_cols, filters)
if data_format='channels_last'.

__异常__

- __ValueError__: if `data_format` is neither
`channels_last` or `channels_first`.

----

### backend


```python
backend.backend()
```


Publicly accessible method
for determining the current backend.

__返回__

String, the name of the backend Keras is currently using.

__例子__

```python
>>> keras.backend.backend()
'tensorflow'
```






