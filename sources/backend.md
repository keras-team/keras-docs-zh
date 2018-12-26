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

在 Keras 中，可以加载比 `"tensorflow"`, `"theano"` 和 `"cntk"` 更多的后端。
Keras 也可以使用外部后端，这可以通过更改 `keras.json` 配置文件和 `"backend"` 设置来执行。 假设您有一个名为 `my_module` 的 Python 模块，您希望将其用作外部后端。`keras.json` 配置文件将更改如下：

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "my_package.my_module"
}
```

必须验证外部后端才能使用，有效的后端必须具有以下函数：`placeholder`, `variable` and `function`.

如果由于缺少必需的条目而导致外部后端无效，则会记录错误，通知缺少哪些条目。

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

您可以通过编辑 `$ HOME/.keras/keras.json` 来更改这些设置。

- `image_data_format`: 字符串，`"channels_last"` 或者 `"channels_first"`。它指定了 Keras 将遵循的数据格式约定。(`keras.backend.image_data_format()` 返回它。)
    - 对于 2D 数据 (例如图像)，`"channels_last"` 假定为 `(rows, cols, channels)`，而 `"channels_first"` 假定为 `(channels, rows, cols)`。
    - 对于 3D 数据， `"channels_last"` 假定为 `(conv_dim1, conv_dim2, conv_dim3, channels)`，而 `"channels_first"` 假定为 `(channels, conv_dim1, conv_dim2, conv_dim3)`。
- `epsilon`: 浮点数，用于避免在某些操作中被零除的数字模糊常量。
- `floatx`: 字符串，`"float16"`, `"float32"`, 或 `"float64"`。默认浮点精度。
- `backend`: 字符串， `"tensorflow"`, `"theano"`, 或 `"cntk"`。

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
b = K.random_uniform_variable(shape=(3, 4), low=0, high=1) # 均匀分布
c = K.random_normal_variable(shape=(3, 4), mean=0, scale=1) # 高斯分布
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


将 Numpy 数组转换为默认的 Keras 浮点类型。

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

### reset_uids


```python
keras.backend.reset_uids()
```

重置图的标识符。


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
>>> K.is_keras_tensor(np_var) # 一个 Numpy 数组不是一个符号张量。
ValueError
>>> k_var = tf.placeholder('float32', shape=(1,1))
>>> K.is_keras_tensor(k_var) # 在 Keras 之外间接创建的变量不是 Keras 张量。
False
>>> keras_var = K.variable(np_var)
>>> K.is_keras_tensor(keras_var)  # Keras 后端创建的变量不是 Keras 张量。
False
>>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
>>> K.is_keras_tensor(keras_placeholder)  # 占位符不是 Keras 张量。
False
>>> keras_input = Input([10])
>>> K.is_keras_tensor(keras_input) # 输入 Input 是 Keras 张量。
True
>>> keras_layer_output = Dense(10)(keras_input)
>>> K.is_keras_tensor(keras_layer_output) # 任何 Keras 层输出都是 Keras 张量。
True
```

----

### is_tensor

```python
keras.backend.is_tensor(x)
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

__Numpy 实现__

```python
def int_shape(x):
    return x.shape
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

整数 (标量), 轴的数量。

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

__Numpy 实现__

```python
def ndim(x):
    return x.ndim
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

- __x__: Keras 变量或张量。
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

使用从均匀分布中抽样出来的值来实例化变量。

__参数__

- __shape__: 整数元组，返回的 Keras 变量的尺寸。
- __low__: 浮点数，输出间隔的下界。
- __high__: 浮点数，输出间隔的上界。
- __dtype__: 字符串，返回的 Keras 变量的数据类型。
- __name__: 字符串，返回的 Keras 变量的名称。
- __seed__: 整数，随机种子。

__返回__

一个 Keras 变量，以抽取的样本填充。

__例子__

```python
# TensorFlow 示例
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

伪代码：

```python
inner_products = []
for xi, yi in zip(x, y):
    inner_products.append(xi.dot(yi))
result = stack(inner_products)
```

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


__Numpy 实现__

```python
def gather(reference, indices):
    return reference[indices]
```

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


__Numpy 实现__

```python
def max(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.max(x, axis=axis, keepdims=keepdims)
```

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

__Numpy 实现__

```python
def min(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.min(x, axis=axis, keepdims=keepdims)
```

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


__Numpy 实现__

```python
def sum(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.sum(x, axis=axis, keepdims=keepdims)
```

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


__Numpy 实现__

```python
def prod(x, axis=None, keepdims=False):
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.prod(x, axis=axis, keepdims=keepdims)
```

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

__Numpy 实现__

```python
def equal(x, y):
    return x == y
```

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

__Numpy 实现__

```python
def not_equal(x, y):
    return x != y
```

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

__Numpy 实现__

```python
def greater(x, y):
    return x > y
```

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

__Numpy 实现__

```python
def greater_equal(x, y):
    return x >= y
```

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

__Numpy 实现__

```python
def less(x, y):
    return x < y
```

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

__Numpy 实现__

```python
def less_equal(x, y):
    return x <= y
```

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

__Numpy 实现__

```python
def maximum(x, y):
    return np.maximum(x, y)
```

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

__Numpy 实现__

```python
def minimum(x, y):
    return np.minimum(x, y)
```

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


计算批次的均值和标准差，然后在批次上应用批次标准化。

__参数__

- __x__: 输入张量或变量。
- __gamma__: 用于缩放输入的张量。
- __beta__: 用于中心化输入的张量。
- __reduction_axes__: 整数迭代，需要标准化的轴。
- __epsilon__: 模糊因子。

__返回__

长度为 3 个元组，`(normalized_tensor, mean, variance)`。

----

### batch_normalization


```python
keras.backend.batch_normalization(x, mean, var, beta, gamma, epsilon=0.001)
```


在给定的 mean，var，beta 和 gamma 上应用批量标准化。

即，返回：
`output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

__参数__

- __x__: 输入张量或变量。
- __mean__: 批次的均值。
- __var__: 批次的方差。
- __beta__: 用于中心化输入的张量。
- __gamma__: 用于缩放输入的张量。
- __epsilon__: 模糊因子。

__返回__

一个张量。

----

### concatenate


```python
keras.backend.concatenate(tensors, axis=-1)
```


基于指定的轴，连接张量的列表。

__参数__

- __tensors__: 需要连接的张量列表。
- __axis__: 连接的轴。

__返回__

一个张量。

----

### reshape


```python
keras.backend.reshape(x, shape)
```


将张量重塑为指定的尺寸。

__参数__

- __x__: 张量或变量。
- __shape__: 目标尺寸元组。

__返回__

一个张量。

----

### permute_dimensions


```python
keras.backend.permute_dimensions(x, pattern)
```


重新排列张量的轴。

__参数__

- __x__: 张量或变量。
- __pattern__: 维度索引的元组，例如 `(0, 2, 1)`。

__返回__

一个张量。

----

### resize_images


```python
keras.backend.resize_images(x, height_factor, width_factor, data_format)
```


调整 4D 张量中包含的图像的大小。

__参数__

- __x__: 需要调整的张量或变量。
- __height_factor__: 正整数。
- __width_factor__: 正整数。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。

__返回__

一个张量。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `"channels_last"` 也不是 `"channels_first"`。

----

### resize_volumes


```python
keras.backend.resize_volumes(x, depth_factor, height_factor, width_factor, data_format)
```


调整 5D 张量中包含的体积。

__参数__

- __x__: 需要调整的张量或变量。
- __depth_factor__: 正整数。
- __height_factor__: 正整数。
- __width_factor__: 正整数。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。

__返回__

一个张量。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `"channels_last"` 也不是 `"channels_first"`。

----

### repeat_elements


```python
keras.backend.repeat_elements(x, rep, axis)
```


沿某一轴重复张量的元素，如 `np.repeat`。

如果 `x` 的尺寸为 `(s1，s2，s3)` 而 `axis` 为 `1`，
则输出尺寸为 `(s1，s2 * rep，s3）`。

__参数__

- __x__: 张量或变量。
- __rep__: Python 整数，重复次数。
- __axis__: 需要重复的轴。

__返回__

一个张量。

----

### repeat


```python
keras.backend.repeat(x, n)
```


重复一个 2D 张量。

如果 `x` 的尺寸为 `(samples, dim)` 并且 `n` 为 `2`，
则输出的尺寸为 `(samples, 2, dim)`。

__参数__

- __x__: 张量或变量。
- __n__: Python 整数，重复次数。

__返回__

一个张量。

----

### arange


```python
keras.backend.arange(start, stop=None, step=1, dtype='int32')
```


创建一个包含整数序列的 1D 张量。

该函数参数与 Theano 的 `arange` 函数的约定相同：
如果只提供了一个参数，那它就是 `stop` 参数。

返回的张量的默认类型是 `int32`，以匹配 TensorFlow 的默认值。

__参数__

- __start__: 起始值。
- __stop__: 结束值。
- __step__: 两个连续值之间的差。
- __dtype__: 要使用的整数类型。

__返回__

一个整数张量。


----

### tile


```python
keras.backend.tile(x, n)
```

创建一个用 `n` 平铺 的 `x` 张量。

__参数__

- __x__: 张量或变量。
- __n__: 整数列表。长度必须与 `x` 中的维数相同。

__返回__

一个平铺的张量。

----

### flatten


```python
keras.backend.flatten(x)
```


展平一个张量。

__参数__

- __x__: 张量或变量。

__返回__

一个重新调整为 1D 的张量。

----

### batch_flatten


```python
keras.backend.batch_flatten(x)
```

将一个 nD 张量变成一个 第 0 维相同的 2D 张量。

换句话说，它将批次中的每一个样本展平。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### expand_dims


```python
keras.backend.expand_dims(x, axis=-1)
```

在索引 `axis` 轴，添加 1 个尺寸的维度。

__参数__

- __x__: 张量或变量。
- __axis__: 需要添加新的轴的位置。

__返回__

一个扩展维度的轴。

----

### squeeze


```python
keras.backend.squeeze(x, axis)
```


在索引 `axis` 轴，移除 1 个尺寸的维度。

__参数__

- __x__: 张量或变量。
- __axis__: 需要丢弃的轴。

__返回__

一个与 `x` 数据相同但维度降低的张量。

----

### temporal_padding


```python
keras.backend.temporal_padding(x, padding=(1, 1))
```


填充 3D 张量的中间维度。

__参数__

- __x__: 张量或变量。
- __padding__: 2 个整数的元组，在第一个维度的开始和结束处添加多少个零。
__返回__

一个填充的 3D 张量。

----

### spatial_2d_padding


```python
keras.backend.spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None)
```

填充 4D 张量的第二维和第三维。

__参数__

- __x__: 张量或变量。
- __padding__: 2 元组的元组，填充模式。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。

__返回__

一个填充的 4D 张量。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `"channels_last"` 也不是 `"channels_first"`。

----

### spatial_3d_padding


```python
keras.backend.spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None)
```


沿着深度、高度宽度三个维度填充 5D 张量。

分别使用 "padding[0]", "padding[1]" 和 "padding[2]" 来左右填充这些维度。

对于 'channels_last' 数据格式，
第 2、3、4 维将被填充。
对于 'channels_first' 数据格式，
第 3、4、5 维将被填充。

__参数__

- __x__: 张量或变量。
- __padding__: 3 元组的元组，填充模式。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。

__返回__

一个填充的 5D 张量。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `"channels_last"` 也不是 `"channels_first"`。


----

### stack


```python
keras.backend.stack(x, axis=0)
```

将秩 为 `R` 的张量列表堆叠成秩为 `R + 1` 的张量。

__参数__

- __x__: 张量列表。
- __axis__: 需要执行堆叠的轴。

__返回__

一个张量。

----

### one_hot


```python
keras.backend.one_hot(indices, num_classes)
```


计算一个整数张量的 one-hot 表示。

__参数__

- __indices__: nD 整数，尺寸为
`(batch_size, dim1, dim2, ... dim(n-1))`
- __num_classes__: 整数，需要考虑的类别数。

__返回__

输入的 (n + 1)D one-hot 表示，
尺寸为 `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`。

----

### reverse


```python
keras.backend.reverse(x, axes)
```


沿指定的轴反转张量。

__参数__

- __x__: 需要反转的张量。
- __axes__: 整数或整数迭代。需要反转的轴。

__返回__

一个张量。

__Numpy 实现__

```python
def reverse(x, axes):
    if isinstance(axes, list):
        axes = tuple(axes)
    return np.flip(x, axes)
```

----

### slice

```python
keras.backend.slice(x, start, size)
```

从张量中提取一个切片。

__参数__

- __x__: 输入张量。
- __start__: 整数列表/元组，表明每个轴的起始切片索引位置。
- __size__: 整数列表/元组，表明每个轴上切片多少维度。

__返回__

一个切片张量：

```python
new_x = x[start[0]: start[0] + size[0], ..., start[-1]: start[-1] + size[-1]]
```

----

### get_value


```python
keras.backend.get_value(x)
```


返回一个变量的值。

__参数__

- __x__: 输入变量。

__返回__

一个 Numpy 数组。

----

### batch_get_value


```python
keras.backend.batch_get_value(ops)
```


返回多个张量变量的值。

__参数__

- __ops__: 要运行的操作列表。

__返回__

一个 Numpy 数组的列表。

----

### set_value


```python
keras.backend.set_value(x, value)
```


使用 Numpy 数组设置变量的值。

__参数__

- __x__: 需要设置新值的张量。
- __value__: 需要设置的值，
一个尺寸相同的 Numpy 数组。

----

### batch_set_value


```python
keras.backend.batch_set_value(tuples)
```


一次设置多个张量变量的值。

__参数__

- __tuples__: 元组 `(tensor, value)` 的列表。
`value` 应该是一个 Numpy 数组。

----

### print_tensor


```python
keras.backend.print_tensor(x, message='')
```

在评估时打印 `message` 和张量的值。

请注意，`print_tensor` 返回一个与 `x` 相同的新张量，应该在后面的代码中使用它。否则在评估过程中不会考虑打印操作。

__例子__

```python
>>> x = K.print_tensor(x, message="x is: ")
```

__参数__

- __x__: 需要打印的张量。
- __message__: 需要与张量一起打印的消息。

__返回__

同一个不变的张量 `x`。

----

### function


```python
keras.backend.function(inputs, outputs, updates=None)
```


实例化 Keras 函数。

__参数__

- __inputs__: 占位符张量列表。
- __outputs__: 输出张量列表。
- __updates__: 更新操作列表。
- __**kwargs__: 需要传递给 `tf.Session.run` 的参数。

__返回__

输出值为 Numpy 数组。

__异常__

- __ValueError__: 如果无效的 kwargs 被传入。

----

### gradients


```python
keras.backend.gradients(loss, variables)
```


返回 `variables` 在 `loss` 上的梯度。

__参数__

- __loss__: 需要最小化的标量张量。
- __variables__: 变量列表。

__返回__

一个梯度张量。

----

### stop_gradient


```python
keras.backend.stop_gradient(variables)
```

返回 `variables`，但是对于其他变量，其梯度为零。

__参数__

- __variables__: 需要考虑的张量或张量列表，任何的其他变量保持不变。

__返回__

单个张量或张量列表（取决于传递的参数），
与任何其他变量具有恒定的梯度。


----

### rnn


```python
keras.backend.rnn(step_function, inputs, initial_states, go_backwards=False, mask=None, constants=None, unroll=False, input_length=None)
```


在张量的时间维度迭代。

__参数__

- __step_function__: RNN 步骤函数，
- __inputs__: 尺寸为 `(samples, ...)` 的张量
(不含时间维度), 表示批次样品在某个时间步的输入。
- __states__: 张量列表。
- __outputs__: 尺寸为 `(samples, output_dim)` 的张量
(不含时间维度)
- __new_states__: 张量列表，与 `states` 长度和尺寸相同。
列表中的第一个状态必须是前一个时间步的输出张量。
- __inputs__: 时序数据张量 `(samples, time, ...)`
(最少 3D)。
- __initial_states__: 尺寸为 `(samples, output_dim)` 的张量
(不含时间维度)，包含步骤函数中使用的状态的初始值。
- __go_backwards__: 布尔值。如果为 True，以相反的顺序在时间维上进行迭代并返回相反的序列。
- __mask__: 尺寸为 `(samples, time, 1)` 的二进制张量，对于被屏蔽的每个元素都为零。
- __constants__: 每个步骤传递的常量值列表。
- __unroll__: 是否展开 RNN 或使用符号循环（依赖于后端的 `while_loop`或 `scan`）。
- __input_length__: 与 TensorFlow 实现不相关。如果使用 Theano 展开，则必须指定。

__返回__

一个元组，`(last_output, outputs, new_states)`。

- __last_output__: rnn 的最后输出，尺寸为 `(samples, ...)`。
- __outputs__: 尺寸为 `(samples, time, ...)` 的张量，其中
每一项 `outputs[s, t]` 是样本 `s` 在时间 `t` 的步骤函数输出值。
- __new_states__: 张量列表，有步骤函数返回的最后状态，
尺寸为 `(samples, ...)`。

__异常__

- __ValueError__: 如果输入的维度小于 3。
- __ValueError__: 如果 `unroll` 为 `True` 但输入时间步并不是固定的数字。
- __ValueError__: 如果提供了 `mask` (非 `None`) 但未提供 `states` (`len(states)` == 0)。

----

### switch


```python
keras.backend.switch(condition, then_expression, else_expression)
```


根据一个标量值在两个操作之间切换。

请注意，`then_expression` 和 `else_expression` 
都应该是*相同尺寸*的符号张量。

__参数__

- __condition__: 张量 (`int` 或 `bool`)。
- __then_expression__: 张量或返回张量的可调用函数。
- __else_expression__: 张量或返回张量的可调用函数。

__返回__

选择的张量。

__异常__

- __ValueError__: 如果 `condition` 的秩大于两个表达式的秩序。

----

### in_train_phase


```python
keras.backend.in_train_phase(x, alt, training=None)
```


在训练阶段选择 `x`，其他阶段选择 `alt`。

请注意 `alt` 应该与 `x` 尺寸相同。

__参数__

- __x__: 在训练阶段需要返回的 x 
(张量或返回张量的可调用函数)。
- __alt__: 在其他阶段需要返回的 alt 
(张量或返回张量的可调用函数)。
- __training__: 可选的标量张量
(或 Python 布尔值，或者 Python 整数)，
以指定学习阶段。

__返回__

基于 `training` 标志，要么返回 `x`，要么返回 `alt`。
`training` 标志默认为 `K.learning_phase()`。

----

### in_test_phase


```python
keras.backend.in_test_phase(x, alt, training=None)
```


在测试阶段选择 `x`，其他阶段选择 `alt`。

请注意 `alt` 应该与 `x` 尺寸相同。

__参数__

- __x__: 在训练阶段需要返回的 x 
(张量或返回张量的可调用函数)。
- __alt__: 在其他阶段需要返回的 alt 
(张量或返回张量的可调用函数)。
- __training__: 可选的标量张量
(或 Python 布尔值，或者 Python 整数)，
以指定学习阶段。

__返回__

基于 `K.learning_phase`，要么返回 `x`，要么返回 `alt`。

----

### relu


```python
keras.backend.relu(x, alpha=0.0, max_value=None)
```


ReLU 整流线性单位。

默认情况下，它返回逐个元素的 `max(x, 0)` 值。

__参数__

- __x__: 一个张量或变量。
- __alpha__: 一个标量，负数部分的斜率（默认为 `0.`）。
- __max_value__: 饱和度阈值。

__返回__

一个张量。

__Numpy 实现__

```python
def relu(x, alpha=0., max_value=None, threshold=0.):
    if max_value is None:
        max_value = np.inf
    above_threshold = x * (x >= threshold)
    above_threshold = np.clip(above_threshold, 0.0, max_value)
    below_threshold = alpha * (x - threshold) * (x < threshold)
    return below_threshold + above_threshold
```

----

### elu


```python
keras.backend.elu(x, alpha=1.0)
```


指数线性单元。

__参数__

- __x__: 用于计算激活函数的张量或变量。
- __alpha__: 一个标量，负数部分的斜率。

__返回__

一个张量。

__Numpy 实现__

```python
def elu(x, alpha=1.):
    return x * (x > 0) + alpha * (np.exp(x) - 1.) * (x < 0)
```

----

### softmax


```python
keras.backend.softmax(x)
```


张量的 Softmax 值。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

__Numpy 实现__

```python
def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)
```

----

### softplus


```python
keras.backend.softplus(x)
```


张量的 Softplus 值。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

__Numpy 实现__

```python
def softplus(x):
    return np.log(1. + np.exp(x))
```

----

### softsign


```python
keras.backend.softsign(x)
```


张量的 Softsign 值。

__参数__

- __x__: 张量或变量。

__返回__

一个张量。

----

### categorical_crossentropy


```python
keras.backend.categorical_crossentropy(target, output, from_logits=False)
```


输出张量与目标张量之间的分类交叉熵。

__参数__

- __target__: 与 `output` 尺寸相同的张量。
- __output__: 由 softmax 产生的张量
(除非 `from_logits` 为 True，
在这种情况下 `output` 应该是对数形式)。
- __from_logits__: 布尔值，`output` 是 softmax 的结果，
还是对数形式的张量。

__返回__

输出张量。

----

### sparse_categorical_crossentropy


```python
keras.backend.sparse_categorical_crossentropy(target, output, from_logits=False)
```


稀疏表示的整数值目标的分类交叉熵。

__参数__

- __target__: 一个整数张量。
- __output__: 由 softmax 产生的张量
(除非 `from_logits` 为 True，
在这种情况下 `output` 应该是对数形式)。
- __from_logits__: 布尔值，`output` 是 softmax 的结果，
还是对数形式的张量。

__返回__

输出张量。

----

### binary_crossentropy


```python
keras.backend.binary_crossentropy(target, output, from_logits=False)
```


输出张量与目标张量之间的二进制交叉熵。

__参数__

- __target__: 与 `output` 尺寸相同的张量。
- __output__: 一个张量。
- __from_logits__: `output` 是否是对数张量。
默认情况下，我们认为 `output` 编码了概率分布。

__返回__

一个张量。

----

### sigmoid


```python
keras.backend.sigmoid(x)
```


逐个元素求 sigmoid 值。

__参数__

- __x__: 一个张量或变量。

__返回__

一个张量。


__Numpy 实现__

```python
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
```

----

### hard_sigmoid


```python
keras.backend.hard_sigmoid(x)
```


分段的 sigmoid 线性近似。速度比 sigmoid 更快。

- 如果 `x < -2.5`，返回 `0`。
- 如果 `x > 2.5`，返回 `1`。
- 如果 `-2.5 <= x <= 2.5`，返回 `0.2 * x + 0.5`。

__参数__

- __x__: 一个张量或变量。

__返回__

一个张量。

__Numpy 实现__

```python
def hard_sigmoid(x):
    y = 0.2 * x + 0.5
    return np.clip(y, 0, 1)
```


----

### tanh


```python
keras.backend.tanh(x)
```


逐个元素求 tanh 值。

__参数__

- __x__: 一个张量或变量。

__返回__

一个张量。

__Numpy 实现__

```python
def tanh(x):
    return np.tanh(x)
```

----

### dropout


```python
keras.backend.dropout(x, level, noise_shape=None, seed=None)
```


将 `x` 中的某些项随机设置为零，同时缩放整个张量。

__参数__

- __x__: 张量
- __level__: 张量中将被设置为 0 的项的比例。
- __noise_shape__: 随机生成的 保留/丢弃 标志的尺寸，
必须可以广播到 `x` 的尺寸。
- __seed__: 保证确定性的随机种子。

__返回__

一个张量。

----

### l2_normalize


```python
keras.backend.l2_normalize(x, axis=None)
```


在指定的轴使用 L2 范式 标准化一个张量。

__参数__

- __x__: 张量或变量。
- __axis__: 需要执行标准化的轴。

__返回__

一个张量。

__Numpy 实现__


```python
def l2_normalize(x, axis=-1):
    y = np.max(np.sum(x ** 2, axis, keepdims=True), axis, keepdims=True)
    return x / np.sqrt(y)
```


----

### in_top_k


```python
keras.backend.in_top_k(predictions, targets, k)
```


判断 `targets` 是否在 `predictions` 的前 `k` 个中。

__参数__

- __predictions__: 一个张量，尺寸为 `(batch_size, classes)`，类型为 `float32`。
- __targets__: 一个 1D 张量，长度为 `batch_size`，类型为 `int32` 或 `int64`。
- __k__: 一个 `int`，要考虑的顶部元素的数量。

__返回__

一个 1D 张量，长度为 `batch_size`，类型为 `bool`。
如果 `predictions[i, targets[i]]` 在 
`predictions[i]` 的 top-`k` 值中，
则 `output[i]` 为 `True`。

----

### conv1d


```python
keras.backend.conv1d(x, kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


1D 卷积。

__参数__

- __x__: 张量或变量。
- __kernel__: 核张量。
- __strides__: 步长整型。
- __padding__: 字符串，`"same"`, `"causal"` 或 `"valid"`。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。
- __dilation_rate__: 整数膨胀率。

__返回__

一个张量，1D 卷积结果。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。

----

### conv2d


```python
keras.backend.conv2d(x, kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


2D 卷积。

__参数__

- __x__: 张量或变量。
- __kernel__: 核张量。
- __strides__: 步长元组。
- __padding__: 字符串，`"same"` 或 `"valid"`。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。
对于输入/卷积核/输出，是否使用 Theano 或 TensorFlow/CNTK数据格式。
- __dilation_rate__: 2 个整数的元组。

__返回__

一个张量，2D 卷积结果。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。

----

### conv2d_transpose


```python
keras.backend.conv2d_transpose(x, kernel, output_shape, strides=(1, 1), padding='valid', data_format=None)
```


2D 反卷积 (即转置卷积)。

__参数__

- __x__: 张量或变量。
- __kernel__: 核张量。
- __output_shape__: 表示输出尺寸的 1D 整型张量。
- __strides__: 步长元组。
- __padding__: 字符串，`"same"` 或 `"valid"`。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。
对于输入/卷积核/输出，是否使用 Theano 或 TensorFlow/CNTK数据格式。

__返回__

一个张量，转置的 2D 卷积的结果。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。

----

### separable_conv1d


```python
keras.backend.separable_conv1d(x, depthwise_kernel, pointwise_kernel, strides=1, padding='valid', data_format=None, dilation_rate=1)
```


带可分离滤波器的 1D 卷积。

__参数__

- __x__: 输入张量。
- __depthwise_kernel__: 用于深度卷积的卷积核。
- __pointwise_kernel__: 1x1 卷积核。
- __strides__: 步长整数。
- __padding__: 字符串，`"same"` 或 `"valid"`。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。
- __dilation_rate__: 整数膨胀率。

__返回__

输出张量。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。

----

### separable_conv2d


```python
keras.backend.separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


带可分离滤波器的 2D 卷积。

__参数__

- __x__: 输入张量。
- __depthwise_kernel__: 用于深度卷积的卷积核。
- __pointwise_kernel__: 1x1 卷积核。
- __strides__: 步长元组 (长度为 2)。
- __padding__: 字符串，`"same"` 或 `"valid"`。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。
- __dilation_rate__: 整数元组，可分离卷积的膨胀率。

__返回__

输出张量。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。

----

### depthwise_conv2d


```python
keras.backend.depthwise_conv2d(x, depthwise_kernel, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1))
```


带可分离滤波器的 2D 卷积。

__参数__

- __x__: 输入张量。
- __depthwise_kernel__: 用于深度卷积的卷积核。
- __strides__: 步长元组 (长度为 2)。
- __padding__: 字符串，`"same"` 或 `"valid"`。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。
- __dilation_rate__: 整数元组，可分离卷积的膨胀率。

__返回__

输出张量。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。

----

### conv3d


```python
keras.backend.conv3d(x, kernel, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1))
```


3D 卷积。

__参数__

- __x__: 张量或变量。
- __kernel__: 核张量。
- __strides__: 步长元组。
- __padding__: 字符串，`"same"` 或 `"valid"`。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。
- __dilation_rate__: 3 个整数的元组。

__返回__

一个张量，3D 卷积的结果。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。

----

### conv3d_transpose


```python
keras.backend.conv3d_transpose(x, kernel, output_shape, strides=(1, 1, 1), padding='valid', data_format=None)
```


3D 反卷积 (即转置卷积)。

__参数__

- __x__: 输入张量。
- __kernel__: 核张量。
- __output_shape__: 表示输出尺寸的 1D 整数张量。
- __strides__: 步长元组。
- __padding__: 字符串，`"same"` 或 `"valid"`。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。
对于输入/卷积核/输出，是否使用 Theano 或 TensorFlow/CNTK数据格式。

__返回__

一个张量，3D 转置卷积的结果。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。

----

### pool2d


```python
keras.backend.pool2d(x, pool_size, strides=(1, 1), padding='valid', data_format=None, pool_mode='max')
```


2D 池化。

__参数__

- __x__: 张量或变量。
- __pool_size__: 2 个整数的元组。
- __strides__: 2 个整数的元组。
- __padding__: 字符串，`"same"` 或 `"valid"`。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。
- __pool_mode__: 字符串，`"max"` 或 `"avg"`。

__返回__

一个张量，2D 池化的结果。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。
- __ValueError__: if `pool_mode` 既不是 `"max"` 也不是 `"avg"`。

----

### pool3d


```python
keras.backend.pool3d(x, pool_size, strides=(1, 1, 1), padding='valid', data_format=None, pool_mode='max')
```


3D 池化。

__参数__

- __x__: 张量或变量。
- __pool_size__: 3 个整数的元组。
- __strides__: 3 个整数的元组。
- __padding__: 字符串，`"same"` 或 `"valid"`。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。
- __pool_mode__: 字符串，`"max"` 或 `"avg"`。

__返回__

一个张量，3D 池化的结果。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。
- __ValueError__: if `pool_mode` 既不是 `"max"` 也不是 `"avg"`。

----

### bias_add


```python
keras.backend.bias_add(x, bias, data_format=None)
```


给张量添加一个偏置向量。

__参数__

- __x__: 张量或变量。
- __bias__: 需要添加的偏置向量。
- __data_format__: 字符串，`"channels_last"` 或 `"channels_first"`。

__返回__

输出张量。

__异常__

- __ValueError__: 以下两种情况之一：
1. 无效的 `data_format` 参数。
2. 无效的偏置向量尺寸。
偏置应该是一个 `ndim(x)-1` 维的向量或张量。

----

### random_normal


```python
keras.backend.random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


返回正态分布值的张量。

__参数__

- __shape__: 一个整数元组，需要创建的张量的尺寸。
- __mean__: 一个浮点数，抽样的正态分布平均值。
- __stddev__: 一个浮点数，抽样的正态分布标准差。
- __dtype__: 字符串，返回的张量的数据类型。
- __seed__: 整数，随机种子。

__返回__

一个张量。

----

### random_uniform


```python
keras.backend.random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None)
```


返回均匀分布值的张量。

__参数__

- __shape__: 一个整数元组，需要创建的张量的尺寸。
- __minval__: 一个浮点数，抽样的均匀分布下界。
- __maxval__: 一个浮点数，抽样的均匀分布上界。
- __dtype__: 字符串，返回的张量的数据类型。
- __seed__: 整数，随机种子。

__返回__

一个张量。

----

### random_binomial


```python
keras.backend.random_binomial(shape, p=0.0, dtype=None, seed=None)
```


返回随机二项分布值的张量。

__参数__

- __shape__: 一个整数元组，需要创建的张量的尺寸。
- __p__: 一个浮点数，`0. <= p <= 1`，二项分布的概率。
- __dtype__: 字符串，返回的张量的数据类型。
- __seed__: 整数，随机种子。

__返回__

一个张量。

----

### truncated_normal


```python
keras.backend.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
```


返回截断的随机正态分布值的张量。

生成的值遵循具有指定平均值和标准差的正态分布，
此外，其中数值大于平均值两个标准差的将被丢弃和重新挑选。

__参数__

- __shape__: 一个整数元组，需要创建的张量的尺寸。
- __mean__: 平均值。
- __stddev__: 标准差。
- __dtype__: 字符串，返回的张量的数据类型。
- __seed__: 整数，随机种子。

__返回__

一个张量。

----

### ctc_label_dense_to_sparse


```python
keras.backend.ctc_label_dense_to_sparse(labels, label_lengths)
```


将 CTC 标签从密集转换为稀疏表示。

__参数__

- __labels__: 密集 CTC 标签。
- __label_lengths__: 标签长度。

__返回__

一个表示标签的稀疏张量。

----

### ctc_batch_cost


```python
keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
```


在每个批次元素上运行 CTC 损失算法。

__参数__

- __y_true__: 张量 `(samples, max_string_length)`，
包含真实标签。
- __y_pred__: 张量 `(samples, time_steps, num_categories)`，
包含预测值，或 softmax 输出。
- __input_length__: 张量 `(samples, 1)`，
包含 `y_pred` 中每个批次样本的序列长度。
- __label_length__: 张量 `(samples, 1)`，
包含 `y_true` 中每个批次样本的序列长度。

__返回__

尺寸为 (samples,1) 的张量，包含每一个元素的 CTC 损失。

----

### ctc_decode


```python
keras.backend.ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1)
```


解码 softmax 的输出。

可以使用贪心搜索（也称为最优路径）或受限字典搜索。

__参数__

- __y_pred__: 张量 `(samples, time_steps, num_categories)`，
包含预测值，或 softmax 输出。
- __input_length__: 张量 `(samples,)`，
包含 `y_pred` 中每个批次样本的序列长度。
- __greedy__: 如果为 `True`，则执行更快速的最优路径搜索，而不使用字典。
- __beam_width__: 如果 `greedy` 为 `false`，将使用该宽度的 beam 搜索解码器搜索。
- __top_paths__: 如果 `greedy` 为 `false`，
将返回多少条最可能的路径。

__返回__

- __Tuple__:
- __List__: 如果 `greedy` 为 `true`，返回包含解码序列的一个元素的列表。
如果为 `false`，返回最可能解码序列的 `top_paths`。
- __Important__: 空白标签返回为 `-1`。包含每个解码序列的对数概率的张量 `(top_paths,)`。

----

### map_fn


```python
keras.backend.map_fn(fn, elems, name=None, dtype=None)
```


将函数fn映射到元素 `elems` 上并返回输出。

__参数__

- __fn__: 将在每个元素上调用的可调用函数。
- __elems__: 张量。
- __name__: 映射节点在图中的字符串名称。
- __dtype__: 输出数据格式。

__返回__

数据类型为 `dtype` 的张量。

----

### foldl


```python
keras.backend.foldl(fn, elems, initializer=None, name=None)
```

使用 fn 归约 elems，以从左到右组合它们。

__参数__

- __fn__: 将在每个元素和一个累加器上调用的可调用函数，例如 `lambda acc, x: acc + x`。
- __elems__: 张量。
- __initializer__: 第一个使用的值 (如果为 None，使用`elems[0]`)。
- __name__: foldl 节点在图中的字符串名称。

__返回__

与 `initializer` 类型和尺寸相同的张量。

----

### foldr


```python
keras.backend.foldr(fn, elems, initializer=None, name=None)
```


使用 fn 归约 elems，以从右到左组合它们。

__参数__

- __fn__: 将在每个元素和一个累加器上调用的可调用函数，例如 `lambda acc, x: acc + x`。
- __elems__: 张量。
- __initializer__: 第一个使用的值 (如果为 None，使用`elems[-1]`)。
- __name__: foldr 节点在图中的字符串名称。

__返回__

与 `initializer` 类型和尺寸相同的张量。

----

### local_conv1d


```python
keras.backend.local_conv1d(inputs, kernel, kernel_size, strides, data_format=None)
```


在不共享权值的情况下，运用 1D 卷积。

__参数__

- __inputs__: 3D 张量，尺寸为 (batch_size, steps, input_dim)
- __kernel__: 卷积的非共享权重,
尺寸为 (output_items, feature_dim, filters)
- __kernel_size__: 一个整数的元组，
指定 1D 卷积窗口的长度。
- __strides__: 一个整数的元组，
指定卷积步长。
- __data_format__: 数据格式，channels_first 或 channels_last。

__返回__

运用不共享权重的 1D 卷积之后的张量，尺寸为 (batch_size, output_length, filters)。

__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。

----

### local_conv2d


```python
keras.backend.local_conv2d(inputs, kernel, kernel_size, strides, output_shape, data_format=None)
```


在不共享权值的情况下，运用 2D 卷积。

__参数__

- __inputs__: 如果 `data_format='channels_first'`，
则为尺寸为 (batch_size, filters, new_rows, new_cols) 的 4D 张量。
如果 `data_format='channels_last'`，
则为尺寸为 (batch_size, new_rows, new_cols, filters) 的 4D 张量。
- __kernel__: 卷积的非共享权重,
尺寸为 (output_items, feature_dim, filters)
- __kernel_size__: 2 个整数的元组，
指定 2D 卷积窗口的宽度和高度。
- __strides__: 2 个整数的元组，
指定 2D 卷积沿宽度和高度方向的步长。
- __output_shape__: 元组 (output_row, output_col) 。
- __data_format__: 数据格式，channels_first 或 channels_last。

__返回__

一个 4D 张量。

- 如果 `data_format='channels_first'`，尺寸为 (batch_size, filters, new_rows, new_cols)。
- 如果 `data_format='channels_last'`，尺寸为 (batch_size, new_rows, new_cols, filters)


__异常__

- __ValueError__: 如果 `data_format` 既不是 `channels_last` 也不是 `channels_first`。

----

### backend


```python
backend.backend()
```

公开可用的方法，以确定当前后端。

__返回__

字符串，Keras 目前正在使用的后端名。

__例子__

```python
>>> keras.backend.backend()
'tensorflow'
```
