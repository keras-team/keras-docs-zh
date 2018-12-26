<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L200)</span>
### Add

```python
keras.layers.Add()
```

计算输入张量列表的和。

它接受一个张量的列表，
所有的张量必须有相同的输入尺寸，
然后返回一个张量（和输入张量尺寸相同）。

__例子__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# 相当于 added = keras.layers.add([x1, x2])
added = keras.layers.Add()([x1, x2])  

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L231)</span>
### Subtract

```python
keras.layers.Subtract()
```

计算两个输入张量的差。

它接受一个长度为 2 的张量列表，
两个张量必须有相同的尺寸，然后返回一个值为 (inputs[0] - inputs[1]) 的张量，
输出张量和输入张量尺寸相同。

__例子__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
# 相当于 subtracted = keras.layers.subtract([x1, x2])
subtracted = keras.layers.Subtract()([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L268)</span>
### Multiply

```python
keras.layers.Multiply()
```

计算输入张量列表的（逐元素间的）乘积。

它接受一个张量的列表，
所有的张量必须有相同的输入尺寸，
然后返回一个张量（和输入张量尺寸相同）。

----

<span style="float:right;">[[source]]<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L283)</span>
### Average

```python
keras.layers.Average()
```

计算输入张量列表的平均值。

它接受一个张量的列表，
所有的张量必须有相同的输入尺寸，
然后返回一个张量（和输入张量尺寸相同）。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L298)</span>
### Maximum

```python
keras.layers.Maximum()
```

计算输入张量列表的（逐元素间的）最大值。

它接受一个张量的列表，
所有的张量必须有相同的输入尺寸，
然后返回一个张量（和输入张量尺寸相同）。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L320)</span>
### Concatenate

```python
keras.layers.Concatenate(axis=-1)
```

连接一个输入张量的列表。

它接受一个张量的列表，
除了连接轴之外，其他的尺寸都必须相同，
然后返回一个由所有输入张量连接起来的输出张量。

__参数__

- __axis__: 连接的轴。
- __**kwargs__: 层关键字参数。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/merge.py#L416)</span>
### Dot

```python
keras.layers.Dot(axes, normalize=False)
```

计算两个张量之间样本的点积。

例如，如果作用于输入尺寸为 `(batch_size, n)` 的两个张量 `a` 和 `b`，
那么输出结果就会是尺寸为 `(batch_size, 1)` 的一个张量。
在这个张量中，每一个条目 `i` 是 `a[i]` 和 `b[i]` 之间的点积。

__参数__

- __axes__: 整数或者整数元组，
一个或者几个进行点积的轴。
- __normalize__: 是否在点积之前对即将进行点积的轴进行 L2 标准化。
如果设置成 `True`，那么输出两个样本之间的余弦相似值。
- __**kwargs__: 层关键字参数。

----

### add


```python
keras.layers.add(inputs)
```

`Add` 层的函数式接口。

__参数__

- __inputs__: 一个输入张量的列表（列表大小至少为 2）。
- __**kwargs__: 层关键字参数。

__返回__

一个张量，所有输入张量的和。

__例子__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
added = keras.layers.add([x1, x2])

out = keras.layers.Dense(4)(added)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

### subtract


```python
keras.layers.subtract(inputs)
```

`Subtract` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小准确为 2）。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，两个输入张量的差。

__例子__


```python
import keras

input1 = keras.layers.Input(shape=(16,))
x1 = keras.layers.Dense(8, activation='relu')(input1)
input2 = keras.layers.Input(shape=(32,))
x2 = keras.layers.Dense(8, activation='relu')(input2)
subtracted = keras.layers.subtract([x1, x2])

out = keras.layers.Dense(4)(subtracted)
model = keras.models.Model(inputs=[input1, input2], outputs=out)
```

----

### multiply


```python
keras.layers.multiply(inputs)
```


`Multiply` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小至少为 2）。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，所有输入张量的逐元素乘积。

----

### average


```python
keras.layers.average(inputs)
```


`Average` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小至少为 2）。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，所有输入张量的平均值。

----

### maximum


```python
keras.layers.maximum(inputs)
```

`Maximum` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小至少为 2）。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，所有张量的逐元素的最大值。

----

### concatenate


```python
keras.layers.concatenate(inputs, axis=-1)
```


`Concatenate` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小至少为 2）。
- __axis__: 串联的轴。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，所有输入张量通过 `axis` 轴串联起来的输出张量。

----

### dot


```python
keras.layers.dot(inputs, axes, normalize=False)
```


`Dot` 层的函数式接口。

__参数__

- __inputs__: 一个列表的输入张量（列表大小至少为 2）。
- __axes__: 整数或者整数元组，
一个或者几个进行点积的轴。
- __normalize__: 是否在点积之前对即将进行点积的轴进行 L2 标准化。
如果设置成 True，那么输出两个样本之间的余弦相似值。
- __**kwargs__: 层的关键字参数。

__返回__

一个张量，所有输入张量样本之间的点积。
