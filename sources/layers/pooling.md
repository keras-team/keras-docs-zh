<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L69)</span>
### MaxPooling1D

```python
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

对于时序数据的最大池化。

__参数__

- __pool_size__: 整数，最大池化的窗口大小。
- __strides__: 整数，或者是 `None`。作为缩小比例的因数。
例如，2 会使得输入张量缩小一半。
如果是 `None`，那么默认值是 `pool_size`。
- __padding__: `"valid"` 或者 `"same"` （区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
    表示输入各维度的顺序。
    `channels_last` 对应输入尺寸为 `(batch, steps, features)`，
    `channels_first` 对应输入尺寸为 `(batch, features, steps)`。


__输入尺寸__

- 如果 `data_format='channels_last'`，
    输入为 3D 张量，尺寸为：
    `(batch_size, steps, features)`
- 如果`data_format='channels_first'`，
    输入为 3D 张量，尺寸为：
    `(batch_size, features, steps)`

__输出尺寸__

- 如果 `data_format='channels_last'`，
    输出为 3D 张量，尺寸为：
    `(batch_size, downsampled_steps, features)`
- 如果 `data_format='channels_first'`，
    输出为 3D 张量，尺寸为：
    `(batch_size, features, downsampled_steps)`

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L217)</span>
### MaxPooling2D

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

对于空间数据的最大池化。

__参数__

- __pool_size__: 整数，或者 2 个整数表示的元组，
    沿（垂直，水平）方向缩小比例的因数。
   （2，2）会把输入张量的两个维度都缩小一半。
    如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
- __strides__: 整数，2 个整数表示的元组，或者是 `None`。
    表示步长值。
    如果是 `None`，那么默认值是 `pool_size`。
- __padding__: `"valid"` 或者 `"same"` （区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
    表示输入各维度的顺序。
    `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量，
    而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。
    默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
    如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, rows, cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, rows, cols)` 的 4D 张量

__输出尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, pooled_rows, pooled_cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, pooled_rows, pooled_cols)` 的 4D 张量

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L386)</span>
### MaxPooling3D

```python
keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```


对于 3D（空间，或时空间）数据的最大池化。

__参数__

- __pool_size__: 3 个整数表示的元组，缩小（dim1，dim2，dim3）比例的因数。
(2, 2, 2) 会把 3D 输入张量的每个维度缩小一半。
- __strides__: 3 个整数表示的元组，或者是 `None`。步长值。
- __padding__: `"valid"` 或者 `"same"`（区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
    表示输入各维度的顺序。
    `channels_last` 代表尺寸是 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的输入张量，
    而 `channels_first` 代表尺寸是 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的输入张量。
    默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
    如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的 5D 张量

__输出尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)` 的 5D 张量

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L117)</span>
### AveragePooling1D

```python
keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

对于时序数据的平均池化。

__参数__

- __pool_size__: 整数，平均池化的窗口大小。
- __strides__: 整数，或者是 `None	`。作为缩小比例的因数。
例如，2 会使得输入张量缩小一半。
如果是 `None`，那么默认值是 `pool_size`。
- __padding__: `"valid"` 或者 `"same"` （区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
    表示输入各维度的顺序。
    `channels_last` 对应输入尺寸为 `(batch, steps, features)`，
    `channels_first` 对应输入尺寸为 `(batch, features, steps)`。

__输入尺寸__

- 如果 `data_format='channels_last'`，
    输入为 3D 张量，尺寸为：
    `(batch_size, steps, features)`
- 如果`data_format='channels_first'`，
    输入为 3D 张量，尺寸为：
    `(batch_size, features, steps)`

__输出尺寸__

- 如果 `data_format='channels_last'`，
    输出为 3D 张量，尺寸为：
    `(batch_size, downsampled_steps, features)`
- 如果 `data_format='channels_first'`，
    输出为 3D 张量，尺寸为：
    `(batch_size, features, downsampled_steps)`

----


<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L272)</span>
### AveragePooling2D

```python
keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```


对于空间数据的平均池化。

__参数__

- __pool_size__: 整数，或者 2 个整数表示的元组，
    沿（垂直，水平）方向缩小比例的因数。
   （2，2）会把输入张量的两个维度都缩小一半。
    如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
- __strides__: 整数，2 个整数表示的元组，或者是 `None`。
    表示步长值。
    如果是 `None`，那么默认值是 `pool_size`。
- __padding__: `"valid"` 或者 `"same"` （区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
    表示输入各维度的顺序。
    `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量，
    而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。
    默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
    如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, rows, cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, rows, cols)` 的 4D 张量

__输出尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, pooled_rows, pooled_cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, pooled_rows, pooled_cols)` 的 4D 张量

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L436)</span>
### AveragePooling3D

```python
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

对于 3D （空间，或者时空间）数据的平均池化。

__参数__

- __pool_size__: 3 个整数表示的元组，缩小（dim1，dim2，dim3）比例的因数。
(2, 2, 2) 会把 3D 输入张量的每个维度缩小一半。
- __strides__: 3 个整数表示的元组，或者是 `None`。步长值。
- __padding__: `"valid"` 或者 `"same"`（区分大小写）。
- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
    表示输入各维度的顺序。
    `channels_last` 代表尺寸是 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的输入张量，
    而 `channels_first` 代表尺寸是 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的输入张量。
    默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
    如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的 5D 张量

__输出尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)` 的 5D 张量

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L557)</span>
### GlobalMaxPooling1D

```python
keras.layers.GlobalMaxPooling1D(data_format='channels_last')
```

对于时序数据的全局最大池化。


__参数__

- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
    表示输入各维度的顺序。
    `channels_last` 对应输入尺寸为 `(batch, steps, features)`，
    `channels_first` 对应输入尺寸为 `(batch, features, steps)`。

__输入尺寸__

尺寸是 `(batch_size, steps, features)` 的 3D 张量。

__输出尺寸__

尺寸是 `(batch_size, features)` 的 2D 张量。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L455)</span>
### GlobalAveragePooling1D

```python
keras.layers.GlobalAveragePooling1D()
```

对于时序数据的全局平均池化。

__输入尺寸__

- 如果 `data_format='channels_last'`，
    输入为 3D 张量，尺寸为：
    `(batch_size, steps, features)`
- 如果`data_format='channels_first'`，
    输入为 3D 张量，尺寸为：
    `(batch_size, features, steps)`

__输出尺寸__

尺寸是 `(batch_size, features)` 的 2D 张量。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L647)</span>
### GlobalMaxPooling2D

```python
keras.layers.GlobalMaxPooling2D(data_format=None)
```

对于空域数据的全局最大池化。

__参数__

- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
    表示输入各维度的顺序。
    `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量，
    而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。
    默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
    如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, rows, cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, rows, cols)` 的 4D 张量

__输出尺寸__

尺寸是 `(batch_size, channels)` 的 2D 张量

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L510)</span>
### GlobalAveragePooling2D

```python
keras.layers.GlobalAveragePooling2D(data_format=None)
```

对于空域数据的全局平均池化。

__参数__

- __data_format__: 一个字符串，`channels_last` （默认值）或者 `channels_first`。
输入张量中的维度顺序。
`channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量，而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。
默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, rows, cols, channels)` 的 4D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, rows, cols)` 的 4D 张量

__输出尺寸__

尺寸是 `(batch_size, channels)` 的 2D 张量

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L742)</span>
### GlobalMaxPooling3D

```python
keras.layers.GlobalMaxPooling3D(data_format=None)
```

对于 3D 数据的全局最大池化。

__参数__

- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
    表示输入各维度的顺序。
    `channels_last` 代表尺寸是 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的输入张量，
    而 `channels_first` 代表尺寸是 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的输入张量。
    默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
    如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的 5D 张量

__输出尺寸__

尺寸是 `(batch_size, channels)` 的 2D 张量

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L707)</span>
### GlobalAveragePooling3D

```python
keras.layers.GlobalAveragePooling3D(data_format=None)
```

对于 3D 数据的全局平均池化。

__参数__

- __data_format__: 字符串，`channels_last` (默认)或 `channels_first` 之一。
    表示输入各维度的顺序。
    `channels_last` 代表尺寸是 `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的输入张量，
    而 `channels_first` 代表尺寸是 `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的输入张量。
    默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。
    如果还没有设置过，那么默认值就是 "channels_last"。

__输入尺寸__

- 如果 `data_format='channels_last'`:
尺寸是 `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)` 的 5D 张量
- 如果 `data_format='channels_first'`:
尺寸是 `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)` 的 5D 张量

__输出尺寸__

尺寸是 `(batch_size, channels)` 的 2D 张量
