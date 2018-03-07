<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L57)</span>
### MaxPooling1D

```python
keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid')
```

对时域1D信号进行最大值池化

参数

pool_size：整数，池化窗口大小
strides：整数或None，下采样因子，例如设2将会使得输出shape为输入的一半，若为None则默认值为pool_size。
padding：‘valid’或者‘same’
输入shape

形如（samples，steps，features）的3D张量
输出shape

形如（samples，downsampled_steps，features）的3D张量

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L170)</span>
### MaxPooling2D

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

空域信号施加最大值池化

参数

pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
strides：整数或长为2的整数tuple，或者None，步长值。
border_mode：‘valid’或者‘same’
data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
输入shape

‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量

‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量

输出shape

‘channels_first’模式下，为形如（samples，channels, pooled_rows, pooled_cols）的4D张量

‘channels_last’模式下，为形如（samples，pooled_rows, pooled_cols，channels）的4D张量

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L339)</span>
### MaxPooling3D

```python
keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

M为3D信号（空域或时空域）施加最大值池化

本层目前只能在使用Theano为后端时可用

参数

pool_size：整数或长为3的整数tuple，代表在三个维度上的下采样因子，如取（2，2，2）将使信号在每个维度都变为原来的一半长。
strides：整数或长为3的整数tuple，或者None，步长值。
padding：‘valid’或者‘same’
data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
输入shape

‘channels_first’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量

‘channels_last’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

输出shape

‘channels_first’模式下，为形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D张量

‘channels_last’模式下，为形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D张量

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L87)</span>
### AveragePooling1D

```python
keras.layers.AveragePooling1D(pool_size=2, strides=None, padding='valid')
```

对时域1D信号进行平均值池化

参数

pool_size：整数，池化窗口大小
strides：整数或None，下采样因子，例如设2将会使得输出shape为输入的一半，若为None则默认值为pool_size。
padding：‘valid’或者‘same’
输入shape

形如（samples，steps，features）的3D张量
输出shape

形如（samples，downsampled_steps，features）的3D张量

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L225)</span>
### AveragePooling2D

```python
keras.layers.AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

为空域信号施加平均值池化

参数

pool_size：整数或长为2的整数tuple，代表在两个方向（竖直，水平）上的下采样因子，如取（2，2）将使图片在两个维度上均变为原长的一半。为整数意为各个维度值相同且为该数字。
strides：整数或长为2的整数tuple，或者None，步长值。
border_mode：‘valid’或者‘same’
data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
输入shape

‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量

‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量

输出shape

‘channels_first’模式下，为形如（samples，channels, pooled_rows, pooled_cols）的4D张量

‘channels_last’模式下，为形如（samples，pooled_rows, pooled_cols，channels）的4D张量

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L389)</span>
### AveragePooling3D

```python
keras.layers.AveragePooling3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

为3D信号（空域或时空域）施加平均值池化

本层目前只能在使用Theano为后端时可用

参数

pool_size：整数或长为3的整数tuple，代表在三个维度上的下采样因子，如取（2，2，2）将使信号在每个维度都变为原来的一半长。
strides：整数或长为3的整数tuple，或者None，步长值。
padding：‘valid’或者‘same’
data_format：字符串，“channels_first”或“channels_last”之一，代表数据的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128x128的数据为例，“channels_first”应将数据组织为（3,128,128,128），而“channels_last”应将数据组织为（128,128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
‘channels_first’模式下，为形如（samples, channels, len_pool_dim1, len_pool_dim2, len_pool_dim3）的5D张量

‘channels_last’模式下，为形如（samples, len_pool_dim1, len_pool_dim2, len_pool_dim3，channels, ）的5D张量

输出shape

‘channels_first’模式下，为形如（samples, channels, pooled_dim1, pooled_dim2, pooled_dim3）的5D张量

‘channels_last’模式下，为形如（samples, pooled_dim1, pooled_dim2, pooled_dim3,channels,）的5D张量

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L470)</span>
### GlobalMaxPooling1D

```python
keras.layers.GlobalMaxPooling1D()
```

对于时间信号的全局最大池化

输入shape

形如（samples，steps，features）的3D张量
输出shape

形如(samples, features)的2D张量
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L455)</span>
### GlobalAveragePooling1D

```python
keras.layers.GlobalAveragePooling1D()
```

为时域信号施加全局平均值池化

输入shape

形如（samples，steps，features）的3D张量
输出shape

形如(samples, features)的2D张量
<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L545)</span>
### GlobalMaxPooling2D

```python
keras.layers.GlobalMaxPooling2D(data_format=None)
```

为空域信号施加全局最大值池化

参数

data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
输入shape

‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量

‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量

输出shape

形如(nb_samples, channels)的2D张量

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/pooling.py#L510)</span>
### GlobalAveragePooling2D

```python
keras.layers.GlobalAveragePooling2D(data_format=None)
```

为空域信号施加全局平均值池化

参数

data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channels_last”对应原本的“tf”，“channels_first”对应原本的“th”。以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
输入shape

‘channels_first’模式下，为形如（samples，channels, rows，cols）的4D张量

‘channels_last’模式下，为形如（samples，rows, cols，channels）的4D张量

输出shape

形如(nb_samples, channels)的2D张量