<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/utils/generic_utils.py#L21)</span>
### CustomObjectScope

```python
keras.utils.CustomObjectScope()
```

提供更改为 `_GLOBAL_CUSTOM_OBJECTS` 无法转义的范围。

`with` 语句中的代码将能够通过名称访问自定义对象。
对全局自定义对象的更改会在封闭的 `with` 语句中持续存在。
在`with`语句结束时，
全局自定义对象将恢复到 `with` 语句开始时的状态。

__例子__


考虑自定义对象 `MyObject` (例如一个类)：

```python
with CustomObjectScope({'MyObject':MyObject}):
    layer = Dense(..., kernel_regularizer='MyObject')
    # 保存，加载等操作将按这个名称来识别自定义对象
```

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/utils/io_utils.py#L25)</span>
### HDF5Matrix

```python
keras.utils.HDF5Matrix(datapath, dataset, start=0, end=None, normalizer=None)
```

使用 HDF5 数据集表示，而不是 Numpy 数组。

__例子__


```python
x_data = HDF5Matrix('input/file.hdf5', 'data')
model.predict(x_data)
```

提供 `start` 和 `end` 将允许使用数据集的一个切片。

你还可以给出标准化函数（或 lambda）（可选）。
这将在检索到的每一个数据切片上调用它。

__参数__

- __datapath__: 字符串，HDF5 文件路径。
- __dataset__: 字符串，datapath指定的文件中的 HDF5 数据集名称。
- __start__: 整数，所需的指定数据集的切片的开始位置。
- __end__: 整数，所需的指定数据集的切片的结束位置。
- __normalizer__: 在检索数据时调用的函数。

__返回__

一个类似于数组的 HDF5 数据集。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302)</span>
### Sequence

```python
keras.utils.Sequence()
```

用于拟合数据序列的基对象，例如一个数据集。

每一个 `Sequence` 必须实现 `__getitem__` 和 `__len__` 方法。
如果你想在迭代之间修改你的数据集，你可以实现 `on_epoch_end`。
`__getitem__` 方法应该范围一个完整的批次。

__注意__


`Sequence` 是进行多进程处理的更安全的方法。这种结构保证网络在每个时期每个样本只训练一次，这与生成器不同。

__例子__


```python
from skimage.io import imread
from skimage.transform import resize
import numpy as np

# 这里，`x_set` 是图像的路径列表
# 以及 `y_set` 是对应的类别

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)
```

----

### to_categorical


```python
keras.utils.to_categorical(y, num_classes=None, dtype='float32')
```


将类向量（整数）转换为二进制类矩阵。

例如，用于 categorical_crossentropy。

__参数__

- __y__: 需要转换成矩阵的类矢量
(从 0 到 num_classes 的整数)。
- __num_classes__: 总类别数。
- __dtype__: 字符串，输入所期望的数据类型 (`float32`, `float64`, `int32`...)

__例子__

```python
# 考虑一组 3 个类 {0,1,2} 中的 5 个标签数组：
> labels
array([0, 2, 1, 2, 0])
# `to_categorical` 将其转换为具有尽可能多表示类别数的列的矩阵。
# 行数保持不变。
> to_categorical(labels)
array([[ 1.,  0.,  0.],
       [ 0.,  0.,  1.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 1.,  0.,  0.]], dtype=float32)
```

__返回__

输入的二进制矩阵表示。

----

### normalize


```python
keras.utils.normalize(x, axis=-1, order=2)
```


标准化一个 Numpy 数组。

__参数__

- __x__: 需要标准化的 Numpy 数组。
- __axis__: 需要标准化的轴。
- __order__: 标准化顺序(例如，2 表示 L2 规范化)。

__Returns__

数组的标准化副本。

----

### get_file


```python
keras.utils.get_file(fname, origin, untar=False, md5_hash=None, file_hash=None, cache_subdir='datasets', hash_algorithm='auto', extract=False, archive_format='auto', cache_dir=None)
```


从一个 URL 下载文件，如果它不存在缓存中。

默认情况下，URL `origin`处的文件
被下载到缓存目录 `〜/.keras` 中，
放在缓存子目录 `datasets`中，并命名为 `fname`。
文件 `example.txt` 的最终位置为 `~/.keras/datasets/example.txt`。

tar, tar.gz, tar.bz, 以及 zip 格式的文件也可以被解压。
传递一个哈希值将在下载后校验文件。
命令行程序 `shasum` 和 `sha256sum` 可以计算哈希。

__参数__

- __fname__: 文件名。如果指定了绝对路径 `/path/to/file.txt`，
那么文件将会保存到那个路径。
- __origin__: 文件的原始 URL。
- __untar__: 由于使用 'extract' 而已被弃用。
布尔值，是否需要解压文件。
- __md5_hash__: 由于使用 'file_hash' 而已被弃用。
用于校验的 md5 哈希值。
- __file_hash__: 下载后的文件的期望哈希字符串。
支持 sha256 和 md5 两个哈希算法。
- __cache_subdir__: 在 Keras 缓存目录下的保存文件的子目录。
如果指定了绝对路径 `/path/to/folder`，则文件将被保存在该位置。
- __hash_algorithm__: 选择哈希算法来校验文件。
可选的有 'md5', 'sha256', 以及 'auto'。
默认的 'auto' 将自动检测所使用的哈希算法。
- __extract__: True 的话会尝试将解压缩存档文件，如tar或zip。
- __archive_format__: 尝试提取文件的存档格式。
可选的有 'auto', 'tar', 'zip', 以及 None。
'tar' 包含 tar, tar.gz, 和 tar.bz 文件。
默认 'auto' 为 ['tar', 'zip']。
None 或 空列表将返回未找到任何匹配。
ke xu az z'auto', 'tar', 'zip', and None.
- __cache_dir__: 存储缓存文件的位置，为 None 时默认为
[Keras 目录](/faq/#where-is-the-keras-configuration-filed-stored).

__返回__

下载的文件的路径。

----

### print_summary


```python
keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)
```


打印模型概况。

__参数__

- __model__: Keras 模型实例。
- __line_length__: 打印的每行的总长度
(例如，设置此项以使其显示适应不同的终端窗口大小)。
- __positions__: 每行中日志元素的相对或绝对位置。
如果未提供，默认为 `[.33, .55, .67, 1.]`。
- __print_fn__: 需要使用的打印函数。
它将在每一行概述时调用。
您可以将其设置为自定义函数以捕获字符串概述。
默认为 `print` (打印到标准输出)。

----

### plot_model


```python
keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
```


将 Keras 模型转换为 dot 格式并保存到文件中。

__参数__

- __model__: 一个 Keras 模型实例。
- __to_file__: 绘制图像的文件名。
- __show_shapes__: 是否显示尺寸信息。
- __show_layer_names__: 是否显示层的名称。
- __rankdir__: 传递给 PyDot 的 `rankdir` 参数，
一个指定绘图格式的字符串：
'TB' 创建一个垂直绘图；
'LR' 创建一个水平绘图。
- __expand_nested__: 是否扩展嵌套模型为聚类。 
- __dpi__: 点 DPI。

----

### multi_gpu_model


```python
keras.utils.multi_gpu_model(model, gpus=None, cpu_merge=True, cpu_relocation=False)
```


将模型复制到不同的 GPU 上。

具体来说，该功能实现了单机多 GPU 数据并行性。
它的工作原理如下：

- 将模型的输入分成多个子批次。
- 在每个子批次上应用模型副本。
每个模型副本都在专用 GPU 上执行。
- 将结果（在 CPU 上）连接成一个大批量。

例如， 如果你的 `batch_size` 是 64，且你使用 `gpus=2`，
那么我们将把输入分为两个 32 个样本的子批次，
在 1 个 GPU 上处理 1 个子批次，然后返回完整批次的 64 个处理过的样本。

这实现了多达 8 个 GPU 的准线性加速。

此功能目前仅适用于 TensorFlow 后端。

__参数__

- __model__: 一个 Keras 模型实例。为了避免OOM错误，该模型可以建立在 CPU 上，
详见下面的使用样例。
- __gpus__: 整数 >= 2 或整数列表，创建模型副本的 GPU 数量，
或 GPU ID 的列表。
- __cpu_merge__: 一个布尔值，用于标识是否强制合并 CPU 范围内的模型权重。
- __cpu_relocation__: 一个布尔值，用来确定是否在 CPU 的范围内创建模型的权重。如果模型没有在任何一个设备范围内定义，您仍然可以通过激活这个选项来拯救它。

__返回__

一个 Keras `Model` 实例，它可以像初始 `model` 参数一样使用，但它将工作负载分布在多个 GPU 上。

__例子__

例 1 - 训练在 CPU 上合并权重的模型

```python
import tensorflow as tf
from keras.applications import Xception
from keras.utils import multi_gpu_model
import numpy as np

num_samples = 1000
height = 224
width = 224
num_classes = 1000

# 实例化基础模型（或者「模版」模型）。
# 我们推荐在 CPU 设备范围内做此操作，
# 这样模型的权重就会存储在 CPU 内存中。
# 否则它们会存储在 GPU 上，而完全被共享。
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)

# 复制模型到 8 个 GPU 上。
# 这假设你的机器有 8 个可用 GPU。
parallel_model = multi_gpu_model(model, gpus=8)
parallel_model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop')

# 生成虚拟数据
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))

# 这个 `fit` 调用将分布在 8 个 GPU 上。
# 由于 batch size 是 256, 每个 GPU 将处理 32 个样本。
parallel_model.fit(x, y, epochs=20, batch_size=256)

# 通过模版模型存储模型（共享相同权重）：
model.save('my_model.h5')
```

例 2 - 训练在 CPU 上利用 cpu_relocation 合并权重的模型

```python
..
# 不需要更改模型定义的设备范围：
model = Xception(weights=None, ..)

try:
    parallel_model = multi_gpu_model(model, cpu_relocation=True)
    print("Training using multiple GPUs..")
except ValueError:
    parallel_model = model
    print("Training using single GPU or CPU..")
parallel_model.compile(..)
..
```

例 3 - 训练在 GPU 上合并权重的模型（建议用于 NV-link）

```python
..
# 不需要更改模型定义的设备范围：
model = Xception(weights=None, ..)

try:
    parallel_model = multi_gpu_model(model, cpu_merge=False)
    print("Training using multiple GPUs..")
except:
    parallel_model = model
    print("Training using single GPU or CPU..")

parallel_model.compile(..)
..
```


__关于模型保存__

要保存多 GPU 模型，请通过模板模型（传递给 `multi_gpu_model` 的参数）调用 `.save(fname)` 或 `.save_weights(fname)` 以进行存储，而不是通过 `multi_gpu_model` 返回的模型。
