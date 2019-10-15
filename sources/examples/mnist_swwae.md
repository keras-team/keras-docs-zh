# 训练基于 MNIST 数据集上残差块的堆叠式自动编码器。

它举例说明了过去几年开发的两种有影响力的方法。

首先是适当地 "分拆" 的想法。在任何最大池化期间，都会丢失合并的接收场中最大值的确切位置（where），但是在输入图像的整体重建中可能非常有用。
因此，如果将 "位置" 从编码器传递到相应的解码器层，则可以将要解码的特征 "放置" 在正确的位置，从而可以实现更高保真度的重构。

# 参考文献

- [Visualizing and Understanding Convolutional Networks, Matthew D Zeiler, Rob Fergus](https://arxiv.org/abs/1311.2901v3)
- [Stacked What-Where Auto-encoders, Junbo Zhao, Michael Mathieu, Ross Goroshin, Yann LeCun](https://arxiv.org/abs/1506.02351v8)

这里利用的第二个想法是残差学习的想法。残差块通过允许跳过连接使网络能够按照数据认为合适的线性（或非线性）能力简化训练过程。
这样可以轻松地训练很多深度的网络。残差元素在该示例的上下文中似乎是有利的，因为它允许编码器和解码器之间的良好对称性。
通常，在解码器中，对重构图像的空间的最终投影是线性的，但是对于残差块，则不必如此，因为其输出是线性还是非线性的程度取决于被馈送的像素数据。
但是，为了限制此示例中的重建，因为我们知道 MNIST 数字映射到 [0, 1]，所以将硬 softmax 用作偏置。

# 参考文献
- [Deep Residual Learning for Image Recognition, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](https://arxiv.org/abs/1512.03385v1)
- [Identity Mappings in Deep Residual Networks, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](https://arxiv.org/abs/1603.05027v3)


```python
from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Activation
from keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers import Input, BatchNormalization, ELU
import matplotlib.pyplot as plt
import keras.backend as K
from keras import layers


def convresblock(x, nfeats=8, ksize=3, nskipped=2, elu=True):
    """[4] 中提出的残差块。

    以 elu=True 运行将使用 ELU 非线性，而以 elu=False 运行将使用 BatchNorm+RELU 非线性。
    尽管 ELU 由于不受 BatchNorm 开销的困扰而很快，但它们可能会过拟合，
    因为它们不提供 BatchNorm 批处理过程的随机因素，而后者是一个很好的正则化工具。

    # 参数
        x: 4D 张量, 穿过块的张量
        nfeats: 整数。卷积层的特征图大小。
        ksize: 整数，第一个卷积中 conv 核的宽度和高度。
        nskipped: 整数，残差函数的卷积层数。
        elu: 布尔值，是使用 ELU 还是 BN+RELU。

    # 输入尺寸
        4D 张量，尺寸为：
        `(batch, channels, rows, cols)`

    # 输出尺寸
        4D 张量，尺寸为：
        `(batch, filters, rows, cols)`
    """
    y0 = Conv2D(nfeats, ksize, padding='same')(x)
    y = y0
    for i in range(nskipped):
        if elu:
            y = ELU()(y)
        else:
            y = BatchNormalization(axis=1)(y)
            y = Activation('relu')(y)
        y = Conv2D(nfeats, 1, padding='same')(y)
    return layers.add([y0, y])


def getwhere(x):
    '''计算包含开关的 'where' 掩码，该掩码指示应用 MaxPool2D 时哪个索引包含最大值。
    使用总和的梯度是使所有内容保持高水平的不错的技巧。'''
    y_prepool, y_postpool = x
    return K.gradients(K.sum(y_postpool), y_prepool)

# 本示例假定 'channels_first' 数据格式。
K.set_image_data_format('channels_first')

# 输入图像尺寸
img_rows, img_cols = 28, 28

# 数据，分为训练集和测试集
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# MaxPooling2D 使用的内核大小
pool_size = 2
# 每层特征图的总数
nfeats = [8, 16, 32, 64, 128]
# 每层池化内核的大小
pool_sizes = np.array([1, 1, 1, 1, 1]) * pool_size
# 卷积核大小
ksize = 3
# 要训练的轮次数
epochs = 5
# 训练期间的批次大小
batch_size = 128

if pool_size == 2:
    # 如果使用 pool_size = 2 的 5 层网络
    x_train = np.pad(x_train, [[0, 0], [0, 0], [2, 2], [2, 2]],
                     mode='constant')
    x_test = np.pad(x_test, [[0, 0], [0, 0], [2, 2], [2, 2]], mode='constant')
    nlayers = 5
elif pool_size == 3:
    # 如果使用 pool_size = 3 的 3 层网
    x_train = x_train[:, :, :-1, :-1]
    x_test = x_test[:, :, :-1, :-1]
    nlayers = 3
else:
    import sys
    sys.exit('Script supports pool_size of 2 and 3.')

# 训练输入的形状（请注意，模型是完全卷积的）
input_shape = x_train.shape[1:]
# axis=1 的所有层的尺寸最终大小，包括输入
nfeats_all = [input_shape[0]] + nfeats

# 首先构建编码器，同时始终跟踪 'where' 掩码
img_input = Input(shape=input_shape)

# 我们将 'where' 掩码推到下面的列表中
wheres = [None] * nlayers
y = img_input
for i in range(nlayers):
    y_prepool = convresblock(y, nfeats=nfeats_all[i + 1], ksize=ksize)
    y = MaxPooling2D(pool_size=(pool_sizes[i], pool_sizes[i]))(y_prepool)
    wheres[i] = layers.Lambda(
        getwhere, output_shape=lambda x: x[0])([y_prepool, y])

# 现在构建解码器，并使用存储的 'where' 掩码放置特征
for i in range(nlayers):
    ind = nlayers - 1 - i
    y = UpSampling2D(size=(pool_sizes[ind], pool_sizes[ind]))(y)
    y = layers.multiply([y, wheres[ind]])
    y = convresblock(y, nfeats=nfeats_all[ind], ksize=ksize)

# 使用 hard_sigmoid 裁剪重建范围
y = Activation('hard_sigmoid')(y)

# 定义模型及其均方误差损失，并使用 Adam 进行编译
model = Model(img_input, y)
model.compile('adam', 'mse')

# 拟合模型
model.fit(x_train, x_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, x_test))

# 绘图
x_recon = model.predict(x_test[:25])
x_plot = np.concatenate((x_test[:25], x_recon), axis=1)
x_plot = x_plot.reshape((5, 10, input_shape[-2], input_shape[-1]))
x_plot = np.vstack([np.hstack(x) for x in x_plot])
plt.figure()
plt.axis('off')
plt.title('Test Samples: Originals/Reconstructions')
plt.imshow(x_plot, interpolation='none', cmap='gray')
plt.savefig('reconstructions.png')
```