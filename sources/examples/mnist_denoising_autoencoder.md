# 在 MNIST 数据集上训练去噪自动编码器。

去噪是自动编码器的经典应用之一。
去噪过程去除了破坏真实信号的有害噪声。

噪声 + 数据 ---> 去噪自动编码器 ---> 数据

给定训练数据集的损坏数据作为输入，输出真实信号作为输出，
去噪自动编码器可以恢复隐藏的结构以生成干净的数据。

此示例具有模块化设计。编码器、解码器和自动编码器是 3 种共享权重的模型。
例如，在训练自动编码器之后，编码器可用于生成输入数据的潜在矢量，以实现低维可视化（如 PCA 或 TSNE）。


```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.random.seed(1337)

# MNIST 数据集
(x_train, _), (x_test, _) = mnist.load_data()

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 通过添加法线距离为 0.5 且 std=0.5 的噪声来生成损坏的 MNIST 图像
noise = np.random.normal(loc=0.5, scale=0.5, size=x_train.shape)
x_train_noisy = x_train + noise
noise = np.random.normal(loc=0.5, scale=0.5, size=x_test.shape)
x_test_noisy = x_test + noise

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# 网络参数
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
latent_dim = 16
# CNN 层和每层过滤器的编码器/解码器数量
layer_filters = [32, 64]

# 建立自动编码器模型
# 首先建立编码器模型
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
# Conv2D块的堆栈
# 注意:
# 1) 在深度网络上的 ReLU 之前使用批处理规范化
# 2) 使用 MaxPooling2D 替代 strides>1
# - 更快但不如 strides>1 好
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               activation='relu',
               padding='same')(x)

# 构建解码器模型所需的形状信息
shape = K.int_shape(x)

# 生成潜在向量
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# 实例化编码器模型
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# 建立解码器模型
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

# 转置 Conv2D 块的堆栈
# 注意:
# 1) 在深度网络上的 ReLU 之前使用批处理规范化
# 2) 使用 UpSampling2D 替代 strides>1
# - 更快但不如 strides>1 好
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        activation='relu',
                        padding='same')(x)

x = Conv2DTranspose(filters=1,
                    kernel_size=kernel_size,
                    padding='same')(x)

outputs = Activation('sigmoid', name='decoder_output')(x)

# 实例化解码器模型
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# Autoencoder = Encoder + Decoder
# 实例化自动编码器模型
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
autoencoder.summary()

autoencoder.compile(loss='mse', optimizer='adam')

# 训练自动编码器
autoencoder.fit(x_train_noisy,
                x_train,
                validation_data=(x_test_noisy, x_test),
                epochs=30,
                batch_size=batch_size)

# 根据损坏的测试图像预测自动编码器的输出
x_decoded = autoencoder.predict(x_test_noisy)

# 显示第 1 个 8 张损坏和去噪的图像
rows, cols = 10, 30
num = rows * cols
imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_decoded[:num]])
imgs = imgs.reshape((rows * 3, cols, image_size, image_size))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, image_size, image_size))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()
```