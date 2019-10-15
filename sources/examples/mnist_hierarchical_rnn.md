# 使用分层 RNN (HRNN) 对 MNIST 数字进行分类的示例。

HRNN 可以跨复杂序列跨多个时间层次学习。
通常，HRNN 的第一循环层将句子（例如单词向量）编码成句子向量。
然后，第二循环层将这样的向量序列（由第一层编码）编码为文档向量。
该文档向量被认为既可以保留上下文的单词级结构也可以保留句子级结构。

# 参考文献

- [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057)
    使用 HRNN 对段落和文档进行编码。
    结果表明，HRNN 优于标准 RNN，并且可能在摘要或问题解答等更复杂的生成任务中发挥某些作用。
- [Hierarchical recurrent neural network for skeleton based action recognition](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7298714)
    通过 3 级双向 HRNN 与完全连接的层相结合，在基于骨骼的动作识别方面取得了最新的成果。

在下面的 MNIST 示例中，第一 LSTM 层首先将形状 (28, 1) 的每一列像素编码为形状 (128,) 的列矢量。
然后，第二个 LSTM 层将形状 (28, 128) 的这 28 个列向量编码为代表整个图像的图像向量。
添加最终的密集层以进行预测。

5 个轮次后：train acc：0.9858, val acc：0.9864


```python
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM

# 训练参数。
batch_size = 32
num_classes = 10
epochs = 5

# 嵌入尺寸。
row_hidden = 128
col_hidden = 128

# 数据，分为训练集和测试集。
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 将数据重塑为 4D 以进行分层 RNN。
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 将类向量转换为二进制类矩阵。
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

row, col, pixel = x_train.shape[1:]

# 4D 输入。
x = Input(shape=(row, col, pixel))

# 使用 TimeDistributed Wrapper 对一行像素进行编码。
encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

# 对已编码行的列进行编码。
encoded_columns = LSTM(col_hidden)(encoded_rows)

# 最终预测和模型。
prediction = Dense(num_classes, activation='softmax')(encoded_columns)
model = Model(x, prediction)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 训练。
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# 评估。
scores = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
```