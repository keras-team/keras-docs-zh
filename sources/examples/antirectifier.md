# 本示例演示了如何为 Keras 编写自定义网络层。

我们构建了一个称为 'Antirectifier' 的自定义激活层，该层可以修改通过它的张量的形状。
我们需要指定两个方法: `compute_output_shape` 和 `call`。

注意，相同的结果也可以通过 Lambda 层取得。

我们的自定义层是使用 Keras 后端 (`K`) 中的基元编写的，因而代码可以在 TensorFlow 和 Theano 上运行。


```python
from __future__ import print_function
import keras
from keras.models import Sequential
from keras import layers
from keras.datasets import mnist
from keras import backend as K


class Antirectifier(layers.Layer):
    '''这是样本级的 L2 标准化与输入的正负部分串联的组合。
    结果是两倍于输入样本的样本张量。
    
    它可以用于替代 ReLU。

    # 输入尺寸
        2D 张量，尺寸为 (samples, n)

    # 输出尺寸
        2D 张量，尺寸为 (samples, 2*n)

    # 理论依据
        在应用 ReLU 时，假设先前输出的分布接近于 0 的中心，
        那么将丢弃一半的输入。这是非常低效的。
        
        Antirectifier 允许像 ReLU 一样返回全正输出，而不会丢弃任何数据。
        
        在 MNIST 上进行的测试表明，Antirectifier 可以训练参数少两倍但具
        有与基于 ReLU 的等效网络相当的分类精度的网络。
    '''

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # 仅对 2D 张量有效
        shape[-1] *= 2
        return tuple(shape)

    def call(self, inputs):
        inputs -= K.mean(inputs, axis=1, keepdims=True)
        inputs = K.l2_normalize(inputs, axis=1)
        pos = K.relu(inputs)
        neg = K.relu(-inputs)
        return K.concatenate([pos, neg], axis=1)

# 全局参数
batch_size = 128
num_classes = 10
epochs = 40

# 切分为训练和测试的数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 将类向量转化为二进制类矩阵
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 构建模型
model = Sequential()
model.add(layers.Dense(256, input_shape=(784,)))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(256))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# 接下来，与具有 2 倍大的密集层
# 和 ReLU 的等效网络进行比较
```
