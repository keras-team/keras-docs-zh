这是由 Tianqi Chen, Ian Goodfellow, and Jonathon Shlens 在 "Net2Net: Accelerating Learning via Knowledge Transfer" 中用 MNIST 进行的 Net2Net 实验的实现。

arXiv:1511.05641v4 [cs.LG] 23 Apr 2016
http://arxiv.org/abs/1511.05641

# 注意

- 什么:
  + Net2Net 是将知识从教师神经网络转移到学生网络的一组方法，因此，与从头开始相比，可以更快地训练学生网络。
  + 本文讨论了 Net2Net 的两种特定方法，即 Net2WiderNet 和 Net2DeeperNet。
  + Net2WiderNet 将模型替换为等效的更宽模型，该模型在每个隐藏层中具有更多单位。
  + Net2DeeperNet 将模型替换为等效的更深模型。
  + 两者都基于“神经网络的功能保留变换”的思想。
- 为什么:
  + 通过创建一系列具有可转移知识的更宽和更深入的模型，在实验和设计过程中快速探索多个神经网络。
  + 通过逐步调整模型的复杂性以适应数据可用性并重用可转让的知识，从而启用“终身学习系统”。

# 实验

- 教师模型：在 MNIST 上训练的 3 个基本 CNN 模型。
- Net2WiderNet 实验：
  + 学生模型具有更宽的 Conv2D 层和更宽的 FC 层。
  + 比较 'random-padding' 和 'net2wider' 权重初始化。
  + 使用这两种方法，在 1 个轮次之后，学生模型的表现应与教师模型相同，但 'net2wider' 要好一些。
- Net2DeeperNet 实验：
  + 学生模型具有额外的 Conv2D 层和额外的 FC 层。
  + 比较 'random-init' 和 'net2deeper' 权重初始化。
  + 1 个轮次后，'net2deeper' 的性能优于 'random-init'。
- 超参数:
  + momentum=0.9 的 SGD 用于训练教师和学生模型。
  + 学习率调整：建议将学生模型的学习率降低到 1/10。
  + 在 'net2wider' 中添加噪声用于打破权重对称性，
    从而实现学生模型的全部容量。使用 Dropout 层时，它是可选的。

# 结果

- 经过 TF 后端和 'channels_last' 的 image_data_format 测试。
- 在 GPU GeForce GTX Titan X Maxwell 上运行
- 性能比较-前 3 个轮次的验证损失值：

教师模型 ...
(0) teacher_model:             0.0537   0.0354   0.0356

Net2WiderNet 实验...
(1) wider_random_pad:          0.0320   0.0317   0.0289
(2) wider_net2wider:           0.0271   0.0274   0.0270

Net2DeeperNet 实验...
(3) deeper_random_init:        0.0682   0.0506   0.0468
(4) deeper_net2deeper:         0.0292   0.0294   0.0286


```python
from __future__ import print_function
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
from keras.datasets import mnist

if K.image_data_format() == 'channels_first':
    input_shape = (1, 28, 28)  # 图像尺寸
else:
    input_shape = (28, 28, 1)  # 图像尺寸
num_classes = 10  # 类别数
epochs = 3


# 加载和预处理数据
def preprocess_input(x):
    return x.astype('float32').reshape((-1,) + input_shape) / 255


def preprocess_output(y):
    return keras.utils.to_categorical(y)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = map(preprocess_input, [x_train, x_test])
y_train, y_test = map(preprocess_output, [y_train, y_test])
print('Loading MNIST data...')
print('x_train shape:', x_train.shape, 'y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape, 'y_test shape', y_test.shape)


# 知识转移算法
def wider2net_conv2d(teacher_w1, teacher_b1, teacher_w2, new_width, init):
    '''通过 'random-padding' 或 'net2wider'，获得具有较大过滤器的更宽 conv2d 层的初始权重。

    # 参数
        teacher_w1: `weight`，conv2d 层需要加宽的权重，
          尺寸为 (filters1, num_channel1, kh1, kw1)
        teacher_b1: `bias`，conv2d 层需要加宽的偏置，
          尺寸为 (filters1, )
        teacher_w2: `weight`，下一个连接的 conv2d 层的权重，
          尺寸为 (filters2, num_channel2, kh2, kw2)
        new_width: 新的 `filters`，对于更宽的 conv2d 层
        init: 新权重的初始化算法，
          'random-pad' 或 'net2wider' 之一
    '''
    assert teacher_w1.shape[0] == teacher_w2.shape[1], (
        'successive layers from teacher model should have compatible shapes')
    assert teacher_w1.shape[3] == teacher_b1.shape[0], (
        'weight and bias from same layer should have compatible shapes')
    assert new_width > teacher_w1.shape[3], (
        'new width (filters) should be bigger than the existing one')

    n = new_width - teacher_w1.shape[3]
    if init == 'random-pad':
        new_w1 = np.random.normal(0, 0.1, size=teacher_w1.shape[:3] + (n,))
        new_b1 = np.ones(n) * 0.1
        new_w2 = np.random.normal(
            0, 0.1,
            size=teacher_w2.shape[:2] + (n, teacher_w2.shape[3]))
    elif init == 'net2wider':
        index = np.random.randint(teacher_w1.shape[3], size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[:, :, :, index]
        new_b1 = teacher_b1[index]
        new_w2 = teacher_w2[:, :, index, :] / factors.reshape((1, 1, -1, 1))
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)

    student_w1 = np.concatenate((teacher_w1, new_w1), axis=3)
    if init == 'random-pad':
        student_w2 = np.concatenate((teacher_w2, new_w2), axis=2)
    elif init == 'net2wider':
        # 添加较小的噪声以破坏对称性，以便学生模型以后可以完全使用
        noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
        student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=2)
        student_w2[:, :, index, :] = new_w2
    student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

    return student_w1, student_b1, student_w2


def wider2net_fc(teacher_w1, teacher_b1, teacher_w2, new_width, init):
    '''通过 'random-padding' 或 'net2wider'，获得具有更大节点的更宽的完全连接（密集）层的初始权重。

    # 参数
        teacher_w1: `weight`，fc 层需要加宽的权重，
          尺寸为 (nin1, nout1)
        teacher_b1: `bias`，fc 层需要加宽的偏置，
          尺寸为 (nout1, )
        teacher_w2: `weight`，下一个连接的 fc 层的权重,
          尺寸为 (nin2, nout2)
        new_width: 更宽的 fc 层的新 `nout`
        init: 新权重的初始化算法，
          'random-pad' 或 'net2wider' 之一
    '''
    assert teacher_w1.shape[1] == teacher_w2.shape[0], (
        'successive layers from teacher model should have compatible shapes')
    assert teacher_w1.shape[1] == teacher_b1.shape[0], (
        'weight and bias from same layer should have compatible shapes')
    assert new_width > teacher_w1.shape[1], (
        'new width (nout) should be bigger than the existing one')

    n = new_width - teacher_w1.shape[1]
    if init == 'random-pad':
        new_w1 = np.random.normal(0, 0.1, size=(teacher_w1.shape[0], n))
        new_b1 = np.ones(n) * 0.1
        new_w2 = np.random.normal(0, 0.1, size=(n, teacher_w2.shape[1]))
    elif init == 'net2wider':
        index = np.random.randint(teacher_w1.shape[1], size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[:, index]
        new_b1 = teacher_b1[index]
        new_w2 = teacher_w2[index, :] / factors[:, np.newaxis]
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)

    student_w1 = np.concatenate((teacher_w1, new_w1), axis=1)
    if init == 'random-pad':
        student_w2 = np.concatenate((teacher_w2, new_w2), axis=0)
    elif init == 'net2wider':
        # 添加较小的噪声以破坏对称性，以便学生模型以后可以完全使用
        noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
        student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=0)
        student_w2[index, :] = new_w2
    student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

    return student_w1, student_b1, student_w2


def deeper2net_conv2d(teacher_w):
    '''通过 "net2deeper' 获得更深层 conv2d 层的初始权重。

    # 参数
        teacher_w: `weight`，前一个 conv2d 层的权重，
          尺寸为 (kh, kw, num_channel, filters)
    '''
    kh, kw, num_channel, filters = teacher_w.shape
    student_w = np.zeros_like(teacher_w)
    for i in range(filters):
        student_w[(kh - 1) // 2, (kw - 1) // 2, i, i] = 1.
    student_b = np.zeros(filters)
    return student_w, student_b


def copy_weights(teacher_model, student_model, layer_names):
    '''将名称从 layer_names 中列出的图层的权重从 teacher_model 复制到 student_model
    '''
    for name in layer_names:
        weights = teacher_model.get_layer(name=name).get_weights()
        student_model.get_layer(name=name).set_weights(weights)


# 构造 teacher_model 和 student_model 的方法
def make_teacher_model(x_train, y_train,
                       x_test, y_test,
                       epochs):
    '''简单 CNN 的训练和基准性能。
    (0) Teacher model
    '''
    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=input_shape,
                     padding='same', name='conv1'))
    model.add(MaxPooling2D(2, name='pool1'))
    model.add(Conv2D(64, 3, padding='same', name='conv2'))
    model.add(MaxPooling2D(2, name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='fc1'))
    model.add(Dense(num_classes, activation='softmax', name='fc2'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=0.01, momentum=0.9),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_test, y_test))
    return model


def make_wider_student_model(teacher_model,
                             x_train, y_train,
                             x_test, y_test,
                             init, epochs):
    '''使用 'random-pad'（基线）或 'net2wider'，基于 teacher_model 训练更广泛的学生模型
    '''
    new_conv1_width = 128
    new_fc1_width = 128

    model = Sequential()
    # 一个比 teacher_model 更宽的 conv1
    model.add(Conv2D(new_conv1_width, 3, input_shape=input_shape,
                     padding='same', name='conv1'))
    model.add(MaxPooling2D(2, name='pool1'))
    model.add(Conv2D(64, 3, padding='same', name='conv2'))
    model.add(MaxPooling2D(2, name='pool2'))
    model.add(Flatten(name='flatten'))
    # 一个比 teacher_model 更宽的 fc1
    model.add(Dense(new_fc1_width, activation='relu', name='fc1'))
    model.add(Dense(num_classes, activation='softmax', name='fc2'))

    # 除了加宽的图层及其直接下游之外，其他图层的权重需要从教师模型复制到学生模型，这将分别进行初始化。
    # 对于此示例，不需要复制其他任何层。

    w_conv1, b_conv1 = teacher_model.get_layer('conv1').get_weights()
    w_conv2, b_conv2 = teacher_model.get_layer('conv2').get_weights()
    new_w_conv1, new_b_conv1, new_w_conv2 = wider2net_conv2d(
        w_conv1, b_conv1, w_conv2, new_conv1_width, init)
    model.get_layer('conv1').set_weights([new_w_conv1, new_b_conv1])
    model.get_layer('conv2').set_weights([new_w_conv2, b_conv2])

    w_fc1, b_fc1 = teacher_model.get_layer('fc1').get_weights()
    w_fc2, b_fc2 = teacher_model.get_layer('fc2').get_weights()
    new_w_fc1, new_b_fc1, new_w_fc2 = wider2net_fc(
        w_fc1, b_fc1, w_fc2, new_fc1_width, init)
    model.get_layer('fc1').set_weights([new_w_fc1, new_b_fc1])
    model.get_layer('fc2').set_weights([new_w_fc2, b_fc2])

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=0.001, momentum=0.9),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_test, y_test))


def make_deeper_student_model(teacher_model,
                              x_train, y_train,
                              x_test, y_test,
                              init, epochs):
    '''使用 'random-pad'（基线）或 'net2wider'，基于 teacher_model 训练更广泛的学生模型
    '''
    model = Sequential()
    model.add(Conv2D(64, 3, input_shape=input_shape,
                     padding='same', name='conv1'))
    model.add(MaxPooling2D(2, name='pool1'))
    model.add(Conv2D(64, 3, padding='same', name='conv2'))
    # 添加另一个 conv2d 层以使原始 conv2 更深
    if init == 'net2deeper':
        prev_w, _ = model.get_layer('conv2').get_weights()
        new_weights = deeper2net_conv2d(prev_w)
        model.add(Conv2D(64, 3, padding='same',
                         name='conv2-deeper', weights=new_weights))
    elif init == 'random-init':
        model.add(Conv2D(64, 3, padding='same', name='conv2-deeper'))
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)
    model.add(MaxPooling2D(2, name='pool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='fc1'))
    # 添加另一个 fc 层以使原始 fc1 更深
    if init == 'net2deeper':
        # 带有 relu 的 fc 层的 net2deeper 只是一个身份初始化器
        model.add(Dense(64, kernel_initializer='identity',
                        activation='relu', name='fc1-deeper'))
    elif init == 'random-init':
        model.add(Dense(64, activation='relu', name='fc1-deeper'))
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)
    model.add(Dense(num_classes, activation='softmax', name='fc2'))

    # 复制其他图层的权重
    copy_weights(teacher_model, model, layer_names=[
                 'conv1', 'conv2', 'fc1', 'fc2'])

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(learning_rate=0.001, momentum=0.9),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=epochs,
              validation_data=(x_test, y_test))


# 实验设置
def net2wider_experiment():
    '''基准表现
    (1) 带有 `random_pad` 初始值设定项的更宽的学生模型
    (2)带有 `Net2WiderNet` 初始化程序的更宽的学生模型
    '''
    print('\nExperiment of Net2WiderNet ...')

    print('\n(1) building wider student model by random padding ...')
    make_wider_student_model(teacher_model,
                             x_train, y_train,
                             x_test, y_test,
                             init='random-pad',
                             epochs=epochs)
    print('\n(2) building wider student model by net2wider ...')
    make_wider_student_model(teacher_model,
                             x_train, y_train,
                             x_test, y_test,
                             init='net2wider',
                             epochs=epochs)


def net2deeper_experiment():
    '''基准表现
    (3) 带有  `random_init` 初始值设定项的更宽的学生模型
    (4) 带有  `Net2DeeperNet` 初始值设定项的更宽的学生模型
    '''
    print('\nExperiment of Net2DeeperNet ...')

    print('\n(3) building deeper student model by random init ...')
    make_deeper_student_model(teacher_model,
                              x_train, y_train,
                              x_test, y_test,
                              init='random-init',
                              epochs=epochs)
    print('\n(4) building deeper student model by net2deeper ...')
    make_deeper_student_model(teacher_model,
                              x_train, y_train,
                              x_test, y_test,
                              init='net2deeper',
                              epochs=epochs)


print('\n(0) building teacher model ...')
teacher_model = make_teacher_model(x_train, y_train,
                                   x_test, y_test,
                                   epochs=epochs)

# 进行实验
net2wider_experiment()
net2deeper_experiment()
```