# 应用 Applications

Keras 的应用模块（keras.applications）提供了带有预训练权值的深度学习模型，这些模型可以用来进行预测、特征提取和微调（fine-tuning）。

当你初始化一个预训练模型时，会自动下载权值到 `~/.keras/models/` 目录下。

## 可用的模型

### 在 ImageNet 上预训练过的用于图像分类的模型：

- [Xception](#xception)
- [VGG16](#vgg16)
- [VGG19](#vgg19)
- [ResNet50](#resnet50)
- [InceptionV3](#inceptionv3)
- [InceptionResNetV2](#inceptionresnetv2)
- [MobileNet](#mobilenet)
- [DenseNet](#densenet)
- [NASNet](#nasnet)

所有的这些模型（除了 Xception 和 MobileNet 外）都兼容Theano和Tensorflow，并会自动按照位于 `~/.keras/keras.json` 的配置文件中设置的图像数据格式来构建模型。举个例子，如果你设置 `image_data_format=channels_last`，则加载的模型将按照 TensorFlow 的维度顺序来构造，即“高度-宽度-深度”（Height-Width-Depth）的顺序。

Xception 模型仅适用于 TensorFlow，因为它依赖于 SeparableConvolution 层。MobileNet 模型仅适用于 TensorFlow，因为它依赖于 DepthwiseConvolution 层。

-----

## 图像分类模型的示例代码

### 使用 ResNet50 进行 ImageNet 分类

```python
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# 把预测结果解码为一个元组数组（类别，描述，概率）
# （这个列表里有一批数据中的每个样本的结果）
print('Predicted:', decode_predictions(preds, top=3)[0])
# 预测结果：[(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
```

### 使用 VGG16 提取特征

```python
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```

### 从VGG19的任意中间层中抽取特征

```python
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

block4_pool_features = model.predict(x)
```

### 在新类上微调 InceptionV3

```python
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K

# 构建不带分类器的预训练模型
base_model = InceptionV3(weights='imagenet', include_top=False)

# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加一个全连接层
x = Dense(1024, activation='relu')(x)

# 添加一个分类器，假设我们有200个类
predictions = Dense(200, activation='softmax')(x)

# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)

# 首先，我们只训练顶部的几层（随机初始化的层）
# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False

# 编译模型（一定要在锁层以后操作）
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 在新的数据集上训练几代
model.fit_generator(...)

# 现在顶层应该训练好了，让我们开始微调 Inception V3 的卷积层。
# 我们会锁住底下的几层，然后训练其余的顶层。

# 让我们看看每一层的名字和层号，看看我们应该锁多少层呢：
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# 我们选择训练最上面的两个 Inception block
也就是说锁住前面249层，然后放开之后的层。
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# 我们需要重新编译模型，才能使上面的修改生效
# 让我们设置一个很低的学习率，使用 SGD 来微调
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# 我们继续训练模型，这次我们训练最后两个 Inception block
# 和两个全连接层
model.fit_generator(...)
```


### 通过自定义输入 tensor 构建 InceptionV3

```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

# 你也可以把 input_tensor 换成其他的 Keras 模型或 Keras 层
input_tensor = Input(shape=(224, 224, 3))  # 这里假设 K.image_data_format() == 'channels_last'

model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
```

-----

# 模型概览

| 模型 | 大小 | Top-1 准确率 | Top-5 准确率 | 参数数量 | 深度 |
| ----- | ----: | --------------: | --------------: | ----------: | -----: |
| [Xception](#xception) | 88 MB | 0.790 | 0.945| 22,910,480 | 126 |
| [VGG16](#vgg16) | 528 MB| 0.715 | 0.901 | 138,357,544 | 23
| [VGG19](#vgg19) | 549 MB | 0.727 | 0.910 | 143,667,240 | 26
| [ResNet50](#resnet50) | 99 MB | 0.759 | 0.929 | 25,636,712 | 168
| [InceptionV3](#inceptionv3) | 92 MB | 0.788 | 0.944 | 23,851,784 | 159 |
| [InceptionResNetV2](#inceptionresnetv2) | 215 MB | 0.804 | 0.953 | 55,873,736 | 572 |
| [MobileNet](#mobilenet) | 17 MB | 0.665 | 0.871 | 4,253,864 | 88
| [DenseNet121](#densenet) | 33 MB | 0.745 | 0.918 | 8,062,504 | 121
| [DenseNet169](#densenet) | 57 MB | 0.759 | 0.928 | 14,307,880 | 169
| [DenseNet201](#densenet) | 80 MB | 0.770 | 0.933 | 20,242,984 | 201


Top-1 准确率和 Top-5 准确率都是在 ImageNet 验证集上的结果。

-----


## Xception


```python
keras.applications.xception.Xception(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

在 ImageNet 上预训练的 Xception V1 模型。

在 ImageNet 上，该模型取得了验证集 top1 0.790 和 top5 0.945 的准确率。

注意，该模型目前仅能在 TensorFlow 后端使用，因为它依赖 `SeparableConvolution` 层，目前该层只支持 `channels_last` 的维度顺序（高度、宽度、通道）。

模型默认输入尺寸是 299x299。

### 参数

- include_top: 是否包括顶层的全连接层。
- weights: `None` 代表随机初始化， `'imagenet'` 代表加载在 ImageNet 上预训练的权值。
- input_tensor: 可选，Keras tensor 作为模型的输入（比如 `layers.Input()` 输出的 tensor）
- input_shape: 可选，输入尺寸元组，仅当`include_top=False`时有效（不然输入形状必须是 `(299, 299, 3)`，因为预训练模型是以这个大小训练的）。输入尺寸必须是三个数字，且宽高必须不小于 71，比如 `(150, 150, 3)` 是一个合法的输入尺寸。
- pooling: 可选，当 `include_top` 为 `False` 时，该参数指定了特征提取时的池化方式。
    - `None` 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    - `'avg'` 代表全局平均池化（GLobalAveragePool2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    - `'max'` 代表全局最大池化
- classes: 可选，图片分类的类别数，仅当 `include_top` 为 `True` 并且不加载预训练权值时可用。

### 返回值

一个 Keras `Model` 对象.

### 参考文献

- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

### License

预训练权值由我们自己训练而来，基于 MIT license 发布。


-----


## VGG16

```python
keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

VGG16 模型，权值由 ImageNet 训练而来。

该模型在 `Theano` 和 `TensorFlow` 后端均可使用，并接受 `channels_first` 和 `channels_last` 两种输入维度顺序（高度，宽度，通道）。

模型默认输入尺寸是 224x224。

### 参数

- include_top: 是否包括顶层的全连接层。
- weights: `None` 代表随机初始化， `'imagenet'` 代表加载在 ImageNet 上预训练的权值。
- input_tensor: 可选，Keras tensor 作为模型的输入（比如 `layers.Input()` 输出的 tensor）
- input_shape: 可选，输入尺寸元组，仅当 `include_top=False` 时有效（不然输入形状必须是 `(224, 224, 3)` （`channels_last` 格式）或 `(3, 224, 224)` （`channels_first` 格式），因为预训练模型是以这个大小训练的）。输入尺寸必须是三个数字，且宽高必须不小于 48，比如 `(200, 200, 3)` 是一个合法的输入尺寸。
- pooling: 可选，当 `include_top` 为 `False` 时，该参数指定了特征提取时的池化方式。
    - `None` 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    - `'avg'` 代表全局平均池化（GLobalAveragePool2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    - `'max'` 代表全局最大池化
- classes: 可选，图片分类的类别数，仅当 `include_top` 为 `True` 并且不加载预训练权值时可用。

### 返回值

一个 Keras `Model` 对象。

### 参考文献

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)：如果在研究中使用了VGG，请引用该论文。

### License

预训练权值由 [VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) 发布的预训练权值移植而来，基于 [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/)。

-----

## VGG19


```python
keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

VGG19 模型，权值由 ImageNet 训练而来。

该模型在 `Theano` 和 `TensorFlow` 后端均可使用，并接受 `channels_first` 和 `channels_last` 两种输入维度顺序（高度，宽度，通道）。

模型默认输入尺寸是 224x224。

### 参数

- include_top: 是否包括顶层的全连接层。
- weights: `None` 代表随机初始化， `'imagenet'` 代表加载在 ImageNet 上预训练的权值。
- input_tensor: 可选，Keras tensor 作为模型的输入（比如 `layers.Input()` 输出的 tensor）
- input_shape: 可选，输入尺寸元组，仅当 `include_top=False` 时有效（不然输入形状必须是 `(224, 224, 3)` （`channels_last` 格式）或 `(3, 224, 224)` （`channels_first` 格式），因为预训练模型是以这个大小训练的）。输入尺寸必须是三个数字，且宽高必须不小于 48，比如 `(200, 200, 3)` 是一个合法的输入尺寸。
- pooling: 可选，当 `include_top` 为 `False` 时，该参数指定了特征提取时的池化方式。
    - `None` 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    - `'avg'` 代表全局平均池化（GLobalAveragePool2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    - `'max'` 代表全局最大池化
- classes: 可选，图片分类的类别数，仅当 `include_top` 为 `True` 并且不加载预训练权值时可用。

### 返回值

一个 Keras `Model` 对象。

### 参考文献

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)：如果在研究中使用了VGG，请引用该论文。

### License

预训练权值由 [VGG at Oxford](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) 发布的预训练权值移植而来，基于 [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/)。

-----

## ResNet50


```python
keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

ResNet50 模型，权值由 ImageNet 训练而来。

该模型在 `Theano` 和 `TensorFlow` 后端均可使用，并接受 `channels_first` 和 `channels_last` 两种输入维度顺序（高度，宽度，通道）。

模型默认输入尺寸是 224x224。

### 参数

- include_top: 是否包括顶层的全连接层。
- weights: `None` 代表随机初始化， `'imagenet'` 代表加载在 ImageNet 上预训练的权值。
- input_tensor: 可选，Keras tensor 作为模型的输入（比如 `layers.Input()` 输出的 tensor）
- input_shape: 可选，输入尺寸元组，仅当 `include_top=False` 时有效（不然输入形状必须是 `(224, 224, 3)` （`channels_last` 格式）或 `(3, 224, 224)` （`channels_first` 格式），因为预训练模型是以这个大小训练的）。输入尺寸必须是三个数字，且宽高必须不小于 197，比如 `(200, 200, 3)` 是一个合法的输入尺寸。
- pooling: 可选，当 `include_top` 为 `False` 时，该参数指定了特征提取时的池化方式。
    - `None` 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    - `'avg'` 代表全局平均池化（GLobalAveragePool2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    - `'max'` 代表全局最大池化
- classes: 可选，图片分类的类别数，仅当 `include_top` 为 `True` 并且不加载预训练权值时可用。

### 返回值

一个 Keras `Model` 对象。

### 参考文献

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### License

预训练权值由 [Kaiming He](https://github.com/KaimingHe/deep-residual-networks) 发布的预训练权值移植而来，基于 [MIT license](https://github.com/KaimingHe/deep-residual-networks/blob/master/LICENSE)。

-----

## InceptionV3


```python
keras.applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

Inception V3 模型，权值由 ImageNet 训练而来。

该模型在 `Theano` 和 `TensorFlow` 后端均可使用，并接受 `channels_first` 和 `channels_last` 两种输入维度顺序（高度，宽度，通道）。

模型默认输入尺寸是 299x299。

### 参数

- include_top: 是否包括顶层的全连接层。
- weights: `None` 代表随机初始化， `'imagenet'` 代表加载在 ImageNet 上预训练的权值。
- input_tensor: 可选，Keras tensor 作为模型的输入（比如 `layers.Input()` 输出的 tensor）
- input_shape: 可选，输入尺寸元组，仅当 `include_top=False` 时有效（不然输入形状必须是 `(299, 299, 3)` （`channels_last` 格式）或 `(3, 299, 299)` （`channels_first` 格式），因为预训练模型是以这个大小训练的）。输入尺寸必须是三个数字，且宽高必须不小于 139，比如 `(150, 150, 3)` 是一个合法的输入尺寸。
- pooling: 可选，当 `include_top` 为 `False` 时，该参数指定了特征提取时的池化方式。
    - `None` 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    - `'avg'` 代表全局平均池化（GLobalAveragePool2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    - `'max'` 代表全局最大池化
- classes: 可选，图片分类的类别数，仅当 `include_top` 为 `True` 并且不加载预训练权值时可用。

### 返回值

一个 Keras `Model` 对象。

### 参考文献		

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

### License

预训练权值基于 [Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)。

-----

## InceptionResNetV2


```python
keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

Inception-ResNet V2 模型，权值由 ImageNet 训练而来。

该模型在 `Theano` 和 `TensorFlow` 后端均可使用，并接受 `channels_first` 和 `channels_last` 两种输入维度顺序（高度，宽度，通道）。

模型默认输入尺寸是 299x299。

### 参数

- include_top: 是否包括顶层的全连接层。
- weights: `None` 代表随机初始化， `'imagenet'` 代表加载在 ImageNet 上预训练的权值。
- input_tensor: 可选，Keras tensor 作为模型的输入（比如 `layers.Input()` 输出的 tensor）
- input_shape: 可选，输入尺寸元组，仅当 `include_top=False` 时有效（不然输入形状必须是 `(299, 299, 3)` （`channels_last` 格式）或 `(3, 299, 299)` （`channels_first` 格式），因为预训练模型是以这个大小训练的）。输入尺寸必须是三个数字，且宽高必须不小于 139，比如 `(150, 150, 3)` 是一个合法的输入尺寸。
- pooling: 可选，当 `include_top` 为 `False` 时，该参数指定了特征提取时的池化方式。
    - `None` 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    - `'avg'` 代表全局平均池化（GLobalAveragePool2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    - `'max'` 代表全局最大池化
- classes: 可选，图片分类的类别数，仅当 `include_top` 为 `True` 并且不加载预训练权值时可用。

### 返回值

一个 Keras `Model` 对象。

### 参考文献		

- [Rethinking the Inception Architecture for Computer Vision](http://arxiv.org/abs/1512.00567)

### License

预训练权值基于 [Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)。

-----

## MobileNet


```python
keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

在 ImageNet 上预训练的 MobileNet 模型。

注意，该模型目前仅能在 TensorFlow 后端使用，因为它依赖 `SeparableConvolution` 层，目前该层只支持 `channels_last` 的维度顺序（高度、宽度、通道）。

要通过 `load_model` 载入 MobileNet 模型，你需要导入自定义对象 `relu6` 和 `DepthwiseConv2D` 并通过 `custom_objects` 传参。

下面是示例代码：

```python
model = load_model('mobilenet.h5', custom_objects={
                   'relu6': mobilenet.relu6,
                   'DepthwiseConv2D': mobilenet.DepthwiseConv2D})
```

模型默认输入尺寸是 224x224.

### 参数

- include_top: 是否包括顶层的全连接层。
- weights: `None` 代表随机初始化， `'imagenet'` 代表加载在 ImageNet 上预训练的权值。
- input_tensor: 可选，Keras tensor 作为模型的输入（比如 `layers.Input()` 输出的 tensor）
- input_shape: 可选，输入尺寸元组，仅当 `include_top=False` 时有效（不然输入形状必须是 `(299, 299, 3)` （`channels_last` 格式）或 `(3, 299, 299)` （`channels_first` 格式），因为预训练模型是以这个大小训练的）。输入尺寸必须是三个数字，且宽高必须不小于 139，比如 `(150, 150, 3)` 是一个合法的输入尺寸。
- pooling: 可选，当 `include_top` 为 `False` 时，该参数指定了特征提取时的池化方式。
    - `None` 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    - `'avg'` 代表全局平均池化（GLobalAveragePool2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    - `'max'` 代表全局最大池化
- classes: 可选，图片分类的类别数，仅当 `include_top` 为 `True` 并且不加载预训练权值时可用。
- input_shape: 可选，输入尺寸元组，仅当 `include_top=False` 时有效（不然输入形状必须是 `(224, 224, 3)` ，因为预训练模型是以这个大小训练的）。输入尺寸必须是三个数字，且宽高必须不小于 32，比如 `(200, 200, 3)` 是一个合法的输入尺寸。
- alpha: 控制网络的宽度：
    - 如果 `alpha` < 1.0，则同比例减少每层的滤波器个数。
    - 如果 `alpha` > 1.0，则同比例增加每层的滤波器个数。
    - 如果 `alpha` = 1，使用论文默认的滤波器个数
- depth_multiplier: depthwise卷积的深度乘子，也称为（分辨率乘子）
- dropout: dropout 概率
- include_top: 是否包括顶层的全连接层。
- weights: `None` 代表随机初始化， `'imagenet'` 代表加载在 ImageNet 上预训练的权值。
- input_tensor: 可选，Keras tensor 作为模型的输入（比如 `layers.Input()` 输出的 tensor）
- pooling: 可选，当 `include_top` 为 `False` 时，该参数指定了特征提取时的池化方式。
    - `None` 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    - `'avg'` 代表全局平均池化（GLobalAveragePool2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    - `'max'` 代表全局最大池化
- classes: 可选，图片分类的类别数，仅当 `include_top` 为 `True` 并且不加载预训练权值时可用。

### 返回

一个 Keras `Model` 对象。

### 参考文献

- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)

### License

预训练权值基于 [Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)。

-----

## DenseNet


```python
keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.densenet.DenseNet169(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
```

可以选择载入在 ImageNet 上的预训练权值。如果你在使用 TensorFlow 为了发挥最佳性能，请在 `~/.keras/keras.json` 的 Keras 配置文件中设置 `image_data_format='channels_last'`。

模型和权值兼容 TensorFlow、Theano 和 CNTK。可以在你的 Keras 配置文件中指定数据格式。

### 参数

- blocks: 四个 Dense Layers 的 block 数量。
- include_top: 是否包括顶层的全连接层。
- weights: `None` 代表随机初始化， `'imagenet'` 代表加载在 ImageNet 上预训练的权值。
- input_tensor: 可选，Keras tensor 作为模型的输入（比如 `layers.Input()` 输出的 tensor）
- input_shape: 可选，输入尺寸元组，仅当 `include_top=False` 时有效（不然输入形状必须是 `(224, 224, 3)` （`channels_last` 格式）或 `(3, 224, 224)` （`channels_first` 格式），因为预训练模型是以这个大小训练的）。输入尺寸必须是三个数字。
- pooling: 可选，当 `include_top` 为 `False` 时，该参数指定了特征提取时的池化方式。
    - `None` 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    - `'avg'` 代表全局平均池化（GLobalAveragePool2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    - `'max'` 代表全局最大池化
- classes: 可选，图片分类的类别数，仅当 `include_top` 为 `True` 并且不加载预训练权值时可用。

### 

A Keras model instance.

### 返回

一个 Keras `Model` 对象。

### 参考文献

预训练权值基于 [BSD 3-clause License](https://github.com/liuzhuang13/DenseNet/blob/master/LICENSE)。

-----

## NASNet


```python
keras.applications.nasnet.NASNetLarge(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
keras.applications.nasnet.NASNetMobile(input_shape=None, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
```

在 ImageNet 上预训练的神经结构搜索网络模型（NASNet）。

注意，该模型目前仅能在 TensorFlow 后端使用，因此它只支持 `channels_last` 的维度顺序（高度、宽度、通道），可以在 `~/.keras/keras.json` Keras 配置文件中设置。

NASNetLarge 默认的输入尺寸是 331x331，NASNetMobile 默认的输入尺寸是 224x224。

### 参数

- input_shape: 可选，输入尺寸元组，仅当 `include_top=False` 时有效（不然对于 NASNetMobile 模型来说，输入形状必须是 `(224, 224, 3)` （`channels_last` 格式），对于 NASNetLarge 来说，输入形状必须是 `(331, 331, 3)` （`channels_last` 格式）。输入尺寸必须是三个数字。
- include_top: 是否包括顶层的全连接层。
- weights: `None` 代表随机初始化， `'imagenet'` 代表加载在 ImageNet 上预训练的权值。
- input_tensor: 可选，Keras tensor 作为模型的输入（比如 `layers.Input()` 输出的 tensor）
- input_shape: 可选，输入尺寸元组，仅当 `include_top=False` 时有效（不然输入形状必须是 `(224, 224, 3)` （`channels_last` 格式）或 `(3, 224, 224)` （`channels_first` 格式），因为预训练模型是以这个大小训练的）。输入尺寸必须是三个数字。
- pooling: 可选，当 `include_top` 为 `False` 时，该参数指定了特征提取时的池化方式。
    - `None` 代表不池化，直接输出最后一层卷积层的输出，该输出是一个四维张量。
    - `'avg'` 代表全局平均池化（GLobalAveragePool2D），相当于在最后一层卷积层后面再加一层全局平均池化层，输出是一个二维张量。
    - `'max'` 代表全局最大池化
- classes: 可选，图片分类的类别数，仅当 `include_top` 为 `True` 并且不加载预训练权值时可用。

### 返回

一个 Keras `Model` 对象。

### 参考文献

- [Learning Transferable Architectures for Scalable Image Recognition](https://arxiv.org/abs/1707.07012)

### License

预训练权值基于 [Apache License](https://github.com/tensorflow/models/blob/master/LICENSE)。
