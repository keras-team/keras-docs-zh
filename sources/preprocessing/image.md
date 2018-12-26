
# 图像预处理

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/image.py#L232)</span>
## ImageDataGenerator 类

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,  
                                             samplewise_center=False, 
                                             featurewise_std_normalization=False, 
                                             samplewise_std_normalization=False, 
                                             zca_whitening=False, 
                                             zca_epsilon=1e-06, 
                                             rotation_range=0, 
                                             width_shift_range=0.0, 
                                             height_shift_range=0.0, 
                                             brightness_range=None, 
                                             shear_range=0.0, 
                                             zoom_range=0.0, 
                                             channel_shift_range=0.0, 
                                             fill_mode='nearest', 
                                             cval=0.0, 
                                             horizontal_flip=False, 
                                             vertical_flip=False, 
                                             rescale=None, 
                                             preprocessing_function=None, 
                                             data_format=None, 
                                             validation_split=0.0, 
                                             dtype=None)
```

通过实时数据增强生成张量图像数据批次。数据将不断循环（按批次）。

__参数__

- __featurewise_center__: 布尔值。将输入数据的均值设置为 0，逐特征进行。
- __samplewise_center__: 布尔值。将每个样本的均值设置为 0。
- __featurewise_std_normalization__: Boolean. 布尔值。将输入除以数据标准差，逐特征进行。
- __samplewise_std_normalization__: 布尔值。将每个输入除以其标准差。
- __zca_epsilon__: ZCA 白化的 epsilon 值，默认为 1e-6。
- __zca_whitening__: 布尔值。是否应用 ZCA 白化。
- __rotation_range__: 整数。随机旋转的度数范围。
- __width_shift_range__: 浮点数、一维数组或整数
    - float: 如果 <1，则是除以总宽度的值，或者如果 >=1，则为像素值。
    - 1-D 数组: 数组中的随机元素。
    - int: 来自间隔 `(-width_shift_range, +width_shift_range)` 之间的整数个像素。
    - `width_shift_range=2` 时，可能值是整数 `[-1, 0, +1]`，与 `width_shift_range=[-1, 0, +1]` 相同；而 `width_shift_range=1.0` 时，可能值是 `[-1.0, +1.0)` 之间的浮点数。
- __height_shift_range__: 浮点数、一维数组或整数
    - float: 如果 <1，则是除以总宽度的值，或者如果 >=1，则为像素值。
    - 1-D array-like: 数组中的随机元素。
    - int: 来自间隔 `(-height_shift_range, +height_shift_range)` 之间的整数个像素。
    - `height_shift_range=2` 时，可能值是整数 `[-1, 0, +1]`，与 `height_shift_range=[-1, 0, +1]` 相同；而 `height_shift_range=1.0` 时，可能值是 `[-1.0, +1.0)` 之间的浮点数。
- __shear_range__: 浮点数。剪切强度（以弧度逆时针方向剪切角度）。
- __zoom_range__: 浮点数 或 `[lower, upper]`。随机缩放范围。如果是浮点数，`[lower, upper] = [1-zoom_range, 1+zoom_range]`。
- __channel_shift_range__: 浮点数。随机通道转换的范围。
- __fill_mode__: {"constant", "nearest", "reflect" or "wrap"} 之一。默认为 'nearest'。输入边界以外的点根据给定的模式填充：
    - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
    - 'nearest': aaaaaaaa|abcd|dddddddd
    - 'reflect': abcddcba|abcd|dcbaabcd
    - 'wrap': abcdabcd|abcd|abcdabcd
- __cval__: 浮点数或整数。用于边界之外的点的值，当 `fill_mode = "constant"` 时。
- __horizontal_flip__: 布尔值。随机水平翻转。
- __vertical_flip__: 布尔值。随机垂直翻转。
- __rescale__: 重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
- __preprocessing_function__: 应用于每个输入的函数。这个函数会在任何其他改变之前运行。这个函数需要一个参数：一张图像（秩为 3 的 Numpy 张量），并且应该输出一个同尺寸的 Numpy 张量。
- __data_format__: 图像数据格式，{"channels_first", "channels_last"} 之一。"channels_last" 模式表示图像输入尺寸应该为 `(samples, height, width, channels)`，"channels_first" 模式表示输入尺寸应该为 `(samples, channels, height, width)`。默认为 在 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值。如果你从未设置它，那它就是 "channels_last"。
- __validation_split__: 浮点数。Float. 保留用于验证的图像的比例（严格在0和1之间）。
- __dtype__: 生成数组使用的数据类型。


__例子__

使用 `.flow(x, y)` 的例子：

```python
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# 计算特征归一化所需的数量
# （如果应用 ZCA 白化，将计算标准差，均值，主成分）
datagen.fit(x_train)

# 使用实时数据增益的批数据对模型进行拟合：
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# 这里有一个更 「手动」的例子
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # 我们需要手动打破循环，
            # 因为生成器会无限循环
            break
```

使用 `.flow_from_directory(directory)` 的例子：

```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
```

同时转换图像和蒙版 (mask) 的例子。

```python
# 创建两个相同参数的实例
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# 为 fit 和 flow 函数提供相同的种子和关键字参数
seed = 1
image_datagen.fit(images, augment=True, seed=seed)
mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/images',
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/masks',
    class_mode=None,
    seed=seed)

# 将生成器组合成一个产生图像和蒙版（mask）的生成器
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
```

---

### ImageDataGenerator 类方法

### apply_transform


```python
apply_transform(x, transform_parameters)
```

根据给定的参数将变换应用于图像。

__参数__

- __x__: 3D 张量，单张图像。
- __transform_parameters__: 字符串 - 参数 对表示的字典，用于描述转换。目前，使用字典中的以下参数：
    - 'theta': 浮点数。旋转角度（度）。
    - 'tx': 浮点数。在 x 方向上移动。
    - 'ty': 浮点数。在 y 方向上移动。
    - shear': 浮点数。剪切角度（度）。
    - 'zx': 浮点数。放大 x 方向。
    - 'zy': 浮点数。放大 y 方向。
    - 'flip_horizontal': 布尔 值。水平翻转。
    - 'flip_vertical': 布尔值。垂直翻转。
    - 'channel_shift_intencity': 浮点数。频道转换强度。
    - 'brightness': 浮点数。亮度转换强度。

__返回__

输入的转换后版本（相同尺寸）。

---

### fit


```python
fit(x, augment=False, rounds=1, seed=None)
```

将数据生成器用于某些示例数据。

它基于一组样本数据，计算与数据转换相关的内部数据统计。

当且仅当 `featurewise_center` 或 `featurewise_std_normalization` 或 `zca_whitening` 设置为 True 时才需要。

__参数__

- __x__: 样本数据。秩应该为 4。对于灰度数据，通道轴的值应该为 1；对于 RGB 数据，值应该为 3。
- __augment__: 布尔值（默认为 False）。是否使用随机样本扩张。
- __rounds__: 整数（默认为 1）。如果数据数据增强（augment=True），表明在数据上进行多少次增强。
- __seed__: 整数（默认 None）。随机种子。

---

### flow


```python
flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)
```

采集数据和标签数组，生成批量增强数据。

__参数__

 - __x__: 输入数据。秩为 4 的 Numpy 矩阵或元组。如果是元组，第一个元素应该包含图像，第二个元素是另一个 Numpy 数组或一列 Numpy 数组，它们不经过任何修改就传递给输出。可用于将模型杂项数据与图像一起输入。对于灰度数据，图像数组的通道轴的值应该为 1，而对于 RGB 数据，其值应该为 3。
- __y__: 标签。
- __batch_size__: 整数 (默认为 32)。
- __shuffle__: 布尔值 (默认为 True)。
- __sample_weight__: 样本权重。
- __seed__: 整数（默认为 None）。
- __save_to_dir__: None 或 字符串（默认为 None）。这使您可以选择指定要保存的正在生成的增强图片的目录（用于可视化您正在执行的操作）。
- __save_prefix__: 字符串（默认 `''`）。保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
- __save_format__: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
- __subset__: 数据子集 ("training" 或 "validation")，如果 在 `ImageDataGenerator` 中设置了 `validation_split`。

__返回__

一个生成元组 `(x, y)` 的 `Iterator`，其中 `x` 是图像数据的 Numpy 数组（在单张图像输入时），或 Numpy 数组列表（在额外多个输入时），`y` 是对应的标签的 Numpy 数组。如果 'sample_weight' 不是 None，生成的元组形式为 `(x, y, sample_weight)`。如果 `y` 是 None, 只有 Numpy 数组 `x` 被返回。

---

### flow_from_dataframe


```python
flow_from_dataframe(dataframe, directory, x_col='filename', y_col='class', has_ext=True, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest')
```


输入 dataframe 和目录的路径，并生成批量的增强/标准化的数据。

这里有一个简单的教程： [http://bit.ly/keras_flow_from_dataframe](http://bit.ly/keras_flow_from_dataframe)


__参数__

- __dataframe__: Pandas dataframe，一列为图像的文件名，另一列为图像的类别，
或者是可以作为原始目标数据多个列。
- __directory__: 字符串，目标目录的路径，其中包含在 dataframe 中映射的所有图像。
- __x_col__: 字符串，dataframe 中包含目标图像文件夹的目录的列。
- __y_col__: 字符串或字符串列表，dataframe 中将作为目标数据的列。
- __has_ext__: 布尔值，如果 dataframe[x_col] 中的文件名具有扩展名则为 True，否则为 False。
- __target_size__: 整数元组 `(height, width)`，默认为 `(256, 256)`。
                 所有找到的图都会调整到这个维度。
- __color_mode__: "grayscale", "rbg" 之一。默认："rgb"。
                图像是否转换为 1 个或 3 个颜色通道。
- __classes__: 可选的类别列表
    (例如， `['dogs', 'cats']`)。默认：None。
     如未提供，类比列表将自动从 y_col 中推理出来，y_col 将会被映射为类别索引）。
     包含从类名到类索引的映射的字典可以通过属性 `class_indices` 获得。
- __class_mode__: "categorical", "binary", "sparse", "input", "other" or None 之一。
     默认："categorical"。决定返回标签数组的类型：
     - `"categorical"` 将是 2D one-hot 编码标签，
     - `"binary"` 将是 1D 二进制标签，
     - `"sparse"` 将是 1D 整数标签，
     - `"input"` 将是与输入图像相同的图像（主要用于与自动编码器一起使用），
     - `"other"` 将是 y_col 数据的 numpy 数组，
     - None, 不返回任何标签（生成器只会产生批量的图像数据，这对使用 `model.predict_generator()`, `model.evaluate_generator()` 等很有用）。
- __batch_size__: 批量数据的尺寸（默认：32）。
- __shuffle__: 是否混洗数据（默认：True）
- __seed__: 可选的混洗和转换的随即种子。
- __save_to_dir__: None 或 str (默认: None).
                 这允许你可选地指定要保存正在生成的增强图片的目录（用于可视化您正在执行的操作）。
- __save_prefix__: 字符串。保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
- __save_format__: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
- __follow_links__: 是否跟随类子目录中的符号链接（默认：False）。
- __subset__: 数据子集 (`"training"` 或 `"validation"`)，如果在 `ImageDataGenerator` 中设置了 `validation_split`。
- __interpolation__: 在目标大小与加载图像的大小不同时，用于重新采样图像的插值方法。
     支持的方法有 `"nearest"`, `"bilinear"`, and `"bicubic"`。
     如果安装了 1.1.3 以上版本的 PIL 的话，同样支持 `"lanczos"`。
     如果安装了 3.4.0 以上版本的 PIL 的话，同样支持 `"box"` 和 `"hamming"`。
     默认情况下，使用 `"nearest"`。

__Returns__

一个生成 `(x, y)` 元组的 DataFrameIterator，
其中 `x` 是一个包含一批尺寸为 `(batch_size, *target_size, channels)` 
的图像样本的 numpy 数组，`y` 是对应的标签的 numpy 数组。

---

### flow_from_directory


```python
flow_from_directory(directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')
```

__参数__

- __directory__: 目标目录的路径。每个类应该包含一个子目录。任何在子目录树下的 PNG, JPG, BMP, PPM 或 TIF 图像，都将被包含在生成器中。更多细节，详见 [此脚本](https://gist.github.com/fchollet/%20%20%20%20%20%20%20%200830affa1f7f19fd47b06d4cf89ed44d)。
- __target_size__: 整数元组 `(height, width)`，默认：`(256, 256)`。所有的图像将被调整到的尺寸。
- __color_mode__: "grayscale", "rbg" 之一。默认："rgb"。图像是否被转换成 1 或 3 个颜色通道。
- __classes__: 可选的类的子目录列表（例如 `['dogs', 'cats']`）。默认：None。如果未提供，类的列表将自动从 `directory` 下的 子目录名称/结构 中推断出来，其中每个子目录都将被作为不同的类（类名将按字典序映射到标签的索引）。包含从类名到类索引的映射的字典可以通过 `class_indices` 属性获得。
- __class_mode__:  "categorical", "binary", "sparse", "input" 或 None 之一。默认："categorical"。决定返回的标签数组的类型：
    - "categorical" 将是 2D one-hot 编码标签，
    - "binary" 将是 1D 二进制标签，"sparse" 将是 1D 整数标签，
    - "input" 将是与输入图像相同的图像（主要用于自动编码器）。
    - 如果为 None，不返回标签（生成器将只产生批量的图像数据，对于 `model.predict_generator()`, `model.evaluate_generator()` 等很有用）。请注意，如果 `class_mode` 为 None，那么数据仍然需要驻留在 `directory` 的子目录中才能正常工作。
- __batch_size__: 一批数据的大小（默认 32）。
- __shuffle__: 是否混洗数据（默认 True）。
- __seed__: 可选随机种子，用于混洗和转换。
- __save_to_dir__: None 或 字符串（默认 None）。这使你可以最佳地指定正在生成的增强图片要保存的目录（用于可视化你在做什么）。
- __save_prefix__: 字符串。 保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
- __save_format__: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
- __follow_links__: 是否跟踪类子目录中的符号链接（默认为 False）。
- __subset__: 数据子集 ("training" 或 "validation")，如果 在 `ImageDataGenerator` 中设置了 `validation_split`。
- __interpolation__: 在目标大小与加载图像的大小不同时，用于重新采样图像的插值方法。
     支持的方法有 `"nearest"`, `"bilinear"`, and `"bicubic"`。
     如果安装了 1.1.3 以上版本的 PIL 的话，同样支持 `"lanczos"`。
     如果安装了 3.4.0 以上版本的 PIL 的话，同样支持 `"box"` 和 `"hamming"`。
     默认情况下，使用 `"nearest"`。

__返回__

一个生成 `(x, y)` 元组的 `DirectoryIterator`，其中 `x` 是一个包含一批尺寸为 `(batch_size, *target_size, channels)`的图像的 Numpy 数组，`y` 是对应标签的 Numpy 数组。

---

### get_random_transform


```python
get_random_transform(img_shape, seed=None)
```

为转换生成随机参数。

__参数__

- __seed__: 随机种子
- __img_shape__: 整数元组。被转换的图像的尺寸。

__返回__

包含随机选择的描述变换的参数的字典。

---


### random_transform


```python
random_transform(x, seed=None)
```

将随机变换应用于图像。

__参数__

- __x__: 3D 张量，单张图像。
- __seed__: 随机种子。

__返回__

输入的随机转换版本（相同形状）。

---

#### standardize

```python
standardize(x)
```

将标准化配置应用于一批输入。

__参数__

- __x__: 需要标准化的一批输入。

__返回__

标准化后的输入。
