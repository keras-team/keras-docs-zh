
## ImageDataGenerator

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())
```

生成批次的带实时数据增益的张量图像数据。数据将按批次无限循环。

- __参数__：
    - __featurewise_center__: 布尔值。将输入数据的均值设置为 0，逐特征进行。
    - __samplewise_center__: 布尔值。将每个样本的均值设置为 0。
    - __featurewise_std_normalization__: 布尔值。将输入除以数据标准差，逐特征进行。
    - __samplewise_std_normalization__: 布尔值。将每个输入除以其标准差。
    - __zca_epsilon__: ZCA 白化的 epsilon 值，默认为 1e-6。
    - __zca_whitening__: 布尔值。应用 ZCA 白化。
    - __rotation_range__: 整数。随机旋转的度数范围。
    - __width_shift_range__: 浮点数（总宽度的比例）。随机水平移动的范围。
    - __height_shift_range__: 浮点数（总高度的比例）。随机垂直移动的范围。
    - __shear_range__: 浮点数。剪切强度（以弧度逆时针方向剪切角度）。
    - __zoom_range__: 浮点数 或 [lower, upper]。随机缩放范围。如果是浮点数，`[lower, upper] = [1-zoom_range, 1+zoom_range]`。
    - __channel_shift_range__: 浮点数。随机通道转换的范围。
    - __fill_mode__: {"constant", "nearest", "reflect" or "wrap"} 之一。输入边界以外的点根据给定的模式填充：
        * "constant": `kkkkkkkk|abcd|kkkkkkkk` (`cval=k`)
        * "nearest":  `aaaaaaaa|abcd|dddddddd`
        * "reflect":  `abcddcba|abcd|dcbaabcd`
        * "wrap":     `abcdabcd|abcd|abcdabcd`
    - __cval__: 浮点数或整数。用于边界之外的点的值，当 `fill_mode = "constant"` 时。
    - __horizontal_flip__: 布尔值。随机水平翻转。
    - __vertical_flip__: 布尔值。随机垂直翻转。
    - __rescale__: 重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
    - __preprocessing_function__: 应用于每个输入的函数。这个函数会在任何其他改变之前运行。这个函数需要一个参数：一张图像（秩为 3 的 Numpy 张量），并且应该输出一个同尺寸的 Numpy 张量。
    - __data_format__: {"channels_first", "channels_last"} 之一。"channels_last" 模式表示输入尺寸应该为 `(samples, height, width, channels)`，"channels_first" 模式表示输入尺寸应该为 `(samples, channels, height, width)`。默认为 在 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值。如果你从未设置它，那它就是 "channels_last"。
- __方法__:
    - __fit(x)__: 根据一组样本数据，计算与数据相关转换有关的内部数据统计信息。当且仅当 featurewise_center 或 featurewise_std_normalization 或 zca_whitening 时才需要。
        - __参数__:
            - __x__: 样本数据。秩应该为 4。在灰度数据的情况下，通道轴的值应该为 1，在 RGB 数据的情况下，它应该为 3。
            - __augment__: 布尔值（默认 False）。是否使用随机样本增益。
            - __rounds__: 整数（默认 1）。如果 augment，在数据上进行多少次增益。
            - __seed__: 整数（默认 None）。随机种子。
    - __flow(x, y)__: 传入 Numpy 数据和标签数组，生成批次的 增益的/标准化的 数据。在生成的批次数据上无限制地无限次循环。
        - __参数__:
            - __x__: 数据。秩应该为 4。在灰度数据的情况下，通道轴的值应该为 1，在 RGB 数据的情况下，它应该为 3。
            - __y__: 标签。
            - __batch_size__: 整数（默认 32）。
            - __shuffle__: 布尔值（默认 True）。
            - __seed__: 整数（默认 None）。
            - __save_to_dir__: None 或 字符串（默认 None）。这使你可以最佳地指定正在生成的增强图片要保存的目录（用于可视化你在做什么）。
            - __save_prefix__: 字符串（默认 `''`）。 保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
            - __save_format__: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
        - __yields__: 元组 `(x, y)`，其中 `x` 是图像数据的 Numpy 数组，`y` 是相应标签的 Numpy 数组。生成器将无限循环。
    - __flow_from_directory(directory)__: 以目录路径为参数，生成批次的 增益的/标准化的 数据。在生成的批次数据上无限制地无限次循环。
        - __参数__:
            - __directory__: 目标目录的路径。每个类应该包含一个子目录。任何在子目录下的 PNG, JPG, BMP 或 PPM 图像，都将被包含在生成器中。更多细节，详见 [此脚本](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)。
            - __target_size__: 整数元组 `(height, width)`，默认：`(256, 256)`。所有的图像将被调整到的尺寸。
            - __color_mode__: "grayscale", "rbg" 之一。默认："rgb"。图像是否被转换成1或3个颜色通道。
            - __classes__: 可选的类的子目录列表（例如 `['dogs', 'cats']`）。默认：None。如果未提供，类的列表将自动从“目录”下的子目录名称/结构中推断出来，其中每个子目录都将被作为不同的类（类名将按字典序映射到标签的索引）。包含从类名到类索引的映射的字典可以通过`class_indices`属性获得。
            - __class_mode__: "categorical", "binary", "sparse", "input" 或 None 之一。默认："categorical"。决定返回的标签数组的类型："categorical" 将是 2D one-hot 编码标签，"binary" 将是 1D 二进制标签，"sparse" 将是 1D 整数标签，"input" 将是与输入图像相同的图像（主要用于与自动编码器一起工作）。如果为 None，不返回标签（生成器将只产生批量的图像数据，对于 `model.predict_generator()`, `model.evaluate_generator()` 等很有用）。请注意，如果 class_mode 为 None，那么数据仍然需要驻留在 `directory` 的子目录中才能正常工作。
            - __batch_size__: 一批数据的大小（默认 32）。
            - __shuffle__: 是否混洗数据（默认 True）。
            - __seed__: 可选随机种子，用于混洗和转换。
            - __save_to_dir__: None 或 字符串（默认 None）。这使你可以最佳地指定正在生成的增强图片要保存的目录（用于可视化你在做什么）。
            - __save_prefix__: 字符串。 保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
            - __save_format__: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
            - __follow_links__: 是否跟踪类子目录下的符号链接（默认 False）。


- __例__:

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
                     rotation_range=90.,
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
