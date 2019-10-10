
# 可视化 VGG16 的过滤器，通过输入空间梯度提升。

该脚本可以在几分钟内在 CPU 上运行完。

结果示例: ![Visualization](http://i.imgur.com/4nj4KjN.jpg)


```python
from __future__ import print_function

import time
import numpy as np
from PIL import Image as pil_image
from keras.preprocessing.image import save_img
from keras import layers
from keras.applications import vgg16
from keras import backend as K


def normalize(x):
    """用于标准化张量的实用函数。

    # 参数
        x: 输入张量。

    # 返回
        标准化的输入张量。
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_image(x):
    """用于将 float 数组转换为有效 uint8 图像的实用函数。

    # 参数
        x: 表示生成图像的 numpy 数组。

    # 返回
        经处理的 numpy 阵列，可用于 imshow 等。
    """
    # 标准化张量: center 为 0., 保证 std 为 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # 裁剪为 [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # 转换为 RGB 数组
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def process_image(x, former):
    """用于将 float 数组转换为有效 uint8 图像转换回 float 数组的实用函数。
       `deprocess_image` 反向操作。

    # 参数
        x: numpy 数组，可用于 imshow 等。
        former: 前身 numpy 数组，
                需要确定前者的均值和方差。

    # 返回
        一个处理后的 numpy 数组，表示一幅生成图像。
    """
    if K.image_data_format() == 'channels_first':
        x = x.transpose((2, 0, 1))
    return (x / 255 - 0.5) * 4 * former.std() + former.mean()


def visualize_layer(model,
                    layer_name,
                    step=1.,
                    epochs=15,
                    upscaling_steps=9,
                    upscaling_factor=1.2,
                    output_dim=(412, 412),
                    filter_range=(0, None)):
    """可视化某个模型中一个转换层的最相关过滤器。

    # 参数
        model: 包含 layer_name 的模型。
        layer_name: 需要可视化的层的名称。
                    必须是模型的一部分。
        step: 梯度提升步长。
        epochs: 梯度提升迭代轮次。
        upscaling_steps: upscaling 步数。
                         起始图像为 (80, 80)。
        upscaling_factor: 将图像缓慢提升到 output_dim 的因子。
        output_dim: [img_width, img_height] 输出图像维度。
        filter_range: 元组 [lower, upper]
                      决定需要计算的过滤器数目。
                      如果第二个值为 `None`,
                      最后一个过滤器将被推断为上边界。
    """

    def _generate_filter_image(input_img,
                               layer_output,
                               filter_index):
        """为一个特定的过滤器生成图像。

        # 参数
            input_img: 输入图像张量。
            layer_output: 输出图像张量。
            filter_index: 需要处理的过滤器数目。
                          假设可用。

        # 返回
            要么是 None，如果无法生成图像。
            要么是图像（数组）本身以及最后的 loss 组成的元组。
        """
        s_time = time.time()

        # 构建一个损失函数，使所考虑的层的第 n 个过滤器的激活最大化
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # 计算这种损失的输入图像的梯度
        grads = K.gradients(loss, input_img)[0]

        # 标准化技巧：将梯度标准化
        grads = normalize(grads)

        # 此函数返回给定输入图片的损失和梯度
        iterate = K.function([input_img], [loss, grads])

        # 从带有一些随机噪音的灰色图像开始
        intermediate_dim = tuple(
            int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random(
                (1, 3, intermediate_dim[0], intermediate_dim[1]))
        else:
            input_img_data = np.random.random(
                (1, intermediate_dim[0], intermediate_dim[1], 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # 缓慢放大原始图像的尺寸可以防止可视化结构的主导高频现象发生
        # （如果我们直接计算 412d-image 时该现象就会发生。）
        # 作为每个后续维度的更好起点，因此它避免了较差的局部最小值
        for up in reversed(range(upscaling_steps)):
            # 执行 20 次梯度提升
            for _ in range(epochs):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                # s一些过滤器被卡在了 0，我们可以跳过它们
                if loss_value <= K.epsilon():
                    return None

            # 计算放大维度
            intermediate_dim = tuple(
                int(x / (upscaling_factor ** up)) for x in output_dim)
            # 放大
            img = deprocess_image(input_img_data[0])
            img = np.array(pil_image.fromarray(img).resize(intermediate_dim,
                                                           pil_image.BICUBIC))
            input_img_data = np.expand_dims(
                process_image(img, input_img_data[0]), 0)

        # 解码生成的输入图像
        img = deprocess_image(input_img_data[0])
        e_time = time.time()
        print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(filter_index,
                                                                  loss_value,
                                                                  e_time - s_time))
        return img, loss_value

    def _draw_filters(filters, n=None):
        """在 nxn 网格中绘制最佳过滤器。

        # 参数
            filters: 每个已处理过滤器的生成图像及其相应的损失的列表。
            n: 网格维度。
               如果为 None，将使用最大可能的方格
        """
        if n is None:
            n = int(np.floor(np.sqrt(len(filters))))

        # 假设损失最大的过滤器看起来更好看。
        # 我们只保留顶部 n*n 过滤器。
        filters.sort(key=lambda x: x[1], reverse=True)
        filters = filters[:n * n]

        # 构建一个有足够空间的黑色图像
        # 例如，8 x 8 个过滤器，总尺寸为 412 x 412，每个过滤器 5px 间隔的图像
        MARGIN = 5
        width = n * output_dim[0] + (n - 1) * MARGIN
        height = n * output_dim[1] + (n - 1) * MARGIN
        stitched_filters = np.zeros((width, height, 3), dtype='uint8')

        # 用我们保存的过滤器填充图像
        for i in range(n):
            for j in range(n):
                img, _ = filters[i * n + j]
                width_margin = (output_dim[0] + MARGIN) * i
                height_margin = (output_dim[1] + MARGIN) * j
                stitched_filters[
                    width_margin: width_margin + output_dim[0],
                    height_margin: height_margin + output_dim[1], :] = img

        # 将结果保存到磁盘
        save_img('vgg_{0:}_{1:}x{1:}.png'.format(layer_name, n), stitched_filters)

    # 这是输入图像的占位符
    assert len(model.inputs) == 1
    input_img = model.inputs[0]

    # 获取每个『关键』图层的符号输出（我们给它们唯一的名称）。
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    output_layer = layer_dict[layer_name]
    assert isinstance(output_layer, layers.Conv2D)

    # 计算要处理的过滤范围
    filter_lower = filter_range[0]
    filter_upper = (filter_range[1]
                    if filter_range[1] is not None
                    else len(output_layer.get_weights()[1]))
    assert(filter_lower >= 0
           and filter_upper <= len(output_layer.get_weights()[1])
           and filter_upper > filter_lower)
    print('Compute filters {:} to {:}'.format(filter_lower, filter_upper))

    # 迭代每个过滤器并生成其相应的图像
    processed_filters = []
    for f in range(filter_lower, filter_upper):
        img_loss = _generate_filter_image(input_img, output_layer.output, f)

        if img_loss is not None:
            processed_filters.append(img_loss)

    print('{} filter processed.'.format(len(processed_filters)))
    # Finally draw and store the best filters to disk
    _draw_filters(processed_filters)


if __name__ == '__main__':
    # 我们想要可视化的图层的名称
    # (see model definition at keras/applications/vgg16.py)
    LAYER_NAME = 'block5_conv1'

    # 构建 ImageNet 权重预训练的 VGG16 网络
    vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')
    vgg.summary()

    # 调用示例函数
    visualize_layer(vgg, LAYER_NAME)
```
