# Keras 神经风格转换。

使用以下命令运行脚本：
```
python neural_style_transfer.py path_to_your_base_image.jpg     path_to_your_reference.jpg prefix_for_results
```
例如：
```
python neural_style_transfer.py img/tuebingen.jpg     img/starry_night.jpg results/my_result
```
可选参数：
```
--iter: 要指定进行样式转移的迭代次数（默认为 10）
--content_weight: 内容损失的权重（默认为 0.025）
--style_weight: 赋予样式损失的权重（默认为 1.0）
--tv_weight: 赋予总变化损失的权重（默认为 1.0）
```

为了提高速度，最好在 GPU 上运行此脚本。

示例结果: https://twitter.com/fchollet/status/686631033085677568

# 详情

样式转换包括生成具有与基本图像相同的 "内容"，但具有不同图片（通常是艺术的）的 "样式" 的图像。

这是通过优化具有 3 个成分的损失函数来实现的：样式损失，内容损失和总变化损失：

- 总变化损失在组合图像的像素之间强加了局部空间连续性，使其具有视觉连贯性。
- 样式损失是深度学习的根源-使用深度卷积神经网络定义深度学习。
精确地，它包括从卷积网络的不同层（在 ImageNet 上训练）提取的基础图像表示
形式和样式参考图像表示形式的 Gram 矩阵之间的 L2 距离之和。
总体思路是在不同的空间比例（相当大的比例-由所考虑的图层的深度定义）上捕获颜色/纹理信息。
 - 内容损失是基础图像（从较深层提取）的特征与组合图像的特征之间的 L2 距离，从而使生成的图像足够接近原始图像。

# 参考文献
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)


```python
from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg19
from keras import backend as K

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter

# 这些是不同损失成分的权重
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# 生成图片的尺寸。
width, height = load_img(base_image_path).size
img_nrows = 400
img_ncols = int(width * img_nrows / height)

# util 函数可将图片打开，调整大小并将其格式化为适当的张量


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# util 函数将张量转换为有效图像


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # 通过平均像素去除零中心
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# 得到我们图像的张量表示
base_image = K.variable(preprocess_image(base_image_path))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# 这将包含我们生成的图像
if K.image_data_format() == 'channels_first':
    combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
else:
    combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

# 将 3 张图像合并为一个 Keras 张量
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

# 以我们的 3 张图像为输入构建 VGG19 网络
# 该模型将加载预训练的 ImageNet 权重
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('Model loaded.')

# 获取每个 "关键" 层的符号输出（我们为它们指定了唯一的名称）。
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# 计算神经风格损失
# 首先，我们需要定义 4 个 util 函数

# 图像张量的 gram 矩阵（按特征量的外部乘积）


def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_data_format() == 'channels_first':
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

# "样式损失" 用于在生成的图像中保持参考图像的样式。
# 它基于来自样式参考图像和生成的图像的特征图的 gram 矩阵（捕获样式）


def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

# 辅助损失函数，用于在生成的图像中维持基本图像的 "内容"


def content_loss(base, combination):
    return K.sum(K.square(combination - base))

# 第 3 个损失函数，总变化损失，旨在使生成的图像保持局部连贯


def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == 'channels_first':
        a = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
        b = K.square(
            x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
        b = K.square(
            x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# 将这些损失函数组合成单个标量
loss = K.variable(0.0)
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss = loss + content_weight * content_loss(base_image_features,
                                            combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss = loss + (style_weight / len(feature_layers)) * sl
loss = loss + total_variation_weight * total_variation_loss(combination_image)

# 获得损失后生成图像的梯度
grads = K.gradients(loss, combination_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# 该 Evaluator 类可以通过两个单独的函数 "loss" 和 "grads" 来一次计算损失和梯度，同时检索它们。
# 这样做是因为 scipy.optimize 需要使用损耗和梯度的单独函数，但是分别计算它们将效率低下。 


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

# 对生成的图像的像素进行基于 Scipy 的优化（L-BFGS），以最大程度地减少神经样式损失
x = preprocess_image(base_image_path)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # 保存当前生成的图像
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
```