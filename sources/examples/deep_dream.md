# Keras 实现的 Deep Dreaming。

按以下命令执行该脚本：
```python
python deep_dream.py path_to_your_base_image.jpg prefix_for_results
```

例如：
```python
python deep_dream.py img/mypic.jpg results/dream
```


```python
from __future__ import print_function

from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
import scipy
import argparse

from keras.applications import inception_v3
from keras import backend as K

parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

args = parser.parse_args()
base_image_path = args.base_image_path
result_prefix = args.result_prefix

# 这些是我们尝试最大化激活的层的名称，以及它们们在我们试图最大化的最终损失中的权重。
# 你可以调整这些设置以获得新的视觉效果。
settings = {
    'features': {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    },
}


def preprocess_image(image_path):
    # 用于打开，调整图片大小并将图片格式化为适当的张量的实用函数。
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # 函数将张量转换为有效图像的实用函数。
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

K.set_learning_phase(0)

# 使用我们的占位符构建 InceptionV3 网络。
# 该模型将加载预先训练的 ImageNet 权重。
model = inception_v3.InceptionV3(weights='imagenet',
                                 include_top=False)
dream = model.input
print('Model loaded.')

# 获取每个『关键』层的符号输出（我们为它们指定了唯一的名称）。
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# 定义损失。
loss = K.variable(0.)
for layer_name in settings['features']:
    # 将层特征的 L2 范数添加到损失中。
    if layer_name not in layer_dict:
        raise ValueError('Layer ' + layer_name + ' not found in model.')
    coeff = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    # 我们通过仅涉及损失中的非边界像素来避免边界伪影。
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if K.image_data_format() == 'channels_first':
        loss = loss + coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
    else:
        loss = loss + coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# 计算 dream 即损失的梯度。
grads = K.gradients(loss, dream)[0]
# 标准化梯度。
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

# 设置函数，以检索给定输入图像的损失和梯度的值。
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


"""Process:

- 载入原始图像。
- 定义一系列预处理规模 (即图像尺寸)，从最小到最大。
- 将原始图像调整为最小尺寸。
- 对于每个规模，从最小的（即当前的）开始：
    - 执行梯度提升
    - 将图像放大到下一个比例
    - 重新投射在提升时丢失的细节
- 当我们回到原始大小时停止。

为了获得在放大过程中丢失的细节，我们只需将原始图像缩小，放大，然后将结果与（调整大小的）原始图像进行比较即可。
"""


# 把玩这些超参数也可以让你获得新的效果
step = 0.01  # 梯度提升步长
num_octave = 3  # 运行梯度提升的规模数
octave_scale = 1.4  # 规模之间的比
iterations = 20  # 每个规模的提升步数
max_loss = 10.

img = preprocess_image(base_image_path)
if K.image_data_format() == 'channels_first':
    original_shape = img.shape[2:]
else:
    original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)

save_img(result_prefix + '.png', deprocess_image(np.copy(img)))
```
