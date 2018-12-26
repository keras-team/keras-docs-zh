## 正则化器的使用

正则化器允许在优化过程中对层的参数或层的激活情况进行惩罚。 网络优化的损失函数也包括这些惩罚项。

惩罚是以层为对象进行的。具体的 API 因层而异，但 `Dense`，`Conv1D`，`Conv2D` 和 `Conv3D` 这些层具有统一的 API。

正则化器开放 3 个关键字参数：

- `kernel_regularizer`: `keras.regularizers.Regularizer` 的实例
- `bias_regularizer`: `keras.regularizers.Regularizer` 的实例
- `activity_regularizer`: `keras.regularizers.Regularizer` 的实例


## 例

```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

## 可用的正则化器

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(l1=0.01, l2=0.01)
```

## 开发新的正则化器

任何输入一个权重矩阵、返回一个损失贡献张量的函数，都可以用作正则化器，例如：

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg))
```

另外，你也可以用面向对象的方式来编写正则化器的代码，例子见 [keras/regularizers.py](https://github.com/keras-team/keras/blob/master/keras/regularizers.py) 模块。
