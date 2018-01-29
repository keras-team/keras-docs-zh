## 正规化的使用

规则化器允许在优化过程中对图层参数或图层活动进行处罚。 这些处罚被纳入网络优化的损失函数中。

罚款是在每个层的基础上进行的。确切的API将取决于层，但层`密集`，`Conv1D`，`Conv2D`和`Conv3D`具有统一的API。

这些图层显示3个关键字参数：

- `kernel_regularizer`: `keras.regularizers.Regularizer`的实例
- `bias_regularizer`: `keras.regularizers.Regularizer`的实例
- `activity_regularizer`: `keras.regularizers.Regularizer`的实例


## 例子

```python
from keras import regularizers
model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

## 可用的惩罚

```python
keras.regularizers.l1(0.)
keras.regularizers.l2(0.)
keras.regularizers.l1_l2(0.)
```

## 开发新的正则化器

任何取得权重矩阵并返回损失贡献张量的函数都可以用作正则化器函数 , e.g.:

```python
from keras import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Dense(64, input_dim=64,
                kernel_regularizer=l1_reg))
```
另外，你也可以用面向对象的方式来写你的正则化器,
例子见[keras / regularizers.py]（https://github.com/keras-team/keras/blob/master/keras/regularizers.py)模块。
