## 约束项的使用

`constraints` 模块的函数允许在优化期间对网络参数设置约束（例如非负性）。

约束是以层为对象进行的。具体的 API 因层而异，但 `Dense`，`Conv1D`，`Conv2D` 和 `Conv3D` 这些层具有统一的 API。

约束层开放 2 个关键字参数：

- `kernel_constraint` 用于主权重矩阵。
- `bias_constraint` 用于偏置。

```python
from keras.constraints import max_norm
model.add(Dense(64, kernel_constraint=max_norm(2.)))
```

## 可用的约束

- __max_norm(max_value=2, axis=0)__: 最大范数约束
- __non_neg()__: 非负性约束
- __unit_norm(axis=0)__: 单位范数约束
- __min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)__:  最小/最大范数约束

