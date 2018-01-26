## 约束使用

来自`constraints`模块的函数允许在优化期间对网络参数设置约束（例如，非负）。

惩罚以每层为基础进行。 确切的API将取决于层，但层`密集`，`Conv1D`，`Conv2D`和`Conv3D`具有统一的API。
这些图层显示2个关键字参数：

- `kernel_constraint` 为主权重矩阵。
- `bias_constraint` 为偏见。


```python
from keras.constraints import max_norm
model.add(Dense(64, kernel_constraint=max_norm(2.)))
```

## 可用的约束

- __max_norm(max_value=2, axis=0)__: 最大范数约束
- __non_neg()__: 非负面约束
- __unit_norm(axis=0)__: 单位规范约束
- __min_max_norm(min_value=0.0, max_value=1.0, rate=1.0, axis=0)__:  最小/最大范数约束
