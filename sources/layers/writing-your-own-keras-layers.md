# 编写自己的Keras图层

对于简单的无状态自定义操作，使用 `layers.core.Lambda` 层可能会更好。 但是对于任何具有可训练权重的自定义操作，您应该实现自己的图层。

这里是Keras层的骨架,**截至 Keras 2.0**（如果你有一个更老的版本，请升级）。 只有三种方法需要实现:

- `build(input_shape)`: 这是你将要定义你的权重的地方。 这个方法必须设置`self.built = True`，这可以通过调用`super（[Layer]，self）.build（）`来完成。
- `call(x)`: 这是图层的逻辑。 除非你想让你的图层支持遮罩，否则你只需要关心传递给`call`的第一个参数： 输入张量。
- `compute_output_shape(input_shape)`: 如果图层修改了其输入的形状，则应在此指定形状转换逻辑。这允许Keras做自动形状推断。

```python
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

现有的Keras层提供了如何实现几乎任何东西的例子。 永远不要犹豫，阅读源代码！
