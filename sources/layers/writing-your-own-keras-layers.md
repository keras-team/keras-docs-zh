# 编写你自己的Keras层

对于简单、无状态的自定义操作，你也许可以通过`layers.core.Lambda`层来实现。但是对于那些包含了可训练权重的自定义层，你应该自己实现这种层。

这是一个**Keras2.0**中，Keras层的骨架（如果你用的是旧的版本，请你更新）。你只需要实现三个方法即可:

- `build(input_shape)`: 这是你定义权重的地方。这个方法必须设`self.built = True`，可以通过调用`super([Layer], self).build()`完成。
- `call(x)`: 这里是编写层的功能逻辑的地方。你只需要关注传入`call`的第一个参数：输入张量，除非你希望你的层支持masking。
- `compute_output_shape(input_shape)`: 如果你的层更改了输入张量的形状，你应该在这里定义形状变化的逻辑，这让Keras能够自动推断各层的形状。

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

已有的Keras层就是实现层的很好例子。不要犹豫阅读源码！
