# 关于 Keras 模型

在 Keras 中有两类主要的模型：[Sequential 顺序模型](/models/sequential) 和 [使用函数式 API 的 Model 类模型](/models/model)。

这些模型有许多共同的方法和属性：

- `model.layers` 是包含模型网络层的展平列表。
- `model.inputs` 是模型输入张量的列表。
- `model.outputs` 是模型输出张量的列表。
- `model.summary()` 打印出模型概述信息。 它是 [utils.print_summary](/utils/#print_summary) 的简捷调用。
- `model.get_config()` 返回包含模型配置信息的字典。通过以下代码，就可以根据这些配置信息重新实例化模型：

```python
config = model.get_config()
model = Model.from_config(config)
# 或者，对于 Sequential:
model = Sequential.from_config(config)
```

- `model.get_weights()` 返回模型中所有权重张量的列表，类型为 Numpy 数组。
- `model.set_weights(weights)` 从 Numpy 数组中为模型设置权重。列表中的数组必须与 `get_weights()` 返回的权重具有相同的尺寸。
- `model.to_json()` 以 JSON 字符串的形式返回模型的表示。请注意，该表示不包括权重，仅包含结构。你可以通过以下方式从 JSON 字符串重新实例化同一模型（使用重新初始化的权重）：

```python
from keras.models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```

- `model.to_yaml()` 以 YAML 字符串的形式返回模型的表示。请注意，该表示不包括权重，只包含结构。你可以通过以下代码，从 YAML 字符串中重新实例化相同的模型（使用重新初始化的权重）：

```python
from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

- `model.save_weights(filepath)` 将模型权重存储为 HDF5 文件。
- `model.load_weights(filepath, by_name=False)`: 从 HDF5 文件（由 `save_weights` 创建）中加载权重。默认情况下，模型的结构应该是不变的。 如果想将权重载入不同的模型（部分层相同）， 设置 `by_name=True` 来载入那些名字相同的层的权重。

注意：另请参阅[如何安装 HDF5 或 h5py 以保存 Keras 模型](/getting-started/faq/#how-can-i-install-HDF5-or-h5py-to-save-my-models-in-Keras)，在常见问题中了解如何安装 `h5py` 的说明。

## Model 类继承

除了这两类模型之外，你还可以通过继承 `Model` 类并在 `call` 方法中实现你自己的前向传播，以创建你自己的完全定制化的模型，（`Model` 类继承 API 引入于 Keras 2.2.0）。

这里是一个用 `Model` 类继承写的简单的多层感知器的例子：

```python
import keras

class SimpleMLP(keras.Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)

model = SimpleMLP()
model.compile(...)
model.fit(...)
```

网络层定义在 `__init__(self, ...)` 中，前向传播在 `call(self, inputs)` 中指定。在 `call` 中，你可以指定自定义的损失函数，通过调用 `self.add_loss(loss_tensor)` （就像你在自定义层中一样）。

在类继承模型中，模型的拓扑结构是由 Python 代码定义的（而不是网络层的静态图）。这意味着该模型的拓扑结构不能被检查或序列化。因此，以下方法和属性**不适用于类继承模型**：

- `model.inputs` 和 `model.outputs`。
- `model.to_yaml()` 和 `model.to_json()`。
- `model.get_config()` 和 `model.save()`。

**关键点**：为每个任务使用正确的 API。`Model` 类继承 API 可以为实现复杂模型提供更大的灵活性，但它需要付出代价（比如缺失的特性）：它更冗长，更复杂，并且有更多的用户错误机会。如果可能的话，尽可能使用函数式 API，这对用户更友好。
