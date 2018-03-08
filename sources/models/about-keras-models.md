# 关于Keras模型

在 Keras 中有两类模型：[Suquential 顺序模型](/models/sequential) 和 [使用函数式 API 的 Model 类模型](/models/model)。

这些模型有许多共同的方法：

- `model.summary()`: 打印出模型概述信息。 它是 [utils.print_summary](/utils/#print_summary) 的简捷调用。
- `model.get_config()`: 返回包含模型配置信息的字典。通过以下代码，就可以根据这些配置信息重新实例化模型：

```python
config = model.get_config()
model = Model.from_config(config)
# or, for Sequential:
model = Sequential.from_config(config)
```

- `model.get_weights()`: 返回模型权重的张量列表，类型为 Numpy array。
- `model.set_weights(weights)`: 从 Nympy array 中为模型设置权重。列表中的数组必须与 `get_weights()` 返回的权重具有相同的尺寸。
- `model.to_json()`: 以 JSON 字符串的形式返回模型的表示。请注意，该表示不包括权重，只包含结构。你可以通过以下代码，从 JSON 字符串中重新实例化相同的模型（带有重新初始化的权重）：

```python
from keras.models import model_from_json

json_string = model.to_json()
model = model_from_json(json_string)
```

- `model.to_yaml()`: 以 YAML 字符串的形式返回模型的表示。请注意，该表示不包括权重，只包含结构。你可以通过以下代码，从 YAML 字符串中重新实例化相同的模型（带有重新初始化的权重）：

```python
from keras.models import model_from_yaml

yaml_string = model.to_yaml()
model = model_from_yaml(yaml_string)
```

- `model.save_weights(filepath)`: 将模型权重存储为 HDF5 文件。
- `model.load_weights(filepath, by_name=False)`: 从 HDF5 文件（由 `save_weights` 创建）中加载权重。默认情况下，模型的结构应该是不变的。 如果想将权重载入不同的模型（部分层相同）， 设置 `by_name=True` 来载入那些名字相同的层的权重。
