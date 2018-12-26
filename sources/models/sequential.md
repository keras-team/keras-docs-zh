# Sequential 模型 API

在阅读这片文档前，请先阅读 [Keras Sequential 模型指引](/getting-started/sequential-model-guide)。

----

## Sequential 模型方法

### compile


```python
compile(optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
```

用于配置训练模型。

__参数__

- __optimizer__: 字符串（优化器名）或者优化器对象。详见 [optimizers](/optimizers)。
- __loss__: 字符串（目标函数名）或目标函数。详见 [losses](/losses)。
如果模型具有多个输出，则可以通过传递损失函数的字典或列表，在每个输出上使用不同的损失。模型将最小化的损失值将是所有单个损失的总和。
- __metrics__: 在训练和测试期间的模型评估标准。通常你会使用 `metrics = ['accuracy']`。
要为多输出模型的不同输出指定不同的评估标准，还可以传递一个字典，如 `metrics = {'output_a'：'accuracy'}`。
- __loss_weights__: 指定标量系数（Python浮点数）的可选列表或字典，用于加权不同模型输出的损失贡献。
    模型将要最小化的损失值将是所有单个损失的加权和，由 `loss_weights` 系数加权。
    如果是列表，则期望与模型的输出具有 1:1 映射。
    如果是张量，则期望将输出名称（字符串）映射到标量系数。
- __sample_weight_mode__: 如果你需要执行按时间步采样权重（2D 权重），请将其设置为 `temporal`。
默认为 `None`，为采样权重（1D）。如果模型有多个输出，则可以通过传递 mode 的字典或列表，以在每个输出上使用不同的 `sample_weight_mode`。
- __weighted_metrics__: 在训练和测试期间，由 sample_weight 或 class_weight 评估和加权的度量标准列表。
- __target_tensors__: 默认情况下，Keras 将为模型的目标创建一个占位符，在训练过程中将使用目标数据。相反，如果你想使用自己的目标张量（反过来说，Keras 在训练期间不会载入这些目标张量的外部 Numpy 数据），您可以通过 `target_tensors` 参数指定它们。它应该是单个张量（对于单输出 Sequential 模型）。
- __**kwargs__: 当使用 Theano/CNTK 后端时，这些参数被传入 `K.function`。当使用 TensorFlow 后端时，这些参数被传递到 `tf.Session.run`。

__异常__

- __ValueError__:  如果 `optimizer`, `loss`, `metrics` 或 `sample_weight_mode` 这些参数不合法。

----

### fit


```python
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
```

以固定数量的轮次（数据集上的迭代）训练模型。

__参数__

- __x__: 训练数据的 Numpy 数组。
如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。
如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，x 可以是 `None`（默认）。
- __y__: 目标（标签）数据的 Numpy 数组。
如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。
如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，y 可以是 `None`（默认）。
- __batch_size__: 整数或 `None`。每次提度更新的样本数。如果未指定，默认为 32.
- __epochs__: 整数。训练模型迭代轮次。一个轮次是在整个 `x` 或 `y` 上的一轮迭代。请注意，与 `initial_epoch` 一起，`epochs` 被理解为 「最终轮次」。模型并不是训练了 `epochs` 轮，而是到第 `epochs` 轮停止训练。
- __verbose__: 0, 1 或 2。日志显示模式。
0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
- __callbacks__: 一系列的 `keras.callbacks.Callback` 实例。一系列可以在训练时使用的回调函数。详见 [callbacks](/callbacks)。
- __validation_split__: 在 0 和 1 之间浮动。用作验证集的训练数据的比例。模型将分出一部分不会被训练的验证数据，并将在每一轮结束时评估这些验证数据的误差和任何其他模型指标。验证数据是混洗之前 `x` 和`y` 数据的最后一部分样本中。
- __validation_data__: 元组 `(x_val，y_val)` 或元组 `(x_val，y_val，val_sample_weights)`，用来评估损失，以及在每轮结束时的任何模型度量指标。模型将不会在这个数据上进行训练。这个参数会覆盖 `validation_split`。
- __shuffle__: 布尔值（是否在每轮迭代之前混洗数据）或者 字符串 (`batch`)。`batch` 是处理 HDF5 数据限制的特殊选项，它对一个 batch 内部的数据进行混洗。当 `steps_per_epoch` 非 `None` 时，这个参数无效。
- __class_weight__: 可选的字典，用来映射类索引（整数）到权重（浮点）值，用于加权损失函数（仅在训练期间）。这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。
- __sample_weight__: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1：1 映射），或者在时序数据的情况下，可以传递尺寸为 `(samples, sequence_length)` 的 2D 数组，以对每个样本的每个时间步施加不同的权重。在这种情况下，你应该确保在 `compile()` 中指定 `sample_weight_mode="temporal"`。
- __initial_epoch__: 开始训练的轮次（有助于恢复之前的训练）。
- __steps_per_epoch__: 在声明一个轮次完成并开始下一个轮次之前的总步数（样品批次）。使用 TensorFlow 数据张量等输入张量进行训练时，默认值 `None` 等于数据集中样本的数量除以 batch 的大小，如果无法确定，则为 1。
- __validation_steps__: 只有在指定了 `steps_per_epoch`时才有用。停止前要验证的总步数（批次样本）。

__返回__

一个 `History` 对象。其 `History.history` 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）。 

__异常__

- __RuntimeError__: 如果模型从未编译。
- __ValueError__: 在提供的输入数据与模型期望的不匹配的情况下。

----

### evaluate


```python
evaluate(x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None)
```

在测试模式，返回误差值和评估标准值。

计算逐批次进行。

__参数__

- __x__: 训练数据的 Numpy 数组。
如果模型中的输入层被命名，你也可以传递一个字典，将输入层名称映射到 Numpy 数组。
如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，x 可以是 `None`（默认）。
- __y__: 目标（标签）数据的 Numpy 数组。
  如果模型中的输出层被命名，你也可以传递一个字典，将输出层名称映射到 Numpy 数组。
  如果从本地框架张量馈送（例如 TensorFlow 数据张量）数据，y 可以是 `None`（默认）。
- __batch_size__: 整数或 `None`。每次提度更新的样本数。如果未指定，默认为 32.
- __verbose__: 0, 1。日志显示模式。0 = 安静模式, 1 = 进度条。
- __sample_weight__: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。
  您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1：1 映射），或者在时序数据的情况下，可以传递尺寸为 `(samples, sequence_length)` 的 2D 数组，以对每个样本的每个时间步施加不同的权重。在这种情况下，你应该确保在 `compile()` 中指定 `sample_weight_mode="temporal"`。
- __steps__: 整数或 `None`。
  声明评估结束之前的总步数（批次样本）。默认值 `None`。

__返回__

标量测试误差（如果模型只有单个输出且没有评估指标）或标量列表（如果模型具有多个输出和/或指标）。
属性 `model.metrics_names` 将提供标量输出的显示标签。

----

### predict


```python
predict(x, batch_size=None, verbose=0, steps=None)
```

为输入样本生成输出预测。

计算逐批次进行。

__参数__

- __x__: 输入数据，Numpy 数组（或者如果模型有多个输入，则为 Numpy 数组列表）。
- __batch_size__: 整数。如未指定，默认为 32。
- __verbose__: 日志显示模式，0 或 1。
- __steps__: 声明预测结束之前的总步数（批次样本）。默认值 `None`。

__返回__

预测的 Numpy 数组。

__异常__

- __ValueError__: 如果提供的输入数据与模型的期望数据不匹配，或者有状态模型收到的数量不是批量大小的倍数。

----

### train_on_batch


```python
train_on_batch(x, y, sample_weight=None, class_weight=None)
```

一批样品的单次梯度更新。

__Arguments__

- __x__: 训练数据的 Numpy 数组，如果模型具有多个输入，则为 Numpy 数组列表。如果模型中的所有输入都已命名，你还可以传入输入名称到 Numpy 数组的映射字典。
- __y__: 目标数据的 Numpy 数组，如果模型具有多个输入，则为 Numpy 数组列表。如果模型中的所有输出都已命名，你还可以传入输出名称到 Numpy 数组的映射字典。
- __sample_weight__: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。
  您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1：1 映射），或者在时序数据的情况下，可以传递尺寸为 `(samples, sequence_length)` 的 2D 数组，以对每个样本的每个时间步施加不同的权重。在这种情况下，你应该确保在 `compile()` 中指定 `sample_weight_mode="temporal"`。
- __class_weight__: 可选的字典，用来映射类索引（整数）到权重（浮点）值，用于加权损失函数（仅在训练期间）。这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。


__返回__

标量训练误差（如果模型只有单个输出且没有评估指标）或标量列表（如果模型具有多个输出和/或指标）。
属性 `model.metrics_names` 将提供标量输出的显示标签。

----

### test_on_batch

```python
test_on_batch(x, y, sample_weight=None)
```

在一批样本上评估模型。

__参数__

- __x__: 训练数据的 Numpy 数组，如果模型具有多个输入，则为 Numpy 数组列表。如果模型中的所有输入都已命名，你还可以传入输入名称到 Numpy 数组的映射字典。
- __y__: 目标数据的 Numpy 数组，如果模型具有多个输入，则为 Numpy 数组列表。如果模型中的所有输出都已命名，你还可以传入输出名称到 Numpy 数组的映射字典。
- __sample_weight__: 训练样本的可选 Numpy 权重数组，用于对损失函数进行加权（仅在训练期间）。
  您可以传递与输入样本长度相同的平坦（1D）Numpy 数组（权重和样本之间的 1：1 映射），或者在时序数据的情况下，可以传递尺寸为 `(samples, sequence_length)` 的 2D 数组，以对每个样本的每个时间步施加不同的权重。在这种情况下，你应该确保在 `compile()` 中指定 `sample_weight_mode="temporal"`。

__返回__

标量测试误差（如果模型只有单个输出且没有评估指标）或标量列表（如果模型具有多个输出和/或指标）。
属性 `model.metrics_names` 将提供标量输出的显示标签。

----

### predict_on_batch


```python
predict_on_batch(x)
```

返回一批样本的模型预测值。

__参数__

- __x__: 输入数据，Numpy 数组或列表（如果模型有多输入）。

__返回__

预测值的 Numpy 数组。

----

### fit_generator


```python
fit_generator(generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
```

使用 Python 生成器或 `Sequence` 实例逐批生成的数据，按批次训练模型。

生成器与模型并行运行，以提高效率。
例如，这可以让你在 CPU 上对图像进行实时数据增强，以在 GPU 上训练模型。

`keras.utils.Sequence` 的使用可以保证数据的顺序， 以及当 `use_multiprocessing=True` 时 ，保证每个输入在每个 epoch 只使用一次。

__参数__

- __generator__: 一个生成器或 Sequence (`keras.utils.Sequence`) 对象的实例，以避免在使用多进程时出现重复数据。
生成器的输出应该为以下之一：
    - 一个 `(inputs, targets)` 元组
    - 一个 `(inputs, targets, sample_weights)` 元组。
  这个元组（生成器的单个输出）表示一个独立批次。因此，此元组中的所有数组必须具有相同的长度（等于此批次的大小）。不同的批次可能具有不同的大小。例如，如果数据集的大小不能被批量大小整除，则最后一批时期通常小于其他批次。生成器将无限地在数据集上循环。当运行到第 `steps_per_epoch` 时，记一个 epoch 结束。
- __steps_per_epoch__: 整数。在声明一个 epoch 完成并开始下一个 epoch 之前从 `generator` 产生的总步数（批次样本）。它通常应该等于你的数据集的样本数量除以批量大小。可选参数 `Sequence`：如果未指定，将使用 `len(generator)` 作为步数。
- __epochs__: 整数，数据的迭代总轮数。一个 epoch 是对所提供的整个数据的一轮迭代，由 `steps_per_epoch` 所定义。请注意，与 `initial_epoch` 一起，参数 `epochs` 应被理解为 「最终轮数」。模型并不是训练了 `epochs` 轮，而是到第 `epochs` 轮停止训练。
- __verbose__: 日志显示模式。0，1 或 2。0 = 安静模式，1 = 进度条，2 = 每轮一行。
- __callbacks__: `keras.callbacks.Callback` 实例列表。在训练时调用的一系列回调。详见 [callbacks](/callbacks)。
- __validation_data__: 它可以是以下之一：
    - 验证数据的生成器或 `Sequence` 实例
    - 一个 `(inputs, targets)` 元组
    - 一个 `(inputs, targets, sample_weights)` 元组。
- __validation_steps__: 仅当 `validation_data` 是一个生成器时才可用。
每个 epoch 结束时验证集生成器产生的步数。它通常应该等于你的数据集的样本数量除以批量大小。可选参数 `Sequence`：如果未指定，将使用 `len(generator)` 作为步数。
- __class_weight__: 可选的字典，用来映射类索引（整数）到权重（浮点）值，用于加权损失函数（仅在训练期间）。这可能有助于告诉模型 「更多关注」来自代表性不足的类的样本。
- __max_queue_size__: 整数。生成器队列的最大尺寸。如果未指定，`max_queue_size` 将默认为 10。
- __workers__: 整数。使用基于进程的多线程时启动的最大进程数。如果未指定，`worker` 将默认为 1。如果为 0，将在主线程上执行生成器。
- __use_multiprocessing__: 如果 True，则使用基于进程的多线程。如果未指定，`use_multiprocessing` 将默认为 `False`。请注意，因为此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
- __shuffle__: 布尔值。是否在每轮迭代之前打乱 batch 的顺序。只能与 `Sequence` (`keras.utils.Sequence`) 实例同用。在 `steps_per_epoch` 不为 `None` 是无效果。
- __initial_epoch__: 整数。开始训练的轮次（有助于恢复之前的训练）。

__返回__

一个 `History` 对象。其 `History.history` 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）。 

__异常__

- __ValueError__: 如果生成器生成的数据格式不正确。

__例子__


```python
def generate_arrays_from_file(path):
    while True:
        with open(path) as f:
            for line in f:
                # 从文件中的每一行生成输入数据和标签的 numpy 数组
                x1, x2, y = process_line(line)
                    yield ({'input_1': x1, 'input_2': x2}, {'output': y})

model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                    steps_per_epoch=10000, epochs=10)
```

----

### evaluate_generator


```python
evaluate_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```

在数据生成器上评估模型。

这个生成器应该返回与 `test_on_batch` 所接收的同样的数据。

__参数__

- __generator__: 生成器，生成 (inputs, targets)
或 (inputs, targets, sample_weights)，或 Sequence (`keras.utils.Sequence`) 对象的实例，以避免在使用多进程时出现重复数据。
- __steps__: 在停止之前，来自 `generator` 的总步数 (样本批次)。
可选参数 `Sequence`：如果未指定，将使用`len(generator)` 作为步数。
- __max_queue_size__: 生成器队列的最大尺寸。
- __workers__: 整数。使用基于进程的多线程时启动的最大进程数。如果未指定，`worker` 将默认为 1。如果为 0，将在主线程上执行生成器。
- __use_multiprocessing__: 如果 True，则使用基于进程的多线程。
请注意，因为此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
- __verbose__：日志显示模式，0 或 1。

__返回__

标量测试误差（如果模型只有单个输出且没有评估指标）或标量列表（如果模型具有多个输出和/或指标）。
属性 `model.metrics_names` 将提供标量输出的显示标签。

__异常__

- __ValueError__: 如果生成器生成的数据格式不正确。

----

### predict_generator


```python
predict_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
```

为来自数据生成器的输入样本生成预测。

这个生成器应该返回与 `predict_on_batch` 所接收的同样的数据。

__参数__

- __generator__: 返回批量输入样本的生成器，或 Sequence (`keras.utils.Sequence`) 对象的实例，以避免在使用多进程时出现重复数据。
- __steps__: 在停止之前，来自 `generator` 的总步数 (样本批次)。
可选参数 `Sequence`：如果未指定，将使用`len(generator)` 作为步数。
- __max_queue_size__: 生成器队列的最大尺寸。
- __workers__: 整数。使用基于进程的多线程时启动的最大进程数。如果未指定，`worker` 将默认为 1。如果为 0，将在主线程上执行生成器。
- __use_multiprocessing__: 如果 True，则使用基于进程的多线程。
请注意，因为此实现依赖于多进程，所以不应将不可传递的参数传递给生成器，因为它们不能被轻易地传递给子进程。
- __verbose__: 日志显示模式， 0 或 1。

__返回__

预测值的 Numpy 数组。

__异常__

- __ValueError__: 如果生成器生成的数据格式不正确。

----

### get_layer


```python
get_layer(name=None, index=None)
```

根据名称（唯一）或索引值查找网络层。

如果同时提供了 `name` 和 `index`，则 `index` 将优先。

根据网络层的名称（唯一）或其索引返回该层。索引是基于水平图遍历的顺序（自下而上）。

__参数__

- __name__: 字符串，层的名字。
- __index__: 整数，层的索引。

__返回__

一个层实例。

__异常__

- __ValueError__: 如果层的名称或索引不正确。
