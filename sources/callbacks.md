## 回调函数使用

回调函数是一个函数的合集，会在训练的阶段中所使用。你可以使用回调函数来查看训练模型的内在状态和统计。你可以传递一个列表的回调函数（作为 `callbacks` 关键字参数）到 `Sequential` 或 `Model` 类型的 `.fit()` 方法。在训练时，相应的回调函数的方法就会被在各自的阶段被调用。

---

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L145)</span>
### Callback

```python
keras.callbacks.Callback()
```

用来组建新的回调函数的抽象基类。

__属性__

- __params__: 字典。训练参数，
(例如，verbosity, batch size, number of epochs...)。
- __model__: `keras.models.Model` 的实例。
指代被训练模型。

被回调函数作为参数的 `logs` 字典，它会含有于当前批量或训练周期相关数据的键。

目前，`Sequential` 模型类的 `.fit()` 方法会在传入到回调函数的 `logs` 里面包含以下的数据：

- __on_epoch_end__: 包括 `acc` 和 `loss` 的日志， 也可以选择性的包括 `val_loss`（如果在 `fit` 中启用验证），和 `val_acc`（如果启用验证和监测精确值）。
- __on_batch_begin__: 包括 `size` 的日志，在当前批量内的样本数量。
- __on_batch_end__: 包括 `loss` 的日志，也可以选择性的包括 `acc`（如果启用监测精确值）。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L201)</span>
### BaseLogger

```python
keras.callbacks.BaseLogger()
```

会积累训练周期平均评估的回调函数。

这个回调函数被自动应用到每一个 Keras 模型上面。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L230)</span>
### TerminateOnNaN

```python
keras.callbacks.TerminateOnNaN()
```

当遇到 NaN 损失会停止训练的回调函数。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L246)</span>
### ProgbarLogger

```python
keras.callbacks.ProgbarLogger(count_mode='samples')
```

会把评估以标准输出打印的回调函数。

__参数__

- __count_mode__: "steps" 或者 "samples"。
进度条是否应该计数看见的样本或步骤（批量）。

__触发__

- __ValueError__: 防止不正确的 `count_mode`

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L313)</span>
### History

```python
keras.callbacks.History()
```

把所有事件都记录到 `History` 对象的回调函数。

这个回调函数被自动启用到每一个 Keras 模型。`History` 对象会被模型的 `fit` 方法返回。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L332)</span>
### ModelCheckpoint

```python
keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
```

在每个训练期之后保存模型。

`filepath` 可以包括命名格式选项，可以由 `epoch` 的值和 `logs` 的键（由 `on_epoch_end` 参数传递）来填充。

例如：如果 `filepath` 是 `weights.{epoch:02d}-{val_loss:.2f}.hdf5`，
那么模型被保存的的文件名就会有训练周期数和验证损失。

__参数__

- __filepath__: 字符串，保存模型的路径。
- __monitor__: 被监测的数据。
- __verbose__: 详细信息模式，0 或者 1 。
- __save_best_only__: 如果 `save_best_only=True`，
被监测数据的最佳模型就不会被覆盖。
- __mode__: {auto, min, max} 的其中之一。
如果 `save_best_only=True`，那么是否覆盖保存文件的决定就取决于被监测数据的最大或者最小值。
对于 `val_acc`，模式就会是 `max`，而对于 `val_loss`，模式就需要是 `min`，等等。
在 `auto` 模式中，方向会自动从被监测的数据的名字中判断出来。
- __save_weights_only__: 如果 True，那么只有模型的参数会被保存 (`model.save_weights(filepath)`)，
否则的话，整个模型会被保存 (`model.save(filepath)`)。
- __period__: 每个检查点之间的间隔（训练周期数）。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L432)</span>
### EarlyStopping

```python
keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
```

当被监测的数量不再优化，则停止训练。

__参数__

- __monitor__: 被监测的数据。
- __min_delta__: 在被监测的数据中被认为是优化的最小变化，
例如，小于 min_delta 的绝对变化会被认为没有优化。
- __patience__: 没有优化的训练周期数，在这之后训练就会被停止。
- __verbose__: 详细信息模式。
- __mode__: {auto, min, max} 其中之一。 在 `min` 模式中，
当被监测的数据停止下降，训练就会停止；在 `max`
模式中，当被监测的数据停止上升，训练就会停止；在 `auto`
模式中，方向会自动从被监测的数据的名字中判断出来。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L514)</span>
### RemoteMonitor

```python
keras.callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None)
```

Callback used to stream events to a server.

Requires the `requests` library.
Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
HTTP POST, with a `data` argument which is a
JSON-encoded dictionary of event data.

__Arguments__

- __root__: String; root url of the target server.
- __path__: String; path relative to `root` to which the events will be sent.
- __field__: String; JSON field under which the data will be stored.
- __headers__: Dictionary; optional custom HTTP headers.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L559)</span>
### LearningRateScheduler

```python
keras.callbacks.LearningRateScheduler(schedule, verbose=0)
```

Learning rate scheduler.

__Arguments__

- __schedule__: a function that takes an epoch index as input
(integer, indexed from 0) and returns a new
learning rate as output (float).
- __verbose__: int. 0: quiet, 1: update messages.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L587)</span>
### TensorBoard

```python
keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
```

Tensorboard basic visualizations.

[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
is a visualization tool provided with TensorFlow.

This callback writes a log for TensorBoard, which allows
you to visualize dynamic graphs of your training and test
metrics, as well as activation histograms for the different
layers in your model.

If you have installed TensorFlow with pip, you should be able
to launch TensorBoard from the command line:
```sh
tensorboard --logdir=/full_path_to_your_logs
```

__Arguments__

- __log_dir__: the path of the directory where to save the log
files to be parsed by TensorBoard.
- __histogram_freq__: frequency (in epochs) at which to compute activation
and weight histograms for the layers of the model. If set to 0,
histograms won't be computed. Validation data (or split) must be
specified for histogram visualizations.
- __write_graph__: whether to visualize the graph in TensorBoard.
The log file can become quite large when
write_graph is set to True.
- __write_grads__: whether to visualize gradient histograms in TensorBoard.
`histogram_freq` must be greater than 0.
- __batch_size__: size of batch of inputs to feed to the network
for histograms computation.
- __write_images__: whether to write model weights to visualize as
image in TensorBoard.
- __embeddings_freq__: frequency (in epochs) at which selected embedding
layers will be saved.
- __embeddings_layer_names__: a list of names of layers to keep eye on. If
None or empty list all the embedding layer will be watched.
- __embeddings_metadata__: a dictionary which maps layer name to a file name
in which metadata for this embedding layer is saved. See the
[details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
about metadata files format. In case if the same metadata file is
used for all embedding layers, string can be passed.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L811)</span>
### ReduceLROnPlateau

```python
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
```

Reduce learning rate when a metric has stopped improving.

Models often benefit from reducing the learning rate by a factor
of 2-10 once learning stagnates. This callback monitors a
quantity and if no improvement is seen for a 'patience' number
of epochs, the learning rate is reduced.

__Example__


```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

__Arguments__

- __monitor__: quantity to be monitored.
- __factor__: factor by which the learning rate will
be reduced. new_lr = lr * factor
- __patience__: number of epochs with no improvement
after which learning rate will be reduced.
- __verbose__: int. 0: quiet, 1: update messages.
- __mode__: one of {auto, min, max}. In `min` mode,
lr will be reduced when the quantity
monitored has stopped decreasing; in `max`
mode it will be reduced when the quantity
monitored has stopped increasing; in `auto`
mode, the direction is automatically inferred
from the name of the monitored quantity.
- __epsilon__: threshold for measuring the new optimum,
to only focus on significant changes.
- __cooldown__: number of epochs to wait before resuming
normal operation after lr has been reduced.
- __min_lr__: lower bound on the learning rate.

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L927)</span>
### CSVLogger

```python
keras.callbacks.CSVLogger(filename, separator=',', append=False)
```

Callback that streams epoch results to a csv file.

Supports all values that can be represented as a string,
including 1D iterables such as np.ndarray.

__Example__


```python
csv_logger = CSVLogger('training.log')
model.fit(X_train, Y_train, callbacks=[csv_logger])
```

__Arguments__

- __filename__: filename of the csv file, e.g. 'run/log.csv'.
- __separator__: string used to separate elements in the csv file.
- __append__: True: append if file exists (useful for continuing
training). False: overwrite existing file,

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L1004)</span>
### LambdaCallback

```python
keras.callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
```

Callback for creating simple, custom callbacks on-the-fly.

This callback is constructed with anonymous functions that will be called
at the appropriate time. Note that the callbacks expects positional
arguments, as:

- `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
`epoch`, `logs`
- `on_batch_begin` and `on_batch_end` expect two positional arguments:
`batch`, `logs`
- `on_train_begin` and `on_train_end` expect one positional argument:
`logs`

__Arguments__

- __on_epoch_begin__: called at the beginning of every epoch.
- __on_epoch_end__: called at the end of every epoch.
- __on_batch_begin__: called at the beginning of every batch.
- __on_batch_end__: called at the end of every batch.
- __on_train_begin__: called at the beginning of model training.
- __on_train_end__: called at the end of model training.

__Example__


```python
# Print the batch number at the beginning of every batch.
batch_print_callback = LambdaCallback(
    on_batch_begin=lambda batch,logs: print(batch))

# Stream the epoch loss to a file in JSON format. The file content
# is not well-formed JSON but rather has a JSON object per line.
import json
json_log = open('loss_log.json', mode='wt', buffering=1)
json_logging_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: json_log.write(
        json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
    on_train_end=lambda logs: json_log.close()
)

# Terminate some processes after having finished model training.
processes = ...
cleanup_callback = LambdaCallback(
    on_train_end=lambda logs: [
        p.terminate() for p in processes if p.is_alive()])

model.fit(...,
          callbacks=[batch_print_callback,
                     json_logging_callback,
                     cleanup_callback])
```


---


# Create a callback

You can create a custom callback by extending the base class `keras.callbacks.Callback`. A callback has access to its associated model through the class property `self.model`.

Here's a simple example saving a list of losses over each batch during training:
```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
```

---

### Example: recording loss history

```python
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, callbacks=[history])

print(history.losses)
# outputs
'''
[0.66047596406559383, 0.3547245744908703, ..., 0.25953155204159617, 0.25901699725311789]
'''
```

---

### Example: model checkpoints

```python
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Dense(10, input_dim=784, kernel_initializer='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=0, validation_data=(X_test, Y_test), callbacks=[checkpointer])
```
