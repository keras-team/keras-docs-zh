
## 评价函数的用法

评价函数用于评估当前训练模型的性能。当模型编译后（compile），评价函数应该作为 `metrics` 的参数来输入。

```python
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```

```python
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

评价函数和 [损失函数](/losses) 相似，只不过评价函数的结果不会用于训练过程中。

我们可以传递已有的评价函数名称，或者传递一个自定义的 Theano/TensorFlow 函数来使用（查阅[自定义评价函数](#custom-metrics)）。

__参数__

  - __y_true__: 真实标签，Theano/Tensorflow 张量。
  - __y_pred__: 预测值。和 y_true 相同尺寸的 Theano/TensorFlow 张量。

__返回__
  
  返回一个表示全部数据点平均值的张量。

----

## 可使用的评价函数


### accuracy


```python
keras.metrics.accuracy(y_true, y_pred)
```


### binary_accuracy


```python
keras.metrics.binary_accuracy(y_true, y_pred, threshold=0.5)
```

----

### categorical_accuracy


```python
keras.metrics.categorical_accuracy(y_true, y_pred)
```

----

### sparse_categorical_accuracy


```python
keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
```

----

### top_k_categorical_accuracy


```python
keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)
```

----

### sparse_top_k_categorical_accuracy


```python
keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
```


----

### cosine_proximity


```python
keras.metrics.cosine_proximity(y_true, y_pred, axis=-1)
```

----

### clone_metric


```python
keras.metrics.clone_metric(metric)
```

若有状态，返回评估指标的克隆，否则返回其本身。


----

### clone_metrics


```python
keras.metrics.clone_metrics(metrics)
```

克隆给定的评估指标序列/字典。

除以上评估指标，你还可以使用在损失函数页描述的损失函数作为评估指标。

----

## 自定义评价函数

自定义评价函数应该在编译的时候（compile）传递进去。该函数需要以 `(y_true, y_pred)` 作为输入参数，并返回一个张量作为输出结果。

```python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```
