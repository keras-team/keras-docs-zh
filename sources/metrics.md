
## 评估标准的用法

一个评估标准是用来判断你的模型的性能的函数。 评估标准函数在模型编译的 `metrics` 参数中提供。

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

一个估标准函数类似于一个[损失函数](/losses), 除了训练模型时不使用评估度量的估标准。

您可以传递现有度量的名称，也可以通过Theano / TensorFlow标记函数 (看到自定义 [定义的评估标准](#custom-metrics))
#### 参数
  - __y_true__: 真正的标签。 Theano/TensorFlow 张量。
  - __y_pred__: 预测。 Theano/TensorFlow 与 y_true 形状相同的张量。

#### 返回

  一个张量代表所有输出数组的平均值
  数据点。

----

## 可用的评估标准


### binary_accuracy


```python
binary_accuracy(y_true, y_pred)
```

----

### categorical_accuracy


```python
categorical_accuracy(y_true, y_pred)
```

----

### sparse_categorical_accuracy


```python
sparse_categorical_accuracy(y_true, y_pred)
```

----

### top_k_categorical_accuracy


```python
top_k_categorical_accuracy(y_true, y_pred, k=5)
```

----

### sparse_top_k_categorical_accuracy


```python
sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
```


----

## 自定义的评估标准

自定义评估标准可以在编译步骤中传递。函数将需要`（y_true，y_pred）`作为参数并返回一个张量值。

```python
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```
