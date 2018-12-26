<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L114)</span>
### TimeDistributed

```python
keras.layers.TimeDistributed(layer)
```

这个封装器将一个层应用于输入的每个时间片。

输入至少为 3D，且第一个维度应该是时间所表示的维度。

考虑 32 个样本的一个 batch，
其中每个样本是 10 个 16 维向量的序列。
那么这个 batch 的输入尺寸为 `(32, 10, 16)`，
而 `input_shape` 不包含样本数量的维度，为 `(10, 16)`。

你可以使用 `TimeDistributed` 来将 `Dense` 层独立地应用到
这 10 个时间步的每一个：

```python
# 作为模型第一层
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# 现在 model.output_shape == (None, 10, 8)
```

输出的尺寸为 `(32, 10, 8)`。

在后续的层中，将不再需要 `input_shape`：

```python
model.add(TimeDistributed(Dense(32)))
# 现在 model.output_shape == (None, 10, 32)
```

输出的尺寸为 `(32, 10, 32)`。

`TimeDistributed` 可以应用于任意层，不仅仅是 `Dense`，
例如运用于 `Conv2D` 层：

```python
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 3)),
                          input_shape=(10, 299, 299, 3)))
```

__参数__

- __layer__: 一个网络层实例。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L333)</span>
### Bidirectional

```python
keras.layers.Bidirectional(layer, merge_mode='concat', weights=None)
```

RNN 的双向封装器，对序列进行前向和后向计算。

__参数__

- __layer__: `Recurrent` 实例。
- __merge_mode__: 前向和后向 RNN 的输出的结合模式。
为 {'sum', 'mul', 'concat', 'ave', None} 其中之一。
如果是 None，输出不会被结合，而是作为一个列表被返回。

__异常__

- __ValueError__: 如果参数 `merge_mode` 非法。

__例__


```python
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```
