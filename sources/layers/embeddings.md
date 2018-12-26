<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/embeddings.py#L16)</span>
### Embedding

```python
keras.layers.Embedding(input_dim, output_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
```

将正整数（索引值）转换为固定尺寸的稠密向量。
例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

该层只能用作模型中的第一层。

__例子__


```python
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# 模型将输入一个大小为 (batch, input_length) 的整数矩阵。
# 输入中最大的整数（即词索引）不应该大于 999 （词汇表大小）
# 现在 model.output_shape == (None, 10, 64)，其中 None 是 batch 的维度。

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)
```

__参数__

- __input_dim__: int > 0。词汇表大小，
即，最大整数 index + 1。
- __output_dim__: int >= 0。词向量的维度。
- __embeddings_initializer__: `embeddings` 矩阵的初始化方法
(详见 [initializers](../initializers.md))。
- __embeddings_regularizer__: `embeddings` matrix 的正则化方法
(详见 [regularizer](../regularizers.md))。
- __embeddings_constraint__: `embeddings` matrix 的约束函数
(详见 [constraints](../constraints.md))。
- __mask_zero__: 是否把 0 看作为一个应该被遮蔽的特殊的 "padding" 值。
这对于可变长的 [循环神经网络层](recurrent.md) 十分有用。
如果设定为 `True`，那么接下来的所有层都必须支持 masking，否则就会抛出异常。
如果 mask_zero 为 `True`，作为结果，索引 0 就不能被用于词汇表中
（input_dim 应该与 vocabulary + 1 大小相同）。
- __input_length__: 输入序列的长度，当它是固定的时。
如果你需要连接 `Flatten` 和 `Dense` 层，则这个参数是必须的
（没有它，dense 层的输出尺寸就无法计算）。

__输入尺寸__

尺寸为 `(batch_size, sequence_length)` 的 2D 张量。

__输出尺寸__

尺寸为 `(batch_size, sequence_length, output_dim)` 的 3D 张量。

__参考文献__

- [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
