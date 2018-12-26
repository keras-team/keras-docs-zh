<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/preprocessing/sequence.py#L16)</span>
### TimeseriesGenerator

```python
keras.preprocessing.sequence.TimeseriesGenerator(data, targets, length, sampling_rate=1, stride=1, start_index=0, end_index=None, shuffle=False, reverse=False, batch_size=128)
```

用于生成批量时序数据的实用工具类。

这个类以一系列由相等间隔以及一些时间序列参数（例如步长、历史长度等）汇集的数据点作为输入，以生成用于训练/验证的批次数据。

__参数__

- __data__: 可索引的生成器（例如列表或 Numpy 数组），包含连续数据点（时间步）。数据应该是 2D 的，且第 0 个轴为时间维度。
- __targets__: 对应于 `data` 的时间步的目标值。它应该与 `data` 的长度相同。
- __length__: 输出序列的长度（以时间步数表示）。
- __sampling_rate__: 序列内连续各个时间步之间的周期。对于周期 `r`, 时间步 `data[i]`, `data[i-r]`, ... `data[i - length]` 被用于生成样本序列。
- __stride__: 连续输出序列之间的周期. 对于周期 `s`, 连续输出样本将为 `data[i]`, `data[i+s]`, `data[i+2*s]` 等。
- __start_index__: 在 `start_index` 之前的数据点在输出序列中将不被使用。这对保留部分数据以进行测试或验证很有用。
- __end_index__: 在 `end_index` 之后的数据点在输出序列中将不被使用。这对保留部分数据以进行测试或验证很有用。
- __shuffle__: 是否打乱输出样本，还是按照时间顺序绘制它们。
- __reverse__: 布尔值: 如果 `true`, 每个输出样本中的时间步将按照时间倒序排列。
- __batch_size__: 每个批次中的时间序列样本数（可能除最后一个外）。

__返回__

一个 [Sequence](https://keras.io/zh/utils/#sequence) 实例。

__例子__

```python
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])

data_gen = TimeseriesGenerator(data, targets,
                               length=10, sampling_rate=2,
                               batch_size=2)
assert len(data_gen) == 20

batch_0 = data_gen[0]
x, y = batch_0
assert np.array_equal(x,
                      np.array([[[0], [2], [4], [6], [8]],
                                [[1], [3], [5], [7], [9]]]))
assert np.array_equal(y,
                      np.array([[10], [11]]))
```

---

### pad_sequences


```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.0)
```

将多个序列截断或补齐为相同长度。

该函数将一个 `num_samples` 的序列（整数列表）转化为一个 2D Numpy 矩阵，其尺寸为 `(num_samples, num_timesteps)`。 `num_timesteps` 要么是给定的 `maxlen` 参数，要么是最长序列的长度。

比 `num_timesteps` 短的序列将在末端以 `value` 值补齐。

比 `num_timesteps` 长的序列将会被截断以满足所需要的长度。补齐或截断发生的位置分别由参数 `pading` 和 `truncating` 决定。

向前补齐为默认操作。

__参数__

- __sequences__: 列表的列表，每一个元素是一个序列。
- __maxlen__: 整数，所有序列的最大长度。
- __dtype__: 输出序列的类型。
要使用可变长度字符串填充序列，可以使用 `object`。
- __padding__: 字符串，'pre' 或 'post' ，在序列的前端补齐还是在后端补齐。
- __truncating__: 字符串，'pre' 或 'post' ，移除长度大于 `maxlen` 的序列的值，要么在序列前端截断，要么在后端。
- __value__: 浮点数，表示用来补齐的值。


__返回__

- __x__: Numpy 矩阵，尺寸为 `(len(sequences), maxlen)`。

__异常__

- ValueError: 如果截断或补齐的值无效，或者序列条目的形状无效。

---

### skipgrams


```python
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size, window_size=4, negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None)
```

生成 skipgram 词对。

该函数将一个单词索引序列（整数列表）转化为以下形式的单词元组：

- （单词, 同窗口的单词），标签为 1（正样本）。
- （单词, 来自词汇表的随机单词），标签为 0（负样本）。

若要了解更多和 Skipgram 有关的知识，请参阅这份由 Mikolov 等人发表的经典论文： [Efficient Estimation of Word Representations in Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

__参数__

- __sequence__: 一个编码为单词索引（整数）列表的词序列（句子）。如果使用一个 `sampling_table`，词索引应该以一个相关数据集的词的排名匹配（例如，10 将会编码为第 10 个最长出现的词）。注意词汇表中的索引 0 是非单词，将被跳过。
- __vocabulary_size__: 整数，最大可能词索引 + 1
- __window_size__: 整数，采样窗口大小（技术上是半个窗口）。词 `w_i` 的窗口是 `[i - window_size, i + window_size+1]`。
- __negative_samples__: 大于等于 0 的浮点数。0 表示非负（即随机）采样。1 表示与正样本数相同。
- __shuffle__: 是否在返回之前将这些词语打乱。
- __categorical__: 布尔值。如果 False，标签将为整数（例如 `[0, 1, 1 .. ]`），如果 True，标签将为分类，例如 `[[1,0],[0,1],[0,1] .. ]`。
- __sampling_table__: 尺寸为 `vocabulary_size` 的 1D 数组，其中第 i 项编码了排名为 i 的词的采样概率。
- __seed__: 随机种子。
    
__返回__

couples, labels: 其中 `couples` 是整数对，`labels` 是 0 或 1。

__注意__

按照惯例，词汇表中的索引 0 是非单词，将被跳过。

---

### make_sampling_table


```python
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-05)
```


生成一个基于单词的概率采样表。

用来生成 `skipgrams` 的 `sampling_table` 参数。`sampling_table[i]` 是数据集中第 i 个最常见词的采样概率（出于平衡考虑，出现更频繁的词应该被更少地采样）。

采样概率根据 word2vec 中使用的采样分布生成：

```python
p(word) = (min(1, sqrt(word_frequency / sampling_factor) /
    (word_frequency / sampling_factor)))
```

我们假设单词频率遵循 Zipf 定律（s=1），来导出 frequency(rank) 的数值近似：

`frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))`，其中 `gamma` 为 Euler-Mascheroni 常量。

__参数__

- __size__: 整数，可能采样的单词数量。
- __sampling_factor__: word2vec 公式中的采样因子。

__返回__

一个长度为 `size` 大小的 1D Numpy 数组，其中第 i 项是排名为 i 的单词的采样概率。
