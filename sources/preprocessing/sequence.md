## pad_sequences

```python
keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32',
    padding='pre', truncating='pre', value=0.)
```

将一个 `num_samples` 的序列（标量的列表）转化为一个二维 Numpy 矩阵，其尺寸为 `(num_samples, num_timesteps)` 。 `num_timesteps` 为 `maxlen`参数，当未提供该参数时，则取最长的序列的长度。比 `num_timesteps`短的序列会被在一端用 `value` 补齐。比 `num_timesteps` 长的序列会被截断使其符合所需要的长度。发生「补齐」和「截断」的位置分别由 `padding` 和 `truncating` 决定。  

- __返回__: 二维 Numpy 矩阵，尺寸为 `(num_samples, num_timesteps)` 。

- __参数__:
    - __sequences__: 一个或多个整数或浮点数列表。
    - __maxlen__: None或者整数。表示最大的序列长度，超过此长度的序列会被截断，短于此长度的序列会被在一端用 `value` 补齐。
    - __dtype__: 返回的 Numpy 矩阵的数据类型。
    - __padding__: 'pre' 或 'post' ，表示长度不足时是在序列的前端补齐还是在后端补齐。
    - __truncating__: 'pre' 或 'post' ，表示长度超出时是在序列前端截断还是在后端截断。
    - __value__: 浮点数，表示用来补齐的值。

---

## skipgrams

```python
keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size,
    window_size=4, negative_samples=1., shuffle=True,
    categorical=False, sampling_table=None)
```

将一个单词索引序列（整数列表）转化为以下形式的组：

- (单词, 同窗口的单词)，标签为1（正样本）。
- (单词, 来自词汇表的随机单词)，标签为0（负样本）。

若要了解更多和 Skipgram 有关的知识，请参阅这份由 Mikolov 等人发表的经典论文： [Efficient Estimation of Word Representations in
Vector Space](http://arxiv.org/pdf/1301.3781v3.pdf)

- __返回__: 元组 `(couples, labels)`.
    - `couples` 是一个元素为两个整型数对的列表： `[word_index, other_word_index]` 。
    - `labels` 是一个由 0 和 1 组成的列表，1 表示 `other_word_index` 与 `word_index` 在同一个窗口中，0 表示 `other_word_index` 是从词汇表中随机选取的。
    - 如果 `categorical` 被设为 `True`，标签为分类标签，即 1 变为 [0, 1] 而 0 变为 [1, 0] 。

- __参数__:
    - __sequence__: 整数索引列表。如果使用一个 sampling_table ，一个词的索引即应为其在数据集中的行号（从 1 开始）。 
    - __vocabulary_size__: 词汇表尺寸，整数。
    - __window_size__: 整数。表示在一个正样本组中两个词的最大距离。
    - __negative_samples__: 大于等于 0 的浮点数，表示负样本（即随机样本）相对于正样本的比例。例如，1 表示生成与正样本一样多的负样本。
    - __shuffle__: 布尔值。表示是否混洗数据。
    - __categorical__: 布尔值。表示是否要将返回的标签设为「分类标签」。 
    - __sampling_table__: 样本表，Numpy 矩阵，尺寸为 `(vocabulary_size,)` ，这里 `sampling_table[i]` 是索引为 i 的词的出现概率（i 应该是数据集中所出现频率第 i 高的词）。

---

## make_sampling_table

```python
keras.preprocessing.sequence.make_sampling_table(size, sampling_factor=1e-5)
```

用来生成 `skipgrams` 的 `sampling_table` 参数。 `sampling_table[i]` 是数据集中出现第 i 多的词的出现概率（出于平衡考虑，出现更频繁的词应该被更少地采样）。

- __返回__: Numpy 矩阵，尺寸为 `(size,)` 。

- __参数__:
    - __size__: 词汇表的大小。
    - __sampling_factor__: 更低的值会产生更高的「概率衰减」（即频繁出现的词会更少被采样）。如果将该参数设为 1 ，则不会进行二次采样（所有的采样率均为 1）。
