
## text_to_word_sequence

```python
keras.preprocessing.text.text_to_word_sequence(text,
                                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                               lower=True,
                                               split=" ")
```

将一个句子划分为词的列表。

- __返回__: 词的列表（字符串）。

- __参数__：
  - __text__: 字符串。
  - __filters__: 需要过滤掉的字符列表（或连接）。
  默认：<code>!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n</code>，
  包含基本标点符号、制表符、换行符。
  - __lower__: 布尔值。是否将文本转换为小写。
  - __split__: 字符串。词的分隔符。

## one_hot

```python
keras.preprocessing.text.one_hot(text,
                                 n,
                                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                 lower=True,
                                 split=" ")
```

One-hot 将文本编码为大小为 n 的词汇表中的词索引列表。

这是使用 `hash` 作为散列函数的 `hashing_trick` 函数的封装器。

- __返回__: 整数列表 [1, n]。每个整数编码一个词（唯一性无法保证）。

- __参数__:
  - __text__: 字符串
  - __n__: 整数。词汇表大小。
  - __filters__: 需要过滤掉的字符列表（或连接）。
  默认：<code>!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n</code>，
  包含基本标点符号、制表符、换行符。
  - __lower__: 布尔值。是否将文本转换为小写。
  - __split__: 字符串。词的分隔符。
    
## hashing_trick

```python
keras.preprocessing.text.hashing_trick(text, 
                                       n,
                                       hash_function=None,
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                       lower=True,
                                       split=' ')
```

将文本转换为固定大小散列空间中的索引序列。

- __返回__: 词索引的列表（唯一性无法保证）。
        
- __参数__:
  - __text__: 字符串。
  - __n__: 散列空间的维度。
  - __hash_function__:默认为 Python `hash` 函数，
  可以是 'md5' 或任何接受输入字符串并返回 int 的函数。
  注意 `hash` 是一个不稳定的散列函数，
  因而它在不同的运行环境下是不一致的，
  而 `md5` 是一个稳定的散列函数。
  - __filters__: 需要过滤掉的字符列表（或连接）。
  默认：<code>!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n</code>，
  包含基本标点符号、制表符、换行符。
  - __lower__: 布尔值。是否将文本转换为小写。
  - __split__: 字符串。词的分隔符。

## Tokenizer

```python
keras.preprocessing.text.Tokenizer(num_words=None,
                                   filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,
                                   split=" ",
                                   char_level=False)
```

将文本向量化的类，或/且 将文本转化为序列（词索引的列表，其中在数据集中的第 i 个首次出现的单词索引为 i，从 1 开始）。

- __参数__: 与上面的 `text_to_word_sequence` 相同。
  - __num_words__: None 或 整型。 要使用的最大词数 （如果设置，标记化过程将会局限在数据集中最常出现的词中）。
  - __char_level__: 如果 True，每一个字符都被作为一个标记。

- __方法__:
  - __fit_on_texts(texts)__: 
    - __参数__:
      - __texts__: 需要训练的文本列表。

  - __texts_to_sequences(texts)__
     - __参数__: 
       - __texts__: 需要转换为序列的文本列表。
     - __返回__: 序列的列表（每个文本输入一个序列）。

  - __texts_to_sequences_generator(texts)__: 以上方法的生成器版本。
    - __返回__: 每一次文本输入返回一个序列。

  - __texts_to_matrix(texts)__:
    - __返回__: numpy array of shape `(len(texts), num_words)`.
    - __参数__:
      - __texts__: 需要向量化的文本列表。
      - __mode__: "binary", "count", "tfidf", "freq" 之一 (默认: "binary")。

  - __fit_on_sequences(sequences)__: 
    - __参数__:
      - __sequences__: 需要训练的文本列表。

  - __sequences_to_matrix(sequences)__:
    - __返回__: 尺寸为 `(len(sequences), num_words)` 的 numpy 数组。
    - __参数__:
      - __sequences__: 需要向量化的序列列表。
      - __mode__: "binary", "count", "tfidf", "freq" 之一 (默认: "binary")。

- __属性__:
  - __word_counts__: 在训练时将词（字符串）映射到其出现次数的字典。只在调用 `fit_on_text` 后才被设置。
  - __word_docs__: 在训练时将词（字符串）映射到其出现的文档/文本数的字典。只在调用 `fit_on_text` 后才被设置。
  - __word_index__: 将词（字符串）映射到索引（整型）的字典。只在调用 `fit_on_text` 后才被设置。
  - __document_count__: 整型。标志器训练的文档（文本/序列）数量。只在调用 `fit_on_text` 或 `fit_on_sequences` 后才被设置。


