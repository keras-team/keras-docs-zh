# 数据集

## CIFAR10 小图像分类数据集

50,000 张 32x32 彩色训练图像数据，以及 10,000 张测试图像数据，总共分为 10 个类别。

### 用法：

```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

- __返回：__
  - 2 个元组：
    - __x_train, x_test__: uint8 数组表示的 RGB 图像数据，尺寸为 (num_samples, 3, 32, 32) 或 (num_samples, 32, 32, 3)，基于 `image_data_format` 后端设定的 `channels_first` 或 `channels_last`。
    - __y_train, y_test__: uint8 数组表示的类别标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)。


---

## CIFAR100 小图像分类数据集

50,000 张 32x32 彩色训练图像数据，以及 10,000 张测试图像数据，总共分为 100 个类别。

### 用法：

```python
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
```

- __返回：__
    - 2 个元组：
        - __x_train, x_test__: uint8 数组表示的 RGB 图像数据，尺寸为 (num_samples, 3, 32, 32) 或 (num_samples, 32, 32, 3)，基于 `image_data_format` 后端设定的 `channels_first` 或 `channels_last`。
        - __y_train, y_test__: uint8 数组表示的类别标签，尺寸为 (num_samples,)。

- __参数：__
    - __label_mode__: "fine" 或者 "coarse"


---

## IMDB 电影评论情感分类数据集

数据集来自 IMDB 的 25,000 条电影评论，以情绪（正面/负面）标记。评论已经过预处理，并编码为词索引（整数）的[序列](preprocessing/sequence.md)表示。为了方便起见，将词按数据集中出现的频率进行索引，例如整数 3 编码数据中第三个最频繁的词。这允许快速筛选操作，例如：「只考虑前 10,000 个最常用的词，但排除前 20 个最常见的词」。

作为惯例，0 不代表特定的单词，而是被用于编码任何未知单词。

### 用法

```python
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
```

- __返回：__
    - 2 个元组：
      - __x_train, x_test__: 序列的列表，即词索引的列表。如果指定了 `num_words` 参数，则可能的最大索引值是 `num_words-1`。如果指定了 `maxlen` 参数，则可能的最大序列长度为 `maxlen`。 
      - __y_train, y_test__: 整数标签列表 (1 或 0)。

- __参数:__
    - __path__: 如果你本地没有该数据集 (在 `'~/.keras/datasets/' + path`)，它将被下载到此目录。
    - __num_words__: 整数或 None。要考虑的最常用的词语。任何不太频繁的词将在序列数据中显示为 `oov_char` 值。
    - __skip_top__: 整数。要忽略的最常见的单词（它们将在序列数据中显示为 `oov_char` 值）。
    - __maxlen__: 整数。最大序列长度。 任何更长的序列都将被截断。
    - __seed__: 整数。用于可重现数据混洗的种子。
    - __start_char__: 整数。序列的开始将用这个字符标记。设置为 1，因为 0 通常作为填充字符。
    - __oov_char__: 整数。由于 `num_words` 或 `skip_top` 限制而被删除的单词将被替换为此字符。
    - __index_from__: 整数。使用此数以上更高的索引值实际词汇索引的开始。


---

## 路透社新闻主题分类

数据集来源于路透社的 11,228 条新闻文本，总共分为 46 个主题。与 IMDB 数据集一样，每条新闻都被编码为一个词索引的序列（相同的约定）。

### 用法：

```python
from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(path="reuters.npz",
                                                         num_words=None,
                                                         skip_top=0,
                                                         maxlen=None,
                                                         test_split=0.2,
                                                         seed=113,
                                                         start_char=1,
                                                         oov_char=2,
                                                         index_from=3)
```

规格与 IMDB 数据集的规格相同，但增加了：

- __test_split__: 浮点型。用作测试集的数据比例。

该数据集还提供了用于编码序列的词索引：

```python
word_index = reuters.get_word_index(path="reuters_word_index.json")
```

- __返回：__ 一个字典，其中键是单词（字符串），值是索引（整数）。 例如，`word_index["giraffe"]` 可能会返回 `1234`。

- __参数：__
    - __path__: 如果在本地没有索引文件 (at `'~/.keras/datasets/' + path`), 它将被下载到该目录。

---

## MNIST 手写字符数据集

训练集为 60,000 张 28x28 像素灰度图像，测试集为 10,000 同规格图像，总共 10 类数字标签。

### 用法：

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

- __返回：__
    - 2 个元组：
        - __x_train, x_test__: uint8 数组表示的灰度图像，尺寸为 (num_samples, 28, 28)。
        - __y_train, y_test__: uint8 数组表示的数字标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)。

- __参数：__
    - __path__: 如果在本地没有索引文件 (at `'~/.keras/datasets/' + path`), 它将被下载到该目录。


---

## Fashion-MNIST 时尚物品数据集

训练集为 60,000 张 28x28 像素灰度图像，测试集为 10,000 同规格图像，总共 10 类时尚物品标签。该数据集可以用作 MNIST 的直接替代品。类别标签是：

| 类别 | 描述 | 中文 |
| --- | --- | --- |
| 0 | T-shirt/top | T恤/上衣 |
| 1 | Trouser | 裤子 |
| 2 | Pullover | 套头衫 |
| 3 | Dress | 连衣裙 |
| 4 | Coat | 外套 |
| 5 | Sandal | 凉鞋 |
| 6 | Shirt | 衬衫 |
| 7 | Sneaker | 运动鞋 |
| 8 | Bag | 背包 |
| 9 | Ankle boot | 短靴 |

### 用法：

```python
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

- __返回：__
    - 2 个元组：
        - __x_train, x_test__: uint8 数组表示的灰度图像，尺寸为 (num_samples, 28, 28)。
        - __y_train, y_test__: uint8 数组表示的数字标签（范围在 0-9 之间的整数），尺寸为 (num_samples,)。


---

## Boston 房价回归数据集


数据集来自卡内基梅隆大学维护的 StatLib 库。

样本包含 1970 年代的在波士顿郊区不同位置的房屋信息，总共有 13 种房屋属性。
目标值是一个位置的房屋的中值（单位：k$）。


### 用法：

```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```

- __参数：__
    - __path__: 缓存本地数据集的位置
    (相对路径 ~/.keras/datasets)。
    - __seed__: 在计算测试分割之前对数据进行混洗的随机种子。
    - __test_split__: 需要保留作为测试数据的比例。

- __返回：__
  Numpy 数组的元组: `(x_train, y_train), (x_test, y_test)`。
    
