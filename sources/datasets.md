# 数据集

## CIFAR10 小图像分类

数据集五万个32x32彩色训练图像, 超过十个类别的标签，以及万个测试图像。

### 用法:

```python
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

- __返回:__
    - 两个元组:
        - __x_train, x_test__: uint8 具有形状的RGB图像数组（num_samples，3，32，32）。
        - __y_train, y_test__: uint8 类别标签（范围0-9中的整数）与形状（num_samples，）的数组。


---

## CIFAR100 小图像分类

数据集五万个32x32彩色训练图像, 超过一百个类别的标签，以及万个测试图像。

### 用法:

```python
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
```

- __返回:__
    - 两个元组:
        - __x_train, x_test__: uint8 具有形状的RGB图像数组 (num_samples, 3, 32, 32)。
        - __y_train, y_test__: uint8 具有形状（num_samples）的类别标签数组。

- __参数:__

    - __label_mode__: `"fine"` 或 `"coarse"`。


---

## IMDB 电影评论情感分类

数据集来自IMDB的二十五千个电影评论，标志着情绪（正面/负面）。 注释已被预处理，每个注释被编码为单词索引（整数) 的[序列](preprocessing/sequence.md)。为了方便起见，单词按数据集中的整体频率进行索引, 例如, 整数“3”被编码为数据中第三最频繁的词。 这允许快速过滤操作，例如: "只考虑前万个最常用的单词，但是排除前二十个最常用的单词"。

按照惯例，“0”不代表一个特定的词，而是编码任何未知的词。

### 用法:

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
- __返回:__
    - 两个元组:
        - __x_train, x_test__: 序列列表，它是索引列表（整数）。如果`num_words`参数是特定的，则可能的最大索引值是`num_words-1`。 如果指定了`maxlen`参数，则可能的最大序列长度是`maxlen`。
        - __y_train, y_test__: 整型标签列表（1或0）。 

- __参数:__

    - __path__: 如果你本地没有数据（在〜/ .keras / datasets /'+ path`），它会被下载到这个位置。
    - __num_words__: 整数或`None`。考虑最常用的词汇。任何不太频繁的单词将在序列数据中显示为`oov_char`值。
    - __skip_top__: 整数。 最常被忽略的单词（它们将在序列数据中显示为`oov_char`值）。
    - __maxlen__: 整数。 最大序列长度。任何更长的序列将被截断。
    - __seed__: 整数。 用于可数据的可重复洗牌。
    - __start_char__: 整数。 一个序列的开始将被标记为这个字符。
        设为1，因为0通常是填充字符。
    - __oov_char__: 整数。 由于`num_words`而被删除的单词
        或 `skip_top`的限制将被替换为这个字符。
    - __index_from__: 整数。 使用此索引或更高的索引实际的单词。


---

## Reuters newswire话题分类

来自路透社的一万二百二十八十个条newswires数据集, 标有超过四十六个话题。 与IMDB数据集一样，每条线都被编码为一系列字索引（相同的约定）。

### 用法:

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

规格与IMDB数据集的规格相同，但增加了：

- __test_split__: 浮点。 要用作测试数据的数据集的子集。

数据集还提供了用于编码序列的字索引：

```python
word_index = reuters.get_word_index(path="reuters_word_index.json")
```

- __返回:__ 一个字典，其中的单词是键（str）和值是索引（整数） 例如 `word_index [“giraffe”]`可能会返回`1234`。 

- __参数:__

    - __path__: 如果你本地没有数据（在〜/ .keras / datasets /'+ path`），它会被下载到这个位置。
    

---

## MNIST 一个手写数字的数据库

数据集包含10个数字的六万个28x28灰度图像，以及万个图像的测试集。

### 用法:

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

- __返回:__
    - 两个元组:
        - __x_train, x_test__: uint8 具有形状的grayscale图像数据阵列（num_samples，28,28）。
        - __y_train, y_test__: uint8 数字标签（整数在0-9范围内）与形状（num_samples，）的数组。

- __参数:__

    - __path__: 如果你在本地没有索引文件（在'〜/ .keras / datasets /'+ path`），它会被下载到这个位置。


---

## Fashion-MNIST 时尚物品数据库

该数据集包含十个时尚类别的六万个28x28灰度图像，以及万个图像的测试集。这个数据集可以用作MNIST的直接替换。 类标签是：

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### 用法:

```python
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

- __Returns:__
    - 2 tuples:
        - __x_train, x_test__: uint8 具有形状的grayscale图像数据阵列（num_samples，28,28）。
        - __y_train, y_test__: uint8 标签数组（整数范围0-9）与形状（num_samples，）。


---

## 波士顿房屋价格回归数据集


数据集取自卡耐基梅隆大学（Carnegie Mellon University）维护的StatLib图书馆

该样本在七十年代后期在波士顿郊区的不同地点包括了十三个房屋属性。
目标是一个地点房屋的中位值（单位：k $）。


### 用法:

```python
from keras.datasets import boston_housing

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
```

- __参数:__
    - __path__: 如果你在本地没有索引文件（在'〜/ .keras / datasets /'+ path`），它会被下载到这个位置。
    - __seed__: 随机种子在计算测试分割之前对数据进行混洗。
    - __test_split__: 将数据的一小部分保留为测试集。

- __返回:__
    Numpy数组的元组：`（x_train，y_train），（x_test，y_test）`。
