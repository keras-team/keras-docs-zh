
# 实现一个用来执行加法的序列到序列学习模型

输入: "535+61"

输出: "596"

使用重复的标记字符（空格）处理填充。

输入可以选择性地反转，它被认为可以提高许多任务的性能，例如：
[Learning to Execute](http://arxiv.org/abs/1410.4615)
以及
[Sequence to Sequence Learning with Neural Networks](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)。

从理论上讲，它引入了源和目标之间的短期依赖关系。

两个反转的数字 + 一个 LSTM 层（128个隐藏单元），在 55 个 epochs 后，5k 的训练样本取得了 99% 的训练/测试准确率。

三个反转的数字 + 一个 LSTM 层（128个隐藏单元），在 100 个 epochs 后，50k 的训练样本取得了 99% 的训练/测试准确率。

四个反转的数字 + 一个 LSTM 层（128个隐藏单元），在 20 个 epochs 后，400k 的训练样本取得了 99% 的训练/测试准确率。

五个反转的数字 + 一个 LSTM 层（128个隐藏单元），在 30 个 epochs 后，550k 的训练样本取得了 99% 的训练/测试准确率。


```python
from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range


class CharacterTable(object):
    """给定一组字符：
    + 将它们编码为 one-hot 整数表示
    + 将 one-hot 或整数表示解码为字符输出
    + 将一个概率向量解码为字符输出
    """
    def __init__(self, chars):
        """初始化字符表。

        # 参数：
            chars: 可以出现在输入中的字符。
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """给定字符串 C 的 one-hot 编码。

        # 参数
            C: 需要编码的字符串。
            num_rows: 返回的 one-hot 编码的行数。
                      这用来保证每个数据的行数相同。
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """将给定的向量或 2D array 解码为它们的字符输出。

        # 参数
            x: 一个向量或 2D 概率数组或 one-hot 表示，
               或 字符索引的向量（如果 `calc_argmax=False`）。
            calc_argmax: 是否根据最大概率来找到字符，默认为 `True`。
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# 模型和数据的参数
TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True

# 输入的最大长度是 'int+int' (例如, '345+678'). int 的最大长度为 DIGITS。
MAXLEN = DIGITS + 1 + DIGITS

# 所有的数字，加上符号，以及用于填充的空格。
chars = '0123456789+ '
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                    for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    # 跳过任何已有的加法问题
    # 同事跳过任何 x+Y == Y+x 的情况(即排序)。
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # 利用空格填充，是的长度始终为 MAXLEN。
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # 答案可能的最长长度为 DIGITS + 1。
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        # 反转查询，例如，'12+345  ' 变成 '  543+21'. 
        # (注意用于填充的空格)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

# 混洗 (x, y)，因为 x 的后半段几乎都是比较大的数字。
indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

# 显式地分离出 10% 的训练数据作为验证集。
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

# 可以尝试更改为 GRU, 或 SimpleRNN。
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

print('Build model...')
model = Sequential()
# 利用 RNN 将输入序列「编码」为一个 HIDDEN_SIZE 长度的输出向量。
# 注意：在输入序列具有可变长度的情况下,
# 使用 input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN, len(chars))))
# 作为解码器 RNN 的输入，为每个时间步重复地提供 RNN 的最后输出。
# 重复 'DIGITS + 1' 次，因为它是最大输出长度。
# 例如，当 DIGITS=3, 最大输出为 999+999=1998。
model.add(layers.RepeatVector(DIGITS + 1))
# 解码器 RNN 可以是多个堆叠的层，或一个单独的层。
for _ in range(LAYERS):
    # 通过设置 return_sequences 为 True, 将不仅返回最后一个输出，而是返回目前的所有输出，形式为(num_samples, timesteps, output_dim)。
    # 这是必须的，因为后面的 TimeDistributed 需要第一个维度是时间步。
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# 将全连接层应用于输入的每个时间片。
# 对于输出序列的每一步，决定应选哪个字符。
model.add(layers.TimeDistributed(layers.Dense(len(chars), activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# 训练模型，并在每一代显示验证数据的预测。
for iteration in range(1, 200):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=1,
              validation_data=(x_val, y_val))
    # 从随机验证集中选择 10 个样本，以便我们可以看到错误。
    for i in range(10):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
        preds = model.predict_classes(rowx, verbose=0)
        q = ctable.decode(rowx[0])
        correct = ctable.decode(rowy[0])
        guess = ctable.decode(preds[0], calc_argmax=False)
        print('Q', q[::-1] if REVERSE else q, end=' ')
        print('T', correct, end=' ')
        if correct == guess:
            print(colors.ok + '☑' + colors.close, end=' ')
        else:
            print(colors.fail + '☒' + colors.close, end=' ')
        print(guess)
```