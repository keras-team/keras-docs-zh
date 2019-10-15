# 用 Keras 实现字符级序列到序列模型。

该脚本演示了如何实现基本的字符级 CNN 序列到序列模型。
我们将其用于将英文短句逐个字符翻译成法语短句。
请注意，进行字符级机器翻译是非比寻常的，因为在此领域中词级模型更为常见。
本示例仅用于演示目的。

**算法总结**

- 我们从一个领域的输入序列（例如英语句子）和另一个领域的对应目标序列（例如法语句子）开始。
- 编码器 CNN 对输入字符序列进行编码。
- 对解码器 CNN 进行训练，以将目标序列转换为相同序列，但以后将偏移一个时间步，在这种情况下，该训练过程称为 "教师强制"。它使用编码器的输出。实际上，解码器会根据输入序列，根据给定的 `targets[...t]` 来学习生成 `target[t+1...]`。 
- 在推理模式下，当我们想解码未知的输入序列时，我们：
    - 对输入序列进行编码；
    - 从大小为1的目标序列开始（仅是序列开始字符）；
    - 将输入序列和 1 个字符的目标序列馈送到解码器，以生成下一个字符的预测；
    - 使用这些预测来采样下一个字符（我们仅使用 argmax）;
    - 将采样的字符附加到目标序列；
    - 重复直到我们达到字符数限制。

**数据下载**

[English to French sentence pairs.](http://www.manythings.org/anki/fra-eng.zip)

[Lots of neat sentence pairs datasets.](http://www.manythings.org/anki/)

**参考**

- lstm_seq2seq.py
- https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html


```python
from __future__ import print_function
import numpy as np
from keras.layers import Input, Convolution1D, Dot, Dense, Activation, Concatenate
from keras.models import Model
batch_size = 64  # 训练批次大小。
epochs = 100  # 训练迭代轮次。
num_samples = 10000  # 训练样本数。
# 磁盘数据文件路径
data_path = 'fra-eng/fra.txt'
# 向量化数据。
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # 我们使用 "tab" 作为 "起始序列" 字符，
    # 对于目标，使用 "\n" 作为 "终止序列" 字符。
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
# 定义输入序列并处理它。
encoder_inputs = Input(shape=(None, num_encoder_tokens))
# Encoder
x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal')(encoder_inputs)
x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=2)(x_encoder)
x_encoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=4)(x_encoder)
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# Decoder
x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal')(decoder_inputs)
x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=2)(x_decoder)
x_decoder = Convolution1D(256, kernel_size=3, activation='relu',
                          padding='causal', dilation_rate=4)(x_decoder)
# Attention
attention = Dot(axes=[2, 2])([x_decoder, x_encoder])
attention = Activation('softmax')(attention)
context = Dot(axes=[2, 1])([attention, x_encoder])
decoder_combined_context = Concatenate(axis=-1)([context, x_decoder])
decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu',
                                padding='causal')(decoder_combined_context)
decoder_outputs = Convolution1D(64, kernel_size=3, activation='relu',
                                padding='causal')(decoder_outputs)
# 输出
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# 定义将 `encoder_input_data` & `decoder_input_data` 
# 转化为 `decoder_target_data`的模型。
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
# 执行训练
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# 保存模型
model.save('cnn_s2s.h5')
# 接下来: 推理模式 (采样)。
# 定义采样模型
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())
nb_examples = 100
in_encoder = encoder_input_data[:nb_examples]
in_decoder = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
in_decoder[:, 0, target_token_index["\t"]] = 1
predict = np.zeros(
    (len(input_texts), max_decoder_seq_length),
    dtype='float32')
for i in range(max_decoder_seq_length - 1):
    predict = model.predict([in_encoder, in_decoder])
    predict = predict.argmax(axis=-1)
    predict_ = predict[:, i].ravel().tolist()
    for j, x in enumerate(predict_):
        in_decoder[j, i + 1, x] = 1
for seq_index in range(nb_examples):
    # 抽取一个序列（训练集的一部分）进行解码。
    output_seq = predict[seq_index, :].ravel().tolist()
    decoded = []
    for x in output_seq:
        if reverse_target_char_index[x] == "\n":
            break
        else:
            decoded.append(reverse_target_char_index[x])
    decoded_sentence = "".join(decoded)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
```
