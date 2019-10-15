
# 还原字符级序列到序列模型以生成预测。

该脚本载入由 [lstm_seq2seq.py](/examples/lstm_seq2seq/) 保存的 ```s2s.h5``` 模型，并从中生成序列。
它假设其未作任何改变（例如，```latent_dim```、输入数据和模型结构均不变）。

有关模型结构细节以及如何训练，参见 [lstm_seq2seq.py](/examples/lstm_seq2seq/)。


```python
from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

batch_size = 64  # 训练批次大小。
epochs = 100  # 训练轮次数。
latent_dim = 256  # 编码空间隐层维度。
num_samples = 10000  # 训练样本数。
# 磁盘中数据文件路径。
data_path = 'fra-eng/fra.txt'

# 向量化数据。使用与训练脚本相同的方法。
# 注意: 数据必须相同，以使字符->整数映射保持一致。
# 我们省略对 target_texts 的编码，因为不需要它们。
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # 我们使用 "tab" 作为目标的 "开始序列" 字符，并使用 "\n" 作为 "结束序列" 字符。
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

for i, input_text in enumerate(input_texts):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.

# 恢复模型并构造编码器和解码器。
model = load_model('s2s.h5')

encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# 反向查询 token 索引可将序列解码回可读的内容。
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


# 解码输入序列。未来的工作应支持波束搜索。
def decode_sequence(input_seq):
    # 将输入编码为状态向量。
    states_value = encoder_model.predict(input_seq)

    # 生成长度为 1 的空目标序列。
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # 用起始字符填充目标序列的第一个字符。
    target_seq[0, 0, target_token_index['\t']] = 1.

    # 一批序列的采样循环
    # (为了简化，这里我们假设一批大小为 1)。
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # 采样一个 token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 退出条件：达到最大长度或找到停止符。
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # 更新目标序列（长度为 1）。
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # 更新状态
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # 抽取一个序列（训练集的一部分）进行解码。
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
```