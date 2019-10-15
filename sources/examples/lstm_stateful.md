
# 如何使用有状态 LSTM 模型，有状态与无状态 LSTM 性能比较

[有关 Keras LSTM 模型的更多文档](/layers/recurrent/#lstm)

在输入/输出对上训练模型，其中输入是生成的长度为 `input_len` 的均匀分布随机序列，
输出是窗口长度为 `tsteps` 的输入的移动平均值。`input_len` 和 `tsteps` 都在 "可编辑参数" 部分中定义。

较大的 `tsteps` 值意味着 LSTM 需要更多的内存来确定输入输出关系。
该内存长度由 `lahead` 变量控制（下面有更多详细信息）。

其余参数为：

- `input_len`: 生成的输入序列的长度
- `lahead`: LSTM 针对每个输出点训练的输入序列长度
- `batch_size`, `epochs`: 与 `model.fit(...)` 函数中的参数相同

当 `lahead > 1` 时，模型输入将预处理为数据的 "滚动窗口视图"，窗口长度为 `lahead`。
这类似于 sklearn 的 `view_as_windows`，
其中 `window_shape` [是一个数字。](http://scikit-image.org/docs/0.10.x/api/skimage.util.html#view-as-windows)

当 `lahead  < tsteps` 时，只有有状态的 LSTM 会收敛，因为它的有状态性使其能够看到超出 lahead 赋予其的 n 点平均值的能力。
无状态 LSTM 不具备此功能，因此受到其 `lahead` 参数的限制，该参数不足以查看 n 点平均值。

当 `lahead >= tsteps` 时，有状态和无状态 LSTM 都会收敛。


```python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM

# ----------------------------------------------------------
# 可编辑参数
# 阅读脚本头中的文档以获取更多详细信息
# ----------------------------------------------------------

# 输入长度
input_len = 1000

# 用于从训练 LSTM 的输入/输出对中的输入生成输出的移动平均值的窗口长度
# 例如，如果 tsteps=2，input=[1, 2, 3, 4, 5],
#      那么 output=[1.5, 2.5, 3.5, 4.5]
tsteps = 2

# LSTM 针对每个输出点训练的输入序列长度
lahead = 1

# 传递给 "model.fit(...)" 的训练参数
batch_size = 1
epochs = 10

# ------------
# 主程序
# ------------

print("*" * 33)
if lahead >= tsteps:
    print("STATELESS LSTM WILL ALSO CONVERGE")
else:
    print("STATELESS LSTM WILL NOT CONVERGE")
print("*" * 33)

np.random.seed(1986)

print('Generating Data...')


def gen_uniform_amp(amp=1, xn=10000):
    """生成 -amp 和 +amp 之间且长度为 xn 的均匀随机数据

    # 参数
        amp: 统一数据的最大/最小范围
        xn: 系列长度
    """
    data_input = np.random.uniform(-1 * amp, +1 * amp, xn)
    data_input = pd.DataFrame(data_input)
    return data_input

# 由于输出是输入的移动平均值，
# 因此输出的前几个点将是 NaN，
# 并且在训练 LSTM 之前将其从生成的数据中删除。
# 同样，当 lahead > 1时，"滚动窗口视图" 
# 后面的预处理步骤也将导致某些点丢失。
# 出于美学原因，为了在预处理后保持生成的数据长度等于 input_len，请添加一些点以说明将丢失的值。
to_drop = max(tsteps - 1, lahead - 1)
data_input = gen_uniform_amp(amp=0.1, xn=input_len + to_drop)

# 将目标设置为输入的 N 点平均值
expected_output = data_input.rolling(window=tsteps, center=False).mean()

# 当 lahead > 1时，需要将输入转换为 "滚动窗口视图"
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
if lahead > 1:
    data_input = np.repeat(data_input.values, repeats=lahead, axis=1)
    data_input = pd.DataFrame(data_input)
    for i, c in enumerate(data_input.columns):
        data_input[c] = data_input[c].shift(i)

# 丢弃 nan
expected_output = expected_output[to_drop:]
data_input = data_input[to_drop:]

print('Input shape:', data_input.shape)
print('Output shape:', expected_output.shape)
print('Input head: ')
print(data_input.head())
print('Output head: ')
print(expected_output.head())
print('Input tail: ')
print(data_input.tail())
print('Output tail: ')
print(expected_output.tail())

print('Plotting input and expected output')
plt.plot(data_input[0][:10], '.')
plt.plot(expected_output[0][:10], '-')
plt.legend(['Input', 'Expected output'])
plt.title('Input')
plt.show()


def create_model(stateful):
    model = Sequential()
    model.add(LSTM(20,
              input_shape=(lahead, 1),
              batch_size=batch_size,
              stateful=stateful))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

print('Creating Stateful Model...')
model_stateful = create_model(stateful=True)


# 切分训练/测试数据
def split_data(x, y, ratio=0.8):
    to_train = int(input_len * ratio)
    # 进行调整以匹配 batch_size
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]

    # 进行调整以匹配 batch_size
    to_drop = x.shape[0] % batch_size
    if to_drop > 0:
        x_test = x_test[:-1 * to_drop]
        y_test = y_test[:-1 * to_drop]

    # 一些重塑
    reshape_3 = lambda x: x.values.reshape((x.shape[0], x.shape[1], 1))
    x_train = reshape_3(x_train)
    x_test = reshape_3(x_test)

    reshape_2 = lambda x: x.values.reshape((x.shape[0], 1))
    y_train = reshape_2(y_train)
    y_test = reshape_2(y_test)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)

print('Training')
for i in range(epochs):
    print('Epoch', i + 1, '/', epochs)
    # 请注意，批次 i 中样品 i 的最后状态将用作下一批中样品 i 的初始状态。
    # 因此，我们同时以低于 data_input 中包含的原始序列的分辨率对 batch_size 系列进行训练。
    # 这些系列中的每一个都偏移一个步骤，并且可以使用 data_input[i::batch_size] 提取。
    model_stateful.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=1,
                       verbose=1,
                       validation_data=(x_test, y_test),
                       shuffle=False)
    model_stateful.reset_states()

print('Predicting')
predicted_stateful = model_stateful.predict(x_test, batch_size=batch_size)

print('Creating Stateless Model...')
model_stateless = create_model(stateful=False)

print('Training')
model_stateless.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    shuffle=False)

print('Predicting')
predicted_stateless = model_stateless.predict(x_test, batch_size=batch_size)

# ----------------------------

print('Plotting Results')
plt.subplot(3, 1, 1)
plt.plot(y_test)
plt.title('Expected')
plt.subplot(3, 1, 2)
# 删除第一个 "tsteps-1"，因为不可能预测它们，因为不存在要使用的 "上一个" 时间步
plt.plot((y_test - predicted_stateful).flatten()[tsteps - 1:])
plt.title('Stateful: Expected - Predicted')
plt.subplot(3, 1, 3)
plt.plot((y_test - predicted_stateless).flatten())
plt.title('Stateless: Expected - Predicted')
plt.show()
```