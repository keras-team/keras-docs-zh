
## 模型可视化

`keras.utils.vis_utils` 模块提供了一些绘制 Keras 模型的实用功能(使用 `graphviz`)。

以下实例，将绘制一张模型图，并保存为文件：
```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model` 有 4 个可选参数:

- `show_shapes` (默认为 False) 控制是否在图中输出各层的尺寸。
- `show_layer_names` (默认为 True) 控制是否在图中显示每一层的名字。
- `expand_dim`（默认为 False）控制是否将嵌套模型扩展为图形中的聚类。
- `dpi`（默认为 96）控制图像 dpi。

此外，你也可以直接取得 `pydot.Graph` 对象并自己渲染它。
例如，ipython notebook 中的可视化实例如下：

```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```

----

## 训练历史可视化

Keras `Model` 上的 `fit()` 方法返回一个 `History` 对象。`History.history` 属性是一个记录了连续迭代的训练/验证（如果存在）损失值和评估值的字典。这里是一个简单的使用 `matplotlib` 来生成训练/验证集的损失和准确率图表的例子：

```python
import matplotlib.pyplot as plt

history = model.fit(x, y, validation_split=0.25, epochs=50, batch_size=16, verbose=1)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
```
