
## 模型可视化

`keras.utils.vis_utils`模块提供了一些绘制Keras模型的实用功能(使用`graphviz`)。

以下实例，将绘制一张模型图，并保存为文件：
```python
from keras.utils import plot_model
plot_model(model, to_file='model.png')
```

`plot_model`有两个可选参数:

- `show_shapes` (默认为False) 控制是否在图中输出各层的shape。
- `show_layer_names` (默认为True) 控制是否在图中显示每一层的名字。

此外，你也可以直接取得`pydot.Graph`对象并自己渲染它。
ipython notebook中的可视化实例如下：
```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
```
