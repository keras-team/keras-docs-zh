# Scikit-Learn API的封装器

你可以使用Keras的顺序模型(仅限单一输入)作为Scikit-Learn工作流程的一部分，通过在此找到的包装器:
`keras.wrappers.scikit_learn.py`.

有两个封装器可用:

`keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)`, 这实现了Scikit-Learn分类器接口,

`keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)`, 这实现了Scikit-Learn regressor界面。

### 参数

- __build_fn__: 可调用函数或类实例
- __sk_params__: 模型参数和拟合参数

`build_fn` 这应该建立，编译，并返回一个Keras模型，将被用来拟合/预测。 以下之一三个值可以传递给`build_fn`

1. 函数
2. 实现`__call__`函数的类的实例
3. 没有。这意味着你实现了一个继承的类KerasClassifier或KerasRegressor。 当前类的__call__函数将被视为默认的`build_fn`。

`sk_params`同时包含模型参数和拟合参数。 法律模型参数是`build_fn`的参数。 类似于其他所有
估计者在Scikit-Learn, `build_fn`应该为其参数提供默认值，这样就可以创建估计器，而不需要将任何值传递给`sk_params`。

`sk_params`也可以被称为`fit`，`predict`，
`predict_proba`和`score`函数 (e.g., `epochs`, `batch_size`).
拟合（预测）参数按以下顺序选择:

1. 传递给`fit`，`predict`，`predict_proba`和`score`函数的字典参数的值
2. 传递给`sk_params`的值
3. `keras.models.Sequential`的默认值
`fit`，`predict`，`predict_proba`和`score`函数

当scikit-learn使用`grid_search` API时，有效参数与`sk_params`相同，包括拟合参数。
换句话说，你可以使用`grid_search`来搜索最好的`batch_size`或`epochs`以及模型参数。
