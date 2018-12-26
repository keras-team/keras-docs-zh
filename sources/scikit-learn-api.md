# Scikit-Learn API 的封装器

你可以使用 Keras 的 `Sequential` 模型（仅限单一输入）作为 Scikit-Learn 工作流程的一部分，通过在此找到的包装器: `keras.wrappers.scikit_learn.py`。

有两个封装器可用:

`keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)`, 这实现了Scikit-Learn 分类器接口,

`keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)`, 这实现了Scikit-Learn 回归接口。

### 参数

- __build_fn__: 可调用函数或类实例
- __sk_params__: 模型参数和拟合参数

`build_fn` 应该建立，编译，并返回一个 Keras 模型，然后被用来训练/预测。以下三个值之一可以传递给`build_fn`

1. 一个函数；
2. 实现 `__call__` 方法的类的实例；
3. None。这意味着你实现了一个继承自 `KerasClassifier` 或 `KerasRegressor` 的类。当前类 `__call__` 方法将被视为默认的 `build_fn`。

`sk_params` 同时包含模型参数和拟合参数。合法的模型参数是 `build_fn` 的参数。请注意，与 scikit-learn 中的所有其他估算器一样，`build_fn` 应为其参数提供默认值，
以便你可以创建估算器而不将任何值传递给 `sk_params`。

`sk_params` 还可以接受用于调用 `fit`，`predict`，`predict_proba` 和 `score` 方法的参数（例如，`epochs`，`batch_size`）。训练（预测）参数按以下顺序选择：

1. 传递给 `fit`，`predict`，`predict_proba` 和 `score` 函数的字典参数的值；
2. 传递给 `sk_params` 的值；
3. `keras.models.Sequential` 的 `fit`，`predict`，`predict_proba` 和 `score` 方法的默认值。

当使用 scikit-learn 的 `grid_search` API 时，合法可调参数是你可以传递给 `sk_params` 的参数，包括训练参数。换句话说，你可以使用 `grid_search` 来搜索最佳的 `batch_size` 或 `epoch` 以及其他模型参数。
