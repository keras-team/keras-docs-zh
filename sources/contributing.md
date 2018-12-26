# 关于 Github Issues 和 Pull Requests

找到一个漏洞？有一个新的功能建议？想要对代码库做出贡献？请务必先阅读这些。

## 漏洞报告

你的代码不起作用，你确定问题在于Keras？请按照以下步骤报告错误。

1. 你的漏洞可能已经被修复了。确保更新到目前的Keras master分支，以及最新的 Theano/TensorFlow/CNTK master 分支。
轻松更新 Theano 的方法：`pip install git+git://github.com/Theano/Theano.git --upgrade`

2. 搜索相似问题。 确保在搜索已经解决的 Issue 时删除 `is:open` 标签。有可能已经有人遇到了这个漏洞。同时记得检查 Keras [FAQ](/faq/)。仍然有问题？在 Github 上开一个 Issue，让我们知道。

3. 确保你向我们提供了有关你的配置的有用信息：什么操作系统？什么 Keras 后端？你是否在 GPU 上运行，Cuda 和 cuDNN 的版本是多少？GPU型号是什么？

4. 为我们提供一个脚本来重现这个问题。该脚本应该可以按原样运行，并且不应该要求下载外部数据（如果需要在某些测试数据上运行模型，请使用随机生成的数据）。我们建议你使用 Github Gists 来张贴你的代码。任何无法重现的问题都会被关闭。

5. 如果可能的话，自己动手修复这个漏洞 - 如果可以的话！

你提供的信息越多，我们就越容易验证存在错误，并且我们可以采取更快的行动。如果你想快速解决你的问题，尊许上述步骤操作是至关重要的。

---

## 请求新功能

你也可以使用 Github Issue 来请求你希望在 Keras 中看到的功能，或者在 Keras API 中的更改。

1. 提供你想要的功能的清晰和详细的解释，以及为什么添加它很重要。请记住，我们需要的功能是对于大多数用户而言的，不仅仅是一小部分人。如果你只是针对少数用户，请考虑为 Keras 编写附加库。对 Keras 来说，避免臃肿的 API 和代码库是至关重要的。

2. 提供代码片段，演示您所需的 API 并说明您的功能的用例。 当然，在这一点上你不需要写任何真正的代码！

3. 讨论完该功能后，您可以选择尝试提一个 Pull Request。如果你完全可以，开始写一些代码。相比时间上，我们总是有更多的工作要做。如果你可以写一些代码，那么这将加速这个过程。


---

## 请求贡献代码

在[这个板块](https://github.com/keras-team/keras/projects/1) 我们会列出当前需要添加的出色的问题和新功能。如果你想要为 Keras 做贡献，这就是可以开始的地方。


---

## Pull Requests 合并请求

**我应该在哪里提交我的合并请求？**

1. **Keras 改进与漏洞修复**， 请到 [Keras `master` 分支](https://github.com/keras-team/keras/tree/master)。

2. **测试新功能**, 例如网络层和数据集，请到 [keras-contrib](https://github.com/farizrahman4u/keras-contrib)。除非它是一个在 [Requests for Contributions](https://github.com/keras-team/keras/projects/1) 中列出的新功能，它属于 Keras 的核心部分。如果你觉得你的功能属于 Keras 核心，你可以提交一个设计文档，来解释你的功能，并争取它（请看以下解释）。

请注意任何有关 **代码风格**（而不是修复修复，改进文档或添加新功能）的 PR 都会被拒绝。

以下是提交你的改进的快速指南：

1. 如果你的 PR 介绍了功能的改变，确保你从撰写设计文档并将其发给 Keras 邮件列表开始，以讨论是否应该修改，以及如何处理。这将拯救你于 PR 关闭。当然，如果你的 PR 只是一个简单的漏洞修复，那就不需要这样做。撰写与提交设计文档的过程如下所示：
      - 从这个 [Google 文档模版](https://docs.google.com/document/d/1ZXNfce77LDW9tFAj6U5ctaJmI5mT7CQXOFMEAZo-mAA/edit#) 开始，将它复制为一个新的 Google 文档。
      - 填写内容。注意你需要插入代码样例。要插入代码，请使用 Google 文档插件，例如 [CodePretty]  (https://chrome.google.com/webstore/detail/code-pretty/igjbncgfgnfpbnifnnlcmjfbnidkndnh?hl=en) (有许多可用的插件)。
      - 将共享设置为 「每个有链接的人都可以发表评论」。
      - 将文档发给 `keras-users@googlegroups.com`，主题从 `[API DESIGN REVIEW]` (全大写) 开始，这样我们才会注意到它。
      - 等待评论，回复评论。必要时修改提案。
      - 该提案最终将被批准或拒绝。一旦获得批准，您可以发出合并请求或要求他人撰写合并请求。


2. 撰写代码（或者让别人写）。这是最难的一部分。

3. 确保你引入的任何新功能或类都有适当的文档。确保你触摸的任何代码仍具有最新的文档。**应该严格遵循 Docstring 风格**。尤其是，它们应该在 MarkDown 中格式化，并且应该有 `Arguments`，`Returns`，`Raises` 部分（如果适用）。查看代码示例中的其他文档以做参考。

4. 撰写测试。你的代码应该有完整的单元测试覆盖。如果你想看到你的 PR 迅速合并，这是至关重要的。 

5. 在本地运行测试套件。这很简单：在 Keras 目录下，直接运行： `py.test tests/`。
      - 您还需要安装测试包： `pip install -e .[tests]`。

6. 确保通过所有测试：
      - 使用 Theano 后端，Python 2.7 和 Python 3.5。确保你有 Theano 的开发版本。
      - 使用 TensorFlow 后端，Python 2.7 和 Python 3.5。确保你有 TensorFlow 的开发版本。
      - 使用 CNTK 后端， Python 2.7 和 Python 3.5。确保你有 CNTK 的开发版本。

7. 我们使用 PEP8 语法约定，但是当涉及到行长时，我们不是教条式的。尽管如此，确保你的行保持合理的大小。为了让您的生活更轻松，我们推荐使用 PEP8 linter：
      - 安装 PEP8 包：`pip install pep8 pytest-pep8 autopep8`
      - 运行独立的 PEP8 检查： `py.test --pep8 -m pep8`
      - 你可以通过运行这个命令自动修复一些 PEP8 错误： `autopep8 -i --select <errors> <FILENAME>`。
    例如： `autopep8 -i --select E128 tests/keras/backend/test_backends.py`

8. 提交时，请使用适当的描述性提交消息。

9. 更新文档。如果引入新功能，请确保包含演示新功能用法的代码片段。

10. 提交你的 PR。如果你的更改已在之前的讨论中获得批准，并且你有完整（并通过）的单元测试以及正确的 docstring/文档，则你的 PR 可能会立即合并。

---

## 添加新的样例

即使你不贡献 Keras 源代码，如果你有一个简洁而强大的 Keras 应用，请考虑将它添加到我们的样例集合中。[现有的例子](https://github.com/keras-team/keras/tree/master/examples)展示惯用的 Keras 代码：确保保持自己的脚本具有相同的风格。
