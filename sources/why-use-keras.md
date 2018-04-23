# 为什么选择 Keras？

在如今无数深度学习框架中，为什么要使用 Keras 而非其他？以下是 Keras 与现有替代品的一些比较。

---

## Keras 优先考虑开发人员的经验

- Keras 是为人类而非机器设计的 API。[Keras 遵循减少认知困难的最佳实践](https://blog.keras.io/user-experience-design-for-apis.html): 它提供一致且简单的 API，它将常见用例所需的用户操作数量降至最低，并且在用户错误时提供清晰和可操作的反馈。

- 这使 Keras 易于学习和使用。作为 Keras 用户，你的工作效率更高，能够比竞争对手更快地尝试更多创意，从而[帮助你赢得机器学习竞赛](https://www.quora.com/Why-has-Keras-been-so-successful-lately-at-Kaggle-competitions)。

- 这种易用性并不以降低灵活性为代价：因为 Keras 与底层深度学习语言（特别是 TensorFlow）集成在一起，所以它可以让你实现任何你可以用基础语言编写的东西。特别是，`tf.keras` 作为 Keras API 可以与 TensorFlow 工作流无缝集成。


---

## Keras 被工业界和学术界广泛采用

截至 2017 年 11 月，拥有超过 200,000 个人用户的 Keras 是除 TensorFlow 外被工业界和学术界最多使用的深度学习框架（且 Keras 往往与 TensorFlow 结合使用）。

你已经在不断与 Keras 构建的功能进行交互 - 它正在被 Netflix, Uber, Yelp, Instacart, Zocdoc, Square 等其他机构使用。它尤其受以深度学习作为产品核心的创业公司的欢迎。

Keras 也大受深度学习研究者的喜爱。在上传到学术论文网站 [arXiv.org](https://arxiv.org/archive/cs) 的论文中被提及的次数位居第二：

<img src='/img/arxiv-mentions.png' style='width:500px; display: block; margin: 0 auto;'/>

Keras 还被大型科研机构的研究者所采用，特别是 CERN 和 NASA。

---

## Keras 可以轻松将模型转化为产品

与任何其他深度学习框架相比，你的 Keras 模型可以轻松部署在更广泛的平台上：

- 在 iOS 上，通过 [Apple’s CoreML](https://developer.apple.com/documentation/coreml)（苹果为 Keras 提供官方支持）。
- 在安卓 上，通过 TensorFlow Android runtime，例如：[Not Hotdog app](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3)。
- 在浏览器上，通过 GPU 加速的 JavaScript 运行时，例如：[Keras.js](https://transcranial.github.io/keras-js/#/) 和 [WebDNN](https://mil-tokyo.github.io/webdnn/)。
- 在 Google Cloud 上，通过 [TensorFlow-Serving](https://www.tensorflow.org/serving/)。
- [在 Python 网页应用后端（比如 Flask app）中]（https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)。
- 在 JVM，通过 [SkyMind 提供的 DL4J 模型导入](https://deeplearning4j.org/model-import-keras)。
- 在 Raspberry Pi 树莓派上。

---

## Keras 支持多个后端引擎，并且不会将你锁定到一个生态系统中

你的 Keras 模型可以基于不同的[深度学习后端](https://keras.io/zh/backend/)开发。重要的是，任何仅利用内置层构建的 Keras 模型，都可以在所有这些后端中移植：用一种后端训练模型，再将它载入另一种后端中（比如为了发布）。支持的后端有：
 
 - 谷歌的 TensorFlow 后端
 - 微软的 CNTK 后端
 - Theano 后端

亚马逊也正在为 Keras 开发 MXNet 后端。

如此一来，你的 Keras 模型可以在 CPU 之外的不同硬件平台上训练：

- [NVIDIA GPU](https://developer.nvidia.com/deep-learning)。
- [Google TPU](https://cloud.google.com/tpu/)，通过 TensorFlow 后端和 Google Cloud。
- OpenGL 支持的 GPU, 比如 AMD, 通过 [PlaidML Keras 后端](https://github.com/plaidml/plaidml)。

---

## Keras 拥有强大的多 GPU 和分布式训练支持

- Keras [内置对多 GPU 数据并行的支持](https://keras.io/zh/utils/#multi_gpu_model)。
- 优步的 [Horovod](https://github.com/uber/horovod) 对 Keras 模型有第一流的支持。
- Keras 模型[可以被转换为 TensorFlow 估计器](https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/estimator/model_to_estimator)并在 [Google Cloud 的 GPU 集群](https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine)上训练。
- Keras 可以在 Spark（通过 CERN 的 [Dist-Keras](https://github.com/cerndb/dist-keras)）和 [Elephas](https://github.com/maxpumperla/elephas) 上运行。

---

## Keras 的发展得到深度学习生态系统中的关键公司的支持

Keras 的开发主要由谷歌支持，Keras API 以 `tf.keras` 的形式包装在 TensorFlow 中。此外，微软维护着 Keras 的 CNTK 后端。亚马逊 AWS 正在开发 MXNet 支持。其他提供支持的公司包括 NVIDIA、优步、苹果（通过 CoreML）等。

<img src='/img/google-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/microsoft-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/nvidia-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/aws-logo.png' style='width:110px; margin-right:15px;'/>
