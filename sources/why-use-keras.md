# 为什么选择Keras？

在如今无数深度学习框架中，为什么选择Keras而不是其他？以下是Keras的优势：

---

## Keras注重开发者体验

- Keras是为人类而不是为机器设计的API。[Keras遵循减轻认知负载的最佳实践。](https://blog.keras.io/user-experience-design-for-apis.html): 它提供一致而简洁的API，尽量减少常用操作所需的步骤，并对误用给予清晰而有指导性的反馈。

- 这让Keras既好学又好用。作为Keras的使用者，你比竞争对手更高效，因为你能更快地尝试更多新点子。这反过来[帮你赢得机器学习竞赛](https://www.quora.com/Why-has-Keras-been-so-successful-lately-at-Kaggle-competitions)。

- 简单易用并不以牺牲灵活性为代价：因为Keras整合了底层深度学习语言，特别是TensorFlow，使你可以构建原本想用底层语言构建的一切。Keras以“tf.keras“的形式与TensorFlow无缝衔接。

---

## Keras被工业界和学术界广泛采用

截至2017年11月，拥有超过200，000个人用户的Keras是除TensorFlow外被工业界和学术界最多使用的深度学习框架（而Keras往往与TensorFlow结合使用）。

其实你早已见识过Keras————Netflix，优步，Yelp，Instacart，Zocdoc，Square，以及其他许多应用上都有用Keras创建的功能。它尤其受以深度学习为产品核心的创业公司的欢迎。

Keras也大受深度学习研究者的喜爱。在上传到学术论文网站[arXiv.org](https://arxiv.org/archive/cs)的论文中被提及的次数位居第二：

<img src='/img/arxiv-mentions.png' style='width:500px; display: block; margin: 0 auto;'/>

Keras还被大型科研机构的研究者所采用，特别是CERN和NASA。

---

## Keras使模型轻松转化为产品

与其他深度学习框架相比，Keras模型可以轻松地发布到更多平台：

- iOS，通过[Apple’s CoreML](https://developer.apple.com/documentation/coreml)（苹果为Keras提供官方支持）
- 安卓，通过TensorFlow Android runtime，例如：[Not Hotdog app](https://medium.com/@timanglade/how-hbos-silicon-valley-built-not-hotdog-with-mobile-tensorflow-keras-react-native-ef03260747f3)
- 浏览器，通过GPU加速的JavaScript runtimes，例如：[Keras.js](https://transcranial.github.io/keras-js/#/)和[WebDNN](https://mil-tokyo.github.io/webdnn/)
- Google Cloud，通过[TensorFlow-Serving](https://www.tensorflow.org/serving/)
- Python网页应用后端（比如Flask app）
- JVM，通过[SkyMind提供的DL4J模型导入](https://deeplearning4j.org/model-import-keras)
- Raspberry Pi

---
## Keras支持众多后端引擎，不限制你于单一生态

你的Keras模型可以用不同的[深度学习后端](https://keras.io/backend/)开发。尤其当模型仅含有Keras内置的层时，它可以在这些后端间移植：用一种后端训练模型，再将它载入另一种后端中（比如为了发布）。支持的后端有：
 
 - 谷歌的TensorFlow后端
 - 微软的CNTK后端
 - Theano后端

亚马逊也正在为Keras开发MXNet后端。

如此一来，你的Keras模型可以在超越CPU的不同硬件平台上训练：

- [NVIDIA GPU](https://developer.nvidia.com/deep-learning)
- [Google TPU](https://cloud.google.com/tpu/)，通过TensorFlow后端和Google Cloud
- OpenGL支持的GPU, 比如AMD那些, 通过[PlaidML Keras后端](https://github.com/plaidml/plaidml)

---

## Keras强力支持多GPU和分布式训练

- Keras[内置对多GPU数据并行的支持](/utils/#multi_gpu_model)
- 优步的[Horovod](https://github.com/uber/horovod)对Keras模型有第一流的支持
- Keras模型[可以被转换为TensorFlow估算器](https://www.tensorflow.org/versions/master/api_docs/python/tf/keras/estimator/model_to_estimator)并在[Google Cloud的GPU集群](https://cloud.google.com/solutions/running-distributed-tensorflow-on-compute-engine)上训练。
- Keras可以在Spark（通过CERN的[Dist-Keras](https://github.com/cerndb/dist-keras)）和 [Elephas](https://github.com/maxpumperla/elephas)上运行

---

## Keras的开发受到深度学习生态圈中关键公司的支持

Keras的开发主要由谷歌支持，Keras API以“tf.keras"的形式打包在TensorFlow中。微软维护着Keras的CNTK后端。亚马逊AWS正在开发MXNet支持。其他提供支持的公司包括NVIDIA、优步、苹果（通过CoreML）。

<img src='/img/google-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/microsoft-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/nvidia-logo.png' style='width:200px; margin-right:15px;'/>
<img src='/img/aws-logo.png' style='width:110px; margin-right:15px;'/>
