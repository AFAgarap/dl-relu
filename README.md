Deep Learning using Rectified Linear Units
===

![](https://img.shields.io/badge/DOI-cs.NE%2F1803.08375-blue.svg)
![](https://img.shields.io/badge/license-Apache--2.0-blue.svg)

## Abstract

We introduce the use of rectified linear units (ReLU) as the classification function in a deep neural network (DNN). Conventionally, ReLU is used as an activation function in DNNs, with Softmax function as their classification function. However, there have been several studies on using a classification function other than Softmax, and this study is an addition to those. We accomplish this by taking the activation of the penultimate layer in a neural network, then multiply it by weight parameters θ to get the raw scores. Afterwards, we threshold the raw scores by 0, i.e. f(o) = max(0, o), where f(o) is the ReLU function. We provide class predictions ŷ through argmax function, i.e. argmax f(x).

## License

```
Copyright 2018 Abien Fred Agarap

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
