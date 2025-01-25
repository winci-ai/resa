<h2 align="center"> <a href="https://arxiv.org/abs/2402.14289">Positive-feedback self-supervised Learning of ReSA</a><h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2405.11788-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2405.11788)[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/winci-ai/resa/blob/main/LICENSE)


<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/postive-feedback.jpg" width="400">
</p>

In this paper, we propose the positive-feedback SSL framework. It involves the model generating representations that possess semantically clustering information. This clustering information is leveraged to design self-supervised loss function, which is then employed to more effectively guide the model's learning process.

## Method
ReSA is built under positive-feedback SSL framework, which directly leverages the stable and semantically meaningful clustering properties of the encoder’s outputs. Through an online self-clustering mechanism, ReSA refines its own
objective during training, outperforming state of-the-art SSL baselines and achieving higher efficiency.

<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/resa.jpg" width="700">
</p>

