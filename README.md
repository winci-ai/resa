<h2 align="center"> <a href="https://arxiv.org/abs/2402.14289">ReSA: positive-feedback self-supervised learning</a><h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2405.11788-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2405.11788)[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](https://github.com/winci-ai/resa/blob/main/LICENSE)


<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/postive-feedback.jpg" width="350">
</p>


## Method
ReSA is developed within our positive-feedback SSL framework, which directly leverages the stable and semantically meaningful clustering properties of the encoder’s outputs. Through an online self-clustering mechanism, ReSA refines its own
objective during training, outperforming state of-the-art SSL baselines and achieving higher efficiency.

<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/resa.jpg" width="600">
</p>

### ReSA Excels at both Fine-grained and Coarse-grained Learning

<p align="center">
    <img src="https://github.com/winci-ai/resa/releases/download/figure/tsne.jpg" width="800">
</p>

## Pretrained models

You can choose to download only the weights of the pretrained encoder used for downstream tasks, or the full checkpoint which contains encoder, projector, and predictor weights for both base and momentum networks.

Our pretrained ResNet-50 models on ImageNet with two $224^{2}$ augmented views:

<table align="center" border="1" style="width:100%; border-collapse:collapse; text-align:center;">
  <tr>
    <th>epochs</th>
    <th>bs</th>
    <th>linear acc</th>
    <th>knn acc</th>
    <th colspan="4">download</th>
  </tr>
  <tr>
    <td>100</td>
    <td>256</td>
    <td>71.9%</td>
    <td>64.6%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_only_bs256_ep100.pth">ResNet-50</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_full_bs256_ep100.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs256_ep100_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs256_ep100_linear.log">eval log</a></td>
  </tr>
  <tr>
    <td>200</td>
    <td>256</td>
    <td>73.4%</td>
    <td>67.1%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_only_bs256_ep200.pth">ResNet-50</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_full_bs256_ep200.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs256_ep200_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs256_ep200_linear.log">eval log</a></td>
  </tr>
  <tr>
    <td>100</td>
    <td>1024</td>
    <td>71.3%</td>
    <td>63.3%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_only_bs1024_ep100.pth">ResNet-50</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_full_bs1024_ep100.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep100_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep100_linear.log">eval log</a></td>
  </tr>
  <tr>
    <td>200</td>
    <td>1024</td>
    <td>73.8%</td>
    <td>67.6%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_only_bs1024_ep200.pth">ResNet-50</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_full_bs1024_ep200.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep200_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep200_linear.log">eval log</a></td>
  </tr>
  <tr>
    <td>800</td>
    <td>1024</td>
    <td>75.2%</td>
    <td>69.9%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_only_bs1024_ep800.pth">ResNet-50</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_full_bs1024_ep800.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep800_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/resnet50/resa_resnet50_bs1024_ep800_linear.log">eval log</a></td>
  </tr>
</table>

Our pretrained ViT-S/16 model on ImageNet with two $224^{2}$ augmented views:
<table align="center" border="1" style="width:100%; border-collapse:collapse; text-align:center;">
  <tr>
    <th>epochs</th>
    <th>bs</th>
    <th>linear acc</th>
    <th>knn acc</th>
    <th colspan="4">download</th>
  </tr>
  <tr>
    <td>300</td>
    <td>1024</td>
    <td>72.7%</td>
    <td>68.3%</td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/vit/resa_vits16_only_bs1024_ep300.pth">ViT-S/16</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/vit/resa_vits16_full_bs1024_ep300.pth">full checkpoint</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/vit/resa_vits16_bs1024_ep300_train.log">train log</a></td>
    <td><a href="https://github.com/winci-ai/resa/releases/download/vit/resa_vits16_bs1024_ep300_linear.log">eval log</a></td>
  </tr>
</table>


## Code Structure

```
.
├── methods                     #   self-supervised methods
│   ├── base.py                 #   base class for self-supervised loss implementation
│   ├── resa_plain.py           #   ReSA without using the momentum network
│   └── resa.py                 #   ReSA using the momentum network, better performance
├── scripts                     #   scripts for executing experiments
│   ├── eval                    #   scripts for evaluation
│   └── train                   #   scripts for pretraining
├── src                         #   the packages
│   ├── imagenet_subset         #   1% and 10% subsets of ImageNet-1K
│   ├── model.py                #   function definition for the encoder, projector, and predictor
│   ├── resnet.py               #   class definition for the ResNet model
│   ├── transforms.py           #   data augmentation for pretraining
│   ├── utils.py                #   shared utilities
│   └── vision_transformer.py   #   class definition for the ViT model
├── args.py                     #   arguments
├── eval_knn.py                 #   evaluate with a weighted k-nn classifier
├── eval_linear.py              #   evaluate with a linear classifier
└── main.py                     #   main function to pretrain with ReSA
```

## Installation and Requirements

1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/winci-ai/resa.git
cd resa
```

2. Create a conda environment, activate it and install packages (newer versions of python are supported)
```Shell
conda create -n resa python=3.8.18
conda activate resa
pip install -r requirements.txt
```