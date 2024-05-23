
![lwganet_arch](docs/lwganet.png)

## This repository is the official implementation of "LWGANet: A Lightweight Group Attention Backbone for Remote Sensing Visual Tasks".
## Abstract
Remote sensing (RS) image recognition has garnered increasing attention in recent years, yet it encounters several challenges. One major issue is the presence of multiple targets with large-scale variations within a single image, posing a difficulty in feature extraction. Research suggests that methods employing dual-branch or multi-branch structures can effectively adapt to large-scale variations in RS targets, thus enhancing accuracy. However, these structures lead to an increase in parameters and computational load, which complicates RS visual tasks. Present lightweight backbone networks for natural images struggle to adeptly extract features of multi-scale targets simultaneously, which impacts their performance in RS visual tasks. To tackle this challenge, this article introduces a lightweight group attention (LWGA) module tailored for RS images. LWGA module efficiently utilizes redundant features to extract local, medium-range, and global information without inflating the input feature dimensions, to efficiently extract features of multi-scale targets in a lightweight setting. The backbone network built on the LWGA module, named LWGANet, was validated across twelve datasets covering four mainstream RS visual tasks: classification, detection, segmentation, and change detection. Experimental results demonstrate that LWGANet, as a lightweight backbone network, exhibits broad applicability, achieving an optimal balance between performance and latency. State-of-the-art performance was achieved in multiple datasets. LWGANet presents a novel solution for resource-constrained devices in RS visual tasks, with its innovative LWGA structure offering valuable insights for the development of lightweight networks.

## Introduction

The master branch is built on MMRotate which works with **PyTorch 1.6+**.

LWGANet backbone code is placed under mmrotate/models/backbones/, and the train/test configure files are placed under configs/lwganet/ 


## Results and models

Imagenet 300-epoch pre-trained LWGANet-L0 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l0_e297.pth)

Imagenet 300-epoch pre-trained LWGANet-L1 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l1_e239.pth)

Imagenet 300-epoch pre-trained LWGANet-L2 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l2_e299.pth)

DOTA1.0

|           Model            |  mAP  | Angle | training mode | Batch Size |                                     Configs                                      |                                                              Download                                                               |
|:--------------------------:|:-----:| :---: |---------------|:----------:|:--------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|
| LWGANet_L2 (1024,1024,200) | 78.64 | le90  | single-scale  |    2\*4    | [lwganet_l2_fpn_30e_dota10_ss_le90](./configs/lwganet/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.py) |          [model](https://github.com/lwCVer/LWGANet/releases/download/weights/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.pth)           |


DOTA1.5

|         Model         |  mAP  | Angle | training mode | Batch Size |                                             Configs                                              |                                                     Download                                                     |
| :----------------------: |:-----:| :---: |---| :------: |:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|
| LWGANet_L2 (1024,1024,200) | 71.72 | le90  | single-scale |    2\*4     | [lwganet_l2_fpn_30e_dota15_ss_le90](./configs/lsknet/ORCNN_LWGANet_L2_fpn_le90_dota15_ss_e30.py) | [model](https://github.com/lwCVer/LWGANet/releases/download/weights/ORCNN_LWGANet_L2_fpn_le90_dota15_ss_e30.pth) |

DIOR-R 

|                    Model                     |  mAP  | Batch Size |
| :------------------------------------------: |:-----:| :--------: |
|                   LWGANet_L2                   | 68.53 |    1\*8    |

## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n LWGANet-Det python=3.8 -y
conda activate LWGANet-Det
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -U openmim
mim install mmcv-full
mim install mmdet
# git clone https://github.com/open-mmlab/mmrotate.git
# cd mmrotate
pip install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)

## Acknowledgement

MMRotate is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks. We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new methods.

## Citation


## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.
