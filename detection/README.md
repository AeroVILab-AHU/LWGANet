# LWGANet: A Lightweight Group Attention Backbone for Remote Sensing Visual Tasks

This is the official Pytorch/Pytorch implementation of the paper: <br/>
> **LWGANet: A Lightweight Group Attention Backbone for Remote Sensing Visual Tasks**
>
> Wei Lu, Si-Bao Chen*, Chris H. Q. Ding, Jin Tang, and Bin Luo, Senior Member, IEEE 
> 
>  *IEEE Transactions on Image Processing (TIP), In peer review.* [arXiv](https://arxiv.org/abs/2501.10040)
> 

----

<p align="center"> 
<img src="../figures/LWGANet.png" width=100% 
class="center">
<p align="center">  Illustration of LWGANet architecture.
</p> 

----

## Results and models

Imagenet 300-epoch pre-trained LWGANet-L0 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l0_e299.pth)

Imagenet 300-epoch pre-trained LWGANet-L1 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l1_e299.pth)

Imagenet 300-epoch pre-trained LWGANet-L2 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l2_e296.pth)

DOTA1.0

|           Model            |  mAP  | Angle | training mode | Batch Size |                                     Configs                                      |                                                              Download                                                               |
|:--------------------------:|:-----:| :---: |---------------|:----------:|:--------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------:|
| LWGANet_L2 (1024,1024,200) | 78.64 | le90  | single-scale  |    2\*4    | [lwganet_l2_fpn_30e_dota10_ss_le90](./configs/lwganet/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.py) |          [model](https://github.com/lwCVer/LWGANet/releases/download/weights/ORCNN_LWGANet_L2_fpn_le90_dota10_ss_e30.pth)           |


DOTA1.5

|         Model         |  mAP  | Angle | training mode | Batch Size |                                             Configs                                              |                                                     Download                                                     |
| :----------------------: |:-----:| :---: |---| :------: |:------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------:|
| LWGANet_L2 (1024,1024,200) | 71.72 | le90  | single-scale |    2\*4     | [lwganet_l2_fpn_30e_dota15_ss_le90](./configs/lwganet/ORCNN_LWGANet_L2_fpn_le90_dota15_ss_e30.py) | [model](https://github.com/lwCVer/LWGANet/releases/download/weights/ORCNN_LWGANet_L2_fpn_le90_dota15_ss_e30.pth) |

DIOR-R 

|                    Model                     |  mAP  | Batch Size |
| :------------------------------------------: |:-----:| :--------: |
|                   LWGANet_L2                   | 68.53 |    1\*8    |

## Dependency

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
git clone https://github.com/lwCVer/LWGANet
cd LWGANet/detection
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
