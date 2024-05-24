# Oriented R-CNN

> [LWGANet: A Lightweight Group Attention Backbone for Remote Sensing Visual Tasks]()

<!-- [ALGORITHM] -->

## Abstract
Remote sensing (RS) image recognition has garnered increasing attention in recent years, yet it encounters several challenges. One major issue is the presence of multiple targets with large-scale variations within a single image, posing a difficulty in feature extraction. Research suggests that methods employing dual-branch or multi-branch structures can effectively adapt to large-scale variations in RS targets, thus enhancing accuracy. However, these structures lead to an increase in parameters and computational load, which complicates RS visual tasks. Present lightweight backbone networks for natural images struggle to adeptly extract features of multi-scale targets simultaneously, which impacts their performance in RS visual tasks. To tackle this challenge, this article introduces a lightweight group attention (LWGA) module tailored for RS images. LWGA module efficiently utilizes redundant features to extract local, medium-range, and global information without inflating the input feature dimensions, to efficiently extract features of multi-scale targets in a lightweight setting. The backbone network built on the LWGA module, named LWGANet, was validated across twelve datasets covering four mainstream RS visual tasks: classification, detection, segmentation, and change detection. Experimental results demonstrate that LWGANet, as a lightweight backbone network, exhibits broad applicability, achieving an optimal balance between performance and latency. State-of-the-art performance was achieved in multiple datasets. LWGANet presents a novel solution for resource-constrained devices in RS visual tasks, with its innovative LWGA structure offering valuable insights for the development of lightweight networks.

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
