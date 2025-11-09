# LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention

This is the official Pytorch/Pytorch implementation of the paper: <br/>
> **LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention**
>
> Wei Lu, Xue Yang, Si-Bao Chen*
>

----

<p align="center"> 
<img src="figures/LWGANet.png" width=100% 
class="center">
<p align="center">  Illustration of LWGANet architecture.
</p> 

----

## News 🆕
- **2025.07.11** Congratulations! Our paper "LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention" has been accepted by [AAAI 2026 (Oral)](https://openaccess.thecvf.com/content/AAAI2026/). 🔥

- **2025.01.17** Update LEGNet original-version paper in [Arxiv](https://arxiv.org/abs/2501.10040). The new code, models and results are uploaded. 🎈



<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>

Remote sensing (RS) visual tasks have gained significant academic and practical importance. However, they encounter numerous challenges that hinder effective feature extraction, including the detection and recognition of multiple objects exhibiting substantial variations in scale within a single image. While prior dual-branch or multi-branch architectural strategies have been effective in managing these object variances, they have concurrently resulted in considerable increases in computational demands and parameter counts. Consequently, these architectures are rendered less viable for deployment on resource-constrained devices. Contemporary lightweight backbone networks, designed primarily for natural images, frequently encounter difficulties in effectively extracting features from multi-scale objects, which compromises their efficacy in RS visual tasks. This article introduces LWGANet, a specialized lightweight backbone network tailored for RS visual tasks, incorporating a novel lightweight group attention (LWGA) module designed to address these specific challenges. The LWGA module, tailored for RS imagery, adeptly harnesses redundant features to extract a wide range of spatial information, from local to global scales, without introducing additional complexity or computational overhead. This facilitates precise feature extraction across multiple scales within an efficient framework. LWGANet was rigorously evaluated across twelve datasets, which span four crucial RS visual tasks: scene classification, oriented object detection, semantic segmentation, and change detection. The results confirm LWGANet's widespread applicability and its ability to maintain an optimal balance between high performance and low complexity, achieving state-of-the-art results across diverse datasets. LWGANet emerged as a novel solution for resource-limited scenarios requiring robust RS image processing capabilities.
</details>




## Pre-train Weights on Imagenet-1k

Imagenet 300-epoch pre-trained LWGANet-L0 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l0_e299.pth)

Imagenet 300-epoch pre-trained LWGANet-L1 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l1_e299.pth)

Imagenet 300-epoch pre-trained LWGANet-L2 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l2_e296.pth)

## Get Started

### [Image Classification](./classification/README.md)

- [Dependency Setup](./classification/README.md#Dependency)
- [Dataset Preparation](./classification/README.md#Dataset)
- [Training Steps](./classification/README.md#Training)
- [Experimental Results](./classification/README.md#Results)

----

### [Object Detection](./detection/README.md)

- [Dependency Setup](./detection/README.md#Dependency)
- [Training Steps](./detection/docs/en/get_started.md)
- [Experimental Results](./detection/README.md#Results)

----

### [Semantic Segmentation](./segmentation/README.md)

- [Dependency Setup](./segmentation/README.md#Dependency)
- [Dataset Preparation](./segmentation/README.md#Dataset)
- [Training Steps](./segmentation/README.md#Training)
- [Experimental Results](./segmentation/README.md#Results)

----

### Change Detection

- [A2Net_LWGANet](./change_detection/A2Net_LWGANet)
- [CLAFA_LWGANet](./change_detection/CLAFA_LWGANet)

----


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lwCVer/LWGANet&type=date&legend=top-left)](https://www.star-history.com/#lwCVer/LWGANet&type=date&legend=top-left)


## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models), [mmrotate](https://github.com/open-mmlab/mmrotate),   [unetformer](https://github.com/WangLibo1995/GeoSeg), [A2Net](https://github.com/guanyuezhen/A2Net), and [CLAFA](https://github.com/xingronaldo/CLAFA) repositories.

If you have any questions about this work, you can contact me. 

Email: [luwei_ahu@qq.com](mailto:luwei_ahu@qq.com); WeChat: lw2858191255.

Your star is the power that keeps us updating github.

## Citation
If LWGANet is useful or relevant to your research, please kindly recognize our contributions by citing our paper:
```
@inproceedings{lu2026lwganet,
  title={LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention},
  author={Lu, Wei and Yang, Xue and Chen, Si-Bao},
  booktitle={AAAI Conference on Artificial Intelligence},
  pages={},
  year={2026}
}
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. 
Any commercial use should get formal permission first.
