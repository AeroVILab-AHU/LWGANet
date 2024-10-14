![LWGANet_arch](figures/LWGANet.png)

## This repository is the official implementation of "LWGANet: A Lightweight Group Attention Backbone for Remote Sensing Visual Tasks"

## Abstract


Remote sensing (RS) visual tasks have gained significant academic and practical importance. However, they encounter numerous challenges that hinder effective feature extraction, including the detection and recognition of multiple objects exhibiting substantial variations in scale within a single image. While prior dual-branch or multi-branch architectural strategies have been effective in managing these object variances, they have concurrently resulted in considerable increases in computational demands and parameter counts. Consequently, these architectures are rendered less viable for deployment on resource-constrained devices. Contemporary lightweight backbone networks, designed primarily for natural images, frequently encounter difficulties in effectively extracting features from multi-scale objects, which compromises their efficacy in RS visual tasks. This article introduces LWGANet, a specialized lightweight backbone network tailored for RS visual tasks, incorporating a novel lightweight group attention (LWGA) module designed to address these specific challenges. The LWGA module, tailored for RS imagery, adeptly harnesses redundant features to extract a wide range of spatial information, from local to global scales, without introducing additional complexity or computational overhead. This facilitates precise feature extraction across multiple scales within an efficient framework.
LWGANet was rigorously evaluated across twelve datasets, which span four crucial RS visual tasks: scene classification, oriented object detection, semantic segmentation, and change detection. The results confirm LWGANet's widespread applicability and its ability to maintain an optimal balance between high performance and low complexity, achieving state-of-the-art results across diverse datasets. LWGANet emerged as a novel solution for resource-limited scenarios requiring robust RS image processing capabilities.


## Introduction

This repository is the official implementation of "LWGANet: A Lightweight Group Attention Backbone for Remote Sensing Visual Tasks"

## Results and models

Imagenet 300-epoch pre-trained LWGANet-L0 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l0_e299.pth)

Imagenet 300-epoch pre-trained LWGANet-L1 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l1_e299.pth)

Imagenet 300-epoch pre-trained LWGANet-L2 backbone: [Download](https://github.com/lwCVer/LWGANet/releases/download/weights/lwganet_l2_e296.pth)

## Get Started
