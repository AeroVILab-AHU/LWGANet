# Code for the TCSVT article 'Cross-Level Attentive Feature Aggregation for Change Detection'.
---------------------------------------------
Here I provide the PyTorch implementation for CLAFA.


## ENVIRONMENT
>RTX 3090<br>
>python 3.8.8<br>
>PyTorch 1.11.0<br>
>mmcv 1.6.0

## Installation
Clone this repo:

```shell
git clone https://github.com/xingronaldo/CLAFA.git
cd CLAFA
```

* Install dependencies

All dependencies can be installed via 'pip'.

## Dataset Preparation
Download data and add them to `./datasets`. 


## Test
Here I provide the trained models for the SV-CD dataset [Baidu Netdisk, code: CLAF](https://pan.baidu.com/s/1nfqqXA3DsZtU4BtHY-3YOg)A.

Put them in `./checkpoints`.


* Test on the SV-CD dataset with the MobileNetV2 backbone

```python
python test.py --backbone mobilenetv2 --name SV_mobilenetv2 --gpu_ids 1
```

* Test on the SV-CD dataset with the ResNet18d backbone

```python
python test.py --backbone resnet18d --name SV_resnet18d --gpu_ids 1
```

## Train & Validation
```python
python trainval.py --gpu_ids 1 
```
All the hyperparameters can be adjusted in `option.py`.


## Contact
Email: guangxingwang@mail.nwpu.edu.cn



