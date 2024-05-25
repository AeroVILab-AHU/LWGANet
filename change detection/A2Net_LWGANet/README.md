
This repository contains a simple Python implementation of the paper [A2Net](https://ieeexplore.ieee.org/abstract/document/10034814).

### 2. Usage
+ Prepare the data:
    - Download datasets LEVIR, WHC-CD, CDD, and SYSU.
    - Crop the datasets into 256x256 patches. 
    - Generate list file as `ls -R ./label/* > test.txt`
    - Prepare datasets into the following structure and set their path in `train.py` and `test.py`
    ```
    ├─Train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─label    ...jpg/png
        └─list     ...txt
    ├─Val
        ├─A
        ├─B
        ├─label
        └─list
    ├─Test
        ├─A
        ├─B
        ├─label
        └─list
    ```

+ Prerequisites for Python:
    - Creating a virtual environment in the terminal: `conda create -n LWGANet-CD python=3.8`
    - Installing necessary packages: `pip install -r requirements.txt `

+ Train/Test
    - `sh ./tools/train.sh`
    - `sh ./tools/test.sh`