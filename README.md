# WinCLIP
Implementation of CVPR'23 paper "WinCLIP: Zero-/few-shot anomaly classification and segmentation". 

updating...

## Setup
- python >= 3.10.11
- torch >= 1.13.0
- torchvision >= 0.14.0
- scipy >= 1.10.1
- scikit-image >= 0.21.0
- numpy >= 1.24.3
- tqdm >= 4.64.0

## Device
Single NVIDIA GeForce RTX 3090

## Run
#### Step 1. Setup the Anomaly Detection dataset

Download the Anomaly Detection Dataset and convert it to MVTec AD format. (For datasets we used in the paper, we provided [the convert script]) 
The dataset folder structure should look like:

```
DATA_PATH/
    subset_1/
        train/
            good/
        test/
            good/
            defect_class_1/
            defect_class_2/
            defect_class_3/
            ...
    ...
```
#### Step 2. Quick Start

```bash
python winclip.py
```
