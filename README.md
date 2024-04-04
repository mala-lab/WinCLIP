# WinCLIP
Unofficial Implementation of CVPR'23 paper ["WinCLIP: Zero-/few-shot anomaly classification and segmentation"](https://openaccess.thecvf.com/content/CVPR2023/papers/Jeong_WinCLIP_Zero-Few-Shot_Anomaly_Classification_and_Segmentation_CVPR_2023_paper.pdf). 

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

Download the Anomaly Detection Dataset and convert it to MVTec AD format. (For datasets we used in the paper, we provided [the convert script](https://github.com/mala-lab/InCTRL/tree/main/datasets/preprocess)) 

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

Change the values of dataset_root_dir, datasetname, shot (in [main.py](https://github.com/mala-lab/WinCLIP/blob/main/main.py)) and OBJECT_TYPE (in [mvtec_dataset.py](https://github.com/mala-lab/WinCLIP/blob/main/datasets/mvtec_dataset.py)). 
For example, if run on the category candle of visa with k=2:
```
dataset_root_dir = "/visa_anomaly_detection"
datasetname = "visa"
shot = 0

OBJECT_TYPE = ["candle"]
```
and run
```bash
python main.py
```

## Citation

This implementation is used to produce the WinCLIP results in our [CVPR'24 InCTRL paper](https://github.com/mala-lab/InCTRL). If you find the implementation useful, we would appreciate your acknowledgement via citing the InCTRL paper:

```bibtex
@inproceedings{zhu2024toward,
  title={Toward Generalist Anomaly Detection via In-context Residual Learning with Few-shot Sample Prompts},
  author={Zhu, Jiawen and Pang, Guansong},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2024}
}
```
