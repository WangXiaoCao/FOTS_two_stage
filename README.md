# FOTS_two_stage

* 中文OCR。
* 检测分支为EAST，识别分支为CRNN。
* 仅是测试代码
* 训练数据集为 RCTW-17

## Pre-requisite

* python 3.6
* torch
* torchvision
* pretrainedmodels
* PIL
* shapely
* lmdb
* 如果使用lanms，而不是locality_aware_nms，则需要python 3.5

## Data

使用RCTW-17中文数据集。

数据集下载：
    
    http://rctw.vlrlab.net/dataset/


## How to use

1. clone：
    * `git clone https://github.com/MagicianQi/FOTS_two_stage`
    * `cd ./FOTS_two_stage/`
    * 下载模型及测试图像 `wget https://github.com/MagicianQi/FOTS_two_stage/releases/download/v0.1/models_and_imgs.zip`
    * 解压 `unzip models_and_imgs.zip`
2. 测试检测分支：
    * `python testing_detection_branch.py`
3. 测试识别分支：
    * `python testing_recognition_branch.py`
4. 测试OCR：
    * `python testing_ocr.py`  
