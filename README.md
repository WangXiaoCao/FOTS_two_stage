# Spam_OCR

* 垃圾图像的文字检测。
* 检测分支为EAST，识别分支为CRNN。
* 不包含训练代码，仅是测试代码
* 训练数据集为 RCTW-17
* 详细信息见：http://wiki.soulapp.cn/pages/viewpage.action?pageId=15436459

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

数据集位置：
    
    "172.30.1.118:/home/datasets/qishuo/RCTW-17/"


## How to use

1. 测试检测分支：
    * `python testing_detection_branch.py`
2. 测试识别分支：
    * `python testing_recognition_branch.py`
3. 测试OCR：
    * `python testing_ocr.py`  
