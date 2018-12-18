# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch

from detection_branch.model import FOTSModel
from detection_branch.bbox import Toolbox

from PIL import Image
import time

import torch
from torch.autograd import Variable

from recognition_branch import utils, alphabet, dataset
import recognition_branch.net as crnn
import rotate

# -------------------路径参数-------------------

detection_model_path = "./models/checkpoint-epoch030-loss-0.0154.pth"
recognition_model_path = './models/crnn_chinese_5529.pth'
img_path = './input/IMG2.png'
alphabet = alphabet.alphabetChinese

# -------------------加载模型-------------------

print("Getting detection model...")
detection_model = FOTSModel()
detection_model = torch.nn.DataParallel(detection_model)
detection_model.load_state_dict(torch.load(detection_model_path, map_location=lambda storage, loc: storage))

print("Getting recognition model...")
recognition_model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)
recognition_model.load_state_dict(torch.load(recognition_model_path, map_location=lambda storage, loc: storage))

if torch.cuda.is_available():
    detection_model = detection_model.cuda()
    recognition_model = recognition_model.cuda()

detection_model.eval()
recognition_model.eval()

# -------------------获取结果-------------------

# 获取检测结果
print("--------------------")
ploys, im = Toolbox.predict(img_path, detection_model, False, with_gpu=torch.cuda.is_available())

# 检测结果转换
crop_images = rotate.rotate_img(ploys, im)

print("--------------------")
# 针对每个crop识别
for img in crop_images:
    converter = utils.strLabelConverter(alphabet)
    # image = Image.open(img_path).convert("RGB")
    image = Image.fromarray(img).convert("RGB")
    image = image.convert('L')

    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    transformer = dataset.resizeNormalize((w, 32))

    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    start = time.time()
    preds = recognition_model(image)
    end = time.time()
    print("Forward time : {}".format(end - start))

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))

    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    print(sim_pred)
