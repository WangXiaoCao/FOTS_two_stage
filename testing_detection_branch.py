# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch

from detection_branch.model import FOTSModel
from detection_branch.bbox import Toolbox

# -------------------路径参数-------------------

model_path = "./models/checkpoint-epoch030-loss-0.0154.pth"
input_dir = "./input/"
output_dir = "./out/ "

# -------------------加载模型-------------------

state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
model = FOTSModel()
model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict)
if torch.cuda.is_available():
    model = model.cuda()
model.eval()

# -------------------预处理-------------------

files = []
for file in os.listdir(input_dir):
    if file != ".DS_Store":
        files.append(input_dir + file)

for image_fn in files:
    with torch.no_grad():
        ploy, im = Toolbox.predict(image_fn, model, True, output_dir, torch.cuda.is_available())
        print(len(ploy))
