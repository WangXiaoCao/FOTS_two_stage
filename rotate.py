# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import math


def rotate_img(ploys, im):
    """
    将文本检测结果裁剪，并旋转(若角度小于10度，则不操作)。
    :param ploys: 检测结果(多边形)
    :param im: 源图像
    :return: 裁剪后的图像列表
    """
    crop_images = []
    for ploy in ploys:
        if judege_ploy_out_of_bounds(ploy, im):
            rect = cv2.boundingRect(ploy)
            x, y, w, h = rect
            croped = im[y:y + h, x:x + w].copy()
            angle, real_w, real_h = get_angle_and_real_width_height(ploy, rect)
            print("{}    {}".format(w, h))
            if angle < -10 or angle > 10:
                croped = rotate_crop(croped, angle, real_w, real_h)
            # cv2.imshow("source", croped)
            # cv2.waitKey(3000)
            crop_images.append(croped)
    return crop_images


def judege_ploy_out_of_bounds(ploy, im):
    """
    判断ploy是否越界，False为越界
    :param ploy: 检测结果，多边形。
    :param im: 源图像
    :return: 判断结果
    """
    width, height, _ = im.shape
    for point in ploy:
        if point[0] < 0 or point[0] > width:
            return False
        if point[1] < 0 or point[1] > height:
            return False
    return True


def rotate_crop(image, angle, real_w, real_h):
    """
    将图像转为正并且根据真实的宽高截取出目标
    :param image: 待旋转图像
    :param angle: 转角角度
    :param real_w: 图像真实宽
    :param real_h: 图像真实高
    :return: 旋转并裁剪后图像
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    im = cv2.warpAffine(image, M, (nW, nH))

    # 长的边永远是宽
    if w <= h:
        w, h = h, w

    im = im[int((h - real_h) * 0.5):int((h + real_h) * 0.5), int((w - real_w) * 0.5):int((w + real_w) * 0.5)].copy()

    return im


def get_angle_and_real_width_height(ploy, rect):
    """
    计算多边形转为正的所需的角度、文本真实的宽与高
    :param ploy: 文本检测的多边形
    :param rect: 外接矩形
    :return: 角度、宽、高
    """
    x, y, w, h = rect

    # 为了永远了保证第一个点是最左边的点。（若是矩形，则是左上角的点）
    if ploy[0][0] != x:
        ploy = list(ploy)
        temp = ploy.pop(-1)
        ploy.insert(0, temp)
        ploy = np.array(ploy)
    # 判断外接矩形的长短边
    if w >= h:
        # 判断是顺时针旋转还是逆时针旋转
        if (ploy[3][0] - x) >= (float(w) / 2.0):
            angle = -(math.atan((y + h - ploy[0][1]) / (ploy[3][0] - x)) * 180 / math.pi)
            real_w = ((y + h - ploy[0][1]) ** 2 + (ploy[3][0] - x) ** 2) ** 0.5
            real_h = ((y + h - ploy[2][1]) ** 2 + (x + w - ploy[3][0]) ** 2) ** 0.5
        else:
            angle = math.atan((y + h - ploy[2][1]) / (x + w - ploy[3][0])) * 180 / math.pi
            real_h = ((y + h - ploy[0][1]) ** 2 + (ploy[3][0] - x) ** 2) ** 0.5
            real_w = ((y + h - ploy[2][1]) ** 2 + (x + w - ploy[3][0]) ** 2) ** 0.5
    else:
        if (ploy[0][1] - y) >= (float(h) / 2.0):
            angle = math.atan((y + h - ploy[2][1]) / (x + w - ploy[3][0])) * 180 / math.pi
            real_h = ((y + h - ploy[0][1]) ** 2 + (ploy[3][0] - x) ** 2) ** 0.5
            real_w = ((y + h - ploy[2][1]) ** 2 + (x + w - ploy[3][0]) ** 2) ** 0.5
        else:
            angle = -(math.atan((y + h - ploy[0][1]) / (ploy[3][0] - x)) * 180 / math.pi)
            real_w = ((y + h - ploy[0][1]) ** 2 + (ploy[3][0] - x) ** 2) ** 0.5
            real_h = ((y + h - ploy[2][1]) ** 2 + (x + w - ploy[3][0]) ** 2) ** 0.5
    return angle, real_w, real_h

