# @Time : 2022/5/15 14:24
# @Author : LiangHao
# @File : BaseDataSet.py

import random
import cv2 as cv
import numpy as np
from torch.utils import data

class BaseDataSetImg(data.Dataset):
    def __init__(self, imgList, patchSize=128):
        super(BaseDataSetImg, self).__init__()
        self.imgList = imgList
        self.patchSize = patchSize

    def __len__(self):
        return len(self.imgList)

    def cropPatch(self, img):
        h, w = img.shape[0:2]
        h = max(h, self.patchSize)
        w = max(w, self.patchSize)
        img = cv.resize(src=img, dsize=(w, h))
        ind_h = random.randint(0, h - self.patchSize)
        ind_w = random.randint(0, w - self.patchSize)
        patch = img[ind_h:ind_h + self.patchSize, ind_w:ind_w + self.patchSize]
        return patch