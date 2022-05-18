# @Time : 2022/5/15 14:21
# @Author : LiangHao
# @File : DataSet.py

import torch
import cv2 as cv
import numpy as np
import DataTool as Tool
from BaseDataSet import BaseDataSetImg

class SimulateTrain(BaseDataSetImg):
    def __init__(self, imgList, peak=4.0, patchSize=128):
        super(SimulateTrain, self).__init__(imgList=imgList, patchSize=patchSize)
        self.peak = peak

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, item):
        im_gt = cv.imread(filename=self.imgList[item], flags=1)[:, :, ::-1]
        im_gt = Tool.rgb2ycbcr(im_gt)[:, :, 0]
        im_gt = Tool.crop_patch(im_gt, self.patchSize)
        im_gt = (im_gt / 255.0).astype(np.float32)
        img_peak = im_gt * float(self.peak)
        im_noisy = (np.random.poisson(img_peak) / self.peak).astype(np.float32)
        im_gt, im_noisy = Tool.rand_batch_augmentation(im_gt, im_noisy)
        im_gt, im_noisy = im_gt[:, :, np.newaxis], im_noisy[:, :, np.newaxis]
        im_gt = torch.from_numpy(im_gt.copy().transpose(2, 0, 1))
        im_noisy = torch.from_numpy(im_noisy.copy().transpose(2, 0, 1))
        return im_gt, im_noisy


class SimulateTest(BaseDataSetImg):
    def __init__(self, imgList, peak=4.0, patchSize=128, depU=None):
        super(SimulateTest, self).__init__(imgList=imgList, patchSize=patchSize)
        self.peak = peak
        self.depU = depU

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, item):
        im_gt = cv.imread(filename=self.imgList[item], flags=1)[:, :, ::-1]

        if self.depU != None :
            h, w, _ = im_gt.shape
            if h % 2 ** self.depU != 0 :
                h -= h % 2 ** self.depU
            if w % 2 ** self.depU != 0 :
                w -= w % 2 ** self.depU
            im_gt = im_gt[:h, :w, :]
            if h > w :
                im_gt = np.rot90(im_gt)

        im_gt = Tool.rgb2ycbcr(im_gt)[:, :, 0]
        im_gt = (im_gt / 255.0).astype(np.float32)
        img_peak = im_gt * float(self.peak)
        im_noisy = (np.random.poisson(img_peak) / self.peak).astype(np.float32)
        im_gt, im_noisy = im_gt[:, :, np.newaxis], im_noisy[:, :, np.newaxis]
        im_gt = torch.from_numpy(im_gt.copy().transpose(2, 0, 1))
        im_noisy = torch.from_numpy(im_noisy.copy().transpose(2, 0, 1))
        return im_gt, im_noisy


