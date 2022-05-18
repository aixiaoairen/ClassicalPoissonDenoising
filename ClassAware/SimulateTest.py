# @Time : 2022/5/15 16:15
# @Author : LiangHao
# @File : SimulateTest.py

import torch
import cv2 as cv
import numpy as np
import DataTool as Tool
from DataSet import SimulateTest
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import img_as_ubyte
from Network.classAwareNet import CADET

def testBatch(model, datasets, batch_size, gpuAcc=True):
    dataloader = DataLoader(dataset=datasets, batch_size=batch_size, shuffle=False)
    if gpuAcc == True :
        model = model.cuda()
    psnr, ssim = 0.0, 0.0
    for idx, data in enumerate(dataloader):
        if gpuAcc == True :
            gt, noisy = [x.cuda() for x in data]
        else:
            gt, noisy = data
        denoise = model(noisy)
        psnr += Tool.batchPSNR(gt, denoise)
        ssim += Tool.batchSSIM(gt, denoise)
    return psnr, ssim

def Main(modelPath, peak, batch_size, nums=160, gpuMode=False, mode=False, depU=None):
    dirPath = "Data/testData"
    datasets = {
        "BSD68" : "*.png"
    }
    model = CADET(in_channels=1, wf=63)
    checkponit = torch.load(modelPath)
    model.load_state_dict(checkponit["model_state_dict"])
    testPaths = np.array(sorted(Tool.get_gt_image(dirPath, datasets)))[: nums]
    testDataSet = SimulateTest(imgList=testPaths, peak=peak, patchSize=128, depU=depU)
    model.eval()
    if mode == False :
        psnrs, ssims = testBatch(model, testDataSet, batch_size, gpuMode)
        avg_psnr = psnrs / nums
        avg_ssim = ssims / nums
        print("AvgPSNR : {} , AvgSSIM : {}".format(avg_psnr, avg_ssim))
    else:
        avgpsnr = 0.0
        avgssim = 0.0
        for i in range(5):
            tmp = testBatch(model, testDataSet, batch_size, gpuMode)
            avgpsnr += tmp[0] / nums
            avgssim += tmp[1] / nums
        print("AvgPSNR : {} , AvgSSIM : {}".format(avgpsnr / 5, avgssim / 5))

if __name__ == '__main__':
    modelPath = "Model/model600.pth"
    Main(modelPath=modelPath, peak=4.0, batch_size=1, nums=68, gpuMode=False, mode=True)