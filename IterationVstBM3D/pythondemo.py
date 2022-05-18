import os
import glob
from queue import PriorityQueue
import cv2 as cv
import numpy as np
import matlab.engine
from pyparsing import nums
from skimage import  img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt


writer = SummaryWriter("log")

def show(*args):
    plt.subplot(131), plt.imshow(args[0], 'gray'), plt.title("Groud Truth")
    plt.subplot(132), plt.imshow(args[1], 'gray'), plt.title("Noisy Image")
    plt.subplot(133), plt.imshow(args[2], 'gray'), plt.title("Denosing Image")
    

def get_gt_image(folder_path, trainsets):
    """
    @Brief: combine images(path) in different folders together
    args:
        folder_path: common floder. Example: "/home/liang/Public/MyWork/data/train"
        trainsets: subset folder. Example : { "BSDS200" : "*.png","T91" : "*.png"}
    return: [
                folder_path/trainsets[0].key()/*.png,
                folder_path/trainsets[1].ket()/*.png,
                ....
                folder_path/trainsets[n].key()/*.png
            ]
    """
    try:
        train_images_path = []
        for key, val in trainsets.items():
            pathName = os.path.join(folder_path, key, val)
            for img in glob.glob(pathname=pathName, recursive=True):
                train_images_path.append(img)
    except:
        print("Erroe in path given")

    return train_images_path

def rgb2ycbcr(rgb_img):
    """
    将RGB图像转为灰度图像
    gray = r * 0.299 + g * 0.587 + b * 0.114
    :param rgb_img: RGB img, dtype=float32, [0.0, 1.0]
    :return: gray 灰度图像， dtype=float32, [0.0, 1.0]
    """
    r, g, b = rgb_img[:, :, 0 : 1], rgb_img[:, :, 1 : 2], rgb_img[:, :, 2 : 3]
    gray = r * 0.299 + g * 0.587 + b * 0.114
    return gray

def vstBm3d(noisy, K):
    eng = matlab.engine.start_matlab()
    noisy = noisy.tolist()
    denoise = eng.demo_python(matlab.double(noisy), K)
    return np.array(denoise)

def genPoisson(im_gt, peak):
    im_noisy = np.random.poisson(im_gt * peak) / peak
    im_noisy = im_noisy.astype(np.float32)
    return im_noisy

def TestOne(imgPath, peak, k):
    # img = cv.imread(imgPath, 0)[:, :, ::-1]
    # im_gt = rgb2ycbcr(img)[:, :, 0]
    """for bsd68"""
    im_gt = cv.imread(imgPath)[:, :, 0]
    im_gt = (im_gt / 255.0).astype(np.float32)
    # im_gt = (im_gt / im_gt.max()).astype(np.float32)
    im_noisy = genPoisson(im_gt, peak)
    im_denoise = vstBm3d(im_noisy, k)
    # show(im_gt, im_noisy, im_denoise)
    # 更改像素范围
    im_gt = img_as_ubyte(im_gt)
    im_denoise = img_as_ubyte(np.array(im_denoise).clip(0, 1))
    psnr = compare_psnr(im_gt, im_denoise, data_range=255)
    ssim = compare_ssim(im_gt, im_denoise, data_range=255, multichannel=False)
    return psnr, ssim

def TestBench(paths, nums, peak, k):
    avg_psnr = 0.0
    avg_ssim = 0.0
    for i in range(0, nums):
        tpsnr, tssim = TestOne(paths[i], peak, k)
        writer.add_scalar(tag="PSNR", scalar_value=tpsnr, global_step=i + 1)
        writer.add_scalar(tag="SSIM", scalar_value=tssim, global_step=i + 1)
        avg_psnr += tpsnr
        avg_ssim += tssim
    avg_psnr = avg_psnr / nums
    avg_ssim = avg_ssim / nums
    print("AvgPSNR : {} , AvgSSIM : {}".format(avg_psnr, avg_ssim))
    
def main():
    k = 1
    nums = 68
    peak = 1.0
    datasets = {
        'BSD68' : '*.png'
    }
    dirpath = "testdata"
    paths = np.array(sorted(get_gt_image(dirpath, datasets)))
    TestBench(paths, nums, peak, k)



if __name__ == "__main__":
    # k = 1
    # nums = 1
    # peak = 4
    # datasets = {
    #     'BSDS500' : '*.jpg'
    # }
    # dirpath = "testdata"
    # paths = np.array(sorted(get_gt_image(dirpath, datasets)))
    # psnr , ssim = TestOne(paths[5], peak, k)
    # plt.text(-800, 600, "psnr : {0}, ssim :  {1}".format(psnr, ssim))
    # plt.show()
    main()
