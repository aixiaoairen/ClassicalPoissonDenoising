import os
import glob

from matplotlib.pyplot import show
import matlab.engine
import cv2 as cv
import numpy as np

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage import img_as_ubyte

import matplotlib.pyplot as plt


def show(*args):
    plt.subplot(131), plt.imshow(args[0], 'gray'), plt.title("Groud Truth")
    plt.subplot(132), plt.imshow(args[1], 'gray'), plt.title("Noisy Image")
    plt.subplot(133), plt.imshow(args[2], 'gray'), plt.title("Denosing Image")

def get_gt_images(folder_path, trainsets):
    """combine image paths in different folders """
    try:
        train_image_paths = []
        for key, val in trainsets.items():
            pathName = os.path.join(folder_path, key, val)
            for img in glob.glob(pathname=pathName, recursive=True):
                train_image_paths.append(img)
    except:
        print("Error in path given")
    return train_image_paths

def rgb2gray(rgb_img):
    r, g, b = rgb_img[:, :, 0 : 1], rgb_img[:, :, 1 : 2], rgb_img[:, :, 2 : 3]
    gray = r * 0.299 + g * 0.587 + b * 0.114
    return gray

def nlmSto(noisy):
    eng = matlab.engine.start_matlab()
    noisy = noisy.tolist()
    matlab_noisy = matlab.double(noisy)
    denoise = eng.nlmsdPoisson(matlab_noisy)
    return np.array(denoise)
    
def genPoisson(im_gt, peak):
    im_noisy = np.random.poisson(im_gt * peak) / float(peak)
    im_noisy = im_noisy.astype(np.float32)
    return im_noisy

def TestOne(imgPath, peak):
    im_gt = cv.imread(imgPath)[:, :, ::-1]
    # [0, 1]的单通道图像 (h, w)
    im_gt = rgb2gray(im_gt)[:, :, 0]
    im_gt = (im_gt / 255.0).astype(np.float32)
    im_noisy = genPoisson(im_gt, peak)
    im_denoise = nlmSto(im_noisy)
    show(im_gt, im_noisy, im_denoise)
    im_gt = img_as_ubyte(im_gt)
    im_denoise = img_as_ubyte(im_denoise.clip(-1, 1))
    psnr = compare_psnr(image_true=im_gt, image_test=im_denoise, data_range=255)
    ssim = compare_ssim(im1=im_gt, im2=im_denoise, data_range=255, multichannel=False)
    return psnr, ssim

def TestBench(paths, nums, peak):
    avg_psnr = 0.0
    avg_ssim = 0.0
    for i in range(0, nums):
        tpsnr, tssim = TestOne(paths[i], peak)
        avg_psnr += tpsnr
        avg_ssim += tssim
    avg_psnr = avg_psnr / nums
    avg_ssim = avg_ssim / nums
    print("AvgPSNR : {} , AvgSSIM : {}".format(avg_psnr, avg_ssim))

def main(peak):
    nums = 1
    dirpath = "testData"
    datasets = {
        'tmpTest' : '*.jpg'
    }
    paths = np.array(sorted(get_gt_images(dirpath, datasets)))
    TestBench(paths, nums, peak)

if __name__ == '__main__':
    main(peak=20.0)
    plt.show()