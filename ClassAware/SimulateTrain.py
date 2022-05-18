# @Time : 2022/5/15 14:55
# @Author : LiangHao
# @File : SimulateTrain.py

import os
import torch
import numpy as np
import Option as op
import torch.nn as nn
import DataTool as Tool
import torch.optim as optim
from DataSet import SimulateTrain
from Utils import weight_init_kaiming
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from Network.classAwareNet import CADET


args = op.option()
writer = SummaryWriter("runlog")

def Train(model, trainDataSets, testDataSets, optimizer, criterion, scheduler):
    trainDataLoader = DataLoader(dataset=trainDataSets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)
    testDataLoader = DataLoader(dataset=testDataSets, batch_size=1, shuffle=True, num_workers=args.num_worker)
    testNums = testDataSets.__len__()
    trainTotal = 0
    for epoch in range(args.epochs):
        model.train()
        for idx, data in enumerate(trainDataLoader):
            gt, noisy = [x.cuda() for x in data]
            optimizer.zero_grad()
            denoise = model(noisy)
            loss = criterion(gt, denoise)
            loss.backward()
            optimizer.step()
            if idx % (len(trainDataLoader) // 2) == 0:
                print("epoch= {0}/{1}, {2}/{3} of train, loss = {4}".format(epoch,
                                                                            args.epochs, idx, len(trainDataLoader), loss.item()))
            writer.add_scalar("Train/Loss", loss.item(), trainTotal)
            trainTotal += 1

        model.eval()
        psnr = 0.0
        ssim = 0.0
        for idx, data in enumerate(testDataLoader):
            gt, noisy = [x.cuda() for x in data]
            denoise = model(noisy)
            psnr += Tool.batchPSNR(gt, denoise)
            ssim += Tool.batchSSIM(gt, denoise)
        writer.add_scalar("Validation / PSNR", psnr / testNums, epoch)
        writer.add_scalar("Validation / SSIM", ssim / testNums, epoch)

        scheduler.step()
        if epoch == args.epochs - 1 or (epoch + 1) % args.interval == 0 :
            torch.save({
                'model_state_dict': model.state_dict()
            }, os.path.join(args.model_save_dir, 'model{}.pth'.format(str(epoch + 1).zfill(2))))

def Main():
    model = CADET(in_channels=args.in_channel, wf=args.wf)
    if args.resume == False:
        model = weight_init_kaiming(model)
    else:
        modelPath = os.path.join(args.model_resume_dir, args.checkpoint)
        checkpoint = torch.load(modelPath)
        state_dict_one = model.state_dict()
        for name, value in checkpoint["model_state_dict"].items():
            state_dict_one[name] = value
        model.load_state_dict(state_dict_one)
    if os.path.exists(args.model_save_dir) == False:
        os.mkdir(args.model_save_dir)
    model = model.cuda()
    trainPaths = np.array(sorted(Tool.get_gt_image(args.trainDataDir, args.trainDataSets)))[ : args.trainNum]
    testPaths = np.array(sorted(Tool.get_gt_image(args.testDataDir, args.testDataSets)))[: args.testNum]
    trainDataSets = SimulateTrain(imgList=trainPaths, peak=args.peak, patchSize=args.patch_size)
    testDataSets = SimulateTrain(imgList=testPaths, peak=args.peak, patchSize=args.patch_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, args.milestones, args.gamma)
    print("Begin Training  ...")
    Train(model, trainDataSets, testDataSets, optimizer, criterion, scheduler)
    print("End Training ...")

if __name__ == '__main__':
    Main()