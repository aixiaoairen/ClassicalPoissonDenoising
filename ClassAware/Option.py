# @Time : 2022/5/15 14:46
# @Author : LiangHao
# @File : Option.py

import os
import argparse

def option():
    parser = argparse.ArgumentParser()
    # Path to Save some files
    parser.add_argument('--model_save_dir', type=str, default="Model",
                        help='The Path to save training model')
    parser.add_argument('--model_resume_dir', type=str, default="Model_Resume")
    parser.add_argument('--trainDataDir', type=str, default="Data/trainData")
    parser.add_argument('--trainDataSets', type=dict, default={
                              "JPEGImages": "*.jpg"
                                },
                        help="The Name and type of DataSets")
    parser.add_argument('--testDataDir', type=str, default="Data/testData")
    parser.add_argument('--testDataSets', type=dict, default={
                              "BSDS500": "*.jpg"
                                })
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--trainNum', type=int, default=3000)
    parser.add_argument('--testNum', type=int, default=30)
    # Make DataSet and Load data to memory
    parser.add_argument('--patch_size', type=int, default=128,
                        help="The Size of Cropping Images (default: 128)")
    parser.add_argument('--num_worker', type=int, default=1,
                        help="The Number of Process")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch Size of Training (default: 64)")
    parser.add_argument('--peak', type=float, default=4.0, help="the peak value of poisson noise")
    # Optimizer
    parser.add_argument('--milestones', type=list, default=[10, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Initialized Learning Rate (default: 1e-4)")
    parser.add_argument('--gamma', type=float, default=0.5)
    # Training
    parser.add_argument('--in_channel', type=int, default=1,
                        help="The Numbers of Channels")
    parser.add_argument('--wf', type=int, default=63,
                        help="The Num of Convolution Kernel")
    parser.add_argument('--epochs', type=int, default=600,
                        help="Train Epochs (default: 120)")
    parser.add_argument('--resume', type=bool, default=False,
                        help="Judge Whether The Training")
    parser.add_argument('--checkpoint', type=str, default="model40.pth",
                        help="The File Name of checkpoint which had been trained")
    args = parser.parse_args(args=[])
    return args