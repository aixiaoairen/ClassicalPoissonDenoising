# ClassicalPoissonDenoising

some classical methods for poisson denoising 

I am trying to use python to run those methods, including traditional and deep learning methods.
## 1. ClassAware
### 1.1 Relative Information
[Class-aware fully convolutional Gaussian and Poisson denoising](https://ieeexplore.ieee.org/abstract/document/8418389/)
```
@article{remez2018class,
  title={Class-aware fully convolutional Gaussian and Poisson denoising},
  author={Remez, Tal and Litany, Or and Giryes, Raja and Bronstein, Alex M},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={11},
  pages={5707--5722},
  year={2018},
  publisher={IEEE}
}
```
### 1.2 How to run codes
- About Training : python SimulateTrain.py
- About Testing : python SimulateTest.py

**You can modify file named 'option.py' to set super parameters about epochs, peak, file paths et.al**.

## 2. Iteration + VST+BM3D (PythonVersion) 
### 2.1 Relative Information
[Variance Stabilization for Noisy+Estimate Combination in Iterative Poisson Denoising](https://ieeexplore.ieee.org/abstract/document/7491301)
```
@article{azzari2016variance,
  title={Variance stabilization for noisy+ estimate combination in iterative poisson denoising},
  author={Azzari, Lucio and Foi, Alessandro},
  journal={IEEE signal processing letters},
  volume={23},
  number={8},
  pages={1086--1090},
  year={2016},
  publisher={IEEE}
}
```
### 2.2 How to run codes
modifying file named "pythonversion/pythondemo.py"

Matlab writes basic denoising codes, and Datasets making are written by python.
**Note**: You should install "~/Matlab/extern/engines/python/steup.py". This is an essential step for the combination of python and Matlab

## 3. snldp
### 3.1 Relative Information
[Simultaneous Nonlocal Low-Rank And Deep Priors For Poisson Denoising](https://ieeexplore.ieee.org/abstract/document/9746870/)
```
@inproceedings{zha2022simultaneous,
  title={Simultaneous Nonlocal Low-Rank And Deep Priors For Poisson Denoising},
  author={Zha, Zhiyuan and Wen, Bihan and Yuan, Xin and Zhou, Jiantao and Zhu, Ce},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2320--2324},
  year={2022},
  organization={IEEE}
}
```
### 3.2 How to run
I modify some codes about image sets and peak-setting. **I did not make any changes to the core algorithm**.

#### 1. Change Test Images Sets
- Put your test images under a folder named "Test_Images"
- Modify the variables named **nums, diroutput** in "SNLDP_Poisson_Denoising_Demo.m" line 83, 85
- Otherwise : you could note "SNLDP_Poisson_Test.m"
#### 2. Run (CPU Mode)
In Matlab command, Input:
```matlab
run matconvnet-1.0-beta25\matlab\vl_setupnn;
mex -setup ; % choose mex -setup C++
vl_compilenn ;
```
## 5. nlmsd
### 5.1 Relative Information
[A new bayesian Poisson denoising algorithm based on nonlocal means 
and stochastic distances](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005434)
```
@article{evangelista2022new,
  title={A new bayesian Poisson denoising algorithm based on nonlocal means and stochastic distances},
  author={Evangelista, Rodrigo C and Salvadeo, Denis HP and Mascarenhas, Nelson DA},
  journal={Pattern Recognition},
  volume={122},
  pages={108363},
  year={2022},
  publisher={Elsevier}
}
```
### 5.2  How to run codes

run a file named "pythonversion.py"
Matlab writes basic denoising codes, and Datasets making are written by python.
**Note**: You should install "~/Matlab/extern/engines/python/steup.py". This is an essential step for the combination of python and Matlab
