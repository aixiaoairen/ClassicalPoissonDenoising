# ClassicalPoissonDenoising

some classical methods for poisson denoising 

I am trying to use python to run those methods, including traditional and deep learning methods.
## 1. ClassAware
### 1.1 Relative Information
[Class-aware fully convolutional Gaussian and Poisson denoising](https://ieeexplore.ieee.org/abstract/document/8418389/)
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
### 1.2 How to run codes
- About Training : python SimulateTrain.py
- About Testing : python SimulateTest.py

**You can modify file named 'option.py' to set super parameters about epochs, peak, file paths et.al**.

## 2. Iteration + VST+BM3D (PythonVersion) 
### 2.1 Relative Information
[Variance Stabilization for Noisy+Estimate Combination in Iterative Poisson Denoising](https://ieeexplore.ieee.org/abstract/document/7491301)
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
### 2.2 How to run codes
modifying file named "pythonversion/pythondemo.py"

Matlab writes basic denoising codes, and Datasets making are written by python.
**Note**: You should install "~/Matlab/extern/engines/python/steup.py". This is an essential step for the combination of python and Matlab

## 5. nlmsd
### 5.1 Relative Information
[A new bayesian Poisson denoising algorithm based on nonlocal means 
and stochastic distances](https://www.sciencedirect.com/science/article/abs/pii/S0031320321005434)
@article{evangelista2022new,
  title={A new bayesian Poisson denoising algorithm based on nonlocal means and stochastic distances},
  author={Evangelista, Rodrigo C and Salvadeo, Denis HP and Mascarenhas, Nelson DA},
  journal={Pattern Recognition},
  volume={122},
  pages={108363},
  year={2022},
  publisher={Elsevier}
}
### 5.2  How to run codes

run a file named "pythonversion.py"
Matlab writes basic denoising codes, and Datasets making are written by python.
**Note**: You should install "~/Matlab/extern/engines/python/steup.py". This is an essential step for the combination of python and Matlab
