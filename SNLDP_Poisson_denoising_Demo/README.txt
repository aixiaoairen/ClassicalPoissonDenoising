# Original Author
This demo is using to SNLDP-based Poisson denoising.

When you use this demo, please first load software package "matconvnet", and make sure
run the function "vl_compilenn.m" succesfully.

Later, you can directly run the function "SNLDP_Poisson_Denoising_Demo.m" 
Currently, this demo can be directly runned by using any CPU.


Note that:

If you cannot conduct the function "vl_compilenn.m" succesfully.

Please do the following operator:

In function "FFD_Net_Denoiser_Poisson.m", 
please transform " res     =   vl_ffdnet_concise(net, input);"
into "res    = vl_ffdnet_matlab(net, input);".

but this operator is very slow.

# Mine
I modify some codes about image sets and peak-setting. **I did not make any changes to the core algorithm**.

## How to run
### 1. Change Test Images Sets
- Put your test images under a folder named "Test_Images"
- Modify the variables named **nums, diroutput** in "SNLDP_Poisson_Denoising_Demo.m" line 83, 85
- Otherwise : you could note "SNLDP_Poisson_Test.m"
### 2. Run (CPU Mode)
In Matlab command, Input:
```matlab
run matconvnet-1.0-beta25\matlab\vl_setupnn;
mex -setup ; % choose mex -setup C++
vl_compilenn ;
```
