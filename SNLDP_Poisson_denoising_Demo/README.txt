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
