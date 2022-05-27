function [filename, p_noise, Inital_SNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final,Time_s]    =  SNLDP_Poisson_Test (filename, p_noise)

 
  randn ('seed',0);
  
 rand ('seed',0);

time0                            =                          clock;
                 
fn                               =                          [filename, '.tif'];

I                                =                           imread(fn);
                
 [~, ~, kk]                      =                           size (I);
         
 colorI                          =                           I;

if kk==3
    
I                                =                          rgb2gray (I);
          
y_yuv                            =                          rgb2ycbcr(colorI);                     
                     
x_2_yuv(:,:,2)                   =                          y_yuv(:,:,2); % Copy U Componet
                  
x_2_yuv(:,:,3)                   =                          y_yuv(:,:,3); % Copy V Componet                   

end



[Opts]                            =                        SNLDP_Poisson_Set ( I, p_noise);
              
              
% Generate Poisson Noise

x                                 =                       Opts. I;

x_ori                             =                        x/max(x(:))*p_noise;


y                                 =                        poissrnd(x_ori); 


nSig                              =                       ImageStdDev(y);


Inital_SNR                        =                       SNR (y,x_ori)


Opts. x_ori                       =                        x_ori;
 
Opts. nim                         =                        double(y);

Opts. nSig                        =                        nSig;

Opts. p_noise                        =                        p_noise;

Opts. x                        =                        x;
 
 
 
 [SNLDP_Results, iter]                 =                         SNLDP_Poisson_Main ( Opts);   
 

im   = SNLDP_Results{iter-1};


Final_SNR                        =                       SNR (im,x_ori);




      im =         im*max(x(:))/p_noise;
 
 
             
             Time_s                      =                          (etime(clock,time0));                          
                
              PSNR_Final                 =                        csnr (im, x,0,0);

              SSIM_Final                 =                          cal_ssim (im, x,0,0);
              
              FSIM_Final                 =                          FeatureSIM(im, x);
              
              
           
      
  
              
              
              
              
            if kk  ==3
              
               x_2_yuv(:,:,1)            =                          uint8(im);
               
            HPnP_Re                  =                          ycbcr2rgb(uint8(x_2_yuv));   
            end      
                    
              
              

if p_noise==5
    

Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');


if kk==3

imwrite(uint8(HPnP_Re),strcat('./5_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./5_Result/',Final_Optsisng));

end




elseif p_noise==10

Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./10_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./10_Result/',Final_Optsisng));

end


elseif p_noise== 15

Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./15_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./15_Result/',Final_Optsisng));

end


elseif p_noise==20

Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./20_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./20_Result/',Final_Optsisng));

end


elseif p_noise==25
    
Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./25_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./25_Result/',Final_Optsisng));

end

elseif p_noise==30
    
Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./30_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./30_Result/',Final_Optsisng));

end


elseif p_noise==35
    
Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./35_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./35_Result/',Final_Optsisng));

end



elseif p_noise==40
    
Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./40_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./40_Result/',Final_Optsisng));

end




elseif p_noise==45
    
Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./45_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./45_Result/',Final_Optsisng));

end




elseif p_noise==50
    
Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./50_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./50_Result/',Final_Optsisng));

end




elseif p_noise==55
    
Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./55_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./55_Result/',Final_Optsisng));

end




elseif p_noise==60
    
Final_Optsisng= strcat(filename,'_SNLDP_P_noise_',num2str(p_noise),'_SNR_',num2str(Final_SNR), '_PSNR_',num2str(PSNR_Final),'_FSIM_',num2str(FSIM_Final), '_SSIM_',num2str(SSIM_Final),'.png');

if kk==3

imwrite(uint8(HPnP_Re),strcat('./60_Result/',Final_Optsisng));

else
imwrite(uint8(im),strcat('./60_Result/',Final_Optsisng));

end







end              
              
end

