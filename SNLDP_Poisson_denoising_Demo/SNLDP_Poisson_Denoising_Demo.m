clc
clear
m_10_1=0; 
m_10_2=0; 
m_10_3=0; 
m_20_1=0; 
m_20_2=0; 
m_20_3=0; 
m_30_1=0; 
m_30_2=0; 
m_30_3=0; 
m_40_1=0;
m_40_2=0; 
m_40_3=0; 
m_50_1=0;
m_50_2=0; 
m_50_3=0; 
m_75_1=0;
m_75_2=0; 
m_75_3=0; 
m_100_1=0;
m_100_2=0; 
m_100_3=0; 


m_110_1=0;
m_110_2=0; 
m_110_3=0; 


m_120_1=0;
m_120_2=0; 
m_120_3=0;

m_130_1=0;
m_130_2=0; 
m_130_3=0;


All_data_Results_10_1 = cell(1,200);
All_data_Results_10_2 = cell(1,200);
All_data_Results_10_3 = cell(1,200);

All_data_Results_20_1 = cell(1,200);
All_data_Results_20_2 = cell(1,200);
All_data_Results_20_3 = cell(1,200);

All_data_Results_30_1 = cell(1,200);
All_data_Results_30_2 = cell(1,200);
All_data_Results_30_3 = cell(1,200);

All_data_Results_40_1 = cell(1,200);
All_data_Results_40_2 = cell(1,200);
All_data_Results_40_3 = cell(1,200);

All_data_Results_50_1 = cell(1,200);
All_data_Results_50_2 = cell(1,200);
All_data_Results_50_3 = cell(1,200);

All_data_Results_75_1 = cell(1,200);
All_data_Results_75_2 = cell(1,200);
All_data_Results_75_3 = cell(1,200);

All_data_Results_100_1 = cell(1,200);
All_data_Results_100_2 = cell(1,200);
All_data_Results_100_3 = cell(1,200);


All_data_Results_110_1 = cell(1,200);
All_data_Results_110_2 = cell(1,200);
All_data_Results_110_3 = cell(1,200);

All_data_Results_120_1 = cell(1,200);
All_data_Results_120_2 = cell(1,200);
All_data_Results_120_3 = cell(1,200);

All_data_Results_130_1 = cell(1,200);
All_data_Results_130_2 = cell(1,200);
All_data_Results_130_3 = cell(1,200);



for i = 1:8
        
ImageNum =i;

switch ImageNum
          case 1
                filename = 'Baboon';
            case 2
                filename = 'Barbara';
            case 3
                filename = 'boats';
            case 4
                filename = 'flag';    
            case 5
                filename = 'J.Bean'; 
           case 6
                filename = 'peppers';
            case 7
                filename = 'straw';
            case 8
                filename = 'tank';
end
for j  =  2:8

        filename
        
    
randn ('seed',0);

Poisson_Noise_Level            =                        [5, 10, 15, 20, 30, 40, 50, 60];


p_noise             =       Poisson_Noise_Level(j);




 if  p_noise<= 5
     

 [filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final, Time_s]     =  SNLDP_Poisson_Test (filename, p_noise);
 

 m_10_1= m_10_1+1;
 
 s=strcat('A',num2str(m_10_1));
 
 All_data_Results_10_1{m_10_1}={filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final, FSIM_Final,SSIM_Final, Time_s};
 

 xlswrite('SNLDP_Poisson_5.xls', All_data_Results_10_1{m_10_1},'sheet1',s);

 
 
 elseif  p_noise<=10
     
     
  [filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final, Time_s]     =  SNLDP_Poisson_Test (filename, p_noise);
 
 m_20_1= m_20_1+1;
 
 s=strcat('A',num2str(m_20_1));
 
 All_data_Results_20_1{m_20_1}={filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final, FSIM_Final,SSIM_Final, Time_s};
 
xlswrite('SNLDP_Poisson_10.xls', All_data_Results_20_1{m_20_1},'sheet1',s);

 
 
  elseif  p_noise<=15
     
     
  [filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final, Time_s]     =  SNLDP_Poisson_Test (filename, p_noise);
 
 m_30_1= m_30_1+1;
 
 s=strcat('A',num2str(m_30_1));
 
 All_data_Results_30_1{m_30_1}={filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final, FSIM_Final,SSIM_Final, Time_s};
 
xlswrite('SNLDP_Poisson_15.xls', All_data_Results_30_1{m_30_1},'sheet1',s);


   elseif  p_noise<=20
     
     
  [filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final, Time_s]     =  SNLDP_Poisson_Test (filename, p_noise);
 
 m_40_1= m_40_1+1;
 
 s=strcat('A',num2str(m_40_1));
 
 All_data_Results_40_1{m_40_1}={filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final, FSIM_Final,SSIM_Final, Time_s};
 
 xlswrite('SNLDP_Poisson_20.xls', All_data_Results_40_1{m_40_1},'sheet1',s);

 elseif  p_noise<=30

     
  [filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final, Time_s]     =  SNLDP_Poisson_Test (filename, p_noise);
 
 m_50_1= m_50_1+1;
 
 s=strcat('A',num2str(m_50_1));
 
 All_data_Results_50_1{m_50_1}={filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final, FSIM_Final,SSIM_Final, Time_s};
 
 xlswrite('SNLDP_Poisson_30.xls', All_data_Results_50_1{m_50_1},'sheet1',s);

 elseif  p_noise<=40
     

  [filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final, Time_s]     =  SNLDP_Poisson_Test (filename, p_noise);
 
 m_75_1= m_75_1+1;
 
 s=strcat('A',num2str(m_75_1));
 
 All_data_Results_75_1{m_75_1}={filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final, FSIM_Final,SSIM_Final, Time_s};
 
 xlswrite('SNLDP_Poisson_40.xls', All_data_Results_75_1{m_75_1},'sheet1',s);


 elseif  p_noise<=45
     
 
  [filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final, Time_s]     =  SNLDP_Poisson_Test (filename, p_noise);
 
 m_100_1= m_100_1+1;
 
 s=strcat('A',num2str(m_100_1));
 
 All_data_Results_100_1{m_100_1}={filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final, FSIM_Final,SSIM_Final, Time_s};
 
 xlswrite('SNLDP_Poisson_45.xls', All_data_Results_100_1{m_100_1},'sheet1',s);
     
 elseif  p_noise<=50
     

  [filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final, Time_s]     =  SNLDP_Poisson_Test (filename, p_noise);
 
 m_110_1= m_110_1+1;
 
 s=strcat('A',num2str(m_110_1));
 
 All_data_Results_110_1{m_110_1}={filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final, FSIM_Final,SSIM_Final, Time_s};
 
 xlswrite('SNLDP_Poisson_50.xls', All_data_Results_110_1{m_110_1},'sheet1',s);
 
   
 elseif  p_noise<=55
     

  [filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final, Time_s]     =  SNLDP_Poisson_Test (filename, p_noise);
 
 m_120_1= m_120_1+1;
 
 s=strcat('A',num2str(m_120_1));
 
 All_data_Results_120_1{m_120_1}={filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final, FSIM_Final,SSIM_Final, Time_s};
 
 xlswrite('SNLDP_Poisson_55.xls', All_data_Results_120_1{m_120_1},'sheet1',s);
 

   elseif  p_noise<=60
     

  [filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final,FSIM_Final,SSIM_Final, Time_s]     =  SNLDP_Poisson_Test (filename, p_noise);
 
 m_130_1= m_130_1+1;
 
 s=strcat('A',num2str(m_130_1));
 
 All_data_Results_130_1{m_130_1}={filename, p_noise, Inital_PSNR, Final_SNR, PSNR_Final, FSIM_Final,SSIM_Final, Time_s};
 
 xlswrite('SNLDP_Poisson_60.xls', All_data_Results_130_1{m_130_1},'sheet1',s);

 end
 

clearvars -except filename i m_10_1 m_10_2 m_10_3 m_20_1 m_20_2 m_20_3 m_30_1 m_30_2 m_30_3 m_40_1 m_40_2 m_40_3...
m_50_1 m_50_2 m_50_3 m_75_1 m_75_2 m_75_3 m_100_1 m_100_2 m_100_3 All_data_Results_10_1...
All_data_Results_10_2 All_data_Results_10_3 All_data_Results_20_1 All_data_Results_20_2...
All_data_Results_20_3 All_data_Results_30_1 All_data_Results_30_2 All_data_Results_30_3...
All_data_Results_40_1 All_data_Results_40_2 All_data_Results_40_3 All_data_Results_50_1...
All_data_Results_50_2 All_data_Results_50_3 All_data_Results_75_1 All_data_Results_75_2...
All_data_Results_75_3 All_data_Results_100_1 All_data_Results_100_2 All_data_Results_100_3...
m_110_1 m_110_2 m_110_3  m_120_1 m_120_2 m_120_3 m_130_1 m_130_2 m_130_3...
All_data_Results_110_1 All_data_Results_110_2 All_data_Results_110_3...
All_data_Results_120_1 All_data_Results_120_2 All_data_Results_120_3...
All_data_Results_130_1 All_data_Results_130_2 All_data_Results_130_3
end
clearvars -except filename m_10_1 m_10_2 m_10_3 m_20_1 m_20_2 m_20_3 m_30_1 m_30_2 m_30_3 m_40_1 m_40_2 m_40_3...
m_50_1 m_50_2 m_50_3 m_75_1 m_75_2 m_75_3 m_100_1 m_100_2 m_100_3 All_data_Results_10_1...
All_data_Results_10_2 All_data_Results_10_3 All_data_Results_20_1 All_data_Results_20_2...
All_data_Results_20_3 All_data_Results_30_1 All_data_Results_30_2 All_data_Results_30_3...
All_data_Results_40_1 All_data_Results_40_2 All_data_Results_40_3 All_data_Results_50_1...
All_data_Results_50_2 All_data_Results_50_3 All_data_Results_75_1 All_data_Results_75_2...
All_data_Results_75_3 All_data_Results_100_1 All_data_Results_100_2 All_data_Results_100_3...
m_110_1 m_110_2 m_110_3  m_120_1 m_120_2 m_120_3 m_130_1 m_130_2 m_130_3...
All_data_Results_110_1 All_data_Results_110_2 All_data_Results_110_3...
All_data_Results_120_1 All_data_Results_120_2 All_data_Results_120_3...
All_data_Results_130_1 All_data_Results_130_2 All_data_Results_130_3
end






         