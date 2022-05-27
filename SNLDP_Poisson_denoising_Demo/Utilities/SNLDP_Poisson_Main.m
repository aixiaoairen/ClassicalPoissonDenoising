function  [HPnP_Results, k]                =                         SNLDP_Poisson_Main ( Opts)


y                                             =                         Opts.nim;


v                                             =                         Opts.nSig;


rec_im                                        =                         y;     % Inital image


[h, w]                                        =                        size(rec_im);  


mu                                            =                        Opts. mu;


alpha                                         =                        Opts. alpha;

beta                                          =                        Opts. beta;


J                                             =                         zeros( size(rec_im) );   %J : multiplier for D

G                                             =                         zeros( size(rec_im) );   %G : multiplier for WNNM


All_PSNR                                      =                         zeros(1,Opts.Iter);

HPnP_Results                                  =                         cell (1,Opts.Iter);  



for   k = 1  :   Opts.Iter
    
    
    
       f                                        =                         rec_im;  

     dif                                        =                         f - y;
    
     vd                                         =                      v^2-(mean(mean(dif.^2)));
    
    if (k ==1)
        
    Opts.nSig                                   =                       sqrt(abs(vd));
    
    else
        
    Opts.nSig                                   =                       sqrt(abs(vd))*Opts.gamma;
        
    end        
       
       
       
   % Deep Prior
   
      D                                         =                       FFD_Net_Denoiser_Poisson ((f + J/alpha), Opts); % 90000*31
      
   % Low-Rank    
    
     [rim, wei]                                 =                       LR_WNNM_Main(f, Opts);         
     
     
      rim                                       =                      (rim+beta*mu*Opts.nSig^2*f + mu*Opts.nSig^2*G)./(wei+beta*mu*Opts.nSig^2); % rim is 90000*31
      
  % x sub-problem    
  
       b_v                                      =                   (2*alpha*v^2* D + 2*beta*v^2*rim - 2*v^2*J- 2*v^2*G- 1);
       
       f                                        =                   (b_v+ sqrt(b_v.^2 +8* (alpha*v^2+beta*v^2)*y))/ (4*(alpha*v^2+beta*v^2));
       
       
      
      J                                          =                     J + 1.05*alpha*(f - D);% 1.1  
     
      
     G                                           =                     G + 1.05*beta*(f - rim);% 1.1       
    
     rec_im                                      =                     f;
     
      aa =         rec_im*max(Opts.x(:))/Opts.p_noise;
       
    
    All_PSNR(k)    =   csnr (aa, Opts.x,0,0);
    
    
    HPnP_Results{k}            =                        rec_im;
    
    
    fprintf( 'Iter = %d,  nSig = %2.2f, PSNR = %2.2f\n', k, Opts.nSig, All_PSNR(k));  
    

    
    if k>5
        
   %  diff      =  norm(abs(HPnP_Results{k}) - abs(HPnP_Results{k-1}),'fro')/norm(abs(HPnP_Results{k-1}), 'fro');
    
    if All_PSNR(k) -All_PSNR(k-1) <=0
        break;
    end
    end   
    
end






end

