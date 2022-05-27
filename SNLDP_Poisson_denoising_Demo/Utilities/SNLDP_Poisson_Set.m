
function  [ Opts]   =    SNLDP_Poisson_Set(  I, p_noise)

randn ('seed',0);


Opts.I         =   double(I);

Opts.M          =                          max(Opts.I(:));

Opts.p_noise         =   p_noise;

Opts.Iter      =   100;

Opts.step      =   4;         

Opts.patch     =  7;   

Opts.Sim       =  80; 


Opts.e         = 0.35;


if p_noise <= 5
   
      
    Opts.gamma      =  0.08;  
    
    Opts. mu        =   0.2;  
    
    Opts. alpha      =   0.001; 
    
    Opts. beta       =   0.03;     

    Opts. c          =   1.3;    
    
elseif p_noise <= 10
    
    Opts.gamma      =  0.1;  
    
    Opts. mu        =   0.2;  
    
    Opts. alpha      =  0.0006; 
    
    Opts. beta       =  0.009;     

    Opts. c          =    0.9;     
     
    
elseif p_noise <= 15
    

    
    Opts.gamma      =  0.1;  
    
    Opts. mu        =   0.4;  
    
    Opts. alpha      =  0.005; 
    
    Opts. beta       =   0.01;     

    Opts. c          =   1.1;    
    
    
 elseif p_noise <= 20
    

    Opts.gamma      =  0.09;  
    
    Opts. mu        =   0.5;  
    
    Opts. alpha      =  0.005; 
    
    Opts. beta       =   0.009;     

    Opts. c          =   1.1;      
    
 
elseif p_noise <= 30
    
    
    Opts.gamma      =  0.08;  
    
    Opts. mu        =   0.4;  
    
    Opts. alpha      =  0.003; 
    
    Opts. beta       =  0.009;     

    Opts. c          =   0.7;          

 
elseif p_noise <= 40
    

    Opts.gamma      =  0.08;  
    
    Opts. mu        =   0.4;  
    
    Opts. alpha      =  0.0008; 
    
    Opts. beta       =  0.009;     

    Opts. c          =   1.1;  
    
elseif p_noise <= 45
     
    Opts.gamma      =  0.08;  
    
    Opts. mu        =   0.4;  
    
    Opts. alpha      =  0.00008; 
    
    Opts. beta       =  0.0008;     

    Opts. c          =   2;   
    
    
elseif p_noise <= 50 
    
    Opts.gamma      =  0.2;  
    
    Opts. mu        =   0.5;  
    
    Opts. alpha      =  0.00006; 
    
    Opts. beta       =  0.0008;     

    Opts. c          =   1.5;    
    
    
 elseif p_noise <= 55
    
    
    Opts.gamma      =  0.1;  
    
    Opts. mu        =   0.3;  
    
    Opts. alpha      =  0.00006; 
    
    Opts. beta       =  0.0007;     

    Opts. c          =   1.1;  
    
    
elseif p_noise <= 60
    
    
    Opts.gamma      =  0.1;  
    
    Opts. mu        =   0.3;  
    
    Opts. alpha      =  0.00006; 
    
    Opts. beta       =  0.0007;     

    Opts. c          =   1.1;              
    
else
    
    Opts.gamma      =  0.1;  
    
    Opts. mu        =   0.3;  
    
    Opts. alpha      =  0.00006; 
    
    Opts. beta       =  0.0007;     

    Opts. c          =   1.1;              
    
end


end

