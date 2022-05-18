%path_sinogram = '/home/cid/OneDrive/Documentos/Matlab/Projeto_NLM_Rodrigo/sinograma/';
%path_reconstruction = '/home/cid/OneDrive/Documentos/Matlab/Projeto_NLM_Rodrigo/resultados/';
 
path_sinogram = 'data/';
path_reconstruction = 'result/';

%string_sinogram = ['assimetrico50';'homogeneo80';'madeira1';'madeira2';'simetrico80 ';'shepplogan'];
string_sinogram = ['shepplogan'];
string_sinogram = cellstr(string_sinogram)'; % Como o Matlab n?o aceita vetor com strings de tamanhos diferentes, fa?o esse ajuste.

search_window_size = 5; %Janela de Busca
patch_size = 2; %Janela de Similaridade
sigma = 0.1; %h
janelamedia = 3; %Janela de suavizaï¿½ï¿½o
 
    fprintf('\nGerando imagens simuladas de Shepp-Logan...');
    %[sinogram_reference, sinogram_noisy] = gera_shepplogan(1);
    DATA = load('base_sinograma.mat');
    sinogram_reference = DATA.sinogram_reference;
    sinogram_noisy     = DATA.sinogram_noisy;

  
    T = tic; %Inicio da tempo
    
    %Filtro de Miavhdfd
    filtradamedia = filtro_media(sinogram_noisy, janelamedia);
   
    %filtro de Varicia
    filtradavar = filtro_variancia(sinogram_noisy, filtradamedia, janelamedia);
    
    %C culo Alpha
    alpha = filtro_alpha(filtradamedia, filtradavar);
    
    %C culo do AlphaLine
    alphaline = filtro_alphaline(alpha, sinogram_noisy, patch_size);
    
    %Cculo do Betha
    betha = filtro_betha(filtradamedia, filtradavar);
    
    %Cï¿½lculo do Bethaline
    bethaline = filtro_bethaline(betha, patch_size); 
    
    %Normalizei parâmetros
    alphaline = alphaline/max(alphaline(:));
    bethaline = bethaline/max(bethaline(:));
    
    % Filtrar com Non-Local Means (sinogram, search_window, patch, sigma)
    sinogram_denoised = nlm_versao_rodrigo(sinogram_noisy, search_window_size, patch_size, sigma, alphaline, bethaline);
    %sinogram_denoised = nlmeans_rodrigo_Cv2(sinogram_noisy, alphaline, bethaline, search_window_size, patch_size, sigma);
    
    %sinogram_denoised = sinogram_noisy;
    
    [phantom_original_retro, phantom_original_ruidoso_retro,phantom_ruidoso_filtrado_retro] = fbpreconstruction(sinogram_reference,sinogram_noisy,sinogram_denoised, string_sinogram{1},path_reconstruction);
    phantom_original_retro = double(phantom_original_retro);
    phantom_original_ruidoso_retro = double(phantom_original_ruidoso_retro);
    phantom_ruidoso_filtrado_retro = double(phantom_ruidoso_filtrado_retro);
    
    psnr_result = psnr(phantom_original_ruidoso_retro,phantom_original_retro);
    ssim_result = ssim(phantom_original_ruidoso_retro,phantom_original_retro);
    epi2_result = EdgePreservationIndex_Laplacian(phantom_original_ruidoso_retro,phantom_original_retro);
    
    time = toc(T);
    name_sinogram = string_sinogram(i);
    fprintf('\nRuidoso:\npsnr: %f; ssim: %f; epi2: %f; time: %f; h: %f; by FBP;',psnr_result, ssim_result, epi2_result, time, sigma); 
    psnr_result = psnr(phantom_ruidoso_filtrado_retro,phantom_original_retro);
    ssim_result = ssim(phantom_ruidoso_filtrado_retro,phantom_original_retro);
    epi2_result = EdgePreservationIndex_Laplacian(phantom_ruidoso_filtrado_retro,phantom_original_retro);
    time = toc(T);
    fprintf('\nFiltrado:\npsnr: %f; ssim: %f; epi2: %f, time: %f; h: %f; by FBP;',psnr_result, ssim_result, epi2_result, time, sigma); 
    
    [phantom_original_pocs, phantom_original_ruidoso_pocs,phantom_ruidoso_filtrado_pocs] = pocsreconstruction(sinogram_reference,sinogram_noisy,sinogram_denoised,string_sinogram{1},path_reconstruction);
    phantom_original_pocs = double(phantom_original_pocs);
    phantom_original_ruidoso_pocs = double(phantom_original_ruidoso_pocs);
    phantom_ruidoso_filtrado_pocs = double(phantom_ruidoso_filtrado_pocs);    
    
    psnr_result = psnr(phantom_original_ruidoso_pocs,phantom_original_pocs);
    ssim_result = ssim(phantom_original_ruidoso_pocs,phantom_original_pocs);   
    epi2_result = EdgePreservationIndex_Laplacian(phantom_original_ruidoso_pocs,phantom_original_pocs);    
    
    time = toc(T);
    fprintf('\nRuidoso:\npsnr: %f; ssim: %f; epi2: %f, time: %f; h: %f; by POCS;',psnr_result, ssim_result, epi2_result, time, sigma); 
    psnr_result = psnr(phantom_ruidoso_filtrado_pocs,phantom_original_pocs);
    ssim_result = ssim(phantom_ruidoso_filtrado_pocs,phantom_original_pocs);
    epi2_result = EdgePreservationIndex_Laplacian(phantom_ruidoso_filtrado_pocs,phantom_original_pocs);    
    time = toc(T);
    fprintf('\nFiltrado:\npsnr: %f; ssim: %f; epi2: %f, time: %f; h: %f; by POCS;',psnr_result, ssim_result, epi2_result, time, sigma); 



    subplot(3,3,1);imshow(sinogram_reference,[]);title('Sin.Original');
    subplot(3,3,2);imshow(sinogram_noisy,[]);title('Sin.Ruidoso');
    subplot(3,3,3);imshow(sinogram_denoised,[]);title('Sin.Filtrado');
    subplot(3,3,4);imshow(phantom_original_retro,[]);title('Imagem Original FBP');
    subplot(3,3,5);imshow(phantom_original_ruidoso_retro,[]);title('Imagem Ruidosa FBP');
    subplot(3,3,6);imshow(phantom_ruidoso_filtrado_retro,[]);title('Imagem Filtrada FBP');
    subplot(3,3,7);imshow(phantom_original_pocs,[]);title('Imgagem Original POCS');
    subplot(3,3,8);imshow(phantom_original_ruidoso_pocs,[]);title('Imagem Ruidosa POCS');
    subplot(3,3,9);imshow(phantom_ruidoso_filtrado_pocs,[]);title('Imagem Filtrada POCS');
    