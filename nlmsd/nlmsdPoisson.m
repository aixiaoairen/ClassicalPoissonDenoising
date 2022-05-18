function denoise = nlmsdPoisson(noisy)
    % noisy image: the size is h ¡Á w£¬ the data type is double 
    % standard variance of gaussian kernel from nlm algorithm
    sigma = 0.1;
    % the size of smooth window
    media_window = 3 ;
    % the size of similarity window
    patch_size = 2 ;
    % the size of search window
    search_window = 5 ;
    % prepare works : media and variance
    media_filter = filtro_media(noisy, media_window) ;
    variance_filter = filtro_variancia(noisy, media_filter, media_window) ;
    % calculate the alpha
    alpha = filtro_alpha(media_filter, variance_filter) ;
    % calculate the alphaline
    alphaline = filtro_alphaline(alpha, noisy, patch_size) ;
    % calculate the beta
    beta = filtro_betha(media_filter, variance_filter) ;
    % calculate the betaline
    betaline = filtro_bethaline(beta, patch_size) ;
    % denoising ...
    denoise = nlm_versao_rodrigo(noisy, search_window, patch_size, sigma, alphaline, betaline) ;
end