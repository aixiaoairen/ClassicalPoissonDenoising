function psnr = SNR(xest,xtrue)

psnr = 10*log10(norm(xtrue(:))^2 /norm(xest(:) - xtrue(:))^2 );
psnr = round(100*psnr)/100;

return;
