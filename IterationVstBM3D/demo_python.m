function denoise = demo_python(noisy, K)
    % Input - noisy: shape Îª h ¡Á w µÄ ÔëÉùÍ¼Ïñ
    % OUtput- denoise
    N_reps = K;
    repCount = 0;
    for kk = 1 : N_reps
        repCount = repCount + 1;
        denoise = iterVSTpoisson(noisy);
    end