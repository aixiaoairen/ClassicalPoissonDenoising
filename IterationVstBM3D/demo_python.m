function denoise = demo_python(noisy, K)
    % Input - noisy: shape Ϊ h �� w �� ����ͼ��
    % OUtput- denoise
    N_reps = K;
    repCount = 0;
    for kk = 1 : N_reps
        repCount = repCount + 1;
        denoise = iterVSTpoisson(noisy);
    end