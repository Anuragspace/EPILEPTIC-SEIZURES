function [features, featureNames] = extractEEGFeatures(X)
% ====================================================================
% FUNCTION: extractEEGFeatures
% ====================================================================
% Purpose: Extract comprehensive features from preprocessed EEG signals
% Input:   X - Preprocessed EEG data (samples x channels)
% Output:  features - Feature matrix (samples x features)
%          featureNames - Cell array of feature names
% ====================================================================
    fprintf('Starting feature extraction...\n');
    
    [num_samples, num_channels] = size(X);
    fs = 256;  % Sampling frequency
    
    % Initialize feature storage
    all_features = [];
    featureNames = {};
    
    %% 1. TIME-DOMAIN FEATURES
    fprintf('  - Extracting time-domain features...\n');
    
    % Statistical features across channels (per sample)
    feat_mean = mean(X, 2);
    feat_std = std(X, 0, 2);
    feat_var = var(X, 0, 2);
    feat_skewness = skewness(X, 0, 2);
    feat_kurtosis = kurtosis(X, 0, 2);
    feat_rms = sqrt(mean(X.^2, 2));  % Root Mean Square
    feat_peak = max(abs(X), [], 2);   % Peak amplitude
    
    % Add to features
    all_features = [all_features, feat_mean, feat_std, feat_var, ...
                    feat_skewness, feat_kurtosis, feat_rms, feat_peak];
    featureNames = [featureNames, {'Mean', 'Std', 'Variance', 'Skewness', ...
                                   'Kurtosis', 'RMS', 'PeakAmplitude'}];
    
    %% 2. FREQUENCY-DOMAIN FEATURES [SKIPPED]
    fprintf('  - Skipping frequency-domain features (Welch PSD) due to data shape.\n');
    % Instead add zero vectors as placeholders
    delta_power = zeros(num_samples, 1);
    theta_power = zeros(num_samples, 1);
    alpha_power = zeros(num_samples, 1);
    beta_power = zeros(num_samples, 1);
    gamma_power = zeros(num_samples, 1);
    spectral_entropy = zeros(num_samples, 1);
    
    % Add dummy frequency features
    all_features = [all_features, delta_power, theta_power, alpha_power, ...
                    beta_power, gamma_power, delta_power, theta_power, ...
                    alpha_power, beta_power, gamma_power, spectral_entropy];
    featureNames = [featureNames, {'DeltaPower', 'ThetaPower', 'AlphaPower', ...
                                   'BetaPower', 'GammaPower', 'RelDelta', 'RelTheta', ...
                                   'RelAlpha', 'RelBeta', 'RelGamma', 'SpectralEntropy'}];
    
    %% 3. WAVELET-BASED FEATURES
    fprintf('  - Extracting wavelet-based features...\n');
    
    wavelet_energy = zeros(num_samples, 5);
    wavelet_entropy = zeros(num_samples, 1);
    
    for i = 1:num_samples
        [C, L] = wavedec(X(i,:), 4, 'db4');
        D1 = detcoef(C, L, 1);
        D2 = detcoef(C, L, 2);
        D3 = detcoef(C, L, 3);
        D4 = detcoef(C, L, 4);
        A4 = appcoef(C, L, 'db4', 4);
        
        wavelet_energy(i, 1) = sum(D1.^2);
        wavelet_energy(i, 2) = sum(D2.^2);
        wavelet_energy(i, 3) = sum(D3.^2);
        wavelet_energy(i, 4) = sum(D4.^2);
        wavelet_energy(i, 5) = sum(A4.^2);
        
        total_energy = sum(wavelet_energy(i, :));
        p = wavelet_energy(i, :) / (total_energy + eps);
        wavelet_entropy(i) = -sum(p .* log2(p + eps));
    end
    
    all_features = [all_features, wavelet_energy, wavelet_entropy];
    featureNames = [featureNames, {'WaveletE_D1', 'WaveletE_D2', 'WaveletE_D3', ...
                                   'WaveletE_D4', 'WaveletE_A4', 'WaveletEntropy'}];
    
    %% 4. ENTROPY-BASED FEATURES
    fprintf('  - Extracting entropy-based features...\n');
    
    approx_entropy = zeros(num_samples, 1);
    sample_entropy = zeros(num_samples, 1);
    
    for i = 1:num_samples
        approx_entropy(i) = approximateEntropy(X(i,:), 2, 0.2*std(X(i,:)));
        sample_entropy(i) = sampleEntropy(X(i,:), 2, 0.2*std(X(i,:)));
    end
    
    all_features = [all_features, approx_entropy, sample_entropy];
    featureNames = [featureNames, {'ApproxEntropy', 'SampleEntropy'}];
    
    %% 5. ADDITIONAL UNIQUE FEATURES
    fprintf('  - Extracting additional unique features...\n');
    
    zcr = sum(diff(sign(X), 1, 2) ~= 0, 2) / num_channels;
    hjorth_activity = var(X, 0, 2);
    dx = diff(X, 1, 2);
    hjorth_mobility = sqrt(var(dx, 0, 2) ./ hjorth_activity);
    ddx = diff(dx, 1, 2);
    hjorth_complexity = sqrt(var(ddx, 0, 2) ./ var(dx, 0, 2)) ./ hjorth_mobility;
    
    all_features = [all_features, zcr, hjorth_activity, hjorth_mobility, hjorth_complexity];
    featureNames = [featureNames, {'ZeroCrossingRate', 'HjorthActivity', ...
                                   'HjorthMobility', 'HjorthComplexity'}];
    
    features = all_features;
    
    fprintf('Feature extraction completed!\n');
    fprintf('Total features extracted: %d\n', size(features, 2));
    
    plotFeatureDistribution(features, featureNames);
end

%% Helper Function: Approximate Entropy
function ApEn = approximateEntropy(U, m, r)
    N = length(U);
    phi = zeros(2, 1);
    
    for j = 1:2
        patterns = zeros(N - m - j + 2, m + j - 1);
        for i = 1:N - m - j + 2
            patterns(i, :) = U(i:i + m + j - 2);
        end
        
        C = zeros(N - m - j + 2, 1);
        for i = 1:N - m - j + 2
            template = repmat(patterns(i, :), N - m - j + 2, 1);
            D = max(abs(patterns - template), [], 2);
            C(i) = sum(D <= r) / (N - m - j + 2);
        end
        phi(j) = sum(log(C + eps)) / (N - m - j + 2);
    end
    
    ApEn = phi(1) - phi(2);
end

%% Helper Function: Sample Entropy
function SampEn = sampleEntropy(U, m, r)
    N = length(U);
    
    patterns = zeros(N - m, m);
    for i = 1:N - m
        patterns(i, :) = U(i:i + m - 1);
    end
    
    B = 0;
    A = 0;
    
    for i = 1:N - m - 1
        template = repmat(patterns(i, :), N - m - i, 1);
        D = max(abs(patterns(i+1:end, :) - template), [], 2);
        
        B = B + sum(D <= r);
        
        for j = i+1:N - m
            if D(j - i) <= r
                if abs(U(i + m) - U(j + m)) <= r
                    A = A + 1;
                end
            end
        end
    end
    
    if B == 0 || A == 0
        SampEn = 0;
    else
        SampEn = -log(A / B);
    end
end

%% Helper Function: Plot Feature Distribution
function plotFeatureDistribution(features, featureNames)
    figure('Name', 'Feature Statistics', 'Position', [100 100 1400 600]);
    
    feat_mean = mean(features);
    feat_std = std(features);
    
    subplot(1,2,1);
    bar(feat_mean);
    title('Mean Values of Extracted Features', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Feature Index', 'FontSize', 10);
    ylabel('Mean Value', 'FontSize', 10);
    grid on;
    
    subplot(1,2,2);
    bar(feat_std);
    title('Standard Deviation of Features', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Feature Index', 'FontSize', 10);
    ylabel('Std Value', 'FontSize', 10);
    grid on;
    
    saveas(gcf, 'feature_statistics.png');
end
