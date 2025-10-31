function [X_preprocessed, preprocessInfo] = preprocessEEG(X)
% ====================================================================
% FUNCTION: preprocessEEG
% ====================================================================
% Purpose: Preprocess raw EEG signals with filtering and normalization
% Input:   X - Raw EEG data matrix (samples x channels)
% Output:  X_preprocessed - Preprocessed EEG signals
%          preprocessInfo - Structure containing preprocessing parameters
% ====================================================================
    fprintf('Starting EEG preprocessing...\n');
    
    % Initialize preprocessing info structure
    preprocessInfo.fs = 256;  % Sampling frequency (Hz)
    preprocessInfo.lowcut = 0.5;   % High-pass cutoff (Hz)
    preprocessInfo.highcut = 40;   % Low-pass cutoff (Hz)
    preprocessInfo.notch_freq = 50; % Notch filter for power line noise
    
    % Get dimensions
    [num_samples, num_channels] = size(X);
    X_preprocessed = zeros(size(X));
    
    %% Step 1: Remove DC offset (demean) column-wise (per channel)
    fprintf('  - Removing DC offset...\n');
    X_demeaned = X - mean(X, 1);
    
    %% Step 2: Bandpass Filter (0.5-40 Hz)
    fprintf('  - Applying bandpass filter (%.1f-%.1f Hz)...\n', ...
            preprocessInfo.lowcut, preprocessInfo.highcut);
    
    % Design Butterworth bandpass filter (4th order)
    fs = preprocessInfo.fs;
    [b_bp, a_bp] = butter(4, [preprocessInfo.lowcut preprocessInfo.highcut]/(fs/2), 'bandpass');
    
    % Apply filter to each channel (column-wise)
    X_filtered = zeros(size(X_demeaned));
    for ch = 1:num_channels
        X_filtered(:, ch) = filtfilt(b_bp, a_bp, X_demeaned(:, ch));
    end
    
    %% Step 3: Notch Filter (50 Hz - Power line interference)
    fprintf('  - Applying notch filter at %d Hz...\n', preprocessInfo.notch_freq);
    
    % Design notch filter
    wo = preprocessInfo.notch_freq/(fs/2);
    bw = wo/35;  % Bandwidth
    [b_notch, a_notch] = iirnotch(wo, bw);
    
    % Apply notch filter column-wise
    X_notched = zeros(size(X_filtered));
    for ch = 1:num_channels
        X_notched(:, ch) = filtfilt(b_notch, a_notch, X_filtered(:, ch));
    end
    
    %% Step 4: Artifact Removal (Simple threshold-based)
    fprintf('  - Detecting and handling artifacts...\n');
    
    % Calculate threshold (3 standard deviations)
    threshold = 3 * std(X_notched(:));
    artifact_count = sum(abs(X_notched(:)) > threshold);
    
    % Clip extreme values instead of removing samples
    X_clipped = X_notched;
    X_clipped(X_clipped > threshold) = threshold;
    X_clipped(X_clipped < -threshold) = -threshold;
    
    fprintf('    Artifacts detected: %d values (%.2f%%)\n', ...
            artifact_count, 100*artifact_count/numel(X_notched));
    
    %% Step 5: Z-score Normalization (column-wise)
    fprintf('  - Applying Z-score normalization...\n');
    
    % Normalize per channel
    X_preprocessed = (X_clipped - mean(X_clipped, 1)) ./ std(X_clipped, 0, 1);
    
    %% Step 6: Visualization (plot sample channel signals)
    fprintf('  - Creating preprocessing visualization...\n');
    
    % Plot comparison for first channel (column)
    figure('Name', 'Preprocessing Results', 'Position', [100 100 1200 800]);
    
    sample_ch = 1;  % First channel for visualization
    time = (0:num_samples-1) / fs;  % time vector in seconds
    
    subplot(4,1,1);
    plot(time, X(:, sample_ch), 'b', 'LineWidth', 1.5);
    title('Raw EEG Signal (Channel 1)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 10);
    grid on;
    
    subplot(4,1,2);
    plot(time, X_filtered(:, sample_ch), 'r', 'LineWidth', 1.5);
    title('After Bandpass Filtering (0.5-40 Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 10);
    grid on;
    
    subplot(4,1,3);
    plot(time, X_notched(:, sample_ch), 'g', 'LineWidth', 1.5);
    title('After Notch Filtering (50 Hz)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Amplitude', 'FontSize', 10);
    grid on;
    
    subplot(4,1,4);
    plot(time, X_preprocessed(:, sample_ch), 'k', 'LineWidth', 1.5);
    title('Final Preprocessed Signal (Normalized)', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Time (s)', 'FontSize', 10);
    ylabel('Normalized Amplitude', 'FontSize', 10);
    grid on;
    
    saveas(gcf, 'preprocessing_comparison.png');
    
    % Save preprocessing parameters
    preprocessInfo.artifact_threshold = threshold;
    preprocessInfo.artifact_count = artifact_count;
    preprocessInfo.artifact_percentage = 100*artifact_count/numel(X_notched);
    
    fprintf('Preprocessing completed successfully!\n\n');
end
