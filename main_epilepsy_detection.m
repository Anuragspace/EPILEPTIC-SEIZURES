% ====================================================================
% MAIN SCRIPT: Epilepsy Detection and Classification System
% ====================================================================
% Project: Automated Detection and Classification of Epileptic Seizures
%          Using EEG Data and Machine Learning Algorithms in MATLAB
% Author: [Your Name]
% Date: October 2025
% Dataset: BEED (Bangalore EEG Epilepsy Dataset)
% ====================================================================

clear all; close all; clc;

%% Step 1: Load Dataset
fprintf('\n=== STEP 1: Loading BEED Dataset ===\n');
data = readtable("D:\Matlab Codes\Biosignal\BEED_Data.csv");
X = table2array(data(:, 1:16));  % EEG channels
y = table2array(data(:, 17));    % Labels (0-3)

fprintf('Dataset loaded successfully!\n');
fprintf('Total samples: %d\n', size(X, 1));
fprintf('Number of channels: %d\n', size(X, 2));
fprintf('Class distribution:\n');
for i = 0:3
    fprintf('  Class %d: %d samples\n', i, sum(y == i));
end

%% Step 2: Preprocessing
fprintf('\n=== STEP 2: Preprocessing EEG Signals ===\n');
[X_preprocessed, preprocessInfo] = preprocessEEG(X);
save('preprocessed_data.mat', 'X_preprocessed', 'y', 'preprocessInfo');
fprintf('Preprocessing completed!\n');

%% Step 3: Feature Extraction
fprintf('\n=== STEP 3: Extracting Features ===\n');
[features, featureNames] = extractEEGFeatures(X_preprocessed);
save('extracted_features.mat', 'features', 'featureNames', 'y');
fprintf('Feature extraction completed!\n');
fprintf('Total features extracted: %d\n', size(features, 2));

%% Step 4: Feature Selection
fprintf('\n=== STEP 4: Feature Selection and Visualization ===\n');
[features_selected, selectedIdx] = selectFeatures(features, y);
save('selected_features.mat', 'features_selected', 'selectedIdx', 'y');

%% Step 5: Classification
fprintf('\n=== STEP 5: Training and Evaluating Classifiers ===\n');
results = classifyEEG(features_selected, y);
save('classification_results.mat', 'results');

%% Step 6: Generate Report
fprintf('\n=== STEP 6: Generating Results Report ===\n');
generateReport(results, featureNames, selectedIdx);

fprintf('\n=== PROJECT COMPLETED SUCCESSFULLY ===\n');
fprintf('All results saved in current directory\n');
