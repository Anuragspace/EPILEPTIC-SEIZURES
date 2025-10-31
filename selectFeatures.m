function [features_selected, selectedIdx] = selectFeatures(features, y)
% ====================================================================
% FUNCTION: selectFeatures
% ====================================================================
% Purpose: Select most relevant features using multiple methods
% Input:   features - Full feature matrix
%          y - Target labels
% Output:  features_selected - Reduced feature matrix
%          selectedIdx - Indices of selected features
% ====================================================================
    fprintf('Starting feature selection...\n');
    
    [num_samples, num_features] = size(features);
    
    %% Method 1: Correlation-Based Feature Selection
    fprintf('  - Computing feature correlations...\n');
    
    % Remove highly correlated features (threshold: 0.95)
    corr_matrix = corr(features);
    highly_corr = [];
    
    for i = 1:num_features-1
        for j = i+1:num_features
            if abs(corr_matrix(i,j)) > 0.95
                highly_corr = [highly_corr; j];
            end
        end
    end
    
    % Remove duplicates
    highly_corr = unique(highly_corr);
    keep_idx = setdiff(1:num_features, highly_corr);
    features_reduced = features(:, keep_idx);
    
    fprintf('    Removed %d highly correlated features\n', length(highly_corr));
    
    %% Method 2: ANOVA F-test for Feature Importance
    fprintf('  - Performing ANOVA F-test...\n');
    
    num_reduced_features = size(features_reduced, 2);
    f_scores = zeros(num_reduced_features, 1);
    p_values = zeros(num_reduced_features, 1);
    
    for i = 1:num_reduced_features
        [p, tbl, stats] = anova1(features_reduced(:,i), y, 'off');
        p_values(i) = p;
        f_scores(i) = tbl{2,5};  % F-statistic
    end
    
    % Select features with p-value < 0.05
    significant_idx = find(p_values < 0.05);
    features_significant = features_reduced(:, significant_idx);
    
    fprintf('    %d features are statistically significant (p < 0.05)\n', ...
            length(significant_idx));
    
    %% Method 3: Mutual Information
    fprintf('  - Computing mutual information scores...\n');
    
    mi_scores = zeros(size(features_significant, 2), 1);
    
    for i = 1:size(features_significant, 2)
        % Discretize continuous features for MI calculation
        feat_discretized = discretize(features_significant(:,i), 10);
        mi_scores(i) = mutualInformation(feat_discretized, y);
    end
    
    % Normalize MI scores
    mi_scores_norm = (mi_scores - min(mi_scores)) / (max(mi_scores) - min(mi_scores));
    
    %% Method 4: Principal Component Analysis (PCA)
    fprintf('  - Performing PCA for dimensionality reduction...\n');
    
    % Standardize features
    features_standardized = (features_significant - mean(features_significant)) ./ ...
                            std(features_significant);
    
    % Perform PCA
    [coeff, score, latent, tsquared, explained, mu] = pca(features_standardized);
    
    % Select components explaining 95% variance
    cumsum_explained = cumsum(explained);
    num_components = find(cumsum_explained >= 95, 1);
    
    fprintf('    %d principal components explain 95%% variance\n', num_components);
    
    % Visualize explained variance
    figure('Name', 'PCA Analysis', 'Position', [100 100 1200 500]);
    
    subplot(1,2,1);
    bar(explained(1:min(20, length(explained))));
    title('Variance Explained by Principal Components', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Principal Component', 'FontSize', 10);
    ylabel('Variance Explained (%)', 'FontSize', 10);
    grid on;
    
    subplot(1,2,2);
    plot(cumsum_explained, 'LineWidth', 2);
    hold on;
    plot([1 length(explained)], [95 95], 'r--', 'LineWidth', 1.5);
    plot(num_components, 95, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
    title('Cumulative Variance Explained', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Number of Components', 'FontSize', 10);
    ylabel('Cumulative Variance (%)', 'FontSize', 10);
    legend('Cumulative Variance', '95% Threshold', ...
           sprintf('%d Components', num_components), 'Location', 'southeast');
    grid on;
    
    saveas(gcf, 'pca_analysis.png');
    
    %% Method 5: Random Forest Feature Importance
    fprintf('  - Training Random Forest for feature importance...\n');
    
    % Train Random Forest
    rf_model = TreeBagger(100, features_significant, y, ...
                         'OOBPredictorImportance', 'on', ...
                         'Method', 'classification');
    
    % Get feature importance
    rf_importance = rf_model.OOBPermutedPredictorDeltaError;
    rf_importance_norm = (rf_importance - min(rf_importance)) / ...
                         (max(rf_importance) - min(rf_importance));
    
    %% Combine Selection Methods
    fprintf('  - Combining selection scores...\n');
    
    % Combine F-scores, MI, and RF importance (weighted average)
    f_scores_norm = (f_scores(significant_idx) - min(f_scores(significant_idx))) / ...
                    (max(f_scores(significant_idx)) - min(f_scores(significant_idx)));
    
    combined_scores = 0.3 * f_scores_norm + 0.3 * mi_scores_norm + ...
                      0.4 * rf_importance_norm';
    
    % Select top features (top 30 or top 50% whichever is smaller)
    num_top_features = min(30, ceil(0.5 * length(combined_scores)));
    [~, sorted_idx] = sort(combined_scores, 'descend');
    top_features_idx = sorted_idx(1:num_top_features);
    
    % Final selected features
    features_selected = features_significant(:, top_features_idx);
    
    % Map back to original indices
    temp_idx = keep_idx(significant_idx);
    selectedIdx = temp_idx(top_features_idx);
    
    fprintf('  - Final selection: %d features\n', size(features_selected, 2));
    
    %% Visualization of Feature Selection Results
    figure('Name', 'Feature Selection Results', 'Position', [100 100 1400 600]);
    
    % Plot 1: Feature importance scores
    subplot(1,2,1);
    [sorted_scores, sort_idx] = sort(combined_scores, 'descend');
    bar(sorted_scores(1:min(30, length(sorted_scores))));
    title('Top Feature Importance Scores', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Feature Rank', 'FontSize', 10);
    ylabel('Combined Importance Score', 'FontSize', 10);
    grid on;
    
    % Plot 2: Comparison of selection methods
    subplot(1,2,2);
    comparison_matrix = [f_scores_norm(top_features_idx), ...
                        mi_scores_norm(top_features_idx), ...
                        rf_importance_norm(top_features_idx)'];
    bar(comparison_matrix);
    title('Feature Selection Method Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Selected Feature Index', 'FontSize', 10);
    ylabel('Normalized Score', 'FontSize', 10);
    legend('F-Score', 'Mutual Information', 'Random Forest', 'Location', 'best');
    grid on;
    
    saveas(gcf, 'feature_selection_results.png');
    
    fprintf('Feature selection completed!\n\n');
end

%% Helper Function: Mutual Information
function mi = mutualInformation(x, y)
    % Calculate mutual information between discrete variables x and y
    
    x_vals = unique(x);
    y_vals = unique(y);
    
    n = length(x);
    mi = 0;
    
    for i = 1:length(x_vals)
        for j = 1:length(y_vals)
            p_xy = sum(x == x_vals(i) & y == y_vals(j)) / n;
            p_x = sum(x == x_vals(i)) / n;
            p_y = sum(y == y_vals(j)) / n;
            
            if p_xy > 0
                mi = mi + p_xy * log2(p_xy / (p_x * p_y));
            end
        end
    end
end
