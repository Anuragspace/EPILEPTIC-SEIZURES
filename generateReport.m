function generateReport(results, featureNames, selectedIdx)
% ====================================================================
% FUNCTION: generateReport
% ====================================================================
% Purpose: Generate comprehensive analysis report with visualizations
% Input:   results - Classification results structure
%          featureNames - Names of all features
%          selectedIdx - Indices of selected features
% ====================================================================
    fprintf('Generating comprehensive report...\n');
    
    classifiers = {'svm', 'knn', 'rf', 'nb', 'nn', 'dt'};
    classifier_names = {'SVM', 'KNN', 'Random Forest', 'Naive Bayes', ...
                       'Neural Network', 'Decision Tree'};
    
    %% Performance Table
    fprintf('  - Creating performance summary table...\n');
    performance_table = table();
    performance_table.Classifier = classifier_names';
    for i = 1:length(classifiers)
        clf = classifiers{i};
        performance_table.Accuracy(i) = results.(clf).accuracy;
        performance_table.Sensitivity(i) = results.(clf).avg_sensitivity;
        performance_table.Specificity(i) = results.(clf).avg_specificity;
        performance_table.Precision(i) = results.(clf).avg_precision;
        performance_table.F1_Score(i) = results.(clf).avg_f1_score;
        performance_table.Training_Time(i) = results.(clf).training_time;
    end
    disp(' ');
    disp('========== CLASSIFICATION PERFORMANCE SUMMARY ==========');
    disp(performance_table);
    writetable(performance_table, 'performance_summary.csv');
    
    [best_accuracy, best_idx] = max(performance_table.Accuracy);
    best_classifier = classifier_names{best_idx};
    fprintf('\n  BEST CLASSIFIER: %s with %.2f%% accuracy\n', best_classifier, best_accuracy);
    
    %% Confusion Matrices
    fprintf('  - Generating confusion matrices...\n');
    class_labels = {'Healthy', 'Generalized', 'Focal', 'Seizure Event'};
    figure('Name', 'Confusion Matrices', 'Position', [50 50 1600 900]);
    for i = 1:length(classifiers)
        clf = classifiers{i};
        confMat = results.(clf).confusion_matrix;
        subplot(2, 3, i);
        imagesc(confMat);
        colormap(flipud(gray));
        colorbar;
        [rows, cols] = size(confMat);
        for r = 1:rows
            for c = 1:cols
                if confMat(r,c) > max(confMat(:))/2
                    text_color = 'white';
                else
                    text_color = 'black';
                end
                text(c, r, sprintf('%d', confMat(r,c)), 'HorizontalAlignment', 'center', ...
                     'Color', text_color, 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
        title(sprintf('%s (Acc: %.2f%%)', classifier_names{i}, results.(clf).accuracy));
        xlabel('Predicted Class');
        ylabel('True Class');
        set(gca, 'XTick', 1:4, 'XTickLabel', class_labels, 'XTickLabelRotation', 45);
        set(gca, 'YTick', 1:4, 'YTickLabel', class_labels);
    end
    saveas(gcf, 'all_confusion_matrices.png');
    
    %% Detailed Best Classifier Analysis
    fprintf('  - Creating detailed analysis for best classifier...\n');
    best_clf_key = classifiers{best_idx};
    best_results = results.(best_clf_key);
    figure('Name', sprintf('Detailed Analysis - %s', best_classifier), 'Position', [100 100 1400 800]);
    
    subplot(2,3,1);
    cm = confusionchart(best_results.confusion_matrix, class_labels);
    cm.Title = sprintf('%s Confusion Matrix', best_classifier);
    % Do NOT set FontSize or FontWeight on cm.Title to avoid type issues

    
    subplot(2,3,2);
    class_metrics = zeros(4, 4);
    for c = 1:4
        class_metrics(c, 1) = best_results.per_class(c).sensitivity;
        class_metrics(c, 2) = best_results.per_class(c).specificity;
        class_metrics(c, 3) = best_results.per_class(c).precision;
        class_metrics(c, 4) = best_results.per_class(c).f1_score;
    end
    bar(class_metrics);
    title('Per-Class Performance Metrics', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Class');
    ylabel('Score (%)');
    set(gca, 'XTickLabel', class_labels);
    legend('Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'Location', 'best', 'FontSize', 8);
    grid on;
    ylim([0 110]);
    
    subplot(2,3,3);
    sens_vals = [best_results.per_class.sensitivity];
    spec_vals = [best_results.per_class.specificity];
    scatter(100-spec_vals, sens_vals, 100, 'filled');
    hold on;
    plot([0 100], [100 0], 'k--');
    for c = 1:4
        text(100-spec_vals(c)+2, sens_vals(c)+2, class_labels{c});
    end
    title('Sensitivity vs Specificity');
    xlabel('False Positive Rate (%)');
    ylabel('True Positive Rate (%)');
    grid on;
    axis([0 100 0 100]);
    
    subplot(2,3,4);
    overall_metrics = [best_results.accuracy, best_results.avg_sensitivity, best_results.avg_specificity,...
                      best_results.avg_precision, best_results.avg_f1_score];
    bar(overall_metrics);
    title('Overall Performance Metrics');
    set(gca, 'XTickLabel', {'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score'});
    ylabel('Score (%)');
    ylim([0 110]);
    grid on;
    for i = 1:length(overall_metrics)
        text(i, overall_metrics(i)+2, sprintf('%.2f%%', overall_metrics(i)));
    end
    
    subplot(2,3,5);
    test_distribution = histcounts(results.y_test, -0.5:1:3.5);
    pie(test_distribution, class_labels);
    title('Test Set Class Distribution');
    
    subplot(2,3,6);
    pred_distribution = histcounts(best_results.predictions, -0.5:1:3.5);
    pie(pred_distribution, class_labels);
    title('Prediction Distribution');
    
    saveas(gcf, sprintf('detailed_analysis_%s.png', best_clf_key));
    
    %% Feature Importance Analysis
    fprintf('  - Analyzing feature importance...\n');
    if isfield(results.rf.model, 'OOBPermutedPredictorDeltaError')
        figure('Name', 'Feature Importance', 'Position', [100 100 1200 500]);
        importance = results.rf.model.OOBPermutedPredictorDeltaError;
        [sorted_importance, ~] = sort(importance, 'descend');
        num_top = min(20, length(sorted_importance));
        bar(sorted_importance(1:num_top));
        title('Top 20 Most Important Features (Random Forest)');
        xlabel('Feature Rank');
        ylabel('Importance Score');
        grid on;
        saveas(gcf, 'feature_importance.png');
    end
    
    %% Generate Text Report
    fprintf('  - Writing text report...\n');
    fileID = fopen('classification_report.txt', 'w');
    
    fprintf(fileID, '================================================================\n');
    fprintf(fileID, 'EPILEPSY DETECTION AND CLASSIFICATION - PROJECT REPORT\n');
    fprintf(fileID, '================================================================\n\n');
    
    fprintf(fileID, 'PROJECT: Automated Detection and Classification of Epileptic\n');
    fprintf(fileID, '         Seizures Using EEG Data and Machine Learning\n\n');
    
    fprintf(fileID, 'DATASET: BEED (Bangalore EEG Epilepsy Dataset)\n');
    fprintf(fileID, '  - Total Samples: %d\n', length(results.y_train) + length(results.y_test));
    fprintf(fileID, '  - Training Samples: %d\n', length(results.y_train));
    fprintf(fileID, '  - Testing Samples: %d\n', length(results.y_test));
    fprintf(fileID, '  - Number of Features: %d (after selection)\n', size(results.X_train, 2));
    fprintf(fileID, '  - Classes: 4 (Healthy, Generalized Seizure, Focal Seizure, Seizure Event)\n\n');
    
    fprintf(fileID, '================================================================\n');
    fprintf(fileID, 'CLASSIFICATION RESULTS\n');
    fprintf(fileID, '================================================================\n\n');
    
    for i = 1:length(classifiers)
        clf = classifiers{i};
        fprintf(fileID, '%d. %s:\n', i, classifier_names{i});
        fprintf(fileID, '   - Accuracy:     %.2f%%\n', results.(clf).accuracy);
        fprintf(fileID, '   - Sensitivity:  %.2f%%\n', results.(clf).avg_sensitivity);
        fprintf(fileID, '   - Specificity:  %.2f%%\n', results.(clf).avg_specificity);
        fprintf(fileID, '   - Precision:    %.2f%%\n', results.(clf).avg_precision);
        fprintf(fileID, '   - F1-Score:     %.2f%%\n', results.(clf).avg_f1_score);
        fprintf(fileID, '   - Training Time: %.2f seconds\n\n', results.(clf).training_time);
    end
    
    fprintf(fileID, '================================================================\n');
    fprintf(fileID, 'BEST CLASSIFIER: %s\n', best_classifier);
    fprintf(fileID, '================================================================\n\n');
    
    fprintf(fileID, 'Overall Performance:\n');
    fprintf(fileID, '  - Accuracy:     %.2f%%\n', best_results.accuracy);
    fprintf(fileID, '  - Sensitivity:  %.2f%%\n', best_results.avg_sensitivity);
    fprintf(fileID, '  - Specificity:  %.2f%%\n', best_results.avg_specificity);
    fprintf(fileID, '  - Precision:    %.2f%%\n', best_results.avg_precision);
    fprintf(fileID, '  - F1-Score:     %.2f%%\n\n', best_results.avg_f1_score);
    
    fprintf(fileID, 'Per-Class Performance:\n');
    for c = 1:4
        fprintf(fileID, '\n  Class %d (%s):\n', c-1, class_labels{c});
        fprintf(fileID, '    - Sensitivity:  %.2f%%\n', best_results.per_class(c).sensitivity);
        fprintf(fileID, '    - Specificity:  %.2f%%\n', best_results.per_class(c).specificity);
        fprintf(fileID, '    - Precision:    %.2f%%\n', best_results.per_class(c).precision);
        fprintf(fileID, '    - F1-Score:     %.2f%%\n', best_results.per_class(c).f1_score);
    end
    
    fprintf(fileID, '\n\n================================================================\n');
    fprintf(fileID, 'CONCLUSION\n');
    fprintf(fileID, '================================================================\n\n');
    
    fprintf(fileID, 'The %s classifier achieved the highest accuracy of %.2f%%\n', best_classifier, best_accuracy);
    fprintf(fileID, 'for automated epilepsy detection and classification.\n\n');
    
    fprintf(fileID, 'This demonstrates the effectiveness of machine learning\n');
    fprintf(fileID, 'algorithms in automated seizure detection from EEG signals.\n\n');
    
    fprintf(fileID, 'Generated files:\n');
    fprintf(fileID, '  - preprocessing_comparison.png\n');
    fprintf(fileID, '  - feature_statistics.png\n');
    fprintf(fileID, '  - pca_analysis.png\n');
    fprintf(fileID, '  - feature_selection_results.png\n');
    fprintf(fileID, '  - classifier_comparison.png\n');
    fprintf(fileID, '  - all_confusion_matrices.png\n');
    fprintf(fileID, '  - detailed_analysis_%s.png\n', best_clf_key);
    fprintf(fileID, '  - feature_importance.png\n');
    fprintf(fileID, '  - performance_summary.csv\n');
    fprintf(fileID, '  - classification_report.txt\n');
    
    fclose(fileID);
    
    fprintf('\nReport generation completed!\n');
    fprintf('Check the following files for results:\n');
    fprintf('  - classification_report.txt\n');
    fprintf('  - performance_summary.csv\n');
    fprintf('  - All PNG visualization files\n\n');
end
