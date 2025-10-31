function results = classifyEEG(features, labels)
% ====================================================================
% FUNCTION: classifyEEG
% ====================================================================
% Purpose: Train and evaluate multiple ML classifiers for epilepsy detection
% Input:   features - Feature matrix (samples x features)
%          labels - Target labels (0-3)
% Output:  results - Structure containing all classification results
% ====================================================================
    fprintf('Starting classification...\n');
    
    %% Data Preparation
    fprintf('  - Preparing data splits...\n');
    
    % Create stratified train-test split (70-30)
    cv = cvpartition(labels, 'HoldOut', 0.3);
    
    X_train = features(training(cv), :);
    y_train = labels(training(cv));
    X_test = features(test(cv), :);
    y_test = labels(test(cv));
    
    fprintf('    Training samples: %d\n', length(y_train));
    fprintf('    Testing samples: %d\n', length(y_test));
    
    % Initialize results structure
    results = struct();
    results.X_train = X_train;
    results.y_train = y_train;
    results.X_test = X_test;
    results.y_test = y_test;
    
    %% Classifier 1: Support Vector Machine (SVM)
    fprintf('\n  - Training SVM Classifier...\n');
    tic;
    
    svm_template = templateSVM('KernelFunction', 'rbf', ...
                               'KernelScale', 'auto', ...
                               'Standardize', true);
    
    svm_model = fitcecoc(X_train, y_train, 'Learners', svm_template, ...
                        'Coding', 'onevsall');
    
    y_pred_svm = predict(svm_model, X_test);
    
    svm_results = calculateMetrics(y_test, y_pred_svm);
    svm_results.training_time = toc;
    svm_results.model = svm_model;
    
    results.svm = svm_results;
    fprintf('    SVM Accuracy: %.2f%%\n', svm_results.accuracy);
    fprintf('    Training time: %.2f seconds\n', svm_results.training_time);
    
    %% Classifier 2: K-Nearest Neighbors (KNN)
    fprintf('\n  - Training KNN Classifier...\n');
    tic;
    
    k_values = [3, 5, 7, 9, 11];
    best_k = 5;
    best_acc = 0;
    
    for k = k_values
        knn_temp = fitcknn(X_train, y_train, 'NumNeighbors', k, ...
                          'Standardize', true, 'Distance', 'euclidean');
        cv_model = crossval(knn_temp, 'KFold', 5);
        acc = 1 - kfoldLoss(cv_model);
        
        if acc > best_acc
            best_acc = acc;
            best_k = k;
        end
    end
    
    fprintf('    Optimal K: %d (CV Accuracy: %.2f%%)\n', best_k, best_acc*100);
    
    knn_model = fitcknn(X_train, y_train, 'NumNeighbors', best_k, ...
                       'Standardize', true, 'Distance', 'euclidean');
    
    y_pred_knn = predict(knn_model, X_test);
    
    knn_results = calculateMetrics(y_test, y_pred_knn);
    knn_results.training_time = toc;
    knn_results.model = knn_model;
    knn_results.optimal_k = best_k;
    
    results.knn = knn_results;
    fprintf('    KNN Accuracy: %.2f%%\n', knn_results.accuracy);
    fprintf('    Training time: %.2f seconds\n', knn_results.training_time);
    
    %% Classifier 3: Random Forest
    fprintf('\n  - Training Random Forest Classifier...\n');
    tic;
    
    rf_model = TreeBagger(200, X_train, y_train, ...
                         'Method', 'classification', ...
                         'OOBPrediction', 'on', ...
                         'MinLeafSize', 5);
    
    [y_pred_rf_cell, scores_rf] = predict(rf_model, X_test);
    y_pred_rf = str2double(y_pred_rf_cell);
    
    rf_results = calculateMetrics(y_test, y_pred_rf);
    rf_results.training_time = toc;
    rf_results.model = rf_model;
    rf_results.oob_error = oobError(rf_model);
    
    results.rf = rf_results;
    fprintf('    Random Forest Accuracy: %.2f%%\n', rf_results.accuracy);
    fprintf('    OOB Error: %.2f%%\n', rf_results.oob_error(end)*100);
    fprintf('    Training time: %.2f seconds\n', rf_results.training_time);
    
    %% Classifier 4: Naive Bayes
    fprintf('\n  - Training Naive Bayes Classifier...\n');
    tic;
    
    nb_model = fitcnb(X_train, y_train, 'DistributionNames', 'kernel');
    
    y_pred_nb = predict(nb_model, X_test);
    
    nb_results = calculateMetrics(y_test, y_pred_nb);
    nb_results.training_time = toc;
    nb_results.model = nb_model;
    
    results.nb = nb_results;
    fprintf('    Naive Bayes Accuracy: %.2f%%\n', nb_results.accuracy);
    fprintf('    Training time: %.2f seconds\n', nb_results.training_time);
    
    %% Classifier 5: Neural Network (Feedforward)
    fprintf('\n  - Training Neural Network Classifier...\n');
    tic;
    
    X_train_nn = X_train';
    X_test_nn = X_test';
    
    y_train_cat = zeros(4, length(y_train));
    y_test_cat = zeros(4, length(y_test));
    
    for i = 1:length(y_train)
        y_train_cat(y_train(i)+1, i) = 1;
    end
    for i = 1:length(y_test)
        y_test_cat(y_test(i)+1, i) = 1;
    end
    
    hiddenLayerSize = [30 20 10];  % 3 hidden layers
    net = patternnet(hiddenLayerSize, 'trainscg');
    
    net.trainParam.epochs = 200;
    net.trainParam.showWindow = false;
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0.0;
    
    [net, tr] = train(net, X_train_nn, y_train_cat);
    
    y_pred_nn_raw = net(X_test_nn);
    [~, y_pred_nn] = max(y_pred_nn_raw);
    y_pred_nn = y_pred_nn' - 1;
    
    nn_results = calculateMetrics(y_test, y_pred_nn);
    nn_results.training_time = toc;
    nn_results.model = net;
    nn_results.performance = tr.best_perf;
    
    results.nn = nn_results;
    fprintf('    Neural Network Accuracy: %.2f%%\n', nn_results.accuracy);
    fprintf('    Training time: %.2f seconds\n', nn_results.training_time);
    
    %% Classifier 6: Decision Tree
    fprintf('\n  - Training Decision Tree Classifier...\n');
    tic;
    
    dt_model = fitctree(X_train, y_train, 'MaxNumSplits', 50, ...
                       'MinLeafSize', 10);
    
    y_pred_dt = predict(dt_model, X_test);
    
    dt_results = calculateMetrics(y_test, y_pred_dt);
    dt_results.training_time = toc;
    dt_results.model = dt_model;
    
    results.dt = dt_results;
    fprintf('    Decision Tree Accuracy: %.2f%%\n', dt_results.accuracy);
    fprintf('    Training time: %.2f seconds\n', dt_results.training_time);
    
    %% Create Comparative Visualizations
    fprintf('\n  - Creating comparison visualizations...\n');
    
    classifiers = {'SVM', 'KNN', 'Random Forest', 'Naive Bayes', 'Neural Net', 'Decision Tree'};
    accuracies = [svm_results.accuracy, knn_results.accuracy, rf_results.accuracy, ...
                  nb_results.accuracy, nn_results.accuracy, dt_results.accuracy];
    training_times = [svm_results.training_time, knn_results.training_time, ...
                      rf_results.training_time, nb_results.training_time, ...
                      nn_results.training_time, dt_results.training_time];
    
    figure('Name', 'Classifier Comparison', 'Position', [100 100 1400 600]);
    
    subplot(1,2,1);
    bar(accuracies);
    set(gca, 'XTickLabel', classifiers, 'XTickLabelRotation', 45);
    title('Classification Accuracy Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Accuracy (%)', 'FontSize', 10);
    ylim([min(accuracies)-5, 100]);
    grid on;
    
    for i = 1:length(accuracies)
        text(i, accuracies(i)+1, sprintf('%.2f%%', accuracies(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    end
    
    subplot(1,2,2);
    bar(training_times);
    set(gca, 'XTickLabel', classifiers, 'XTickLabelRotation', 45);
    title('Training Time Comparison', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Time (seconds)', 'FontSize', 10);
    grid on;
    
    saveas(gcf, 'classifier_comparison.png');
    
    fprintf('\nClassification completed successfully!\n');
end

%% Helper Function: Calculate Performance Metrics
function metrics = calculateMetrics(y_true, y_pred)
    confMat = confusionmat(y_true, y_pred);
    metrics.confusion_matrix = confMat;
    
    metrics.accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;
    
    num_classes = size(confMat, 1);
    metrics.per_class = struct();
    
    for i = 1:num_classes
        TP = confMat(i, i);
        FP = sum(confMat(:, i)) - TP;
        FN = sum(confMat(i, :)) - TP;
        TN = sum(confMat(:)) - TP - FP - FN;
        
        if (TP + FN) > 0
            metrics.per_class(i).sensitivity = TP / (TP + FN) * 100;
        else
            metrics.per_class(i).sensitivity = 0;
        end
        
        if (TN + FP) > 0
            metrics.per_class(i).specificity = TN / (TN + FP) * 100;
        else
            metrics.per_class(i).specificity = 0;
        end
        
        if (TP + FP) > 0
            metrics.per_class(i).precision = TP / (TP + FP) * 100;
        else
            metrics.per_class(i).precision = 0;
        end
        
        if metrics.per_class(i).precision + metrics.per_class(i).sensitivity > 0
            metrics.per_class(i).f1_score = 2 * (metrics.per_class(i).precision * ...
                                        metrics.per_class(i).sensitivity) / ...
                                        (metrics.per_class(i).precision + ...
                                        metrics.per_class(i).sensitivity);
        else
            metrics.per_class(i).f1_score = 0;
        end
    end
    
    metrics.avg_sensitivity = mean([metrics.per_class.sensitivity]);
    metrics.avg_specificity = mean([metrics.per_class.specificity]);
    metrics.avg_precision = mean([metrics.per_class.precision]);
    metrics.avg_f1_score = mean([metrics.per_class.f1_score]);
    metrics.predictions = y_pred;
end
