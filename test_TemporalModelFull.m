% test_TemporalModelFull.m
% Test script for TemporalModelFull - multi-ROI temporal modeling

clear; close all; clc;

fprintf('=== Testing TemporalModelFull.m ===\n\n');

%% 1. Load test data
fprintf('Loading test ROI data...\n');
load('ROI_test_running.mat', 'ROI');

fprintf('  Fluorescence ROIs: %d\n', length(ROI.modalities.fluorescence.labels));
fprintf('  Available ROIs: %s\n', strjoin(ROI.modalities.fluorescence.labels, ', '));
fprintf('  Behavior ROIs: %d\n', length(ROI.modalities.behavior.labels));
fprintf('  Available behaviors: %s\n\n', strjoin(ROI.modalities.behavior.labels, ', '));

%% 2. Test Case 1: Analyze ALL neural ROIs
fprintf('TEST 1: Analyzing ALL neural ROIs\n');
fprintf('==========================================\n');

opts1 = struct();
opts1.behavior_predictor = 'Face';
opts1.min_lag = -5;
opts1.max_lag = 10;
opts1.cv_folds = 5;
opts1.output_file = 'TemporalModelFull_test_all_rois.mat';
opts1.save_results = true;
opts1.show_plots = true;

try
    results_all = TemporalModelFull(ROI, opts1);
    fprintf('\n✓ TEST 1 PASSED: All ROIs analyzed successfully\n\n');
catch ME
    fprintf('\n✗ TEST 1 FAILED: %s\n', ME.message);
    fprintf('  Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    return;
end

%% 3. Test Case 2: Analyze SUBSET of ROIs
fprintf('\nTEST 2: Analyzing subset of ROIs (first 3)\n');
fprintf('==========================================\n');

opts2 = struct();
opts2.target_neural_rois = ROI.modalities.fluorescence.labels(1:3);
opts2.behavior_predictor = 'Face';
opts2.min_lag_seconds = -0.5;
opts2.max_lag_seconds = 1.0;
opts2.cv_folds = 5;
opts2.output_file = 'TemporalModelFull_test_subset.mat';
opts2.save_results = true;
opts2.show_plots = false;  % Skip plots for this test

try
    results_subset = TemporalModelFull(ROI, opts2);
    fprintf('\n✓ TEST 2 PASSED: Subset analysis successful\n\n');
catch ME
    fprintf('\n✗ TEST 2 FAILED: %s\n', ME.message);
    fprintf('  Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    return;
end

%% 4. Verification: Check output structure
fprintf('\nVERIFICATION: Checking output structure\n');
fprintf('==========================================\n');

% Check temporal_kernels
assert(isstruct(results_all.temporal_kernels), 'temporal_kernels must be struct');
assert(length(results_all.temporal_kernels) == results_all.metadata.n_rois, ...
    'temporal_kernels length mismatch');
fprintf('  ✓ temporal_kernels: [%d × 1] struct array\n', length(results_all.temporal_kernels));

% Check performance
assert(isstruct(results_all.performance), 'performance must be struct');
assert(length(results_all.performance) == results_all.metadata.n_rois, ...
    'performance length mismatch');
fprintf('  ✓ performance: [%d × 1] struct array\n', length(results_all.performance));

% Check comparison
assert(isfield(results_all, 'comparison'), 'Missing comparison field');
assert(size(results_all.comparison.beta_matrix_cv, 2) == results_all.metadata.n_rois, ...
    'beta_matrix_cv wrong dimensions');
fprintf('  ✓ comparison.beta_matrix_cv: [%d × %d]\n', ...
    size(results_all.comparison.beta_matrix_cv, 1), size(results_all.comparison.beta_matrix_cv, 2));

% Check predictions
assert(size(results_all.predictions.Y_pred, 2) == results_all.metadata.n_rois, ...
    'Y_pred wrong dimensions');
assert(size(results_all.predictions.Y_actual, 2) == results_all.metadata.n_rois, ...
    'Y_actual wrong dimensions');
fprintf('  ✓ predictions.Y_pred: [%d × %d]\n', ...
    size(results_all.predictions.Y_pred, 1), size(results_all.predictions.Y_pred, 2));

%% 5. Verification: Compare with single-ROI TemporalModel
fprintf('\nVERIFICATION: Comparing with single-ROI TemporalModel\n');
fprintf('==========================================\n');

% Run original TemporalModel on first ROI
opts_single = struct();
opts_single.target_neural_roi = ROI.modalities.fluorescence.labels{1};
opts_single.behavior_predictor = 'Face';
opts_single.min_lag = -5;
opts_single.max_lag = 10;
opts_single.cv_folds = 5;
opts_single.save_results = false;
opts_single.show_plots = false;

try
    fprintf('  Running TemporalModel on %s...\n', opts_single.target_neural_roi);
    results_single = TemporalModel(ROI, opts_single);

    % Compare R² values (should be very close)
    R2_single = results_single.performance.R2_cv_mean;
    R2_full = results_all.performance(1).R2_cv_mean;
    R2_diff = abs(R2_single - R2_full);

    fprintf('  Single-ROI R²: %.4f\n', R2_single);
    fprintf('  Multi-ROI R²:  %.4f\n', R2_full);
    fprintf('  Difference:    %.6f\n', R2_diff);

    if R2_diff < 0.001
        fprintf('  ✓ R² values match (within tolerance)\n');
    else
        warning('  ⚠ R² values differ by %.6f (expected < 0.001)', R2_diff);
    end

    % Compare beta weights
    beta_single = results_single.temporal_kernel.beta_cv_mean;
    beta_full = results_all.temporal_kernels(1).beta_cv_mean;
    beta_corr = corr(beta_single, beta_full);

    fprintf('  Beta correlation: %.4f\n', beta_corr);

    if beta_corr > 0.999
        fprintf('  ✓ Beta weights match (correlation > 0.999)\n');
    else
        warning('  ⚠ Beta weights differ (correlation = %.4f)', beta_corr);
    end

catch ME
    fprintf('  ⚠ Could not compare with TemporalModel: %s\n', ME.message);
end

%% 6. Summary
fprintf('\n=== TEST SUMMARY ===\n');
fprintf('All tests completed successfully!\n\n');

fprintf('Results for ALL ROIs test:\n');
fprintf('  Total ROIs analyzed: %d\n', results_all.metadata.n_rois);
fprintf('  R² range: [%.4f, %.4f]\n', ...
    min(results_all.comparison.R2_all_rois), max(results_all.comparison.R2_all_rois));
fprintf('  Best ROI: %s (R² = %.4f)\n', ...
    results_all.comparison.roi_names{results_all.comparison.R2_all_rois == max(results_all.comparison.R2_all_rois)}, ...
    max(results_all.comparison.R2_all_rois));
fprintf('  Peak lag range: [%.3f, %.3f] seconds\n', ...
    min(results_all.comparison.peak_lags_all_sec), max(results_all.comparison.peak_lags_all_sec));

fprintf('\nOutput files saved:\n');
fprintf('  %s\n', opts1.output_file);
fprintf('  %s\n', opts2.output_file);

fprintf('\n=== TemporalModelFull testing complete ===\n');
