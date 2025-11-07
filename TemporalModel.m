function results = TemporalModel(ROI, opts)
% TemporalModel Bidirectional lag ridge regression using pre-extracted ROI data
%
%   Fits temporal model Y(t) = Σ βᵢ·X(t-i) with bidirectional lags (leads
%   and lags). All data is z-scored, so no intercept is included. Negative
%   lags test whether fluorescence PREDICTS future motion (motor planning),
%   while positive lags test whether motion DRIVES fluorescence (sensory
%   feedback).
%
%   Uses cross-validation to obtain robust estimates of temporal kernel
%   coefficients (beta weights) with uncertainty (SEM), and fits full-data
%   model for trace predictions.
%
%   results = TemporalModel(ROI, opts)
%
%   REQUIRED INPUTS:
%       ROI     - ROI struct from rois_to_mat.m containing:
%                 .modalities.fluorescence.data [T x N]
%                 .modalities.fluorescence.labels {1 x N} (ROI names)
%                 .modalities.fluorescence.sample_rate (Hz)
%                 .modalities.behavior.data [Tb x Nb]
%                 .modalities.behavior.labels {1 x Nb} (behavior names)
%                 .modalities.behavior.sample_rate (Hz)
%
%   OPTIONAL INPUTS (opts struct):
%       target_neural_roi    - Name of neural ROI to predict (default: first ROI)
%       behavior_predictor   - Name of behavior ROI (default: 'Face')
%       min_lag              - Minimum lag in frames (negative = leads, default -5)
%       max_lag              - Maximum lag in frames (positive = lags, default 10)
%       min_lag_seconds      - Alternative: min lag in seconds (overrides min_lag)
%       max_lag_seconds      - Alternative: max lag in seconds (overrides max_lag)
%       cv_folds             - Number of cross-validation folds (default 5)
%       output_file          - Where to save results .mat (default: auto-generated)
%       save_results         - Save results to file (default true)
%       show_plots           - Generate diagnostic plots (default true)
%
%   OUTPUTS (results struct):
%       temporal_kernel     - Temporal kernel structure:
%           .beta_cv_mean       - CV-averaged beta coefficients (primary estimate)
%           .beta_cv_sem        - Standard error across CV folds
%           .beta_full_data     - Beta from full-data fit (reference)
%           .beta_cv_folds      - Beta weights from each CV fold [n_lags x n_folds]
%           .lag_indices        - Lag values in frames
%           .lag_times_sec      - Lag values in seconds
%           .peak_lag_sec       - Peak response lag (from CV mean)
%           .peak_beta          - Peak beta value (from CV mean)
%       performance         - Model performance metrics:
%           .R2_cv_mean         - Mean R² across CV folds (realistic estimate)
%           .R2_cv_sem          - Standard error of CV R²
%           .R2_cv_folds        - R² for each fold
%           .R2_full_data       - R² from full-data fit (optimistic)
%       predictions         - Model predictions:
%           .Y_pred             - Predicted fluorescence (full-data model)
%           .neural_trace_z     - Z-scored neural trace (truncated for lags)
%           .behavior_trace_z   - Z-scored behavior trace
%       metadata            - Analysis configuration and provenance

if nargin < 2 || isempty(opts)
    opts = struct();
end

% Default options
defaults = struct(...
    'target_neural_roi', '', ...
    'behavior_predictor', 'Face', ...
    'min_lag', -5, ...
    'max_lag', 10, ...
    'min_lag_seconds', [], ...
    'max_lag_seconds', [], ...
    'cv_folds', 5, ...
    'output_file', '', ...
    'save_results', true, ...
    'show_plots', true);

opts = populate_defaults(opts, defaults);

% Validate ROI structure
validate_roi_structure(ROI);

script_dir = fileparts(mfilename('fullpath'));
ensure_ridgeMML_on_path(script_dir);

fprintf('=== TemporalModel: Bidirectional lag ridge regression ===\n');

%% 1. Extract neural ROI trace
fluo_data = ROI.modalities.fluorescence.data;
fluo_labels = ROI.modalities.fluorescence.labels;
fluo_rate = ROI.modalities.fluorescence.sample_rate;

% Select target neural ROI
if isempty(opts.target_neural_roi)
    neural_idx = 1;
    opts.target_neural_roi = fluo_labels{1};
    fprintf('No target_neural_roi specified, using first ROI: %s\n', opts.target_neural_roi);
else
    neural_idx = find(strcmpi(fluo_labels, opts.target_neural_roi), 1);
    if isempty(neural_idx)
        error('Neural ROI "%s" not found.\nAvailable: %s', ...
            opts.target_neural_roi, strjoin(fluo_labels, ', '));
    end
end

neural_trace = fluo_data(:, neural_idx);
fprintf('Neural ROI: %s (%.1f Hz, %d frames)\n', ...
    opts.target_neural_roi, fluo_rate, length(neural_trace));

%% 2. Extract behavioral predictor trace
behav_data = ROI.modalities.behavior.data;
behav_labels = ROI.modalities.behavior.labels;
behav_rate = ROI.modalities.behavior.sample_rate;

behav_idx = find(strcmpi(behav_labels, opts.behavior_predictor), 1);
if isempty(behav_idx)
    error('Behavioral predictor "%s" not found.\nAvailable: %s', ...
        opts.behavior_predictor, strjoin(behav_labels, ', '));
end

behavior_trace = behav_data(:, behav_idx);
fprintf('Behavioral predictor: %s (%.1f Hz, %d frames)\n', ...
    opts.behavior_predictor, behav_rate, length(behavior_trace));

%% 3. Handle sampling rate mismatch
if abs(behav_rate - fluo_rate) > 0.01
    fprintf('Resampling behavior from %.1f Hz to %.1f Hz...\n', behav_rate, fluo_rate);
    behavior_trace = resample(behavior_trace, round(fluo_rate*1000), round(behav_rate*1000));
    fprintf('  Resampled to %d frames\n', length(behavior_trace));
end

sampling_rate = fluo_rate;

%% 4. Convert lag parameters from seconds to frames if needed
if ~isempty(opts.min_lag_seconds)
    opts.min_lag = round(opts.min_lag_seconds * sampling_rate);
    fprintf('Min lag: %.2f s → %d frames\n', opts.min_lag_seconds, opts.min_lag);
end
if ~isempty(opts.max_lag_seconds)
    opts.max_lag = round(opts.max_lag_seconds * sampling_rate);
    fprintf('Max lag: %.2f s → %d frames\n', opts.max_lag_seconds, opts.max_lag);
end

min_lag = opts.min_lag;
max_lag = opts.max_lag;

if min_lag >= max_lag
    error('min_lag (%d) must be less than max_lag (%d)', min_lag, max_lag);
end

%% 5. Match trace lengths and z-score
min_length = min(length(neural_trace), length(behavior_trace));
neural_trace = neural_trace(1:min_length);
behavior_trace = behavior_trace(1:min_length);

fprintf('\nMatched timepoints: %d frames (~%.1f s)\n', min_length, min_length / sampling_rate);

neural_trace_z = zscore(neural_trace);
behavior_trace_z = zscore(behavior_trace);

%% 6. Build bidirectional lag design matrix
fprintf('\nBuilding bidirectional lag design matrix (%d to +%d frames)...\n', min_lag, max_lag);
fprintf('  Lag range: %.3f s to +%.3f s @ %.1f Hz\n', ...
    min_lag / sampling_rate, max_lag / sampling_rate, sampling_rate);

n_lags_total = max_lag - min_lag + 1;
n_frames_lost_start = max_lag;
n_frames_lost_end = abs(min_lag);
n_valid = min_length - n_frames_lost_start - n_frames_lost_end;

if n_valid <= 0
    error('Not enough frames for requested lag range. Need at least %d frames, have %d.', ...
        n_frames_lost_start + n_frames_lost_end + 1, min_length);
end

X = zeros(n_valid, n_lags_total);

% Fill each column with appropriate lag
lag_idx = 0;
for lag = min_lag:max_lag
    lag_idx = lag_idx + 1;
    start_idx = max_lag + 1 - lag;
    end_idx = min_length - abs(min_lag) - lag;
    X(:, lag_idx) = behavior_trace_z(start_idx:end_idx);
end

% Truncate Y to match (middle section of data)
Y = neural_trace_z(max_lag+1 : min_length-abs(min_lag));

fprintf('  Timepoints after truncation: %d (lost %d at start, %d at end)\n', ...
    n_valid, n_frames_lost_start, n_frames_lost_end);

% Create lag time vectors
lag_values = (min_lag:max_lag)';
lag_times_sec = lag_values / sampling_rate;

%% 7. Check design matrix conditioning
fprintf('\nDesign matrix diagnostics:\n');
corr_matrix = corr(X);
off_diag_corrs = corr_matrix(~eye(size(corr_matrix)));
max_corr = max(off_diag_corrs);
mean_corr = mean(off_diag_corrs);
fprintf('  Max correlation between lags: %.3f\n', max_corr);
fprintf('  Mean correlation between lags: %.3f\n', mean_corr);

condition_num = cond(X'*X);
fprintf('  Condition number: %.2f\n', condition_num);
if condition_num > 30
    fprintf('  → High collinearity expected (using ridge regression)\n');
end

%% 8. Cross-validation with beta weight tracking
fprintf('\nPerforming %d-fold blocked time-series cross-validation...\n', opts.cv_folds);

cv_folds = opts.cv_folds;
fold_size = floor(n_valid / cv_folds);

beta_cv_folds = zeros(n_lags_total, cv_folds);
lambda_cv_folds = zeros(cv_folds, 1);
R2_cv_folds = zeros(cv_folds, 1);
convergence_cv = zeros(cv_folds, 1);

for fold = 1:cv_folds
    % Define test indices (contiguous block)
    test_start = (fold - 1) * fold_size + 1;
    test_end = min(fold * fold_size, n_valid);
    test_idx = test_start:test_end;
    train_idx = setdiff(1:n_valid, test_idx);

    % Extract train/test data
    X_train = X(train_idx, :);
    Y_train = Y(train_idx);
    X_test = X(test_idx, :);
    Y_test = Y(test_idx);

    % Fit ridge regression on training data
    [lambda_fold, beta_fold, conv_fail] = ridgeMML(Y_train, X_train, 0);

    % Store results
    beta_cv_folds(:, fold) = beta_fold;
    lambda_cv_folds(fold) = mean(lambda_fold(:));
    convergence_cv(fold) = conv_fail;

    % Predict on test data
    Y_pred_test = X_test * beta_fold;

    % Compute test R²
    TSS_test = sum((Y_test - mean(Y_test)).^2);
    RSS_test = sum((Y_test - Y_pred_test).^2);
    R2_cv_folds(fold) = max(0, 1 - RSS_test / TSS_test);

    fprintf('  Fold %d/%d: R² = %.4f, lambda = %.4f, n_train = %d, n_test = %d\n', ...
        fold, cv_folds, R2_cv_folds(fold), lambda_cv_folds(fold), ...
        length(train_idx), length(test_idx));
end

% Compute CV statistics
R2_cv_mean = mean(R2_cv_folds);
R2_cv_sem = std(R2_cv_folds) / sqrt(cv_folds);

fprintf('\nCV Results:\n');
fprintf('  R² (CV mean): %.4f ± %.4f (%.2f%% ± %.2f%%)\n', ...
    R2_cv_mean, R2_cv_sem, R2_cv_mean*100, R2_cv_sem*100);

if any(convergence_cv)
    warning('%d/%d CV folds reported convergence failure', sum(convergence_cv), cv_folds);
end

%% 9. Fit full-data model for predictions
fprintf('\nFitting full-data model (for predictions)...\n');
[lambda_full, beta_full, convergence_full] = ridgeMML(Y, X, 1);

if convergence_full
    warning('Full-data ridgeMML reported convergence failure');
end

lambda_full_mean = mean(lambda_full(:));
fprintf('  Optimal lambda: %.4f\n', lambda_full_mean);

% Generate predictions using full-data model
Y_pred_full = X * beta_full;

% Compute full-data R²
TSS_full = sum((Y - mean(Y)).^2);
RSS_full = sum((Y - Y_pred_full).^2);
R2_full = max(0, 1 - RSS_full / TSS_full);

fprintf('  R² (full-data fit): %.4f (%.2f%%) [optimistic]\n', R2_full, R2_full*100);

%% 10. Compute temporal kernel statistics from CV folds
fprintf('\nTemporal kernel statistics:\n');

% Primary estimate: CV-averaged betas
beta_cv_mean = mean(beta_cv_folds, 2);
beta_cv_sem = std(beta_cv_folds, 0, 2) / sqrt(cv_folds);

% Find peak response (from CV mean)
[peak_beta, peak_idx] = max(abs(beta_cv_mean));
peak_lag_frames = lag_values(peak_idx);
peak_lag_sec = lag_times_sec(peak_idx);

fprintf('  Peak response at lag %d (%.3f s)\n', peak_lag_frames, peak_lag_sec);
fprintf('  Peak beta (CV mean): %.4f ± %.4f\n', beta_cv_mean(peak_idx), beta_cv_sem(peak_idx));

if peak_lag_frames < 0
    fprintf('  → PREDICTIVE: Fluorescence leads motion by %.3f s\n', abs(peak_lag_sec));
elseif peak_lag_frames > 0
    fprintf('  → REACTIVE: Fluorescence follows motion by %.3f s\n', peak_lag_sec);
else
    fprintf('  → INSTANTANEOUS: Peak at zero lag\n');
end

%% 11. Assemble results structure
results = struct();

% Temporal kernel (primary scientific output)
results.temporal_kernel = struct();
results.temporal_kernel.beta_cv_mean = beta_cv_mean;
results.temporal_kernel.beta_cv_sem = beta_cv_sem;
results.temporal_kernel.beta_full_data = beta_full;
results.temporal_kernel.beta_cv_folds = beta_cv_folds;
results.temporal_kernel.lag_indices = lag_values;
results.temporal_kernel.lag_times_sec = lag_times_sec;
results.temporal_kernel.peak_lag_frames = peak_lag_frames;
results.temporal_kernel.peak_lag_sec = peak_lag_sec;
results.temporal_kernel.peak_beta = beta_cv_mean(peak_idx);
results.temporal_kernel.peak_beta_sem = beta_cv_sem(peak_idx);

% Performance metrics
results.performance = struct();
results.performance.R2_cv_mean = R2_cv_mean;
results.performance.R2_cv_sem = R2_cv_sem;
results.performance.R2_cv_folds = R2_cv_folds;
results.performance.R2_full_data = R2_full;
results.performance.lambda_cv_folds = lambda_cv_folds;
results.performance.lambda_full_data = lambda_full_mean;
results.performance.convergence_cv_failures = sum(convergence_cv);
results.performance.convergence_full_failure = convergence_full;

% Predictions (from full-data model)
results.predictions = struct();
results.predictions.Y_pred = Y_pred_full;
results.predictions.neural_trace_z = Y;
results.predictions.behavior_trace_z = behavior_trace_z;

% Metadata
results.metadata = struct();
results.metadata.target_neural_roi = opts.target_neural_roi;
results.metadata.behavior_predictor = opts.behavior_predictor;
results.metadata.min_lag = min_lag;
results.metadata.max_lag = max_lag;
results.metadata.min_lag_seconds = min_lag / sampling_rate;
results.metadata.max_lag_seconds = max_lag / sampling_rate;
results.metadata.sampling_rate = sampling_rate;
results.metadata.n_timepoints_total = min_length;
results.metadata.n_timepoints_used = n_valid;
results.metadata.n_timepoints_lost_start = n_frames_lost_start;
results.metadata.n_timepoints_lost_end = n_frames_lost_end;
results.metadata.cv_folds = cv_folds;
results.metadata.timestamp = datestr(now);
results.metadata.diagnostics = struct('max_correlation', max_corr, ...
    'mean_correlation', mean_corr, 'condition_number', condition_num);

% Source information
if isfield(ROI, 'metadata') && isfield(ROI.metadata, 'source')
    results.metadata.source_roi_file = ROI.metadata.source;
end

%% 12. Generate plots
if opts.show_plots
    fprintf('\nGenerating plots...\n');
    plot_temporal_kernel(results);
    plot_model_predictions(results);
    fprintf('Plots generated.\n');
end

%% 13. Save results
if opts.save_results
    if isempty(opts.output_file)
        safe_roi = regexprep(opts.target_neural_roi, '\W+', '');
        safe_behav = regexprep(opts.behavior_predictor, '\W+', '');
        default_output = sprintf('TemporalModel_%s_vs_%s_results.mat', safe_roi, safe_behav);
        output_file = fullfile(script_dir, default_output);
    else
        output_file = opts.output_file;
    end

    save(output_file, 'results', '-v7.3');
    fprintf('\nResults saved to:\n  %s\n', output_file);
else
    fprintf('\nResults not saved (opts.save_results == false)\n');
end

fprintf('\n=== TemporalModel complete ===\n');

end

%% ================= Helper Functions =================

function opts = populate_defaults(opts, defaults)
    fields = fieldnames(defaults);
    for i = 1:numel(fields)
        name = fields{i};
        if ~isfield(opts, name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
    end
end

function validate_roi_structure(ROI)
    % Check required fields exist
    if ~isstruct(ROI)
        error('ROI input must be a struct from rois_to_mat.m');
    end

    if ~isfield(ROI, 'modalities')
        error('ROI struct missing "modalities" field');
    end

    if ~isfield(ROI.modalities, 'fluorescence')
        error('ROI.modalities missing "fluorescence" field');
    end

    if ~isfield(ROI.modalities, 'behavior')
        error('ROI.modalities missing "behavior" field. Behavioral data required.');
    end

    % Check fluorescence fields
    fluo = ROI.modalities.fluorescence;
    required_fluo = {'data', 'labels', 'sample_rate'};
    for i = 1:length(required_fluo)
        if ~isfield(fluo, required_fluo{i})
            error('ROI.modalities.fluorescence missing "%s" field', required_fluo{i});
        end
    end

    % Check behavior fields
    behav = ROI.modalities.behavior;
    required_behav = {'data', 'labels', 'sample_rate'};
    for i = 1:length(required_behav)
        if ~isfield(behav, required_behav{i})
            error('ROI.modalities.behavior missing "%s" field', required_behav{i});
        end
    end
end

function ensure_ridgeMML_on_path(script_dir)
    if exist('ridgeMML', 'file') == 2
        return;
    end

    % Search common locations
    candidate_paths = {
        fullfile(script_dir)
        fullfile(script_dir, '..')
        'C:\Users\shires\Downloads'
        'H:\IsaacAndGarrettMatlabScripts\glm code\Puff_Dataset'
    };

    for i = 1:numel(candidate_paths)
        if exist(candidate_paths{i}, 'dir')
            addpath(candidate_paths{i});
        end
        if exist('ridgeMML', 'file') == 2
            return;
        end
    end

    error('ridgeMML.m not found. Please add it to the MATLAB path.');
end

%% ================= Plotting Functions =================

function plot_temporal_kernel(results)
    % Plot CV-averaged temporal kernel with error bars

    tk = results.temporal_kernel;
    perf = results.performance;

    fig_title = sprintf('Temporal Kernel: %s vs %s', ...
        results.metadata.target_neural_roi, results.metadata.behavior_predictor);

    figure('Name', fig_title, 'Position', [100 100 900 600]);

    % Plot CV-averaged beta with error bars
    errorbar(tk.lag_times_sec, tk.beta_cv_mean, tk.beta_cv_sem, ...
        'o-', 'LineWidth', 2, 'MarkerSize', 8, ...
        'MarkerFaceColor', [0.2 0.4 0.8], 'Color', [0.2 0.4 0.8], ...
        'DisplayName', 'CV mean ± SEM');

    hold on;

    % Overlay full-data beta as reference (dashed)
    plot(tk.lag_times_sec, tk.beta_full_data, '--', ...
        'LineWidth', 1.5, 'Color', [0.7 0.7 0.7], ...
        'DisplayName', 'Full-data fit');

    % Zero lines
    plot([min(tk.lag_times_sec), max(tk.lag_times_sec)], [0 0], ...
        'k--', 'LineWidth', 1, 'HandleVisibility', 'off');

    yl = ylim;
    plot([0, 0], yl, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

    % Mark peak
    plot(tk.peak_lag_sec, tk.peak_beta, 'r*', ...
        'MarkerSize', 15, 'LineWidth', 2, ...
        'DisplayName', sprintf('Peak (%.3f s)', tk.peak_lag_sec));

    % Vertical line at peak
    plot([tk.peak_lag_sec, tk.peak_lag_sec], yl, 'r--', ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');

    % Shade predictive region (negative lags)
    if results.metadata.min_lag < 0
        x_pred = [min(tk.lag_times_sec), 0, 0, min(tk.lag_times_sec)];
        y_pred = [yl(1), yl(1), yl(2), yl(2)];
        patch(x_pred, y_pred, [1 0.9 0.9], 'FaceAlpha', 0.3, ...
            'EdgeColor', 'none', 'DisplayName', 'Predictive (leads)');
    end

    hold off;

    xlabel('Lag time (seconds)', 'FontSize', 11);
    ylabel('Beta coefficient', 'FontSize', 11);
    title(fig_title, 'FontSize', 12, 'Interpreter', 'none');
    legend('Location', 'best');
    grid on;

    % Add performance annotation
    anno_str = sprintf(['R² (CV): %.2f%% ± %.2f%%\n' ...
                        'R² (full-data): %.2f%%\n' ...
                        'Peak: %.3f s (β = %.3f ± %.3f)'], ...
        perf.R2_cv_mean*100, perf.R2_cv_sem*100, ...
        perf.R2_full_data*100, ...
        tk.peak_lag_sec, tk.peak_beta, tk.peak_beta_sem);

    annotation(gcf, 'textbox', [0.15 0.75 0.25 0.18], ...
        'String', anno_str, 'EdgeColor', 'k', 'BackgroundColor', 'w', ...
        'FitBoxToText', 'on', 'FontSize', 9);

    % Region labels
    if results.metadata.min_lag < 0
        text(min(tk.lag_times_sec)/2, yl(2)*0.9, 'PREDICTIVE', ...
            'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', [0.5 0 0]);
    end
    text(max(tk.lag_times_sec)/2, yl(2)*0.9, 'REACTIVE', ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'Color', [0 0 0.5]);
end

function plot_model_predictions(results)
    % Plot model predictions vs actual data

    meta = results.metadata;
    pred = results.predictions;

    n_valid = meta.n_timepoints_used;
    n_lost_start = meta.n_timepoints_lost_start;
    n_lost_end = meta.n_timepoints_lost_end;

    % Time vector for truncated data
    t_truncated = (n_lost_start:(n_lost_start + n_valid - 1)) / meta.sampling_rate;

    % Time vector for full behavior trace
    n_total = length(pred.behavior_trace_z);
    t_full = (0:(n_total-1)) / meta.sampling_rate;

    fig_title = sprintf('Model Predictions: %s vs %s', ...
        meta.target_neural_roi, meta.behavior_predictor);

    figure('Name', fig_title, 'Position', [150 150 1000 600]);
    tiled = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tiled, fig_title, 'Interpreter', 'none');

    % Top panel: Neural trace and prediction
    ax1 = nexttile(tiled);
    plot(t_truncated, pred.neural_trace_z, 'Color', [0.2 0.2 0.8], ...
        'DisplayName', sprintf('%s (z-score)', meta.target_neural_roi));
    hold(ax1, 'on');
    plot(t_truncated, pred.Y_pred, 'Color', [0.85 0.33 0.1], ...
        'LineWidth', 1.25, 'DisplayName', 'Prediction (full-data model)');
    hold(ax1, 'off');
    ylabel(ax1, 'Fluorescence (z-score)');
    legend(ax1, 'Location', 'best');
    grid(ax1, 'on');

    % R² annotation
    text(ax1, 0.02, 0.98, ...
        sprintf('R² (CV): %.2f%% ± %.2f%%\nR² (full): %.2f%%', ...
            results.performance.R2_cv_mean*100, results.performance.R2_cv_sem*100, ...
            results.performance.R2_full_data*100), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 9, 'BackgroundColor', 'w', 'EdgeColor', 'k');

    % Bottom panel: Behavior with truncation regions
    ax2 = nexttile(tiled);
    plot(t_full, pred.behavior_trace_z, 'Color', [0.13 0.55 0.13], ...
        'DisplayName', sprintf('%s (z-score)', meta.behavior_predictor));

    hold(ax2, 'on');

    % Shade truncated regions
    if n_lost_start > 0
        t_trunc_start = t_full(1:n_lost_start);
        behav_trunc_start = pred.behavior_trace_z(1:n_lost_start);
        patch([t_trunc_start, fliplr(t_trunc_start)], ...
            [behav_trunc_start', fliplr(zeros(1, n_lost_start))], ...
            [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none', ...
            'DisplayName', sprintf('Truncated start (%d frames)', n_lost_start));
    end

    if n_lost_end > 0
        t_trunc_end = t_full(end-n_lost_end+1:end);
        behav_trunc_end = pred.behavior_trace_z(end-n_lost_end+1:end);
        patch([t_trunc_end, fliplr(t_trunc_end)], ...
            [behav_trunc_end', fliplr(zeros(1, n_lost_end))], ...
            [0.95 0.95 0.95], 'FaceAlpha', 0.5, 'EdgeColor', 'none', ...
            'DisplayName', sprintf('Truncated end (%d frames)', n_lost_end));
    end

    hold(ax2, 'off');

    ylabel(ax2, sprintf('%s (z-score)', meta.behavior_predictor));
    xlabel(ax2, 'Time (s)');
    grid(ax2, 'on');
    legend(ax2, 'Location', 'best');

    linkaxes([ax1, ax2], 'x');

    % Footer annotation
    footer_str = sprintf('Sampling: %.1f Hz | Lags: %d to +%d frames (%.3f to +%.3f s) | CV folds: %d', ...
        meta.sampling_rate, meta.min_lag, meta.max_lag, ...
        meta.min_lag_seconds, meta.max_lag_seconds, meta.cv_folds);
    annotation(gcf, 'textbox', [0.01 0.01 0.7 0.04], ...
        'String', footer_str, 'EdgeColor', 'none', ...
        'Interpreter', 'none', 'FontSize', 8);
end
