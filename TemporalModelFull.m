function results = TemporalModelFull(ROI, opts)
% TemporalModelFull Multi-ROI bidirectional lag ridge regression
%
%   Fits temporal models for ALL neural ROIs simultaneously:
%   Y_i(t) = Σ β_i,j·X(t-j) with bidirectional lags (leads and lags)
%   where i indexes neural ROIs and j indexes lag values.
%
%   All ROIs share the same lagged behavior design matrix, but each gets
%   its own temporal kernel (beta weights) and regularization (lambda).
%   This is ~10-20x faster than fitting ROIs separately and enables direct
%   cross-ROI comparisons of temporal dynamics.
%
%   results = TemporalModelFull(ROI, opts)
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
%       target_neural_rois   - Cell array of ROI names to analyze (default: all)
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
%       temporal_kernels    - [n_ROIs × 1] struct array with fields:
%           .roi_name           - Name of this ROI
%           .beta_cv_mean       - CV-averaged beta coefficients [n_lags × 1]
%           .beta_cv_sem        - Standard error across CV folds
%           .beta_full_data     - Beta from full-data fit
%           .beta_cv_folds      - Beta weights from each CV fold [n_lags × n_folds]
%           .lag_indices        - Lag values in frames
%           .lag_times_sec      - Lag values in seconds
%           .peak_lag_sec       - Peak response lag (from CV mean)
%           .peak_beta          - Peak beta value (from CV mean)
%       performance         - [n_ROIs × 1] struct array with fields:
%           .roi_name           - Name of this ROI
%           .R2_cv_mean         - Mean R² across CV folds (realistic estimate)
%           .R2_cv_sem          - Standard error of CV R²
%           .R2_cv_folds        - R² for each fold [1 × n_folds]
%           .R2_full_data       - R² from full-data fit (optimistic)
%           .lambda_cv_mean     - Mean lambda across CV folds
%           .lambda_full_data   - Lambda from full-data fit
%       comparison          - Cross-ROI comparison structure:
%           .roi_names          - {1 × n_ROIs} cell array of names
%           .beta_matrix_cv     - [n_lags × n_ROIs] all CV-averaged kernels
%           .R2_all_rois        - [1 × n_ROIs] R² for each ROI
%           .peak_lags_all_sec  - [1 × n_ROIs] peak lag for each ROI
%           .peak_betas_all     - [1 × n_ROIs] peak beta for each ROI
%       predictions         - Model predictions (from full-data model)
%           .Y_pred             - [n_valid × n_ROIs] predicted fluorescence
%           .Y_actual           - [n_valid × n_ROIs] actual z-scored fluorescence
%           .behavior_trace_z   - Z-scored behavior trace
%       metadata            - Analysis configuration and provenance

if nargin < 2 || isempty(opts)
    opts = struct();
end

% Default options
defaults = struct(...
    'target_neural_rois', {{}}, ...
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

fprintf('=== TemporalModelFull: Multi-ROI bidirectional lag ridge regression ===\n');

%% 1. Extract ALL neural ROI traces
fluo_data = ROI.modalities.fluorescence.data;
fluo_labels = ROI.modalities.fluorescence.labels;
fluo_rate = ROI.modalities.fluorescence.sample_rate;

% Select target neural ROIs
if isempty(opts.target_neural_rois)
    % Use all ROIs
    neural_indices = 1:length(fluo_labels);
    opts.target_neural_rois = fluo_labels;
    fprintf('No target_neural_rois specified, using all %d ROIs\n', length(fluo_labels));
else
    % Find specified ROIs
    neural_indices = [];
    for i = 1:length(opts.target_neural_rois)
        idx = find(strcmpi(fluo_labels, opts.target_neural_rois{i}), 1);
        if isempty(idx)
            error('Neural ROI "%s" not found.\nAvailable: %s', ...
                opts.target_neural_rois{i}, strjoin(fluo_labels, ', '));
        end
        neural_indices(end+1) = idx;
    end
end

n_rois = length(neural_indices);
neural_traces = fluo_data(:, neural_indices);
roi_names = fluo_labels(neural_indices);

fprintf('Analyzing %d neural ROIs (%.1f Hz, %d frames):\n', ...
    n_rois, fluo_rate, size(neural_traces, 1));
for i = 1:n_rois
    fprintf('  %d. %s\n', i, roi_names{i});
end

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
min_length = min(size(neural_traces, 1), length(behavior_trace));
neural_traces = neural_traces(1:min_length, :);
behavior_trace = behavior_trace(1:min_length);

fprintf('\nMatched timepoints: %d frames (~%.1f s)\n', min_length, min_length / sampling_rate);

% Z-score each ROI independently
neural_traces_z = zeros(size(neural_traces));
for i = 1:n_rois
    neural_traces_z(:, i) = zscore(neural_traces(:, i));
end

behavior_trace_z = zscore(behavior_trace);

%% 6. Build bidirectional lag design matrix (SHARED across all ROIs)
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
% Y is now [n_valid × n_rois] - ALL ROIs together
Y_all = neural_traces_z(max_lag+1 : min_length-abs(min_lag), :);

fprintf('  Design matrix X: [%d × %d] (timepoints × lags)\n', size(X, 1), size(X, 2));
fprintf('  Outcome matrix Y: [%d × %d] (timepoints × ROIs)\n', size(Y_all, 1), size(Y_all, 2));
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

%% 8. Cross-validation with multi-ROI simultaneous fitting
fprintf('\nPerforming %d-fold blocked time-series cross-validation...\n', opts.cv_folds);
fprintf('  Fitting all %d ROIs simultaneously in each fold\n', n_rois);

cv_folds = opts.cv_folds;
fold_size = floor(n_valid / cv_folds);

% Pre-allocate storage for CV results
beta_cv_folds = zeros(n_lags_total, n_rois, cv_folds);  % [lags × ROIs × folds]
lambda_cv_folds = zeros(n_rois, cv_folds);              % [ROIs × folds]
R2_cv_folds = zeros(n_rois, cv_folds);                  % [ROIs × folds]
convergence_cv = zeros(n_rois, cv_folds);               % [ROIs × folds]

for fold = 1:cv_folds
    % Define test indices (contiguous block) - SAME for all ROIs
    test_start = (fold - 1) * fold_size + 1;
    test_end = min(fold * fold_size, n_valid);
    test_idx = test_start:test_end;
    train_idx = setdiff(1:n_valid, test_idx);

    % Extract train/test data
    X_train = X(train_idx, :);
    Y_train_all = Y_all(train_idx, :);  % All ROIs
    X_test = X(test_idx, :);
    Y_test_all = Y_all(test_idx, :);    % All ROIs

    % Fit ridge regression on training data for ALL ROIs simultaneously
    % ridgeMML returns [n_predictors × n_outcomes] betas and [1 × n_outcomes] lambdas
    [lambda_fold, beta_fold, conv_fail] = ridgeMML(Y_train_all, X_train, 1);

    % Store results for each ROI
    beta_cv_folds(:, :, fold) = beta_fold;      % [n_lags × n_rois]
    lambda_cv_folds(:, fold) = lambda_fold';    % [n_rois × 1]
    convergence_cv(:, fold) = conv_fail';       % [n_rois × 1]

    % Predict on test data for all ROIs
    Y_pred_test_all = X_test * beta_fold;       % [n_test × n_rois]

    % Compute test R² for each ROI independently
    for roi = 1:n_rois
        Y_test_roi = Y_test_all(:, roi);
        Y_pred_test_roi = Y_pred_test_all(:, roi);

        TSS_test = sum((Y_test_roi - mean(Y_test_roi)).^2);
        RSS_test = sum((Y_test_roi - Y_pred_test_roi).^2);
        R2_cv_folds(roi, fold) = max(0, 1 - RSS_test / TSS_test);
    end

    fprintf('  Fold %d/%d: R² range [%.4f, %.4f], mean lambda = %.4f\n', ...
        fold, cv_folds, min(R2_cv_folds(:, fold)), max(R2_cv_folds(:, fold)), ...
        mean(lambda_fold));
end

% Compute CV statistics per ROI
R2_cv_mean = mean(R2_cv_folds, 2);              % [n_rois × 1]
R2_cv_sem = std(R2_cv_folds, 0, 2) / sqrt(cv_folds);  % [n_rois × 1]
lambda_cv_mean = mean(lambda_cv_folds, 2);     % [n_rois × 1]

fprintf('\nCV Results Summary:\n');
fprintf('  R² (CV mean) range: [%.4f, %.4f]\n', min(R2_cv_mean), max(R2_cv_mean));
fprintf('  Best ROI: %s (R² = %.4f ± %.4f)\n', ...
    roi_names{R2_cv_mean == max(R2_cv_mean)}, max(R2_cv_mean), R2_cv_sem(R2_cv_mean == max(R2_cv_mean)));
fprintf('  Worst ROI: %s (R² = %.4f ± %.4f)\n', ...
    roi_names{R2_cv_mean == min(R2_cv_mean)}, min(R2_cv_mean), R2_cv_sem(R2_cv_mean == min(R2_cv_mean)));

if any(convergence_cv(:))
    n_failures = sum(convergence_cv(:));
    total = n_rois * cv_folds;
    warning('%d/%d CV fits reported convergence failure', n_failures, total);
end

%% 9. Fit full-data model for predictions (all ROIs simultaneously)
fprintf('\nFitting full-data model for all ROIs (for predictions)...\n');
[lambda_full, beta_full, convergence_full] = ridgeMML(Y_all, X, 1);

lambda_full = lambda_full';  % [n_rois × 1]

if any(convergence_full)
    warning('%d/%d ROIs reported convergence failure in full-data fit', ...
        sum(convergence_full), n_rois);
end

fprintf('  Lambda range: [%.4f, %.4f]\n', min(lambda_full), max(lambda_full));

% Generate predictions using full-data model
Y_pred_full = X * beta_full;  % [n_valid × n_rois]

% Compute full-data R² for each ROI
R2_full = zeros(n_rois, 1);
for roi = 1:n_rois
    Y_roi = Y_all(:, roi);
    Y_pred_roi = Y_pred_full(:, roi);

    TSS_full = sum((Y_roi - mean(Y_roi)).^2);
    RSS_full = sum((Y_roi - Y_pred_roi).^2);
    R2_full(roi) = max(0, 1 - RSS_full / TSS_full);
end

fprintf('  R² (full-data) range: [%.4f, %.4f]\n', min(R2_full), max(R2_full));

%% 10. Compute temporal kernel statistics from CV folds (per ROI)
fprintf('\nComputing temporal kernel statistics for each ROI...\n');

% Primary estimate: CV-averaged betas
beta_cv_mean_all = mean(beta_cv_folds, 3);  % [n_lags × n_rois]
beta_cv_sem_all = std(beta_cv_folds, 0, 3) / sqrt(cv_folds);  % [n_lags × n_rois]

% Find peak response for each ROI (from CV mean)
peak_lags_frames = zeros(n_rois, 1);
peak_lags_sec = zeros(n_rois, 1);
peak_betas = zeros(n_rois, 1);
peak_betas_sem = zeros(n_rois, 1);

for roi = 1:n_rois
    [peak_beta_abs, peak_idx] = max(abs(beta_cv_mean_all(:, roi)));
    peak_lags_frames(roi) = lag_values(peak_idx);
    peak_lags_sec(roi) = lag_times_sec(peak_idx);
    peak_betas(roi) = beta_cv_mean_all(peak_idx, roi);
    peak_betas_sem(roi) = beta_cv_sem_all(peak_idx, roi);
end

%% 11. Assemble results structure
results = struct();

% --- Per-ROI temporal kernels ---
results.temporal_kernels = repmat(struct(), n_rois, 1);
for roi = 1:n_rois
    results.temporal_kernels(roi).roi_name = roi_names{roi};
    results.temporal_kernels(roi).beta_cv_mean = beta_cv_mean_all(:, roi);
    results.temporal_kernels(roi).beta_cv_sem = beta_cv_sem_all(:, roi);
    results.temporal_kernels(roi).beta_full_data = beta_full(:, roi);
    results.temporal_kernels(roi).beta_cv_folds = squeeze(beta_cv_folds(:, roi, :));  % [n_lags × n_folds]
    results.temporal_kernels(roi).lag_indices = lag_values;
    results.temporal_kernels(roi).lag_times_sec = lag_times_sec;
    results.temporal_kernels(roi).peak_lag_frames = peak_lags_frames(roi);
    results.temporal_kernels(roi).peak_lag_sec = peak_lags_sec(roi);
    results.temporal_kernels(roi).peak_beta = peak_betas(roi);
    results.temporal_kernels(roi).peak_beta_sem = peak_betas_sem(roi);
end

% --- Per-ROI performance metrics ---
results.performance = repmat(struct(), n_rois, 1);
for roi = 1:n_rois
    results.performance(roi).roi_name = roi_names{roi};
    results.performance(roi).R2_cv_mean = R2_cv_mean(roi);
    results.performance(roi).R2_cv_sem = R2_cv_sem(roi);
    results.performance(roi).R2_cv_folds = R2_cv_folds(roi, :);
    results.performance(roi).R2_full_data = R2_full(roi);
    results.performance(roi).lambda_cv_mean = lambda_cv_mean(roi);
    results.performance(roi).lambda_full_data = lambda_full(roi);
    results.performance(roi).convergence_cv_failures = sum(convergence_cv(roi, :));
    results.performance(roi).convergence_full_failure = convergence_full(roi);
end

% --- Cross-ROI comparison ---
results.comparison = struct();
results.comparison.roi_names = roi_names;
results.comparison.beta_matrix_cv = beta_cv_mean_all;       % [n_lags × n_rois]
results.comparison.R2_all_rois = R2_cv_mean';               % [1 × n_rois]
results.comparison.peak_lags_all_sec = peak_lags_sec';      % [1 × n_rois]
results.comparison.peak_betas_all = peak_betas';            % [1 × n_rois]

% --- Predictions (from full-data model) ---
results.predictions = struct();
results.predictions.Y_pred = Y_pred_full;                   % [n_valid × n_rois]
results.predictions.Y_actual = Y_all;                       % [n_valid × n_rois]
results.predictions.behavior_trace_z = behavior_trace_z;

% --- Metadata ---
results.metadata = struct();
results.metadata.n_rois = n_rois;
results.metadata.roi_names = roi_names;
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
    PlotTemporalModelFull(results);
end

%% 13. Save results
if opts.save_results
    if isempty(opts.output_file)
        safe_behav = regexprep(opts.behavior_predictor, '\W+', '');
        default_output = sprintf('TemporalModelFull_%dROIs_vs_%s.mat', n_rois, safe_behav);
        output_file = fullfile(script_dir, default_output);
    else
        output_file = opts.output_file;
    end

    save(output_file, 'results', '-v7.3');
    fprintf('\nResults saved to:\n  %s\n', output_file);
else
    fprintf('\nResults not saved (opts.save_results == false)\n');
end

fprintf('\n=== TemporalModelFull complete ===\n');

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

