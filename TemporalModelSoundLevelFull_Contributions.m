function results = TemporalModelSoundLevelFull_Contributions(ROI, session_file, opts)
% TemporalModelSoundLevelFull_Contributions_v2  Motion + sound-level multi-ROI model with contributions.
%
%   Mirrors TemporalModelSoundLevelFull but additionally computes per-group
%   explained and unique contributions (single-variable fits + shuffle drops)
%   for the motion predictor, each sound-level block, and each lick block.
%
%   results = TemporalModelSoundLevelFull_Contributions_v2(ROI, session_file)
%   results = TemporalModelSoundLevelFull_Contributions_v2(ROI, session_file, opts)
%
%   See file header for option descriptions.

if nargin < 3 || isempty(opts)
    opts = struct();
end

defaults = struct(...
    'target_neural_rois', {{}}, ...
    'behavior_predictor', 'Face', ...
    'stim_kernel', 0:0.1:2, ...
    'lick_kernel', -0.5:0.1:5, ...
    'event_protocol', 'WNoiseLickWater', ...
    'min_lag', -5, ...
    'max_lag', 10, ...
    'min_lag_seconds', [], ...
    'max_lag_seconds', [], ...
    'remove_initial_seconds', 0, ...
    'cv_folds', 5, ...
    'zscore_design', true, ...
    'design_debug_file', '', ...
    'lambda_cap', 1e4, ...
    'ridge_time_limit', 5, ...
    'output_file', '', ...
    'save_results', true, ...
    'show_plots', true);

opts = populate_defaults(opts, defaults);

validate_roi_structure(ROI);
if ~exist(session_file, 'file')
    error('Session file "%s" not found.', session_file);
end

script_dir = fileparts(mfilename('fullpath'));
ensure_ridgeMML_on_path(script_dir);

fprintf('=== TemporalModelSoundLevelFull_Contributions_v2: multi-ROI motion+sound-level regression ===\n');
t_start = tic;
%% 1. Neural ROI selection
fluo_data = ROI.modalities.fluorescence.data;
fluo_labels = ROI.modalities.fluorescence.labels;
sampling_rate = ROI.modalities.fluorescence.sample_rate;

if isempty(opts.target_neural_rois)
    target_idx = 1:numel(fluo_labels);
else
    target_idx = find(ismember(lower(fluo_labels), lower(opts.target_neural_rois)));
    if isempty(target_idx)
        error('None of the requested target_neural_rois found.');
    end
end

roi_names = fluo_labels(target_idx);
neural_traces = fluo_data(:, target_idx);
n_rois = numel(roi_names);

fprintf('Neural ROIs: %d selected (%s%s)\n', n_rois, ...
    strjoin(roi_names(1:min(5, n_rois)), ', '), ternary(n_rois>5, ', ...', ''));

%% 2. Behavior predictor
behav_data = ROI.modalities.behavior.data;
behav_labels = ROI.modalities.behavior.labels;
behav_rate = ROI.modalities.behavior.sample_rate;
behav_idx = find(strcmpi(behav_labels, opts.behavior_predictor), 1);
if isempty(behav_idx)
    error('Behavior predictor "%s" not found.', opts.behavior_predictor);
end

behavior_trace = behav_data(:, behav_idx);
if abs(behav_rate - sampling_rate) > 0.01
    fprintf('Resampling behavior from %.2f Hz to %.2f Hz...\n', behav_rate, sampling_rate);
    behavior_trace = resample(behavior_trace, round(sampling_rate*1000), round(behav_rate*1000));
end

%% 3. Match trace lengths and build event design
min_length = min(size(neural_traces,1), numel(behavior_trace));
neural_traces = neural_traces(1:min_length, :);
behavior_trace = behavior_trace(1:min_length);
min_length_initial = min_length;

fprintf('Matched timepoints: %d frames (~%.1f s)\n', min_length_initial, min_length_initial / sampling_rate);
fprintf('Constructing event design from %s...\n', session_file);
design = build_event_design_matrix(session_file, opts.event_protocol, ...
    opts.stim_kernel, opts.lick_kernel, sampling_rate, min_length_initial);

trim_seconds = max(0, opts.remove_initial_seconds);
trim_frames = round(trim_seconds * sampling_rate);
if trim_frames > 0
    if trim_frames >= min_length_initial
        error('remove_initial_seconds (%.1f s) exceeds recording length (%.1f s).', ...
            trim_seconds, min_length_initial / sampling_rate);
    end
    fprintf('Trimming initial %.1f s (%d frames) from all signals/design.\n', trim_seconds, trim_frames);
    neural_traces = neural_traces(trim_frames+1:end, :);
    behavior_trace = behavior_trace(trim_frames+1:end);
    min_length = min_length_initial - trim_frames;
    design = trim_initial_design_segment(design, trim_frames, trim_seconds, sampling_rate);
else
    min_length = min_length_initial;
end

X_events_full = design.matrix;
regressor_names_events = design.regressor_names;
group_info = design.group_info;
n_event_regressors = size(X_events_full, 2);
%% 4. Lag configuration
if ~isempty(opts.min_lag_seconds)
    opts.min_lag = round(opts.min_lag_seconds * sampling_rate);
    fprintf('Min lag %.3f s -> %d frames\n', opts.min_lag_seconds, opts.min_lag);
end
if ~isempty(opts.max_lag_seconds)
    opts.max_lag = round(opts.max_lag_seconds * sampling_rate);
    fprintf('Max lag %.3f s -> %d frames\n', opts.max_lag_seconds, opts.max_lag);
end

min_lag = opts.min_lag;
max_lag = opts.max_lag;
if min_lag >= max_lag
    error('min_lag (%d) must be less than max_lag (%d).', min_lag, max_lag);
end

lag_values = (min_lag:max_lag)';
lag_times_sec = lag_values / sampling_rate;
n_lags_total = numel(lag_values);
n_frames_lost_start = max_lag;
n_frames_lost_end = abs(min_lag);
n_valid = min_length - n_frames_lost_start - n_frames_lost_end;
if n_valid <= 0
    error('Not enough frames for requested lags. Need at least %d frames, have %d.', ...
        n_frames_lost_start + n_frames_lost_end + 1, min_length);
end

%% 5. Build design matrix
fprintf('Z-scoring neural traces (%d ROIs) and behavior predictor...\n', n_rois);
neural_z = zscore(neural_traces);
behavior_z_full = zscore(behavior_trace);

Y_all = neural_z(n_frames_lost_start+1 : min_length - n_frames_lost_end, :);
X_motion = zeros(n_valid, n_lags_total);

lag_idx = 0;
for lag = min_lag:max_lag
    lag_idx = lag_idx + 1;
    start_idx = n_frames_lost_start + 1 - lag;
    end_idx = min_length - n_frames_lost_end - lag;
    X_motion(:, lag_idx) = behavior_z_full(start_idx:end_idx);
end

X_events = X_events_full(n_frames_lost_start+1 : min_length - n_frames_lost_end, :);
X = [X_motion, X_events];

if opts.zscore_design
    X = zscore(X);
else
    X = bsxfun(@minus, X, mean(X, 1));
end
X(~isfinite(X)) = 0;

n_regressors = size(X, 2);
motion_group_label = sprintf('%s motion (lags)', opts.behavior_predictor);
group_labels = [{motion_group_label}; arrayfun(@(g) g.label, group_info, 'UniformOutput', false)];
group_indices = cell(numel(group_labels),1);
group_indices{1} = 1:n_lags_total;
for g = 1:numel(group_info)
    group_info(g).indices = group_info(g).indices + n_lags_total;
    group_indices{g+1} = group_info(g).indices;
end
n_groups_total = numel(group_labels);

if ~isempty(opts.design_debug_file)
    design_debug = struct();
    design_debug.design = design;
    design_debug.X_motion = X_motion;
    design_debug.X_events = X_events;
    design_debug.X = X;
    design_debug.group_labels = group_labels;
    design_debug.group_indices = group_indices;
    design_debug.lag_values = lag_values;
    design_debug.lag_times_sec = lag_times_sec;
    design_debug.n_valid = n_valid;
    design_debug.n_motion_regressors = n_lags_total;
    design_debug.n_event_regressors = n_event_regressors;
    design_debug.behavior_predictor = opts.behavior_predictor;
    design_debug.zscore_design = opts.zscore_design;
    try
        save(opts.design_debug_file, 'design_debug', '-v7.3');
        fprintf('Design debug dump saved to %s\n', opts.design_debug_file);
    catch ME
        warning('Failed to save design debug dump (%s).', ME.message);
    end
end

fprintf('Combined design matrix: %d timepoints x %d regressors (%d motion + %d event)\n', ...
    n_valid, n_regressors, n_lags_total, n_event_regressors);

%% 6. Diagnostics
if n_regressors > 1
    corr_matrix = corr(X);
    off_diag = corr_matrix(~eye(n_regressors));
    fprintf('Design corr: max %.3f, mean %.3f\n', max(off_diag), mean(off_diag));
else
    fprintf('Design corr: single regressor\n');
end
condition_num = cond(X'*X);
fprintf('Condition number (X''X): %.2f\n', condition_num);
%% 7. Cross-validation
cv_folds = min(opts.cv_folds, max(1, n_valid));
if cv_folds ~= opts.cv_folds
    fprintf('Reducing cv_folds from %d to %d due to limited samples.\n', opts.cv_folds, cv_folds);
end
fold_size = max(1, floor(n_valid / cv_folds));

beta_cv_folds = zeros(n_regressors, n_rois, cv_folds);
lambda_cv_folds = zeros(n_rois, cv_folds);
R2_cv_folds = zeros(n_rois, cv_folds);
convergence_cv = zeros(n_rois, cv_folds);
group_single_R2 = nan(n_groups_total, n_rois, cv_folds);
group_shuffle_R2 = nan(n_groups_total, n_rois, cv_folds);
var_tol = 1e-6;

fprintf('\nPerforming %d-fold blocked CV across all ROIs simultaneously...\n', cv_folds);
for fold = 1:cv_folds
    test_start = (fold - 1) * fold_size + 1;
    test_end = min(fold * fold_size, n_valid);
    fprintf('[diag %.2fs] Starting fold %d/%d (test %d-%d)\n', toc(t_start), fold, cv_folds, test_start, test_end);
    test_idx = test_start:test_end;
    train_idx = setdiff(1:n_valid, test_idx);

    X_train = X(train_idx, :);
    Y_train = Y_all(train_idx, :);
    X_test = X(test_idx, :);
    Y_test = Y_all(test_idx, :);

    X_train_mean = mean(X_train, 1);
    Y_train_mean = mean(Y_train, 1);

    [lambda_fold, beta_fold, conv_fail] = ridgeMML_with_cap(Y_train, X_train, 1, opts.lambda_cap, ...
        sprintf('fold%d_full', fold), opts.ridge_time_limit);

    beta_cv_folds(:, :, fold) = beta_fold;
    lambda_cv_folds(:, fold) = lambda_fold(:);
    convergence_cv(:, fold) = conv_fail(:);

    X_test_centered = bsxfun(@minus, X_test, X_train_mean);
    Y_pred = bsxfun(@plus, X_test_centered * beta_fold, Y_train_mean);

    for r = 1:n_rois
        Y_t = Y_test(:, r);
        Y_p = Y_pred(:, r);
        TSS = sum((Y_t - mean(Y_t)).^2);
        RSS = sum((Y_t - Y_p).^2);
        R2_cv_folds(r, fold) = max(0, 1 - RSS / max(TSS, eps));
    end

    % Contribution metrics
    for g = 1:n_groups_total
        fprintf('[diag %.2fs]   Fold %d: processing group %d/%d (%s)\n', ...
            toc(t_start), fold, g, n_groups_total, group_labels{g});
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end

        X_train_single = X_train(:, idx);
        X_test_single = X_test(:, idx);
        col_std = std(X_train_single, 0, 1);
        fprintf('[diag %.2fs]     Fold %d %s column std (min %.3g, max %.3g, %d/%d >= %.1e)\n', ...
            toc(t_start), fold, group_labels{g}, min(col_std), max(col_std), ...
            sum(col_std >= var_tol), numel(col_std), var_tol);

        valid_cols = col_std >= var_tol;
        if isempty(X_train_single) || all(col_std < var_tol)
            fprintf('[diag %.2fs]     Fold %d %s near-zero variance; skipping group.\n', ...
                toc(t_start), fold, group_labels{g});
            group_single_R2(g, :, fold) = 0;
            group_shuffle_R2(g, :, fold) = NaN;
            continue;
        elseif ~all(valid_cols)
            fprintf('[diag %.2fs]     Fold %d %s dropping %d low-variance columns.\n', ...
                toc(t_start), fold, group_labels{g}, sum(~valid_cols));
            X_train_single = X_train_single(:, valid_cols);
            X_test_single = X_test_single(:, valid_cols);
            idx = idx(valid_cols);
        end

        if size(X_train_single, 2) > 1
            corr_mat = corr(X_train_single);
            corr_mat(~isfinite(corr_mat)) = 0;
            mask = ~eye(size(corr_mat,1));
            if any(mask(:))
                max_corr = max(abs(corr_mat(mask)));
            else
                max_corr = 0;
            end
            cond_val = cond(X_train_single' * X_train_single);
            fprintf('[diag %.2fs]     Fold %d %s corr diag: max |r|=%.3f, cond(X''X)=%.3g\n', ...
                toc(t_start), fold, group_labels{g}, max_corr, cond_val);
        else
            fprintf('[diag %.2fs]     Fold %d %s corr diag: single column\n', ...
                toc(t_start), fold, group_labels{g});
        end

        diag_single_tic = tic;
        X_train_single_mean = mean(X_train_single, 1);
        [~, beta_single, ~] = ridgeMML_with_cap(Y_train, X_train_single, 1, opts.lambda_cap, ...
            sprintf('fold%d_%s_single', fold, group_labels{g}), opts.ridge_time_limit);
        fprintf('[diag %.2fs]     Fold %d %s single ridge done (%d cols, %.2fs)\n', ...
            toc(t_start), fold, group_labels{g}, numel(idx), toc(diag_single_tic));
        X_test_single_centered = bsxfun(@minus, X_test_single, X_train_single_mean);
        Y_pred_single = bsxfun(@plus, X_test_single_centered * beta_single, Y_train_mean);
        for r = 1:n_rois
            Y_t = Y_test(:, r);
            Y_p = Y_pred_single(:, r);
            TSS = sum((Y_t - mean(Y_t)).^2);
            RSS = sum((Y_t - Y_p).^2);
            group_single_R2(g, r, fold) = max(0, 1 - RSS / max(TSS, eps)) * 100;
        end

        orig_train_cols = X_train(:, idx);
        orig_test_cols = X_test(:, idx);
        perm_train = randperm(size(X_train, 1));
        perm_test = randperm(size(X_test, 1));
        X_train(:, idx) = orig_train_cols(perm_train, :);
        X_test(:, idx) = orig_test_cols(perm_test, :);
        X_train_shuff_mean = mean(X_train, 1);
        diag_shuffle_tic = tic;
        [~, beta_shuff, ~] = ridgeMML_with_cap(Y_train, X_train, 1, opts.lambda_cap, ...
            sprintf('fold%d_%s_shuffle', fold, group_labels{g}), opts.ridge_time_limit);
        fprintf('[diag %.2fs]     Fold %d %s shuffle ridge done (%d cols, %.2fs)\n', ...
            toc(t_start), fold, group_labels{g}, numel(idx), toc(diag_shuffle_tic));
        X_test_shuff_centered = bsxfun(@minus, X_test, X_train_shuff_mean);
        Y_pred_shuff = bsxfun(@plus, X_test_shuff_centered * beta_shuff, Y_train_mean);
        for r = 1:n_rois
            Y_t = Y_test(:, r);
            Y_p = Y_pred_shuff(:, r);
            TSS = sum((Y_t - mean(Y_t)).^2);
            RSS = sum((Y_t - Y_p).^2);
            group_shuffle_R2(g, r, fold) = max(0, 1 - RSS / max(TSS, eps)) * 100;
        end
        X_train(:, idx) = orig_train_cols;
        X_test(:, idx) = orig_test_cols;
    end

    fprintf('  Fold %d/%d: R^2 range %.4f-%.4f\n', fold, cv_folds, ...
        min(R2_cv_folds(:, fold)), max(R2_cv_folds(:, fold)));
end
fprintf('[diag %.2fs] Cross-validation finished.\n', toc(t_start));

R2_cv_mean = mean(R2_cv_folds, 2);
R2_cv_sem = std(R2_cv_folds, 0, 2) / sqrt(cv_folds);
lambda_cv_mean = mean(lambda_cv_folds, 2);

%% 8. Full-data fit
[lambda_full, beta_full, convergence_full] = ridgeMML_with_cap(Y_all, X, 1, opts.lambda_cap, 'full_data', opts.ridge_time_limit);
lambda_full = lambda_full(:);

X_mean_full = mean(X, 1);
Y_mean_full = mean(Y_all, 1);
Y_pred_full = bsxfun(@plus, bsxfun(@minus, X, X_mean_full) * beta_full, Y_mean_full);
R2_full = zeros(n_rois, 1);
for r = 1:n_rois
    Y_r = Y_all(:, r);
    Y_hat = Y_pred_full(:, r);
    TSS = sum((Y_r - mean(Y_r)).^2);
    RSS = sum((Y_r - Y_hat).^2);
    R2_full(r) = max(0, 1 - RSS / max(TSS, eps));
end
fprintf('[diag %.2fs] Full-data fit completed.\n', toc(t_start));
%% 9. Motion kernel summaries
beta_motion_cv = beta_cv_folds(1:n_lags_total, :, :);
beta_motion_mean = mean(beta_motion_cv, 3);
beta_motion_sem = std(beta_motion_cv, 0, 3) / sqrt(cv_folds);

temporal_kernels = repmat(struct(), n_rois, 1);
for r = 1:n_rois
    beta_cv_mean = beta_motion_mean(:, r);
    beta_cv_sem = beta_motion_sem(:, r);
    beta_full_r = beta_full(1:n_lags_total, r);
    [~, peak_idx] = max(abs(beta_cv_mean));
    temporal_kernels(r).roi_name = roi_names{r};
    temporal_kernels(r).beta_cv_mean = beta_cv_mean;
    temporal_kernels(r).beta_cv_sem = beta_cv_sem;
    temporal_kernels(r).beta_full_data = beta_full_r;
    temporal_kernels(r).beta_cv_folds = squeeze(beta_motion_cv(:, r, :));
    temporal_kernels(r).lag_indices = lag_values;
    temporal_kernels(r).lag_times_sec = lag_times_sec;
    temporal_kernels(r).peak_lag_sec = lag_times_sec(peak_idx);
    temporal_kernels(r).peak_beta = beta_cv_mean(peak_idx);
end

%% 10. Event kernel summaries per ROI
event_kernels = cell(n_rois, 1);
for r = 1:n_rois
    beta_cv_roi = squeeze(beta_cv_folds(:, r, :));
    beta_full_roi = beta_full(:, r);
    event_kernels{r} = compute_event_kernels_multi(beta_cv_roi, beta_full_roi, group_info, cv_folds);
end

%% 11. Performance structs
performance = repmat(struct(), n_rois, 1);
for r = 1:n_rois
    performance(r).roi_name = roi_names{r};
    performance(r).R2_cv_mean = R2_cv_mean(r);
    performance(r).R2_cv_sem = R2_cv_sem(r);
    performance(r).R2_cv_folds = R2_cv_folds(r, :);
    performance(r).R2_full_data = R2_full(r);
    performance(r).lambda_cv_mean = lambda_cv_mean(r);
    performance(r).lambda_full_data = lambda_full(r);
    performance(r).convergence_cv_failures = sum(convergence_cv(r, :) ~= 0);
    performance(r).convergence_full_failure = logical(convergence_full(r));
end

%% 12. Cross-ROI comparison
comparison = struct();
comparison.roi_names = roi_names;
comparison.beta_matrix_cv = beta_motion_mean;
comparison.R2_all_rois = R2_cv_mean;
comparison.peak_lags_all_sec = arrayfun(@(tk) tk.peak_lag_sec, temporal_kernels);
comparison.peak_betas_all = arrayfun(@(tk) tk.peak_beta, temporal_kernels);

%% 13. Predictions and metadata
predictions = struct();
predictions.Y_pred = Y_pred_full;
predictions.Y_actual = Y_all;
predictions.behavior_trace_z = behavior_z_full;
predictions.time_vector = ((n_frames_lost_start):(n_frames_lost_start + n_valid - 1))' / sampling_rate;
predictions.truncated_behavior = behavior_z_full(n_frames_lost_start+1 : min_length - n_frames_lost_end);

results = struct();
results.temporal_kernels = temporal_kernels;
results.event_kernels = event_kernels;
results.performance = performance;
results.predictions = predictions;
results.comparison = comparison;
results.design_matrix = struct();
motion_regressor_names = arrayfun(@(lag) sprintf('%s_lag%+d', opts.behavior_predictor, lag), ...
    lag_values, 'UniformOutput', false);
motion_regressor_names = motion_regressor_names(:)';
event_regressor_names = regressor_names_events(:)';
results.design_matrix.regressor_names = [motion_regressor_names, event_regressor_names];
results.design_matrix.motion = struct('lag_indices', lag_values, 'lag_times_sec', lag_times_sec, ...
    'column_indices', 1:n_lags_total, 'predictor', opts.behavior_predictor);
results.design_matrix.events = design.events;
results.design_matrix.group_info = group_info;
results.design_matrix.group_labels = group_labels;
results.design_matrix.group_indices = group_indices;
results.design_matrix.event_counts = design.event_counts;
results.design_matrix.n_zero_variance = design.n_zero_variance;

results.metadata = struct();
results.metadata.session_file = session_file;
results.metadata.target_neural_rois = roi_names;
results.metadata.behavior_predictor = opts.behavior_predictor;
results.metadata.stim_kernel = opts.stim_kernel(:);
results.metadata.lick_kernel = opts.lick_kernel(:);
results.metadata.sampling_rate = sampling_rate;
results.metadata.min_lag = min_lag;
results.metadata.max_lag = max_lag;
results.metadata.min_lag_seconds = lag_times_sec(1);
results.metadata.max_lag_seconds = lag_times_sec(end);
results.metadata.n_timepoints_total = min_length;
results.metadata.n_timepoints_original = min_length_initial;
results.metadata.n_timepoints_removed_initial = trim_frames;
results.metadata.initial_trim_seconds = trim_seconds;
results.metadata.frames_lost_start = n_frames_lost_start;
results.metadata.frames_lost_end = n_frames_lost_end;
results.metadata.cv_folds = cv_folds;
results.metadata.n_motion_regressors = n_lags_total;
results.metadata.n_event_regressors = n_event_regressors;
results.metadata.n_rois = n_rois;
results.metadata.group_labels = group_labels;
results.metadata.n_groups_total = n_groups_total;
results.metadata.condition_number = condition_num;
results.metadata.timestamp = datestr(now);
[mouse_label_meta, recording_label_meta] = parse_session_labels(session_file);
results.metadata.session_mouse_label = mouse_label_meta;
results.metadata.session_recording_label = recording_label_meta;
if isfield(ROI, 'metadata') && isfield(ROI.metadata, 'source')
    results.metadata.source_roi_file = ROI.metadata.source;
end

baseline_percent = R2_cv_folds * 100;
group_explained_mean = nan(n_groups_total, n_rois);
group_explained_std = nan(n_groups_total, n_rois);
group_unique_mean = nan(n_groups_total, n_rois);
group_unique_std = nan(n_groups_total, n_rois);

for g = 1:n_groups_total
    for r = 1:n_rois
        single_vals = squeeze(group_single_R2(g, r, :));
        valid_single = ~isnan(single_vals);
        if any(valid_single)
            vals = single_vals(valid_single);
            group_explained_mean(g, r) = mean(vals);
            group_explained_std(g, r) = std(vals);
        end

        shuffle_vals = squeeze(group_shuffle_R2(g, r, :));
        valid_shuffle = ~isnan(shuffle_vals);
        if any(valid_shuffle)
            vals = shuffle_vals(valid_shuffle);
            vals = vals(:);
            baseline_vals = baseline_percent(r, valid_shuffle)';
            diff_vals = baseline_vals - vals;
            diff_vals(diff_vals < 0) = 0;
            group_unique_mean(g, r) = mean(diff_vals);
            group_unique_std(g, r) = std(diff_vals);
        end
    end
end

results.contributions = struct();
results.contributions.group_labels = group_labels;
results.contributions.group_indices = group_indices;
results.contributions.group_single_R2 = group_single_R2;
results.contributions.group_shuffle_R2 = group_shuffle_R2;
results.contributions.group_explained_mean = group_explained_mean;
results.contributions.group_explained_std = group_explained_std;
results.contributions.group_unique_mean = group_unique_mean;
results.contributions.group_unique_std = group_unique_std;
results.contributions.R2_cv_percent = baseline_percent;
results.contributions.R2_mean_percent = R2_cv_mean * 100;

if opts.show_plots
    try
        plot_multi_roi_kernels(results);
        plot_multi_roi_heatmap(results);
        plot_multi_roi_performance(results);
    catch ME
        warning('Plotting failed: %s', ME.message);
    end
end

if opts.save_results
    save_timestamp = datestr(now, 'yymmdd_HHMMSS');
    [mouse_label, recording_label] = parse_session_labels(session_file);
    if isempty(opts.output_file)
        default_file = sprintf('TemporalModelSoundLevelFull_Contributions_v2_%s_%s_%s.mat', ...
            mouse_label, recording_label, save_timestamp);
        output_file = fullfile(script_dir, default_file);
    else
        output_file = opts.output_file;
    end
    results.metadata.output_save_timestamp = save_timestamp;
    results.metadata.output_mouse_label = mouse_label;
    results.metadata.output_recording_label = recording_label;
    save(output_file, 'results', '-v7.3');
    fprintf('Results saved to %s\n', output_file);
else
    fprintf('Results not saved (opts.save_results == false)\n');
end

fprintf('=== TemporalModelSoundLevelFull_Contributions_v2 complete ===\n');
end

function [lambda_out, beta_out, conv_fail] = ridgeMML_with_cap(Y, X, recenter, lambda_cap, context_label, time_limit)
if nargin < 5 || isempty(context_label)
    context_label = '';
end
if nargin < 6 || isempty(time_limit)
    time_limit = Inf;
end
[lambda_out, beta_out, conv_fail] = ridgeMML(Y, X, recenter, [], time_limit);
if any(conv_fail) && isfinite(time_limit)
    warning('TemporalModelSoundLevelFull_Contributions_v2:RidgeTimeout', ...
        '[%s] ridgeMML reported convergence issues (time limit %.2fs).', ...
        context_label, time_limit);
end

if nargin < 4 || isempty(lambda_cap) || ~isfinite(lambda_cap)
    return;
end
exceed_mask = lambda_out > lambda_cap;
if any(exceed_mask)
    warning('TemporalModelSoundLevelFull_Contributions_v2:LambdaCap', ...
        '[%s] lambda exceeded cap %.3g for %d/%d ROIs; clamping and recomputing betas.', ...
        context_label, lambda_cap, nnz(exceed_mask), numel(lambda_out));
    lambda_clamped = lambda_out;
    lambda_clamped(exceed_mask) = lambda_cap;
    [~, beta_recalc] = ridgeMML(Y, X, recenter, lambda_clamped, time_limit);
    beta_out = beta_recalc;
    lambda_out = lambda_clamped;
end
end

function [mouse_label, recording_label] = parse_session_labels(session_file)
[~, base_name, ~] = fileparts(session_file);
tokens = regexp(base_name, '_', 'split');
mouse_label = 'Mouse';
recording_label = 'Recording';
if ~isempty(tokens)
    candidate = regexprep(tokens{1}, '\W+', '');
    if ~isempty(candidate)
        mouse_label = candidate;
    end
end
recording_parts = {};
for i = 2:numel(tokens)
    token = tokens{i};
    if all(isstrprop(token, 'digit'))
        recording_parts{end+1} = token; %#ok<AGROW>
    elseif isempty(recording_parts)
        continue;
    else
        break;
    end
end
if ~isempty(recording_parts)
    recording_label = strjoin(recording_parts, '');
end
mouse_label = regexprep(mouse_label, '\W+', '');
recording_label = regexprep(recording_label, '\W+', '');
if isempty(mouse_label)
    mouse_label = 'Mouse';
end
if isempty(recording_label)
    recording_label = 'Recording';
end
end

%% ================= Helper functions =================
function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end

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
    error('ROI.modalities missing "behavior" field');
end
fluo = ROI.modalities.fluorescence;
for fld = {'data','labels','sample_rate'}
    if ~isfield(fluo, fld{1})
        error('ROI.modalities.fluorescence missing "%s"', fld{1});
    end
end
behav = ROI.modalities.behavior;
for fld = {'data','labels','sample_rate'}
    if ~isfield(behav, fld{1})
        error('ROI.modalities.behavior missing "%s"', fld{1});
    end
end
end

function ensure_ridgeMML_on_path(script_dir)
if exist('ridgeMML', 'file') == 2
    return;
end
candidate_paths = {
    fullfile(script_dir)
    fullfile(script_dir, '..')
    'C:\\Users\\shires\\Downloads'
    'H:\\IsaacAndGarrettMatlabScripts\\glm code\\Puff_Dataset'
};
for i = 1:numel(candidate_paths)
    if exist(candidate_paths{i}, 'dir')
        addpath(candidate_paths{i});
    end
    if exist('ridgeMML', 'file') == 2
        return;
    end
end
error('ridgeMML.m not found on MATLAB path.');
end

function design = build_event_design_matrix(session_file, protocol_label, stim_kernel, lick_kernel, sampling_rate, n_timepoints)
total_time_s = n_timepoints / sampling_rate;
session_data = load(session_file);
if isfield(session_data, 'SessionData')
    [stimulus_events, lick_events] = extract_behavioral_events_session( ...
        session_data.SessionData, protocol_label, total_time_s);
elseif all(isfield(session_data, {'eventID','timestamps','eventNameList','state'}))
    [stimulus_events, lick_events] = extract_behavioral_events_log(session_data, total_time_s);
else
    error('Session file %s missing SessionData or event log fields.', session_file);
end

sound_cfg = get_sound_level_info();
design_matrix = [];
regressor_names = {};
column_group = [];
column_lags = [];
group_defs = struct('label', {}, 'prefix', {});

for s = 1:numel(sound_cfg.field_names)
    field_name = sound_cfg.field_names{s};
    group_defs(end+1).label = sound_cfg.labels{s}; %#ok<AGROW>
    group_defs(end).prefix = sound_cfg.prefixes{s};
    group_idx = numel(group_defs);

    level_times = getfield_or_empty(stimulus_events, field_name);
    level_intensity = ones(size(level_times));
    sound_regressors = create_stimulus_regressors( ...
        level_times, level_intensity, n_timepoints, sampling_rate, stim_kernel);
    design_matrix = [design_matrix, sound_regressors]; %#ok<AGROW>
    for i = 1:length(stim_kernel)
        regressor_names{end+1} = sprintf('%s_t%.1fs', field_name, stim_kernel(i)); %#ok<AGROW>
        column_group(end+1) = group_idx; %#ok<AGROW>
        column_lags(end+1) = stim_kernel(i); %#ok<AGROW>
    end
end

lick_groups = {
    'post_stimulus',        'Lick post-stimulus',        'lick_post_stim_';
    'post_water_cued',      'Lick post-water (cued)',    'lick_water_cued_';
    'post_water_uncued',    'Lick post-water (uncued)',  'lick_water_uncued_';
    'post_water_omission',  'Lick post-water (omission)','lick_water_omission_'};
for lf = 1:size(lick_groups,1)
    field_name = lick_groups{lf,1};
    label = lick_groups{lf,2};
    prefix = lick_groups{lf,3};
    group_defs(end+1).label = label; %#ok<AGROW>
    group_defs(end).prefix = prefix;
    group_idx = numel(group_defs);

    lick_reg = create_lick_regressors( ...
        lick_events.(field_name), n_timepoints, sampling_rate, lick_kernel);
    design_matrix = [design_matrix, lick_reg]; %#ok<AGROW>
    base_name = prefix(1:end-1);
    for i = 1:length(lick_kernel)
        regressor_names{end+1} = sprintf('%s_t%.1fs', base_name, lick_kernel(i)); %#ok<AGROW>
        column_group(end+1) = group_idx; %#ok<AGROW>
        column_lags(end+1) = lick_kernel(i); %#ok<AGROW>
    end
end

reg_std = std(design_matrix, 0, 1);
zero_mask = reg_std < 1e-10;
if any(zero_mask)
    design_matrix(:, zero_mask) = [];
    regressor_names(zero_mask) = [];
    column_group(zero_mask) = [];
    column_lags(zero_mask) = [];
end

group_info = repmat(struct('label','', 'prefix','', 'indices',[], 'lag_times_sec',[]), numel(group_defs), 1);
for g = 1:numel(group_defs)
    idx = find(column_group == g);
    group_info(g).label = group_defs(g).label;
    group_info(g).prefix = group_defs(g).prefix;
    group_info(g).indices = idx;
    group_info(g).lag_times_sec = column_lags(idx);
end

lick_post_water_all = [lick_events.post_water_cued(:); lick_events.post_water_uncued(:); lick_events.post_water_omission(:)];
lick_post_water_all = lick_post_water_all(~isnan(lick_post_water_all));
lick_post_water_all = sort(lick_post_water_all);

event_counts = struct();
event_struct = struct();
noise_onsets_all = [];
noise_intensities_all = [];
for s = 1:numel(sound_cfg.field_names)
    field_name = sound_cfg.field_names{s};
    values = getfield_or_empty(stimulus_events, field_name);
    event_counts.(field_name) = numel(values);
    event_struct.(sprintf('%s_onsets', field_name)) = values(:);
    noise_onsets_all = [noise_onsets_all; values(:)]; %#ok<AGROW>
    noise_intensities_all = [noise_intensities_all; repmat(sound_cfg.db_values(s), numel(values), 1)]; %#ok<AGROW>
end
[noise_onsets_all, sort_idx] = sort(noise_onsets_all);
if isempty(sort_idx)
    noise_intensities_all = zeros(0,1);
else
    noise_intensities_all = noise_intensities_all(sort_idx);
end
event_counts.noise_primary = numel(noise_onsets_all);
event_counts.lick_post_stimulus = numel(lick_events.post_stimulus);
event_counts.lick_post_water_cued = numel(lick_events.post_water_cued);
event_counts.lick_post_water_uncued = numel(lick_events.post_water_uncued);
event_counts.lick_post_water_omission = numel(lick_events.post_water_omission);

event_struct.noise_onsets = noise_onsets_all;
event_struct.noise_intensities = noise_intensities_all;
event_struct.lick_post_stimulus = lick_events.post_stimulus(:);
event_struct.lick_post_water_cued = lick_events.post_water_cued(:);
event_struct.lick_post_water_uncued = lick_events.post_water_uncued(:);
event_struct.lick_post_water_omission = lick_events.post_water_omission(:);
event_struct.lick_post_water_all = lick_post_water_all;

design = struct();
design.matrix = design_matrix;
design.regressor_names = regressor_names;
design.group_info = group_info;
design.n_zero_variance = sum(zero_mask);
design.event_counts = event_counts;
design.events = event_struct;
end
function event_kernels = compute_event_kernels_multi(beta_cv_roi, beta_full_roi, group_info, cv_folds)
    n_groups = numel(group_info);
    event_kernels = repmat(struct('label','', 'lag_times_sec',[], 'beta_cv_mean',[], ...
        'beta_cv_sem',[], 'beta_full',[]), n_groups, 1);
    for g = 1:n_groups
        idx = group_info(g).indices;
        if isempty(idx)
            continue;
        end
    beta_cv = beta_cv_roi(idx, :);
    beta_mean = mean(beta_cv, 2);
    if cv_folds > 1
        beta_sem = std(beta_cv, 0, 2) / sqrt(cv_folds);
    else
        beta_sem = zeros(size(beta_mean));
    end
    event_kernels(g).label = group_info(g).label;
    event_kernels(g).lag_times_sec = group_info(g).lag_times_sec(:);
    event_kernels(g).beta_cv_mean = beta_mean;
    event_kernels(g).beta_cv_sem = beta_sem;
    event_kernels(g).beta_full = beta_full_roi(idx);
end
end
function design = trim_initial_design_segment(design, trim_frames, trim_seconds, sampling_rate)
if trim_frames <= 0 || isempty(design)
    return;
end
if size(design.matrix, 1) <= trim_frames
    error('Initial trim removes all event regressors.');
end
design.matrix = design.matrix(trim_frames+1:end, :);
total_time_s = size(design.matrix, 1) / sampling_rate;
if isfield(design, 'events') && ~isempty(design.events)
    design.events = shift_and_trim_events(design.events, trim_seconds, total_time_s);
end
design.event_counts = summarize_event_counts(design.events);
end

function events = shift_and_trim_events(events, trim_seconds, total_time_s)
if isempty(events)
    return;
end
sound_cfg = get_sound_level_info();
noise_onsets = [];
noise_intensities = [];
for s = 1:numel(sound_cfg.field_names)
    onset_field = sprintf('%s_onsets', sound_cfg.field_names{s});
    if isfield(events, onset_field)
        times = events.(onset_field)(:) - trim_seconds;
        keep = times >= 0 & times <= total_time_s;
        trimmed_times = times(keep);
        events.(onset_field) = trimmed_times;
        noise_onsets = [noise_onsets; trimmed_times]; %#ok<AGROW>
        noise_intensities = [noise_intensities; repmat(sound_cfg.db_values(s), numel(trimmed_times), 1)]; %#ok<AGROW>
    end
end
if isempty(noise_onsets)
    events.noise_onsets = zeros(0,1);
    events.noise_intensities = zeros(0,1);
else
    [noise_onsets, sort_idx] = sort(noise_onsets);
    if isempty(sort_idx)
        events.noise_intensities = noise_intensities;
    else
        events.noise_intensities = noise_intensities(sort_idx);
    end
    events.noise_onsets = noise_onsets;
end
lick_fields = {'lick_post_stimulus','lick_post_water_cued','lick_post_water_uncued','lick_post_water_omission'};
for i = 1:numel(lick_fields)
    field = lick_fields{i};
    if isfield(events, field)
        times = events.(field)(:);
        times = times - trim_seconds;
        keep = times >= 0 & times <= total_time_s;
        events.(field) = times(keep);
    end
end
if isfield(events, 'lick_post_water_all')
    events.lick_post_water_all = sort([ ...
        getfield_or_empty(events, 'lick_post_water_cued'); ...
        getfield_or_empty(events, 'lick_post_water_uncued'); ...
        getfield_or_empty(events, 'lick_post_water_omission')]);
end
end

function vals = getfield_or_empty(struct_in, field)
if isfield(struct_in, field)
    vals = struct_in.(field)(:);
else
    vals = [];
end
end

function counts = summarize_event_counts(events)
sound_cfg = get_sound_level_info();
counts = struct();
sound_total = 0;
for s = 1:numel(sound_cfg.field_names)
    counts.(sound_cfg.field_names{s}) = 0;
end
counts.noise_primary = 0;
counts.lick_post_stimulus = 0;
counts.lick_post_water_cued = 0;
counts.lick_post_water_uncued = 0;
counts.lick_post_water_omission = 0;
if isempty(events)
    return;
end
for s = 1:numel(sound_cfg.field_names)
    onset_field = sprintf('%s_onsets', sound_cfg.field_names{s});
    if isfield(events, onset_field)
        counts.(sound_cfg.field_names{s}) = numel(events.(onset_field));
        sound_total = sound_total + counts.(sound_cfg.field_names{s});
    end
end
if isfield(events, 'noise_onsets')
    counts.noise_primary = numel(events.noise_onsets);
else
    counts.noise_primary = sound_total;
end
if isfield(events, 'lick_post_stimulus')
    counts.lick_post_stimulus = numel(events.lick_post_stimulus);
end
if isfield(events, 'lick_post_water_cued')
    counts.lick_post_water_cued = numel(events.lick_post_water_cued);
end
if isfield(events, 'lick_post_water_uncued')
    counts.lick_post_water_uncued = numel(events.lick_post_water_uncued);
end
if isfield(events, 'lick_post_water_omission')
    counts.lick_post_water_omission = numel(events.lick_post_water_omission);
end
end

function cfg = get_sound_level_info()
cfg = struct();
cfg.db_values = [20, 40, 60, 80];
cfg.field_names = arrayfun(@(db) sprintf('sound%ddb', db), cfg.db_values, 'UniformOutput', false);
cfg.prefixes = cellfun(@(name) sprintf('%s_', name), cfg.field_names, 'UniformOutput', false);
cfg.labels = arrayfun(@(db) sprintf('Sound %d dB stimulus', db), cfg.db_values, 'UniformOutput', false);
cfg.event_names = arrayfun(@(db) sprintf('Sound%ddB', db), cfg.db_values, 'UniformOutput', false);
cfg.event_ids = containers.Map({'Sound20dB','Sound40dB','Sound60dB','Sound80dB'}, cfg.field_names);
end

function lookup = build_sound_event_lookup(event_names)
if isstring(event_names)
    event_names = cellstr(event_names);
end
cfg = get_sound_level_info();
lookup = repmat({''}, numel(event_names), 1);
for i = 1:numel(event_names)
    if isempty(event_names{i})
        continue;
    end
    lookup{i} = match_sound_level_field_from_event_name(event_names{i});
    if isempty(lookup{i})
        token = regexp(event_names{i}, 'Sound\s*(?<val>\d+(\.\d+)?)', 'names', 'once');
        if ~isempty(token)
            lookup{i} = match_sound_level_field_from_intensity(str2double(token.val));
        end
    end
end
end

function field_name = determine_session_sound_field(SessionData, trial_idx)
field_name = '';
intensity = get_sound_level_value_from_session(SessionData, trial_idx);
if ~isnan(intensity)
    field_name = match_sound_level_field_from_intensity(intensity);
end
end

function intensity = get_sound_level_value_from_session(SessionData, trial_idx)
cfg = get_sound_level_info();
intensity = NaN;
if isfield(SessionData, 'TrialData') && numel(SessionData.TrialData) >= trial_idx
    trial_entry = SessionData.TrialData(trial_idx);
    candidate_fields = {'VolumeLevel','SoundLevel','NoiseLevel','StimulusLevel'};
    for c = 1:numel(candidate_fields)
        field = candidate_fields{c};
        if isfield(trial_entry, field)
            entry_val = trial_entry.(field);
            val = resolve_trial_level_value(entry_val, cfg, 0);
            if ~isnan(val)
                intensity = val;
                return;
            end
        end
    end
end
if isnan(intensity) && isfield(SessionData, 'TrialSettings')
    trial_settings = SessionData.TrialSettings;
    trial_entry = [];
    if iscell(trial_settings)
        if numel(trial_settings) >= trial_idx
            trial_entry = trial_settings{trial_idx};
        end
    elseif numel(trial_settings) >= trial_idx
        trial_entry = trial_settings(trial_idx);
    end
    candidate_fields = {'NoiseLevel','SoundLevel','NoiseAmplitude','SoundAmplitude','TargetdB'};
    intensity = extract_scalar_from_struct(trial_entry, candidate_fields);
end
if isnan(intensity) && isfield(SessionData, 'Custom') && ~isempty(SessionData.Custom)
    custom_struct = SessionData.Custom;
    candidate_fields = {'NoiseLevel','SoundLevel','NoiseLevels','SoundLevels','TargetdB'};
    for c = 1:numel(candidate_fields)
        field = candidate_fields{c};
        if isfield(custom_struct, field)
            intensity = pick_trial_value(custom_struct.(field), SessionData, trial_idx);
            if ~isnan(intensity)
                return;
            end
        end
    end
end
if isnan(intensity) && isfield(SessionData, 'SoundLevels')
    intensity = pick_trial_value(SessionData.SoundLevels, SessionData, trial_idx);
end
end

function val = extract_scalar_from_struct(entry, candidate_fields)
val = NaN;
if ~isstruct(entry) || isempty(entry)
    return;
end
for c = 1:numel(candidate_fields)
    field = candidate_fields{c};
    if isfield(entry, field)
        data = entry.(field);
        if iscell(data) && ~isempty(data)
            data = data{1};
        end
        if isnumeric(data) && ~isempty(data)
            val = double(data(1));
            return;
        end
    end
end
end

function val = resolve_trial_level_value(entry_val, cfg, depth)
if nargin < 3
    depth = 0;
end
val = NaN;
if depth > 6 || isempty(entry_val)
    return;
end
if isstruct(entry_val)
    if isfield(entry_val, 'Index') && ~isempty(entry_val.Index)
        idx_val = round(double(entry_val.Index(1)));
        if idx_val >= 1 && idx_val <= numel(cfg.db_values)
            val = cfg.db_values(idx_val);
            return;
        end
    end
    nested_fields = {'dBValue','NoiseLevel','SoundLevel','StimulusLevel','Voltage','Amplitude'};
    for k = 1:numel(nested_fields)
        field = nested_fields{k};
        if isfield(entry_val, field)
            nested_val = resolve_trial_level_value(entry_val.(field), cfg, depth+1);
            if ~isnan(nested_val)
                val = nested_val;
                return;
            end
        end
    end
    return;
end
if iscell(entry_val)
    entry_val = entry_val{1};
    val = resolve_trial_level_value(entry_val, cfg, depth+1);
    return;
end
if isnumeric(entry_val) && ~isempty(entry_val)
    val = double(entry_val(1));
end
end

function val = pick_trial_value(values, SessionData, trial_idx)
val = NaN;
if iscell(values)
    try
        values = cellfun(@double, values);
    catch
        return;
    end
end
if ~isnumeric(values) || isempty(values)
    return;
end
values = double(values);
if numel(values) >= trial_idx
    val = values(trial_idx);
    return;
end
if isfield(SessionData, 'TrialTypes') && numel(SessionData.TrialTypes) >= trial_idx
    type_idx = SessionData.TrialTypes(trial_idx);
    if ~isempty(type_idx) && type_idx >= 1 && type_idx <= numel(values)
        val = values(type_idx);
    end
end
end

function field_name = match_sound_level_field_from_intensity(intensity)
field_name = '';
if isempty(intensity) || isnan(intensity)
    return;
end
val = abs(double(intensity));
if val == 0
    return;
end
scale_iter = 0;
while val < 10 && scale_iter < 4
    val = val * 10;
    scale_iter = scale_iter + 1;
end
val = round(val);
cfg = get_sound_level_info();
[diff_min, idx] = min(abs(cfg.db_values - val));
if diff_min <= 5
    field_name = cfg.field_names{idx};
else
    persistent warned_invalid_intensity;
    if isempty(warned_invalid_intensity)
        warning('Sound intensity %.3f does not match configured sound levels.', double(intensity));
        warned_invalid_intensity = true;
    end
end
end

function field_name = match_sound_level_field_from_event_name(event_name)
field_name = '';
if isempty(event_name)
    return;
end
cfg = get_sound_level_info();
event_key = regexprep(event_name, '\s+', '');
if isKey(cfg.event_ids, event_key)
    field_name = cfg.event_ids(event_key);
    return;
end
token = regexp(event_name, 'Sound\s*(?<val>\d+(\.\d+)?)', 'names', 'once');
if ~isempty(token)
    field_name = match_sound_level_field_from_intensity(str2double(token.val));
end
end
function regressors = create_stimulus_regressors(event_times, event_intensities, n_t, sampling_rate, kernel_window)
stimulus_vector = zeros(n_t, 1);
for i = 1:length(event_times)
    onset_frame = round(event_times(i) * sampling_rate);
    if onset_frame > 0 && onset_frame <= n_t
        stimulus_vector(onset_frame) = event_intensities(i);
    end
end
regressors = apply_temporal_kernel(stimulus_vector, kernel_window, sampling_rate);
end

function regressors = create_lick_regressors(lick_times, n_t, sampling_rate, kernel_window)
lick_vector = zeros(n_t, 1);
for i = 1:length(lick_times)
    lick_frame = round(lick_times(i) * sampling_rate);
    if lick_frame > 0 && lick_frame <= n_t
        lick_vector(lick_frame) = 1;
    end
end
regressors = apply_temporal_kernel(lick_vector, kernel_window, sampling_rate);
end

function regressors = apply_temporal_kernel(input_vector, kernel_window, sampling_rate)
n_t = length(input_vector);
n_frames = length(kernel_window);
regressors = zeros(n_t, n_frames);
for i = 1:n_frames
    shift_frames = round(kernel_window(i) * sampling_rate);
    if shift_frames <= 0
        shift_samples = abs(shift_frames);
        if shift_samples == 0
            shifted_vector = input_vector;
        else
            shifted_vector = [zeros(shift_samples, 1); input_vector(1:end-shift_samples)];
        end
    else
        shifted_vector = [input_vector(shift_frames+1:end); zeros(shift_frames, 1)];
    end
    regressors(:, i) = shifted_vector;
end
end
function [stimulus_events, lick_events] = extract_behavioral_events_session(SessionData, protocol_type, total_time_s)
fprintf('    Extracting behavioral events from SessionData (%s)...\n', protocol_type);
n_trials = SessionData.nTrials;
sound_cfg = get_sound_level_info();
stimulus_events = struct();
for s = 1:numel(sound_cfg.field_names)
    stimulus_events.(sound_cfg.field_names{s}) = [];
end
lick_events = struct();
lick_events.post_stimulus = [];
lick_events.post_water_cued = [];
lick_events.post_water_uncued = [];
lick_events.post_water_omission = [];

for trial = 1:n_trials
    if ~isfield(SessionData.RawEvents.Trial{trial}, 'States')
        continue;
    end
    states = SessionData.RawEvents.Trial{trial}.States;
    events = SessionData.RawEvents.Trial{trial}.Events;
    trial_start = SessionData.TrialStartTimestamp(trial);

    if isfield(states, 'NoiseDelivery') && ~isempty(states.NoiseDelivery)
        level_field = determine_session_sound_field(SessionData, trial);
        if ~isempty(level_field)
            noise_time = trial_start + states.NoiseDelivery(1);
            stimulus_events.(level_field)(end+1,1) = noise_time; %#ok<AGROW>
        end
    end

    has_water = isfield(states, 'WaterReward') && ~isempty(states.WaterReward);
    if has_water
        water_time = trial_start + states.WaterReward(1);
    end

    if isfield(events, 'Port1In') && ~isempty(events.Port1In)
        lick_times_trial = trial_start + events.Port1In;
        for lick_time = lick_times_trial
            if isfield(states, 'NoiseDelivery') && ~isempty(states.NoiseDelivery)
                noise_time = trial_start + states.NoiseDelivery(1);
                if lick_time >= noise_time && lick_time < (noise_time + 2.0)
                    lick_events.post_stimulus(end+1) = lick_time; %#ok<AGROW>
                    continue;
                end
            end
            if has_water && lick_time >= water_time && lick_time < (water_time + 2.0)
                lick_events.post_water_cued(end+1) = lick_time; %#ok<AGROW>
            end
        end
        if ~has_water && isfield(states, 'NoiseDelivery') && ~isempty(states.NoiseDelivery)
            expected_water = trial_start + states.NoiseDelivery(1) + 2.0;
            for lick_time = lick_times_trial
                if lick_time >= expected_water && lick_time < (expected_water + 2.0)
                    lick_events.post_water_omission(end+1) = lick_time; %#ok<AGROW>
                end
            end
        end
    end
end

cutoff_time = total_time_s;
fields_to_trim = {'post_stimulus','post_water_cued','post_water_uncued','post_water_omission'};
for f = fields_to_trim
    field = f{1};
    lick_events.(field) = lick_events.(field)(lick_events.(field) <= cutoff_time);
end
for s = 1:numel(sound_cfg.field_names)
    field_name = sound_cfg.field_names{s};
    times = stimulus_events.(field_name);
    stimulus_events.(field_name) = times(times <= cutoff_time);
end
end

function [stimulus_events, lick_events] = extract_behavioral_events_log(log_struct, total_time_s)
fprintf('    Extracting events from imaging event log...\n');
timestamps = double(log_struct.timestamps(:));
event_ids = double(log_struct.eventID(:));
states = double(log_struct.state(:));
event_names = log_struct.eventNameList;
onset_mask = (states == 1);
timestamps = timestamps(onset_mask);
event_ids = event_ids(onset_mask);

sound_cfg = get_sound_level_info();
sound_lookup = build_sound_event_lookup(event_names);
first_lick_id = find(strcmpi(event_names, 'FirstLick'), 1);
first_lick_post_water_id = find(strcmpi(event_names, 'FirstLickPostWater'), 1);
stimulus_events = struct();
for s = 1:numel(sound_cfg.field_names)
    stimulus_events.(sound_cfg.field_names{s}) = [];
end
lick_events = struct();
lick_events.post_stimulus = [];
lick_events.post_water_cued = [];
lick_events.post_water_uncued = [];
lick_events.post_water_omission = [];

for i = 1:numel(timestamps)
    id = event_ids(i);
    if id < 1 || id > numel(event_names)
        continue;
    end
    time = timestamps(i);
    sound_field = '';
    if id <= numel(sound_lookup)
        sound_field = sound_lookup{id};
    end
    if ~isempty(sound_field)
        stimulus_events.(sound_field)(end+1,1) = time; %#ok<AGROW>
    elseif ~isempty(first_lick_id) && id == first_lick_id
        lick_events.post_stimulus(end+1,1) = time; %#ok<AGROW>
    elseif ~isempty(first_lick_post_water_id) && id == first_lick_post_water_id
        lick_events.post_water_cued(end+1,1) = time; %#ok<AGROW>
    end
end

cutoff_time = total_time_s;
fields_to_trim = {'post_stimulus','post_water_cued','post_water_uncued','post_water_omission'};
for f = fields_to_trim
    field = f{1};
    lick_events.(field) = lick_events.(field)(lick_events.(field) <= cutoff_time);
end
for s = 1:numel(sound_cfg.field_names)
    field_name = sound_cfg.field_names{s};
    times = stimulus_events.(field_name);
    stimulus_events.(field_name) = times(times <= cutoff_time);
end
end
%% ========== Plotting helpers ==========
function plot_multi_roi_kernels(results)
tk = results.temporal_kernels;
n_rois = numel(tk);
colors = lines(n_rois);
figure('Name','TemporalModelSoundLevelFull_Contributions_v2 Kernels','Position',[80 80 900 500]);
hold on;
for r = 1:n_rois
    plot(tk(r).lag_times_sec, tk(r).beta_cv_mean, 'LineWidth', 1.5, 'Color', colors(r,:), ...
        'DisplayName', sprintf('%s (R^2=%.2f%%)', tk(r).roi_name, results.performance(r).R2_cv_mean*100));
end
xlabel('Lag (s)');
ylabel('\beta');
title('Temporal kernels (motion predictor)');
grid on;
legend('Location','bestoutside');
end

function plot_multi_roi_heatmap(results)
beta_matrix = results.comparison.beta_matrix_cv;
lag_times = results.temporal_kernels(1).lag_times_sec;
figure('Name','TemporalModelSoundLevelFull_Contributions_v2 Heatmap','Position',[100 100 700 500]);
imagesc(lag_times, 1:size(beta_matrix,2), beta_matrix');
colormap(redblue(256));
colorbar;
yticks(1:numel(results.temporal_kernels));
yticklabels({results.temporal_kernels.roi_name});
xlabel('Lag (s)');
ylabel('ROI');
title('Temporal kernel heatmap');
end

function plot_multi_roi_performance(results)
perf = results.performance;
figure('Name','TemporalModelSoundLevelFull_Contributions_v2 Performance','Position',[120 120 800 400]);
bar([perf.R2_cv_mean]*100);
hold on;
errorbar(1:numel(perf), [perf.R2_cv_mean]*100, [perf.R2_cv_sem]*100, 'k.', 'LineWidth', 1);
hold off;
xticks(1:numel(perf));
xticklabels({perf.roi_name});
ylabel('R^2 (CV, %)');
title('Cross-validated performance per ROI');
grid on;
end

function cmap = redblue(m)
if nargin < 1, m = 256; end
r = [(0:(m/2-1))/(m/2), ones(1,m/2)];
g = [zeros(1,m/2), (0:(m/2-1))/(m/2)];
b = [ones(1,m/2), (m/2-1:-1:0)/(m/2)];
cmap = [r' g' b'];
end
