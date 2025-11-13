function results = TemporalModelEvents(ROI, session_file, opts)
% TemporalModelEvents  Event-driven ridge regression using ROI-ready data
%
%   Extends the bidirectional temporal model framework to include both the
%   movement predictor lag design (as in TemporalModel) and the stimulus/
%   lick event kernels used in ridge_regression_events_only_mml.m. The
%   neural trace comes from the ROI struct, behavioral predictors are
%   lagged into bidirectional regressors, and event regressors are rebuilt
%   from the provided session file. Outputs mirror TemporalModel by
%   providing cross-validated performance metrics, full-data predictions,
%   motion-lag kernels, and event-kernel summaries.
%
%   results = TemporalModelEvents(ROI, session_file, opts)
%
%   REQUIRED INPUTS:
%       ROI          - ROI struct from rois_to_mat.m (same requirements as TemporalModel)
%       session_file - Bpod SessionData MAT file or imaging-aligned event log
%
%   OPTIONAL INPUTS (opts struct):
%       target_neural_roi - Name of neural ROI to predict (default: first ROI)
%       behavior_predictor- Behavior ROI name to lag (default 'Face')
%       stim_kernel       - Kernel (s) for stimulus regressors (default 0:0.1:2)
%       lick_kernel       - Kernel (s) for lick regressors (default -0.5:0.1:5)
%       event_protocol    - SessionData protocol label (default 'WNoiseLickWater')
%       min_lag/max_lag   - Lag bounds in frames (default -5/+10)
%       min_lag_seconds / max_lag_seconds - Alternative specification
%       cv_folds          - Number of blocked folds (default 5)
%       zscore_design     - Z-score design columns (default true)
%       output_file       - MAT file for saving results (default auto)
%       save_results      - Save results to disk (default true)
%       show_plots        - Generate diagnostic plots (default true)
%       remove_initial_seconds - Discard leading seconds before fitting (default 0)

if nargin < 2
    error('TemporalModelEvents requires ROI struct input plus a session_file.');
end

if nargin < 3 || isempty(opts)
    opts = struct();
end

defaults = struct(...
    'target_neural_roi', '', ...
    'behavior_predictor', 'Face', ...
    'stim_kernel', 0:0.1:2, ...
    'lick_kernel', -0.5:0.1:2, ...
    'event_protocol', 'WNoiseLickWater', ...
    'min_lag', -5, ...
    'max_lag', 10, ...
    'min_lag_seconds', [], ...
    'max_lag_seconds', [], ...
    'cv_folds', 5, ...
    'zscore_design', true, ...
    'output_file', '', ...
    'save_results', true, ...
    'show_plots', true, ...
    'remove_initial_seconds', 0);

opts = populate_defaults(opts, defaults);

validate_roi_structure(ROI);

if ~exist(session_file, 'file')
    error('Session file "%s" was not found.', session_file);
end

script_dir = fileparts(mfilename('fullpath'));
ensure_ridgeMML_on_path(script_dir);

fprintf('=== TemporalModelEvents: ROI-based event ridge regression ===\n');

%% 1. Extract neural ROI trace (identical to TemporalModel)
fluo_data = ROI.modalities.fluorescence.data;
fluo_labels = ROI.modalities.fluorescence.labels;
fluo_rate = ROI.modalities.fluorescence.sample_rate;

if isempty(opts.target_neural_roi)
    neural_idx = 1;
    opts.target_neural_roi = fluo_labels{1};
    fprintf('No target_neural_roi specified, using first ROI: %s\n', opts.target_neural_roi);
else
    neural_idx = find(strcmpi(fluo_labels, opts.target_neural_roi), 1);
    if isempty(neural_idx)
        error('Neural ROI "%s" not found. Available: %s', ...
            opts.target_neural_roi, strjoin(fluo_labels, ', '));
    end
end

neural_trace = fluo_data(:, neural_idx);
sampling_rate = fluo_rate;

fprintf('Neural ROI: %s (%.1f Hz, %d frames)\n', ...
    opts.target_neural_roi, sampling_rate, length(neural_trace));

%% 2. Extract behavioral predictor trace
behav_data = ROI.modalities.behavior.data;
behav_labels = ROI.modalities.behavior.labels;
behav_rate = ROI.modalities.behavior.sample_rate;

behav_idx = find(strcmpi(behav_labels, opts.behavior_predictor), 1);
if isempty(behav_idx)
    error('Behavioral predictor "%s" not found. Available: %s', ...
        opts.behavior_predictor, strjoin(behav_labels, ', '));
end

behavior_trace = behav_data(:, behav_idx);
if abs(behav_rate - sampling_rate) > 0.01
    fprintf('Resampling behavior from %.1f Hz to %.1f Hz...\n', behav_rate, sampling_rate);
    behavior_trace = resample(behavior_trace, round(sampling_rate*1000), round(behav_rate*1000));
end

fprintf('Behavior predictor: %s (%.1f Hz, %d frames)\n', ...
    opts.behavior_predictor, sampling_rate, length(behavior_trace));

%% 3. Match trace lengths and build event design
min_length = min(length(neural_trace), length(behavior_trace));
neural_trace = neural_trace(1:min_length);
behavior_trace = behavior_trace(1:min_length);
min_length_initial = min_length;

fprintf('\nMatched timepoints: %d frames (~%.1f s)\n', min_length_initial, min_length_initial / sampling_rate);
fprintf('Constructing events-only design matrix from %s...\n', session_file);
design = build_event_design_matrix(session_file, opts.event_protocol, ...
    opts.stim_kernel, opts.lick_kernel, sampling_rate, min_length_initial);

trim_seconds = max(0, opts.remove_initial_seconds);
trim_frames = round(trim_seconds * sampling_rate);
if trim_frames > 0
    if trim_frames >= min_length_initial
        error('remove_initial_seconds (%g s) exceeds available recording length (%g s).', ...
            trim_seconds, min_length_initial / sampling_rate);
    end
    fprintf('Removing initial habituation segment: %.1f s (%d frames)\n', trim_seconds, trim_frames);
    neural_trace = neural_trace(trim_frames+1:end);
    behavior_trace = behavior_trace(trim_frames+1:end);
    min_length = min_length_initial - trim_frames;
    design = trim_initial_design_segment(design, trim_frames, trim_seconds, sampling_rate);
else
    trim_seconds = 0;
    trim_frames = 0;
    min_length = min_length_initial;
end

X_events_full = design.matrix;
regressor_names_events = design.regressor_names;
group_info = design.group_info;

%% 4. Lag configuration (borrowed from TemporalModel)
if ~isempty(opts.min_lag_seconds)
    opts.min_lag = round(opts.min_lag_seconds * sampling_rate);
    fprintf('Min lag: %.3f s -> %d frames\n', opts.min_lag_seconds, opts.min_lag);
end
if ~isempty(opts.max_lag_seconds)
    opts.max_lag = round(opts.max_lag_seconds * sampling_rate);
    fprintf('Max lag: %.3f s -> %d frames\n', opts.max_lag_seconds, opts.max_lag);
end

min_lag = opts.min_lag;
max_lag = opts.max_lag;
if min_lag >= max_lag
    error('min_lag (%d) must be less than max_lag (%d)', min_lag, max_lag);
end

lag_values = (min_lag:max_lag)';
lag_times_sec = lag_values / sampling_rate;
n_lags_total = numel(lag_values);
n_frames_lost_start = max_lag;
n_frames_lost_end = abs(min_lag);
n_valid = min_length - n_frames_lost_start - n_frames_lost_end;
if n_valid <= 0
    error('Not enough frames for requested lag range. Need at least %d frames, have %d.', ...
        n_frames_lost_start + n_frames_lost_end + 1, min_length);
end

%% 5. Z-score traces and build combined design matrix
fprintf('\nZ-scoring neural and behavior traces...\n');
neural_trace_z_full = zscore(neural_trace);
behavior_trace_z_full = zscore(behavior_trace);

Y = neural_trace_z_full(n_frames_lost_start+1 : min_length - n_frames_lost_end);
X_motion = zeros(n_valid, n_lags_total);

lag_idx = 0;
for lag = min_lag:max_lag
    lag_idx = lag_idx + 1;
    start_idx = n_frames_lost_start + 1 - lag;
    end_idx = min_length - n_frames_lost_end - lag;
    X_motion(:, lag_idx) = behavior_trace_z_full(start_idx:end_idx);
end

X_events = X_events_full(n_frames_lost_start+1 : min_length - n_frames_lost_end, :);
n_event_regressors = size(X_events, 2);
X = [X_motion, X_events];

regressor_names_motion = arrayfun(@(lag) sprintf('%s_lag%+d', opts.behavior_predictor, lag), ...
    lag_values, 'UniformOutput', false);
regressor_names = [regressor_names_motion(:); regressor_names_events(:)];

for g = 1:numel(group_info)
    group_info(g).indices = group_info(g).indices + n_lags_total;
end

motion_group_label = sprintf('%s motion (lags)', opts.behavior_predictor);
group_labels = [{motion_group_label}; arrayfun(@(g) g.label, group_info, 'UniformOutput', false)];
group_indices = cell(numel(group_labels), 1);
group_indices{1} = 1:n_lags_total;
for g = 1:numel(group_info)
    group_indices{g+1} = group_info(g).indices;
end
n_groups_total = numel(group_labels);

if opts.zscore_design
    X = zscore(X);
else
    X = X - mean(X, 1);
end
X(~isfinite(X)) = 0;

n_regressors = size(X, 2);
fprintf('Combined design matrix: %d timepoints x %d regressors (%d motion + %d event)\n', ...
    n_valid, n_regressors, n_lags_total, n_event_regressors);
fprintf('  Zero-variance event regressors removed: %d\n', design.n_zero_variance);

%% 6. Diagnostics
fprintf('\nDesign matrix diagnostics:\n');
if n_regressors > 1
    corr_matrix = corr(X);
    off_diag_corrs = corr_matrix(~eye(n_regressors));
    max_corr = max(off_diag_corrs);
    mean_corr = mean(off_diag_corrs);
else
    max_corr = 1;
    mean_corr = 1;
end
fprintf('  Max correlation between columns: %.3f\n', max_corr);
fprintf('  Mean correlation between columns: %.3f\n', mean_corr);
condition_num = cond(X' * X);
fprintf('  Condition number (X''X): %.2f\n', condition_num);

%% 7. Cross-validation (blocked, as in TemporalModel)
cv_folds = min(opts.cv_folds, max(1, n_valid));
if cv_folds ~= opts.cv_folds
    fprintf('\nReducing cv_folds from %d to %d due to limited samples.\n', opts.cv_folds, cv_folds);
end
fold_size = max(1, floor(n_valid / cv_folds));
fprintf('\nPerforming %d-fold blocked cross-validation...\n', cv_folds);

beta_cv_folds = zeros(n_regressors, cv_folds);
lambda_cv_folds = zeros(cv_folds, 1);
R2_cv_folds = zeros(cv_folds, 1);
convergence_cv = zeros(cv_folds, 1);
group_single_R2 = nan(n_groups_total, cv_folds);
group_shuffle_R2 = nan(n_groups_total, cv_folds);

for fold = 1:cv_folds
    test_start = (fold - 1) * fold_size + 1;
    test_end = min(fold * fold_size, n_valid);
    test_idx = test_start:test_end;
    train_idx = setdiff(1:n_valid, test_idx);

    X_train = X(train_idx, :);
    Y_train = Y(train_idx);
    X_test = X(test_idx, :);
    Y_test = Y(test_idx);

    [lambda_fold, beta_fold, conv_fail] = ridgeMML(Y_train, X_train, 1);

    beta_cv_folds(:, fold) = beta_fold;
    lambda_cv_folds(fold) = mean(lambda_fold(:));
    convergence_cv(fold) = conv_fail;

    Y_pred_test = X_test * beta_fold;
    TSS_test = sum((Y_test - mean(Y_test)).^2);
    RSS_test = sum((Y_test - Y_pred_test).^2);
    R2_cv_folds(fold) = max(0, 1 - RSS_test / TSS_test);

    TSS_common = TSS_test;
    if TSS_common < eps
        TSS_common = eps;
    end

    for g = 1:n_groups_total
        idx = group_indices{g};
        if isempty(idx)
            continue;
        end

        X_train_single = X_train(:, idx);
        X_test_single = X_test(:, idx);
        if isempty(X_train_single) || all(std(X_train_single, 0, 1) < 1e-8)
            group_single_R2(g, fold) = 0;
        else
            [~, beta_single, ~] = ridgeMML(Y_train, X_train_single, 1);
            Y_pred_single = X_test_single * beta_single;
            RSS_single = sum((Y_test - Y_pred_single).^2);
            group_single_R2(g, fold) = max(0, 1 - RSS_single / TSS_common) * 100;
        end

        X_train_shuff = X_train;
        X_test_shuff = X_test;
        perm_train = randperm(size(X_train, 1));
        perm_test = randperm(size(X_test, 1));
        X_train_shuff(:, idx) = X_train(perm_train, idx);
        X_test_shuff(:, idx) = X_test(perm_test, idx);

        [~, beta_shuff, ~] = ridgeMML(Y_train, X_train_shuff, 1);
        Y_pred_shuff = X_test_shuff * beta_shuff;
        RSS_shuff = sum((Y_test - Y_pred_shuff).^2);
        group_shuffle_R2(g, fold) = max(0, 1 - RSS_shuff / TSS_common) * 100;
    end

    fprintf('  Fold %d/%d: R^2 = %.4f, lambda = %.4f, n_train = %d, n_test = %d\n', ...
        fold, cv_folds, R2_cv_folds(fold), lambda_cv_folds(fold), ...
        length(train_idx), length(test_idx));
end

R2_cv_mean = mean(R2_cv_folds);
R2_cv_sem = std(R2_cv_folds) / sqrt(cv_folds);

fprintf('\nCV summary: R^2 = %.4f +/- %.4f (%.2f%% +/- %.2f%%)\n', ...
    R2_cv_mean, R2_cv_sem, R2_cv_mean * 100, R2_cv_sem * 100);
if any(convergence_cv)
    warning('%d/%d CV folds reported convergence failure', sum(convergence_cv), cv_folds);
end

%% 8. Full-data model and predictions
fprintf('\nFitting full-data ridge model...\n');
[lambda_full, beta_full, convergence_full] = ridgeMML(Y, X, 1);
if convergence_full
    warning('Full-data ridgeMML reported convergence issues.');
end

lambda_full_mean = mean(lambda_full(:));
Y_pred_full = X * beta_full;
TSS_full = sum((Y - mean(Y)).^2);
RSS_full = sum((Y - Y_pred_full).^2);
R2_full = max(0, 1 - RSS_full / TSS_full);

fprintf('  Full-data R^2 = %.4f (%.2f%%), lambda = %.4f\n', R2_full, R2_full * 100, lambda_full_mean);

%% 9. Motion lag kernel summary
fprintf('\nSummarizing motion lag kernel...\n');
beta_motion_cv = beta_cv_folds(1:n_lags_total, :);
beta_motion_mean = mean(beta_motion_cv, 2);
if cv_folds > 1
    beta_motion_sem = std(beta_motion_cv, 0, 2) / sqrt(cv_folds);
else
    beta_motion_sem = zeros(size(beta_motion_mean));
end
beta_motion_full = beta_full(1:n_lags_total);
[peak_beta, peak_idx] = max(abs(beta_motion_mean));
peak_lag_frames = lag_values(peak_idx);
peak_lag_sec = lag_times_sec(peak_idx);

temporal_kernel = struct();
temporal_kernel.beta_cv_mean = beta_motion_mean;
temporal_kernel.beta_cv_sem = beta_motion_sem;
temporal_kernel.beta_full_data = beta_motion_full;
temporal_kernel.beta_cv_folds = beta_motion_cv;
temporal_kernel.lag_indices = lag_values;
temporal_kernel.lag_times_sec = lag_times_sec;
temporal_kernel.peak_lag_frames = peak_lag_frames;
temporal_kernel.peak_lag_sec = peak_lag_sec;
temporal_kernel.peak_beta = beta_motion_mean(peak_idx);
temporal_kernel.peak_beta_sem = beta_motion_sem(peak_idx);

if peak_lag_frames < 0
    fprintf('  Peak predictive lag: %d frames (%.3f s)\n', peak_lag_frames, peak_lag_sec);
elseif peak_lag_frames > 0
    fprintf('  Peak reactive lag: %d frames (%.3f s)\n', peak_lag_frames, peak_lag_sec);
else
    fprintf('  Peak at zero lag\n');
end

%% 10. Event kernel summaries
fprintf('Summarizing event kernels...\n');
event_kernels = compute_event_kernels(beta_cv_folds, beta_full, group_info, cv_folds);

%% 11. Contribution statistics (single-variable and shuffle unique)
baseline_percent = R2_cv_folds(:)' * 100;
group_explained_mean = nan(n_groups_total, 1);
group_explained_std = nan(n_groups_total, 1);
group_unique_mean = nan(n_groups_total, 1);
group_unique_std = nan(n_groups_total, 1);

for g = 1:n_groups_total
    single_vals = group_single_R2(g, :);
    valid_single = ~isnan(single_vals);
    if any(valid_single)
        vals = single_vals(valid_single);
        group_explained_mean(g) = mean(vals);
        group_explained_std(g) = std(vals);
    end

    shuffle_vals = group_shuffle_R2(g, :);
    valid_shuffle = ~isnan(shuffle_vals);
    if any(valid_shuffle)
        vals = shuffle_vals(valid_shuffle);
        diff_vals = baseline_percent(valid_shuffle) - vals;
        diff_vals = max(diff_vals, 0);
        group_unique_mean(g) = mean(diff_vals);
        group_unique_std(g) = std(diff_vals);
    end
end

%% 8. Assemble results structure
results = struct();

results.temporal_kernel = temporal_kernel;
results.event_kernels = event_kernels;
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

results.performance = struct();
results.performance.R2_cv_mean = R2_cv_mean;
results.performance.R2_cv_sem = R2_cv_sem;
results.performance.R2_cv_folds = R2_cv_folds;
results.performance.R2_full_data = R2_full;
results.performance.lambda_cv_folds = lambda_cv_folds;
results.performance.lambda_full_data = lambda_full_mean;
results.performance.convergence_cv_failures = sum(convergence_cv);
results.performance.convergence_full_failure = convergence_full;

results.predictions = struct();
results.predictions.Y_pred = Y_pred_full;
results.predictions.neural_trace_z = Y;
results.predictions.behavior_trace_z = behavior_trace_z_full;
results.predictions.behavior_trace_truncated = behavior_trace_z_full(n_frames_lost_start+1 : min_length - n_frames_lost_end);
results.predictions.time_vector = ((n_frames_lost_start):(n_frames_lost_start + n_valid - 1))' / sampling_rate;
results.events = design.events;

results.design_matrix = struct();
results.design_matrix.regressor_names = regressor_names;
results.design_matrix.motion = struct( ...
    'predictor', opts.behavior_predictor, ...
    'lag_indices', lag_values, ...
    'lag_times_sec', lag_times_sec, ...
    'column_indices', 1:n_lags_total);
results.design_matrix.events = design.events;
results.design_matrix.event = struct( ...
    'group_info', group_info, ...
    'event_counts', design.event_counts, ...
    'n_zero_variance', design.n_zero_variance, ...
    'column_offset', n_lags_total);
results.design_matrix.group_info = group_info;
results.design_matrix.event_counts = design.event_counts;
results.design_matrix.n_zero_variance = design.n_zero_variance;
results.design_matrix.group_indices = group_indices;
results.design_matrix.group_labels = group_labels;

results.metadata = struct();
results.metadata.target_neural_roi = opts.target_neural_roi;
results.metadata.behavior_predictor = opts.behavior_predictor;
results.metadata.session_file = session_file;
results.metadata.event_protocol = opts.event_protocol;
results.metadata.stim_kernel = opts.stim_kernel(:);
results.metadata.lick_kernel = opts.lick_kernel(:);
results.metadata.min_lag = min_lag;
results.metadata.max_lag = max_lag;
results.metadata.min_lag_seconds = lag_times_sec(1);
results.metadata.max_lag_seconds = lag_times_sec(end);
results.metadata.sampling_rate = sampling_rate;
results.metadata.cv_folds = cv_folds;
results.metadata.n_timepoints_total = min_length;
results.metadata.n_timepoints_original = min_length_initial;
results.metadata.n_timepoints_used = n_valid;
results.metadata.n_timepoints_lost_start = n_frames_lost_start;
results.metadata.n_timepoints_lost_end = n_frames_lost_end;
results.metadata.n_timepoints_removed_initial = trim_frames;
results.metadata.initial_trim_seconds = trim_seconds;
results.metadata.initial_trim_applied = trim_frames > 0;
results.metadata.n_motion_regressors = n_lags_total;
results.metadata.n_event_regressors = n_event_regressors;
results.metadata.n_regressors = n_regressors;
results.metadata.timestamp = datestr(now);
results.metadata.zscore_design = opts.zscore_design;
results.metadata.condition_number = condition_num;
results.metadata.group_labels = group_labels;
results.metadata.n_groups = n_groups_total;

if isfield(ROI, 'metadata') && isfield(ROI.metadata, 'source')
    results.metadata.source_roi_file = ROI.metadata.source;
end

%% 9. Plots
if opts.show_plots
    fprintf('\nGenerating plots...\n');
    plot_temporal_kernel(results);
    plot_event_kernels(results);
    plot_event_model_predictions(results);
    fprintf('Plots generated.\n');
end

%% 10. Save
if opts.save_results
    if isempty(opts.output_file)
        safe_roi = regexprep(opts.target_neural_roi, '\W+', '');
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        default_output = sprintf('TemporalModelEvents_%s_%s.mat', safe_roi, timestamp);
        output_file = fullfile(script_dir, default_output);
    else
        output_file = opts.output_file;
    end

    save(output_file, 'results', '-v7.3');
    fprintf('\nResults saved to %s\n', output_file);
else
    fprintf('\nResults not saved (opts.save_results == false)\n');
end

fprintf('\n=== TemporalModelEvents complete ===\n');

end

%% ================ Core Helpers ================

function design = build_event_design_matrix(session_file, protocol_label, stim_kernel, lick_kernel, sampling_rate, n_timepoints)
    total_time_s = n_timepoints / sampling_rate;
    session_data = load(session_file);

    if isfield(session_data, 'SessionData')
        [stimulus_events, lick_events] = extract_behavioral_events_session( ...
            session_data.SessionData, protocol_label, total_time_s);
    elseif all(isfield(session_data, {'eventID', 'timestamps', 'eventNameList', 'state'}))
        [stimulus_events, lick_events] = extract_behavioral_events_log(session_data, total_time_s);
    else
        error(['Session file %s does not contain SessionData or event log fields ' ...
            '(eventID, timestamps, eventNameList, state).'], session_file);
    end

    design_matrix = [];
    regressor_names = {};
    column_group = [];
    column_lags = [];

    group_defs = struct( ...
        'label', {'Noise stimulus', 'Lick post-stimulus', 'Lick post-water (cued)', ...
                  'Lick post-water (uncued)', 'Lick post-water (omission)'}, ...
        'prefix', {'noise_primary_', 'lick_post_stim_', 'lick_water_cued_', ...
                   'lick_water_uncued_', 'lick_water_omission_'});
    group_defs = group_defs(:);

    fprintf('  Stim kernel: %d lags (%0.2f to %0.2f s)\n', numel(stim_kernel), ...
        min(stim_kernel), max(stim_kernel));
    fprintf('  Lick kernel: %d lags (%0.2f to %0.2f s)\n', numel(lick_kernel), ...
        min(lick_kernel), max(lick_kernel));

    % Stimulus regressors
    noise_regressors = create_stimulus_regressors( ...
        stimulus_events.noise_primary.times, ...
        stimulus_events.noise_primary.intensities, ...
        n_timepoints, sampling_rate, stim_kernel);
    design_matrix = [design_matrix, noise_regressors]; %#ok<AGROW>
    for i = 1:length(stim_kernel)
        regressor_names{end+1} = sprintf('noise_primary_t%.1fs', stim_kernel(i)); %#ok<AGROW>
        column_group(end+1) = 1; %#ok<AGROW>
        column_lags(end+1) = stim_kernel(i); %#ok<AGROW>
    end

    % Lick regressors (four categories share lick_kernel)
    lick_fields = {
        'post_stimulus', 'lick_post_stim_', 2; ...
        'post_water_cued', 'lick_water_cued_', 3; ...
        'post_water_uncued', 'lick_water_uncued_', 4; ...
        'post_water_omission', 'lick_water_omission_', 5};

    for lf = 1:size(lick_fields, 1)
        field_name = lick_fields{lf, 1};
        prefix = lick_fields{lf, 2};
        group_id = lick_fields{lf, 3};

        lick_reg = create_lick_regressors( ...
            lick_events.(field_name), n_timepoints, sampling_rate, lick_kernel);
        design_matrix = [design_matrix, lick_reg]; %#ok<AGROW>
        for i = 1:length(lick_kernel)
            regressor_names{end+1} = sprintf('%s_t%.1fs', prefix(1:end-1), lick_kernel(i)); %#ok<AGROW>
            column_group(end+1) = group_id; %#ok<AGROW>
            column_lags(end+1) = lick_kernel(i); %#ok<AGROW>
        end
    end

    fprintf('  Raw design matrix: %d timepoints x %d regressors\n', ...
        size(design_matrix, 1), size(design_matrix, 2));

    % Remove zero-variance columns
    reg_std = std(design_matrix, 0, 1);
    zero_mask = reg_std < 1e-10;
    n_zero = sum(zero_mask);
    if n_zero > 0
        fprintf('  Removing %d zero-variance regressors caused by empty event bins...\n', n_zero);
        design_matrix(:, zero_mask) = [];
        regressor_names(zero_mask) = [];
        column_group(zero_mask) = [];
        column_lags(zero_mask) = [];
    end

    group_info = repmat(struct('label', '', 'prefix', '', 'indices', [], 'lag_times_sec', []), numel(group_defs), 1);
    for g = 1:numel(group_defs)
        idx = find(column_group == g);
        group_info(g).label = group_defs(g).label;
        group_info(g).prefix = group_defs(g).prefix;
        group_info(g).indices = idx;
        group_info(g).lag_times_sec = column_lags(idx);
    end

    design = struct();
    design.matrix = design_matrix;
    design.regressor_names = regressor_names;
    design.group_info = group_info;
    design.n_zero_variance = n_zero;
    design.event_counts = struct(...
        'noise_primary', numel(stimulus_events.noise_primary.times), ...
        'lick_post_stimulus', numel(lick_events.post_stimulus), ...
        'lick_post_water_cued', numel(lick_events.post_water_cued), ...
        'lick_post_water_uncued', numel(lick_events.post_water_uncued), ...
        'lick_post_water_omission', numel(lick_events.post_water_omission));
    lick_post_water_all = [lick_events.post_water_cued(:); ...
                           lick_events.post_water_uncued(:); ...
                           lick_events.post_water_omission(:)];
    lick_post_water_all = lick_post_water_all(~isnan(lick_post_water_all));
    lick_post_water_all = sort(lick_post_water_all);

    design.events = struct( ...
        'noise_onsets', stimulus_events.noise_primary.times(:), ...
        'noise_intensities', stimulus_events.noise_primary.intensities(:), ...
        'lick_post_stimulus', lick_events.post_stimulus(:), ...
        'lick_post_water_cued', lick_events.post_water_cued(:), ...
        'lick_post_water_uncued', lick_events.post_water_uncued(:), ...
        'lick_post_water_omission', lick_events.post_water_omission(:), ...
        'lick_post_water_all', lick_post_water_all);
end

function event_kernels = compute_event_kernels(beta_cv_folds, beta_full, group_info, cv_folds)
    n_groups = numel(group_info);
    event_kernels = repmat(struct('label', '', 'lag_times_sec', [], ...
        'beta_cv_mean', [], 'beta_cv_sem', [], 'beta_full', [], 'indices', []), n_groups, 1);

    for g = 1:n_groups
        idx = group_info(g).indices;
        if isempty(idx)
            continue;
        end
        group_beta_cv = beta_cv_folds(idx, :);
        beta_mean = mean(group_beta_cv, 2);
        if cv_folds > 1
            beta_sem = std(group_beta_cv, 0, 2) / sqrt(cv_folds);
        else
            beta_sem = zeros(size(beta_mean));
        end

        event_kernels(g).label = group_info(g).label;
        event_kernels(g).lag_times_sec = group_info(g).lag_times_sec(:);
        event_kernels(g).beta_cv_mean = beta_mean(:);
        event_kernels(g).beta_cv_sem = beta_sem(:);
        event_kernels(g).beta_full = beta_full(idx);
        event_kernels(g).indices = idx;
    end
end

%% ================ Plotting ================

function plot_temporal_kernel(results)
    tk = results.temporal_kernel;
    perf = results.performance;
    meta = results.metadata;

    fig_title = sprintf('Temporal Kernel: %s vs %s', ...
        meta.target_neural_roi, meta.behavior_predictor);

    figure('Name', fig_title, 'Position', [100 100 900 600]);
    hold on;

    kernel_color = [0.15, 0.35, 0.8];
    sem_upper = tk.beta_cv_mean + tk.beta_cv_sem;
    sem_lower = tk.beta_cv_mean - tk.beta_cv_sem;

    fill([tk.lag_times_sec; flipud(tk.lag_times_sec)], ...
         [sem_upper; flipud(sem_lower)], kernel_color, ...
         'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
         'DisplayName', 'SEM envelope');

    plot(tk.lag_times_sec, tk.beta_cv_mean, ...
        'LineWidth', 2.5, 'Color', kernel_color, ...
        'DisplayName', 'CV mean');

    plot([min(tk.lag_times_sec), max(tk.lag_times_sec)], [0 0], ...
        'k--', 'LineWidth', 1, 'HandleVisibility', 'off');

    yl = ylim;
    plot([0, 0], yl, 'k:', 'LineWidth', 1.25, 'HandleVisibility', 'off');
    plot([tk.peak_lag_sec, tk.peak_lag_sec], yl, '--', ...
        'LineWidth', 1.25, 'Color', [0.4 0.4 0.4], ...
        'DisplayName', sprintf('Peak lag (%.3f s)', tk.peak_lag_sec));

    hold off;

    xlabel('Lag time (s)');
    ylabel('Beta coefficient');
    title(fig_title, 'Interpreter', 'none');
    legend('Location', 'best');
    grid on;

    anno_str = sprintf(['R^2 (CV): %.2f%% +/- %.2f%%\n' ...
                        'R^2 (full-data): %.2f%%\n' ...
                        'Peak: %.3f s (beta = %.3f +/- %.3f)'], ...
        perf.R2_cv_mean * 100, perf.R2_cv_sem * 100, ...
        perf.R2_full_data * 100, ...
        tk.peak_lag_sec, tk.peak_beta, tk.peak_beta_sem);

    annotation(gcf, 'textbox', [0.15 0.75 0.25 0.18], ...
        'String', anno_str, 'EdgeColor', 'k', 'BackgroundColor', 'w', ...
        'FitBoxToText', 'on', 'FontSize', 11);

    if meta.min_lag < 0
        text(min(tk.lag_times_sec) * 0.9, yl(2) * 0.9, 'Predictive', ...
            'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', [0.3 0.2 0.2]);
    end
    text(max(tk.lag_times_sec) * 0.9, yl(2) * 0.9, 'Reactive', ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', [0.2 0.2 0.2]);
end

function plot_event_kernels(results)
    kernels = results.event_kernels;
    valid_mask = arrayfun(@(k) ~isempty(k.indices), kernels);
    kernels = kernels(valid_mask);
    if isempty(kernels)
        warning('No event kernels available for plotting.');
        return;
    end

    n_groups = numel(kernels);
    fig = figure('Name', 'Event Kernels', 'Position', [80 80 900 250 + 150*n_groups]);
    tiled = tiledlayout(fig, n_groups, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tiled, sprintf('TemporalModelEvents Kernels: %s', results.metadata.target_neural_roi), 'Interpreter', 'none');

    for g = 1:n_groups
        k = kernels(g);
        ax = nexttile(tiled);
        hold(ax, 'on');
        upper = k.beta_cv_mean + k.beta_cv_sem;
        lower = k.beta_cv_mean - k.beta_cv_sem;
        fill(ax, [k.lag_times_sec; flipud(k.lag_times_sec)], [upper; flipud(lower)], ...
            [0.2 0.4 0.8], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        plot(ax, k.lag_times_sec, k.beta_cv_mean, 'LineWidth', 2, 'Color', [0.1 0.2 0.6]);
        yline(ax, 0, 'k:');
        xlabel(ax, 'Lag (s)');
        ylabel(ax, '\beta');
        title(ax, k.label, 'Interpreter', 'none');
        grid(ax, 'on');
        hold(ax, 'off');
    end
end

function plot_event_model_predictions(results)
    meta = results.metadata;
    perf = results.performance;
    pred = results.predictions;

    n_valid = meta.n_timepoints_used;
    n_lost_start = meta.n_timepoints_lost_start;
    n_lost_end = meta.n_timepoints_lost_end;

    t_truncated = pred.time_vector(:);
    if numel(t_truncated) ~= n_valid
        t_truncated = (n_lost_start:(n_lost_start + n_valid - 1))' / meta.sampling_rate;
    end

    neural_trace = pred.neural_trace_z(:);
    prediction = pred.Y_pred(:);
    behavior_trace = pred.behavior_trace_z(:);
    t_full = (0:(numel(behavior_trace) - 1))' / meta.sampling_rate;

    fig_title = sprintf('TemporalModelEvents Predictions: %s vs %s', ...
        meta.target_neural_roi, meta.behavior_predictor);

    figure('Name', 'TemporalModelEvents Predictions', 'Position', [120 120 1000 600]);
    tiled = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tiled, fig_title, 'Interpreter', 'none');

    ax1 = nexttile(tiled);
    plot(ax1, t_truncated, neural_trace, 'Color', [0.2 0.2 0.8], ...
        'DisplayName', sprintf('%s (z-score)', meta.target_neural_roi));
    hold(ax1, 'on');
    plot(ax1, t_truncated, prediction, 'Color', [0.85 0.33 0.1], ...
        'LineWidth', 1.25, 'DisplayName', 'Prediction (full-data model)');
    hold(ax1, 'off');
    ylabel(ax1, 'Neural z-score');
    legend(ax1, 'Location', 'best');
    grid(ax1, 'on');

    text(ax1, 0.02, 0.98, ...
        sprintf('R^2 (CV): %.2f%% +/- %.2f%%\nR^2 (full): %.2f%%', ...
            perf.R2_cv_mean * 100, perf.R2_cv_sem * 100, ...
            perf.R2_full_data * 100), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 9, 'BackgroundColor', 'w', 'EdgeColor', 'k');

    ax2 = nexttile(tiled);
    plot(ax2, t_full, behavior_trace, 'Color', [0.13 0.55 0.13], ...
        'DisplayName', sprintf('%s (z-score)', meta.behavior_predictor));
    hold(ax2, 'on');

    if n_lost_start > 0
        t_trunc_start = t_full(1:n_lost_start);
        behav_trunc_start = behavior_trace(1:n_lost_start);
        patch(ax2, [t_trunc_start(:); flipud(t_trunc_start(:))], ...
            [behav_trunc_start(:); zeros(n_lost_start, 1)], ...
            [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none', ...
            'DisplayName', sprintf('Truncated start (%d frames)', n_lost_start));
    end

    if n_lost_end > 0
        t_trunc_end = t_full(end - n_lost_end + 1:end);
        behav_trunc_end = behavior_trace(end - n_lost_end + 1:end);
        patch(ax2, [t_trunc_end(:); flipud(t_trunc_end(:))], ...
            [behav_trunc_end(:); zeros(n_lost_end, 1)], ...
            [0.95 0.95 0.95], 'FaceAlpha', 0.5, 'EdgeColor', 'none', ...
            'DisplayName', sprintf('Truncated end (%d frames)', n_lost_end));
    end

    % Event markers (noise onset, first lick post-stimulus, first lick post water)
    event_data = struct();
    if isfield(results, 'events')
        event_data = results.events;
    elseif isfield(results, 'design_matrix') && isfield(results.design_matrix, 'events')
        event_data = results.design_matrix.events;
    end

    if ~isempty(fieldnames(event_data))
        event_specs = {
            'noise_onsets',        [0.35 0.35 0.35], 'Noise stimulus start';
            'lick_post_stimulus',  [0.80 0.30 0.30], 'First lick post stimulus';
            'lick_post_water_all', [0.20 0.55 0.75], 'First lick post water'};

        for es = 1:size(event_specs, 1)
            field_name = event_specs{es, 1};
            color = event_specs{es, 2};
            label = event_specs{es, 3};

            if ~isfield(event_data, field_name) || isempty(event_data.(field_name))
                continue;
            end

            times = double(event_data.(field_name)(:));
            times = times(isfinite(times));
            times = times(times >= t_full(1) & times <= t_full(end));
            if isempty(times)
                continue;
            end

            first_marker = true;
            for t_evt = times(:)'
                h = xline(ax2, t_evt, ':', 'Color', color, 'LineWidth', 1.2);
                if first_marker
                    set(h, 'DisplayName', label);
                    first_marker = false;
                else
                    set(h, 'HandleVisibility', 'off');
                end
            end
        end
    end

    hold(ax2, 'off');
    ylabel(ax2, sprintf('%s (z-score)', meta.behavior_predictor));
    xlabel(ax2, 'Time (s)');
    grid(ax2, 'on');
    legend(ax2, 'Location', 'best');

    linkaxes([ax1, ax2], 'x');

    footer_str = sprintf('Sampling: %.1f Hz | Lags: %d to +%d frames (%.3f to +%.3f s) | CV folds: %d', ...
        meta.sampling_rate, meta.min_lag, meta.max_lag, ...
        meta.min_lag_seconds, meta.max_lag_seconds, meta.cv_folds);
    annotation(gcf, 'textbox', [0.01 0.01 0.7 0.04], ...
        'String', footer_str, 'EdgeColor', 'none', ...
        'Interpreter', 'none', 'FontSize', 8);
end

%% ================ Shared helpers from TemporalModel ================

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
        error('ROI.modalities missing "behavior" field (required for compatibility)');
    end

    fluo = ROI.modalities.fluorescence;
    required_fluo = {'data', 'labels', 'sample_rate'};
    for i = 1:numel(required_fluo)
        if ~isfield(fluo, required_fluo{i})
            error('ROI.modalities.fluorescence missing "%s"', required_fluo{i});
        end
    end

    behav = ROI.modalities.behavior;
    required_behav = {'data', 'labels', 'sample_rate'};
    for i = 1:numel(required_behav)
        if ~isfield(behav, required_behav{i})
            error('ROI.modalities.behavior missing "%s"', required_behav{i});
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

    error('ridgeMML.m not found on the MATLAB path.');
end

%% ================ Event extraction helpers (from ridge_regression_events_only_mml) ================

function [stimulus_events, lick_events] = extract_behavioral_events_session(SessionData, protocol_type, total_time_s)
    fprintf('    Extracting behavioral events from SessionData (%s)...\n', protocol_type);
    n_trials = SessionData.nTrials;

    stimulus_events = struct();
    stimulus_events.noise_primary = struct('times', [], 'intensities', []);

    lick_events = struct();
    lick_events.post_stimulus = [];
    lick_events.post_water_cued = [];
    lick_events.post_water_uncued = [];
    lick_events.post_water_omission = [];

    trial_start_times = SessionData.TrialStartTimestamp - SessionData.TrialStartTimestamp(1);

    for trial = 1:n_trials
        trial_type = SessionData.TrialData(trial).TrialType;
        states = SessionData.RawEvents.Trial{trial}.States;
        events = SessionData.RawEvents.Trial{trial}.Events;
        trial_start_abs = trial_start_times(trial);

        if isfield(states, 'NoiseDelivery') && ~isempty(states.NoiseDelivery)
            noise_time = trial_start_abs + states.NoiseDelivery(1);
            stimulus_events.noise_primary.times(end+1) = noise_time; %#ok<AGROW>

            if isfield(SessionData, 'TrialSettings') && ...
               length(SessionData.TrialSettings) >= trial && ...
               isfield(SessionData.TrialSettings(trial), 'NoiseAmp')
                intensity = SessionData.TrialSettings(trial).NoiseAmp;
            else
                intensity = 60;
            end
            stimulus_events.noise_primary.intensities(end+1) = intensity; %#ok<AGROW>
        end

        has_water = isfield(states, 'WaterReward') && ~isempty(states.WaterReward);
        if has_water
            water_time = trial_start_abs + states.WaterReward(1);
        end

        if isfield(events, 'Port1In') && ~isempty(events.Port1In)
            lick_times_trial = trial_start_abs + events.Port1In;

            for lick_time = lick_times_trial
                if isfield(states, 'NoiseDelivery') && ~isempty(states.NoiseDelivery)
                    noise_time = trial_start_abs + states.NoiseDelivery(1);
                    if lick_time >= noise_time && lick_time < (noise_time + 2.0)
                        lick_events.post_stimulus(end+1) = lick_time; %#ok<AGROW>
                        continue;
                    end
                end

                if has_water && lick_time >= water_time && lick_time < (water_time + 2.0)
                    if trial_type == 2
                        lick_events.post_water_uncued(end+1) = lick_time; %#ok<AGROW>
                    else
                        lick_events.post_water_cued(end+1) = lick_time; %#ok<AGROW>
                    end
                end
            end

            if ~has_water && isfield(states, 'NoiseDelivery') && ~isempty(states.NoiseDelivery)
                expected_water_time = trial_start_abs + states.NoiseDelivery(1) + 2.0;
                for lick_time = lick_times_trial
                    if lick_time >= expected_water_time && lick_time < (expected_water_time + 2.0)
                        lick_events.post_water_omission(end+1) = lick_time; %#ok<AGROW>
                    end
                end
            end
        end
    end

    stimulus_events.noise_primary.times = stimulus_events.noise_primary.times(:);
    stimulus_events.noise_primary.intensities = stimulus_events.noise_primary.intensities(:);
    lick_events.post_stimulus = lick_events.post_stimulus(:);
    lick_events.post_water_cued = lick_events.post_water_cued(:);
    lick_events.post_water_uncued = lick_events.post_water_uncued(:);
    lick_events.post_water_omission = lick_events.post_water_omission(:);

    cutoff_time = total_time_s;
    fields_to_trim = {'post_stimulus', 'post_water_cued', 'post_water_uncued', 'post_water_omission'};
    for f = fields_to_trim
        field_name = f{1};
        lick_events.(field_name) = lick_events.(field_name)(lick_events.(field_name) <= cutoff_time);
    end
    keep_idx = stimulus_events.noise_primary.times <= cutoff_time;
    stimulus_events.noise_primary.times = stimulus_events.noise_primary.times(keep_idx);
    stimulus_events.noise_primary.intensities = stimulus_events.noise_primary.intensities(keep_idx);

    fprintf('      Noise stimuli: %d\n', numel(stimulus_events.noise_primary.times));
    fprintf('      Post-stimulus licks: %d\n', numel(lick_events.post_stimulus));
    fprintf('      Post-water (cued) licks: %d\n', numel(lick_events.post_water_cued));
    fprintf('      Post-water (uncued) licks: %d\n', numel(lick_events.post_water_uncued));
    fprintf('      Post-water (omission) licks: %d\n', numel(lick_events.post_water_omission));
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

    stimulus_events = struct();
    stimulus_events.noise_primary = struct('times', [], 'intensities', []);

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

        name = event_names{id};
        time = timestamps(i);

        if startsWith(name, 'Sound', 'IgnoreCase', true) || startsWith(name, 'Whisker', 'IgnoreCase', true)
            token = regexp(name, '(?<val>\d+(\.\d+)?)', 'names');
            if ~isempty(token)
                intensity = str2double(token.val);
            else
                intensity = 1;
            end
            stimulus_events.noise_primary.times(end+1,1) = time; %#ok<AGROW>
            stimulus_events.noise_primary.intensities(end+1,1) = intensity; %#ok<AGROW>
        elseif strcmpi(name, 'FirstLick')
            lick_events.post_stimulus(end+1,1) = time; %#ok<AGROW>
        elseif strcmpi(name, 'FirstLickPostWater')
            lick_events.post_water_cued(end+1,1) = time; %#ok<AGROW>
        end
    end

    cutoff_time = total_time_s;
    fields_to_trim = {'post_stimulus', 'post_water_cued', 'post_water_uncued', 'post_water_omission'};
    for f = fields_to_trim
        field_name = f{1};
        lick_events.(field_name) = lick_events.(field_name)(lick_events.(field_name) <= cutoff_time);
    end
    keep_idx = stimulus_events.noise_primary.times <= cutoff_time;
    stimulus_events.noise_primary.times = stimulus_events.noise_primary.times(keep_idx);
    stimulus_events.noise_primary.intensities = stimulus_events.noise_primary.intensities(keep_idx);

    fprintf('      Noise/whisker stimuli: %d\n', numel(stimulus_events.noise_primary.times));
    fprintf('      Post-stimulus licks: %d\n', numel(lick_events.post_stimulus));
    fprintf('      Post-water (cued) licks: %d\n', numel(lick_events.post_water_cued));
    fprintf('      Post-water (uncued) licks: %d\n', numel(lick_events.post_water_uncued));
    fprintf('      Post-water (omission) licks: %d\n', numel(lick_events.post_water_omission));
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

function regressors = apply_temporal_kernel(stimulus_vector, kernel_window, sampling_rate)
    n_t = length(stimulus_vector);
    n_frames = length(kernel_window);
    regressors = zeros(n_t, n_frames);

    for i = 1:n_frames
        shift_frames = round(kernel_window(i) * sampling_rate);
        if shift_frames <= 0
            shift_samples = abs(shift_frames);
            if shift_samples == 0
                shifted_vector = stimulus_vector;
            else
                shifted_vector = [zeros(shift_samples, 1); stimulus_vector(1:end-shift_samples)];
            end
        else
            shifted_vector = [stimulus_vector(shift_frames+1:end); zeros(shift_frames, 1)];
        end
        regressors(:, i) = shifted_vector;
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

    if isfield(events, 'noise_onsets')
        times = events.noise_onsets(:) - trim_seconds;
        keep = times >= 0 & times <= total_time_s;
        events.noise_onsets = times(keep);
        if isfield(events, 'noise_intensities')
            intens = events.noise_intensities(:);
            events.noise_intensities = intens(keep);
        end
    end

    lick_fields = {'lick_post_stimulus', 'lick_post_water_cued', ...
        'lick_post_water_uncued', 'lick_post_water_omission'};
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
        events.lick_post_water_all = events.lick_post_water_all(:);
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
    counts = struct('noise_primary', 0, ...
        'lick_post_stimulus', 0, ...
        'lick_post_water_cued', 0, ...
        'lick_post_water_uncued', 0, ...
        'lick_post_water_omission', 0);

    if isempty(events)
        return;
    end

    if isfield(events, 'noise_onsets')
        counts.noise_primary = numel(events.noise_onsets);
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
