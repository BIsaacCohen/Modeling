function results = face_only_temporal_lag_isaac(fluo_file, motion_file, behav_roi_file, neural_roi_file, vascular_mask_file, brain_mask_file, opts)
% face_only_temporal_lag_isaac Temporal lag ridge regression (ridgeMML) of face motion -> fluorescence
%
%   Fits temporal lag model Y(t) = Σ βᵢ·X(t-i) + intercept using ridge
%   regression with automatic lambda selection via marginal maximum likelihood.
%   Reveals how past face motion predicts current fluorescence.
%
%   results = face_only_temporal_lag_isaac(fluo_file, motion_file, ...
%       behav_roi_file, neural_roi_file, vascular_mask_file, brain_mask_file, opts)
%
%   REQUIRED INPUTS:
%       fluo_file           - Hemodynamic-corrected movie (.dat)
%       motion_file         - Motion energy movie (.dat)
%       behav_roi_file      - Behavioral ROI masks (must contain 'Face')
%       neural_roi_file     - Fluorescence ROI masks (contains target ROI)
%       vascular_mask_file  - Vascular mask (.roimsk) to remove vessels
%       brain_mask_file     - Brain mask (.roimsk) for anatomical bounds
%
%   OPTIONAL INPUTS (opts struct):
%       target_roi_name          - ROI name from fluoROIs.roimsk (default 'AU_L')
%       chunk_size_frames        - Frames per IO chunk when reading .dat (default 500)
%       max_lag                  - Maximum lag in frames (default 10)
%       max_lag_seconds          - Alternative: max lag in seconds (converted to frames)
%       motion_smoothing_window  - Gaussian smoothing for motion (default 3)
%       output_file              - Where to save results .mat file (default auto-generated)
%
%   OUTPUTS (results struct):
%       beta                - Regression coefficients [β₀, β₁, ..., βₗ]
%       lambda_mml          - Optimal lambda from ridgeMML
%       intercept           - Regression intercept
%       R2_temporal         - R² using temporal lags (0 to max_lag)
%       R2_nolag            - R² using only current timepoint (baseline)
%       delta_R2            - Improvement from adding temporal lags
%       temporal_kernel     - Struct with lag times and beta values for plotting
%       Y_pred              - Predicted fluorescence (temporal model)
%       Y_pred_nolag        - Predicted fluorescence (no-lag baseline)
%       neural_trace        - Original fluorescence trace
%       neural_trace_z      - Z-scored fluorescence (truncated to match lags)
%       face_motion         - Face motion trace
%       face_motion_z       - Z-scored face motion trace
%       regressor_names     - Names for each lag predictor
%       diagnostics         - Multicollinearity metrics
%       convergence_failure - ridgeMML convergence status

if nargin < 7 || isempty(opts)
    opts = struct();
end

defaults = struct('target_roi_name', 'AU_L', 'chunk_size_frames', 500, ...
    'max_lag', 10, 'max_lag_seconds', [], 'motion_smoothing_window', 3, ...
    'output_file', '', 'save_results', true);
opts = populate_defaults(opts, defaults);

% Validate required files
assert_file_exists(fluo_file, 'fluo_file');
assert_file_exists(motion_file, 'motion_file');
assert_file_exists(behav_roi_file, 'behav_roi_file');
assert_file_exists(neural_roi_file, 'neural_roi_file');
if ~isempty(vascular_mask_file)
    assert_file_exists(vascular_mask_file, 'vascular_mask_file');
end
if ~isempty(brain_mask_file)
    assert_file_exists(brain_mask_file, 'brain_mask_file');
end

script_dir = fileparts(mfilename('fullpath'));

% Determine output file
if opts.save_results
    if isempty(opts.output_file)
        safe_roi = regexprep(opts.target_roi_name, '\W+', '');
        default_output = sprintf('face_temporal_lag_isaac_%s_results.mat', safe_roi);
        output_file = fullfile(script_dir, default_output);
    else
        output_file = opts.output_file;
    end
else
    output_file = '';
end

log_cfg(fluo_file, motion_file, neural_roi_file, opts);

ensure_ridgeMML_on_path(script_dir);

%% 1. Load fluorescence metadata
[fluo_Y, fluo_X, fluo_T, sampling_rate] = load_dat_metadata(fluo_file);
fprintf('Fluorescence data: %d x %d x %d frames @ %.1f Hz\n', fluo_Y, fluo_X, fluo_T, sampling_rate);

% Handle max_lag_seconds if provided
if ~isempty(opts.max_lag_seconds)
    opts.max_lag = round(opts.max_lag_seconds * sampling_rate);
    fprintf('Max lag: %.2f s → %d frames\n', opts.max_lag_seconds, opts.max_lag);
end

max_lag = opts.max_lag;

%% 2. Build anatomical mask (brain minus vascular)
final_mask = true(fluo_Y, fluo_X);
if ~isempty(vascular_mask_file)
    vascular_mask = load_single_roi_mask(vascular_mask_file, [fluo_Y, fluo_X]);
    final_mask = final_mask & ~vascular_mask;
    fprintf('  Vascular pixels removed: %d\n', sum(vascular_mask(:)));
end

if ~isempty(brain_mask_file)
    brain_mask = load_single_roi_mask(brain_mask_file, [fluo_Y, fluo_X]);
    final_mask = final_mask & brain_mask;
    fprintf('  Brain mask applied: %d pixels kept\n', sum(brain_mask(:)));
end

%% 3. Load fluorescence ROI and extract target ROI
[roi_info, roi_source_path] = load_neural_roi_info(neural_roi_file);
roi_names = {roi_info.Name};
roi_idx = find(strcmpi(roi_names, opts.target_roi_name), 1);

if isempty(roi_idx)
    error('ROI "%s" not found in %s.\nAvailable: %s', opts.target_roi_name, roi_source_path, strjoin(roi_names, ', '));
end

roi_mask = logical(roi_info(roi_idx).Stats.ROI_binary_mask);
if ~isequal(size(roi_mask), [fluo_Y, fluo_X])
    error('ROI mask size mismatch for %s', opts.target_roi_name);
end
roi_mask = roi_mask & final_mask;

n_roi_pixels = sum(roi_mask(:));
if n_roi_pixels == 0
    error('ROI "%s" contains 0 pixels after masking. Check masks.', opts.target_roi_name);
end

fprintf('Target ROI "%s": %d pixels (%.2f%% of frame)\n', opts.target_roi_name, n_roi_pixels, 100 * n_roi_pixels / (fluo_Y * fluo_X));
fprintf('  ROI masks loaded from: %s\n', roi_source_path);

%% 4. Extract ROI-averaged fluorescence timeseries
fprintf('Extracting ROI timeseries...\n');
neural_trace = extract_roi_trace(fluo_file, [fluo_Y, fluo_X], fluo_T, roi_mask, opts.chunk_size_frames);

%% 5. Load behavioral motion and extract Face only
[me_y, me_x, me_T, me_freq] = load_dat_metadata(motion_file);
fprintf('Motion energy: %d x %d x %d frames @ %.1f Hz\n', me_y, me_x, me_T, me_freq);

behav_data = load(behav_roi_file, '-mat');
face_mask = find_roi_mask(behav_data, 'Face', me_y, me_x);

fprintf('Extracting Face motion trace...\n');
face_motion = extract_motion_trace_single(motion_file, [me_y, me_x], me_T, face_mask, opts.chunk_size_frames);

%% 6. Downsample motion to match fluorescence sampling rate
fprintf('Resampling motion trace from %.1f Hz to %.1f Hz using MATLAB resample...\n', me_freq, sampling_rate);

if opts.motion_smoothing_window > 1 && me_freq > sampling_rate
    fprintf('  Applying pre-resample smoothing (window = %d frames)\n', opts.motion_smoothing_window);
    face_motion = smoothdata(face_motion, 'gaussian', opts.motion_smoothing_window);
end

face_motion = resample(face_motion, sampling_rate, me_freq);
downsample_factor = me_freq / sampling_rate;
fprintf('  Original frames: %d, Resampled frames: %d (ratio %.3f)\n', me_T, length(face_motion), sampling_rate / me_freq);

%% 7. Match lengths
min_length = min([fluo_T, length(face_motion)]);
neural_trace = neural_trace(1:min_length);
face_motion = face_motion(1:min_length);
fprintf('Matched timepoints: %d frames (~%.1f s)\n', min_length, min_length / sampling_rate);

%% 8. Z-score data
face_motion_z = zscore(face_motion);
neural_trace_z = zscore(neural_trace);

%% 9. Build temporal lag design matrix
fprintf('\nBuilding temporal lag design matrix (0-%d frames back)...\n', max_lag);
fprintf('  Max lag: %d frames (%.3f s @ %.1f Hz)\n', max_lag, max_lag / sampling_rate, sampling_rate);

n_valid = min_length - max_lag;  % Number of valid timepoints after truncation
X = zeros(n_valid, max_lag + 1);

% Fill each column with appropriate lag
for lag = 0:max_lag
    start_idx = max_lag + 1 - lag;
    end_idx = min_length - lag;
    X(:, lag+1) = face_motion_z(start_idx:end_idx);
end

% Truncate Y to match
Y = neural_trace_z(max_lag+1:end);

fprintf('  Timepoints after truncation: %d (lost first %d)\n', n_valid, max_lag);

% Create regressor names
regressor_names = cell(1, max_lag+1);
for lag = 0:max_lag
    lag_time_sec = lag / sampling_rate;
    regressor_names{lag+1} = sprintf('face_lag%.3fs', lag_time_sec);
end

%% 10. Check for multicollinearity
fprintf('\nChecking design matrix conditioning...\n');

% Correlation matrix between all lags
corr_matrix = corr(X);
off_diag_corrs = corr_matrix(~eye(size(corr_matrix)));
max_corr = max(off_diag_corrs);
mean_corr = mean(off_diag_corrs);
fprintf('  Max correlation between lags: %.3f\n', max_corr);
fprintf('  Mean correlation between lags: %.3f\n', mean_corr);

% Condition number
condition_num = cond(X'*X);
fprintf('  Condition number: %.2f\n', condition_num);
if condition_num > 30
    fprintf('  → High collinearity expected for lagged predictors (using ridge regression)\n');
end

%% 11. Temporal lag ridge regression (ridgeMML)
fprintf('\nPerforming ridge regression (ridgeMML)...\n');

[lambda_mml, beta, convergenceFailure] = ridgeMML(Y, X, 1);

if convergenceFailure
    warning('ridgeMML reported convergence failure. Results may be unreliable.');
end

lambda_final = mean(lambda_mml(:));
fprintf('  Optimal lambda (MML): %.4f\n', lambda_final);
fprintf('  Number of predictors: %d (lags 0-%d)\n', max_lag+1, max_lag);

% Calculate intercept
% For each lag, compute mean of original (non-z-scored) values
X_mean_orig = zeros(1, max_lag+1);
for lag = 0:max_lag
    start_idx = max_lag + 1 - lag;
    end_idx = min_length - lag;
    X_mean_orig(lag+1) = mean(face_motion(start_idx:end_idx));
end
Y_mean_orig = mean(neural_trace(max_lag+1:end));
intercept = Y_mean_orig - X_mean_orig * beta;

% Generate predictions
Y_pred = X * beta;  % Using z-scored data

% Calculate R²
TSS = sum((Y - mean(Y)).^2);
RSS = sum((Y - Y_pred).^2);
R2_temporal = max(0, 1 - RSS / TSS);

fprintf('  R² (temporal model): %.4f (%.2f%%)\n', R2_temporal, R2_temporal * 100);

%% 12. Compare to no-lag baseline
fprintf('\nComparing to no-lag baseline...\n');

% Fit no-lag model: Y(t) ~ X(t) only
X_nolag = face_motion_z(max_lag+1:end);  % Match truncated Y (lag=0 column)
[lambda_nolag, beta_nolag, convergence_nolag] = ridgeMML(Y, X_nolag, 1);

if convergence_nolag
    warning('ridgeMML convergence failure for no-lag baseline.');
end

Y_pred_nolag = X_nolag * beta_nolag;

RSS_nolag = sum((Y - Y_pred_nolag).^2);
R2_nolag = max(0, 1 - RSS_nolag / TSS);

delta_R2 = R2_temporal - R2_nolag;

fprintf('Model comparison:\n');
fprintf('  R² (no lags, t=0 only):  %.4f (%.2f%%)\n', R2_nolag, R2_nolag * 100);
fprintf('  R² (temporal, 0-%d lags): %.4f (%.2f%%)\n', max_lag, R2_temporal, R2_temporal * 100);
fprintf('  ΔR² improvement:          %.4f (%.2f%% points)\n', delta_R2, delta_R2 * 100);

if delta_R2 < 0.01
    warning('Temporal lags add minimal explanatory power (ΔR² = %.4f). No-lag model may be sufficient.', delta_R2);
end

%% 13. Temporal kernel analysis
fprintf('\nTemporal kernel analysis:\n');

% Convert lag indices to time (seconds)
lag_times = (0:max_lag)' / sampling_rate;

% Find peak response lag
[peak_beta, peak_idx] = max(abs(beta));
peak_lag_time = lag_times(peak_idx);
peak_lag_frames = peak_idx - 1;

fprintf('  Peak response at lag %d (%.3f s)\n', peak_lag_frames, peak_lag_time);
fprintf('  Peak beta value: %.4f\n', beta(peak_idx));
fprintf('  Sign of peak: %s\n', ternary(beta(peak_idx) > 0, 'positive', 'negative'));

% Create temporal kernel structure for easy plotting
temporal_kernel = struct();
temporal_kernel.lag_indices = (0:max_lag)';
temporal_kernel.lag_times_sec = lag_times;
temporal_kernel.beta_values = beta;
temporal_kernel.regressor_names = regressor_names';
temporal_kernel.peak_lag_frames = peak_lag_frames;
temporal_kernel.peak_lag_sec = peak_lag_time;
temporal_kernel.peak_beta = beta(peak_idx);

%% 14. Save results
results = struct();
results.target_roi_name = opts.target_roi_name;
results.max_lag = max_lag;
results.max_lag_seconds = max_lag / sampling_rate;
results.regressor_names = regressor_names;
results.beta = beta;
results.lambda_mml = lambda_final;
results.lambda_nolag = mean(lambda_nolag(:));
results.intercept = intercept;
results.convergence_failure = convergenceFailure;
results.convergence_failure_nolag = convergence_nolag;
results.R2_temporal = R2_temporal;
results.R2_nolag = R2_nolag;
results.delta_R2 = delta_R2;
results.Y_pred = Y_pred;
results.Y_pred_nolag = Y_pred_nolag;
results.temporal_kernel = temporal_kernel;
results.neural_trace = neural_trace;
results.neural_trace_z = neural_trace_z(max_lag+1:end);  % Truncated to match lags
results.face_motion = face_motion;
results.face_motion_z = face_motion_z;
results.sampling_rate = sampling_rate;
results.motion_sampling_rate = me_freq;
results.downsample_factor = downsample_factor;
results.n_timepoints_total = min_length;
results.n_timepoints_used = n_valid;
results.n_timepoints_lost = max_lag;
results.diagnostics = struct('max_correlation', max_corr, 'mean_correlation', mean_corr, 'condition_number', condition_num);
results.paths = struct('fluo_file', fluo_file, 'motion_file', motion_file, 'behav_roi_file', behav_roi_file, 'neural_roi_file', neural_roi_file);
results.timestamp = datestr(now);

%% 15. Generate plots
fprintf('\nGenerating plots...\n');
plot_temporal_kernel_isaac(results);
plot_temporal_prediction_isaac(results);
fprintf('Plots generated.\n');

if opts.save_results
    save(output_file, 'results', '-v7.3');
    fprintf('\nResults saved to %s\n', output_file);
else
    fprintf('\nResults not saved (opts.save_results == false)\n');
end

end

%% ================= Helper functions =================

function opts = populate_defaults(opts, defaults)
    fields = fieldnames(defaults);
    for i = 1:numel(fields)
        name = fields{i};
        if ~isfield(opts, name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
    end
end

function assert_file_exists(path_str, label)
    if exist(path_str, 'file') ~= 2
        error('Required %s not found: %s', label, path_str);
    end
end

function log_cfg(fluo_file, motion_file, neural_roi_file, opts)
    fprintf('=== Face temporal lag ridge regression configuration ===\n');
    fprintf('  Fluorescence: %s\n', fluo_file);
    fprintf('  Motion:       %s\n', motion_file);
    fprintf('  Neural ROI:   %s (target = %s)\n', neural_roi_file, opts.target_roi_name);
    fprintf('  Model:        Y(t) = Σ βᵢ·X(t-i) + intercept, lags 0-%d\n', opts.max_lag);
    fprintf('=========================================================\n');
end

function ensure_ridgeMML_on_path(script_dir)
    if exist('ridgeMML', 'file') == 2
        return;
    end
    candidate_paths = {fullfile(script_dir), fullfile(script_dir, '..'), 'C:\Users\shires\Downloads', 'H:\IsaacAndGarrettMatlabScripts\glm code\Puff_Dataset'};
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

function [dimY, dimX, n_frames, freq] = load_dat_metadata(dat_path)
    [dat_dir, dat_name, ~] = fileparts(dat_path);
    meta_file = fullfile(dat_dir, [dat_name '.mat']);
    meta = load(meta_file);
    dimY = meta.datSize(1);
    dimX = meta.datSize(2);
    freq = meta.Freq;

    info = dir(dat_path);
    n_frames = info.bytes / (dimY * dimX * 4);
    if abs(n_frames - round(n_frames)) > 1e-6
        error('File %s has non-integer frame count.', dat_path);
    end
    n_frames = round(n_frames);
end

function mask = load_single_roi_mask(mask_file, dims)
    data = load(mask_file, '-mat');
    mask = logical(data.ROI_info(1).Stats.ROI_binary_mask);
    if ~isequal(size(mask), dims)
        error('Mask dimension mismatch for %s', mask_file);
    end
end

function [roi_info, roi_source_path] = load_neural_roi_info(neural_roi_file)
    data = load(neural_roi_file, '-mat');

    if isfield(data, 'ROI_info')
        roi_info = data.ROI_info;
        roi_source_path = neural_roi_file;
        return;
    end

    if isfield(data, 'SessionROIData')
        session_roi = data.SessionROIData;
        if isstruct(session_roi) && isfield(session_roi, 'AnalysisOptions') && isfield(session_roi.AnalysisOptions, 'FluoROI_filename')

            roi_reference = session_roi.AnalysisOptions.FluoROI_filename;
            roi_source_path = resolve_referenced_roi_file(neural_roi_file, roi_reference);

            roi_data = load(roi_source_path, '-mat');
            if ~isfield(roi_data, 'ROI_info')
                error('Referenced fluorescence ROI file %s does not contain ROI_info.', roi_source_path);
            end
            roi_info = roi_data.ROI_info;
            return;
        end

        error('SessionROIData in %s does not specify AnalysisOptions.FluoROI_filename.', neural_roi_file);
    end

    error('Neural ROI file %s does not contain ROI_info or SessionROIData.', neural_roi_file);
end

function roi_path = resolve_referenced_roi_file(base_file, roi_reference)
    if iscell(roi_reference)
        roi_reference = roi_reference{1};
    end
    if isa(roi_reference, 'string')
        roi_reference = char(roi_reference);
    end

    roi_reference = strtrim(roi_reference);
    if isempty(roi_reference)
        error('Empty fluorescence ROI filename referenced from %s.', base_file);
    end

    if exist(roi_reference, 'file') == 2
        roi_path = roi_reference;
        return;
    end

    base_dir = fileparts(base_file);
    candidate = fullfile(base_dir, roi_reference);
    if exist(candidate, 'file') == 2
        roi_path = candidate;
        return;
    end

    current_dir = base_dir;
    max_levels = 5;
    for depth = 1:max_levels
        parent_dir = fileparts(current_dir);
        if strcmp(parent_dir, current_dir)
            break;
        end
        candidate = fullfile(parent_dir, roi_reference);
        if exist(candidate, 'file') == 2
            roi_path = candidate;
            return;
        end
        current_dir = parent_dir;
    end

    error('Unable to resolve fluorescence ROI file "%s" referenced from %s.', roi_reference, base_file);
end

function trace = extract_roi_trace(dat_file, dims, n_frames, roi_mask, chunk_size)
    pixels_per_frame = prod(dims);
    roi_idx = find(roi_mask(:));
    trace = zeros(n_frames, 1);

    fid = fopen(dat_file, 'r');
    if fid < 0
        error('Unable to open %s', dat_file);
    end

    try
        n_chunks = ceil(n_frames / chunk_size);
        for chunk = 1:n_chunks
            start_idx = (chunk - 1) * chunk_size + 1;
            end_idx = min(chunk * chunk_size, n_frames);
            frames_this_chunk = end_idx - start_idx + 1;

            raw = fread(fid, pixels_per_frame * frames_this_chunk, 'single');
            if numel(raw) < pixels_per_frame * frames_this_chunk
                error('Unexpected EOF while reading %s', dat_file);
            end
            raw = reshape(raw, pixels_per_frame, frames_this_chunk);
            trace(start_idx:end_idx) = mean(raw(roi_idx, :), 1)';
            if mod(chunk, 10) == 0 || chunk == n_chunks
                fprintf('  Fluorescence chunk %d/%d processed\n', chunk, n_chunks);
            end
        end
    catch ME
        fclose(fid);
        rethrow(ME);
    end

    fclose(fid);
end

function mask = find_roi_mask(roi_data, roi_name, dimY, dimX)
    idx = find(strcmpi({roi_data.ROI_info.Name}, roi_name), 1);
    if isempty(idx)
        error('Behavioral ROI "%s" not found.', roi_name);
    end
    mask = logical(roi_data.ROI_info(idx).Stats.ROI_binary_mask);
    if ~isequal(size(mask), [dimY, dimX])
        error('Behavioral ROI "%s" has mismatched dimensions.', roi_name);
    end
end

function trace = extract_motion_trace_single(dat_file, dims, n_frames, roi_mask, chunk_size)
    pixels_per_frame = prod(dims);
    roi_idx = find(roi_mask(:));
    trace = zeros(n_frames, 1);

    fid = fopen(dat_file, 'r');
    if fid < 0
        error('Unable to open %s', dat_file);
    end

    try
        n_chunks = ceil(n_frames / chunk_size);
        for chunk = 1:n_chunks
            start_idx = (chunk - 1) * chunk_size + 1;
            end_idx = min(chunk * chunk_size, n_frames);
            frames_this_chunk = end_idx - start_idx + 1;

            raw = fread(fid, pixels_per_frame * frames_this_chunk, 'single');
            if numel(raw) < pixels_per_frame * frames_this_chunk
                error('Unexpected EOF while reading %s', dat_file);
            end
            raw = reshape(raw, pixels_per_frame, frames_this_chunk);
            trace(start_idx:end_idx) = mean(raw(roi_idx, :), 1)';

            if mod(chunk, 50) == 0 || chunk == n_chunks
                fprintf('  Motion chunk %d/%d processed\n', chunk, n_chunks);
            end
        end
    catch ME
        fclose(fid);
        rethrow(ME);
    end

    fclose(fid);
end

function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

%% ================= Plotting Helper Functions =================

function plot_temporal_kernel_isaac(results, opts)
% plot_temporal_kernel_isaac Plot temporal kernel (beta vs lag time)
%
%   plot_temporal_kernel_isaac(results)
%   plot_temporal_kernel_isaac(results, opts)
%
%   results : struct produced by face_only_temporal_lag_isaac
%             Required field: temporal_kernel
%   opts    : optional struct with fields:
%             .title_suffix - char/string appended to figure title

    if nargin < 2 || isempty(opts)
        opts = struct();
    end

    if ~isfield(results, 'temporal_kernel')
        error('Results struct is missing temporal_kernel field.');
    end

    tk = results.temporal_kernel;

    % Build figure title
    fig_title = 'Temporal Kernel';
    if isfield(opts, 'title_suffix') && ~isempty(opts.title_suffix)
        fig_title = sprintf('%s - %s', fig_title, opts.title_suffix);
    elseif isfield(results, 'target_roi_name')
        fig_title = sprintf('%s (%s)', fig_title, results.target_roi_name);
    end

    % Create figure
    figure('Name', fig_title);

    % Plot beta values vs lag time
    plot(tk.lag_times_sec, tk.beta_values, 'o-', 'LineWidth', 2, ...
        'MarkerSize', 8, 'MarkerFaceColor', [0.2 0.4 0.8], ...
        'Color', [0.2 0.4 0.8], 'DisplayName', 'Beta coefficients');

    hold on;

    % Zero line
    plot([min(tk.lag_times_sec), max(tk.lag_times_sec)], [0 0], ...
        'k--', 'LineWidth', 1, 'HandleVisibility', 'off');

    % Mark peak
    plot(tk.peak_lag_sec, tk.peak_beta, 'r*', 'MarkerSize', 15, ...
        'LineWidth', 2, 'DisplayName', sprintf('Peak (%.3f s)', tk.peak_lag_sec));

    % Add vertical line at peak
    yl = ylim;
    plot([tk.peak_lag_sec, tk.peak_lag_sec], yl, 'r--', ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');

    hold off;

    xlabel('Lag time (seconds)', 'FontSize', 11);
    ylabel('Beta coefficient', 'FontSize', 11);
    title(fig_title, 'FontSize', 12, 'Interpreter', 'none');
    legend('Location', 'best');
    grid on;

    % Add annotation with model comparison
    if isfield(results, 'R2_temporal') && isfield(results, 'R2_nolag')
        anno_str = sprintf('R² (temporal): %.2f%%\nR² (no lag): %.2f%%\nΔR²: +%.2f%%', ...
            results.R2_temporal*100, results.R2_nolag*100, results.delta_R2*100);
        annotation(gcf, 'textbox', [0.15 0.75 0.25 0.15], ...
            'String', anno_str, 'EdgeColor', 'k', 'BackgroundColor', 'w', ...
            'FitBoxToText', 'on', 'FontSize', 9);
    end
end

function plot_temporal_prediction_isaac(results, opts)
% plot_temporal_prediction_isaac Plot neural data, prediction, and face motion
%
%   plot_temporal_prediction_isaac(results)
%   plot_temporal_prediction_isaac(results, opts)
%
%   results : struct produced by face_only_temporal_lag_isaac
%             Required fields: neural_trace_z, Y_pred, face_motion_z,
%             sampling_rate (Hz), n_timepoints_used, max_lag
%   opts    : optional struct with fields:
%             .title_suffix - char/string appended to figure title
%
%   The function creates a two-panel plot with linked x-axes:
%       Top:  neural_trace_z (observed) and Y_pred (predicted)
%       Bottom: face_motion_z (full trace with shaded truncated region)

    if nargin < 2 || isempty(opts)
        opts = struct();
    end

    required_fields = {'neural_trace_z', 'Y_pred', 'face_motion_z', ...
        'sampling_rate', 'n_timepoints_used', 'max_lag'};
    missing = setdiff(required_fields, fieldnames(results));
    if ~isempty(missing)
        error('Results struct is missing required fields: %s', strjoin(missing, ', '));
    end

    % Note: neural_trace_z and Y_pred are truncated (n_timepoints_used)
    % face_motion_z is full length (n_timepoints_total or n_timepoints_used + max_lag)
    n_valid = results.n_timepoints_used;
    max_lag = results.max_lag;

    if numel(results.neural_trace_z) ~= n_valid || numel(results.Y_pred) ~= n_valid
        error('Length mismatch: neural_trace_z and Y_pred should have %d samples.', n_valid);
    end

    % Time vector for truncated data (starts at max_lag / sampling_rate)
    t_truncated = (max_lag:(max_lag + n_valid - 1)) ./ results.sampling_rate;

    % Time vector for full face motion
    n_total_motion = numel(results.face_motion_z);
    t_full = (0:(n_total_motion-1)) ./ results.sampling_rate;

    % Build figure title
    fig_title = 'Temporal Lag Prediction';
    if isfield(opts, 'title_suffix') && ~isempty(opts.title_suffix)
        fig_title = sprintf('%s - %s', fig_title, opts.title_suffix);
    elseif isfield(results, 'target_roi_name')
        fig_title = sprintf('%s (%s)', fig_title, results.target_roi_name);
    end

    % Create figure
    figure('Name', fig_title);
    tiled = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tiled, fig_title, 'Interpreter', 'none');

    % Top panel: Neural trace and prediction
    ax1 = nexttile(tiled);
    plot(t_truncated, results.neural_trace_z, 'Color', [0.2 0.2 0.8], ...
        'DisplayName', 'Neural (z-score)');
    hold(ax1, 'on');
    plot(t_truncated, results.Y_pred, 'Color', [0.85 0.33 0.1], ...
        'LineWidth', 1.25, 'DisplayName', 'Prediction (z-score)');
    hold(ax1, 'off');
    ylabel(ax1, 'Fluorescence (z-score)');
    legend(ax1, 'Location', 'best');
    grid(ax1, 'on');

    % Add R² to title or legend
    if isfield(results, 'R2_temporal')
        text(ax1, 0.02, 0.98, sprintf('R² = %.2f%%', results.R2_temporal*100), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k');
    end

    % Bottom panel: Face motion (full trace)
    ax2 = nexttile(tiled);
    plot(t_full, results.face_motion_z, 'Color', [0.13 0.55 0.13], ...
        'DisplayName', 'Face motion (z-score)');

    % Shade the truncated region (first max_lag frames)
    if max_lag > 0
        hold(ax2, 'on');
        t_truncated_region = t_full(1:max_lag);
        face_truncated = results.face_motion_z(1:max_lag);
        patch([t_truncated_region, fliplr(t_truncated_region)], ...
            [face_truncated', fliplr(zeros(1, max_lag))], ...
            [0.9 0.9 0.9], 'FaceAlpha', 0.5, 'EdgeColor', 'none', ...
            'DisplayName', sprintf('Truncated (%d frames)', max_lag));
        hold(ax2, 'off');
    end

    ylabel(ax2, 'Face motion (z-score)');
    xlabel(ax2, 'Time (s)');
    grid(ax2, 'on');
    legend(ax2, 'Location', 'best');

    % Link x-axes
    linkaxes([ax1, ax2], 'x');

    % Annotation with sampling info
    if isfield(results, 'downsample_factor')
        dim_info = sprintf('Sampling: %.1f Hz (motion %.1f Hz) | Max lag: %d frames (%.3f s)', ...
            results.sampling_rate, results.motion_sampling_rate, ...
            max_lag, max_lag / results.sampling_rate);
    else
        dim_info = sprintf('Sampling: %.1f Hz | Max lag: %d frames (%.3f s)', ...
            results.sampling_rate, max_lag, max_lag / results.sampling_rate);
    end

    annotation(gcf, 'textbox', [0.01 0.01 0.5 0.05], 'String', dim_info, ...
        'EdgeColor', 'none', 'Interpreter', 'none', 'FontSize', 8);
end
