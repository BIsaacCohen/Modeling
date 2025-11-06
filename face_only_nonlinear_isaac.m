function results = face_only_nonlinear_isaac(fluo_file, motion_file, behav_roi_file, neural_roi_file, vascular_mask_file, brain_mask_file, opts)
% face_only_nonlinear_isaac Polynomial OLS regression (X + X²) of face motion -> fluorescence
%
%   Fits polynomial regression Y = β₁X + β₂X² + intercept using ordinary
%   least squares. Includes multicollinearity diagnostics and comparison to
%   linear-only model to assess whether the quadratic term improves fit.
%
%   results = face_only_nonlinear_isaac(fluo_file, motion_file, ...
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
%       target_roi_name     - ROI name from fluoROIs.roimsk (default 'AU_L')
%       chunk_size_frames   - Frames per IO chunk when reading .dat (default 500)
%       output_file         - Where to save results .mat file (default auto-generated)
%       motion_smoothing_window - Gaussian smoothing window for motion (default 3)
%
%   OUTPUTS (results struct):
%       beta_linear         - Coefficient for X (linear term)
%       beta_quadratic      - Coefficient for X² (quadratic term)
%       beta                - Full coefficient vector [β₁; β₂]
%       intercept           - Regression intercept
%       R2_linear_only      - R² using only linear term
%       R2_polynomial       - R² using both linear + quadratic terms
%       delta_R2            - Improvement from adding X² term
%       Y_pred              - Predicted fluorescence (polynomial model)
%       Y_pred_linear       - Predicted fluorescence (linear-only model)
%       neural_trace        - Original fluorescence trace
%       neural_trace_z      - Z-scored fluorescence trace
%       face_motion         - Face motion trace
%       face_motion_z       - Z-scored face motion trace
%       regressor_names     - {'face_motion_linear', 'face_motion_squared'}
%       diagnostics         - Struct with correlation, VIF, condition number
%       sampling_rate       - Fluorescence sampling rate
%       target_roi_name     - Name of analyzed ROI

if nargin < 7 || isempty(opts)
    opts = struct();
end

defaults = struct('target_roi_name', 'AU_L', 'chunk_size_frames', 500, 'output_file', '', 'motion_smoothing_window', 3);
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

% Determine output file
if isempty(opts.output_file)
    [output_dir, ~, ~] = fileparts(mfilename('fullpath'));
    safe_roi = regexprep(opts.target_roi_name, '\W+', '');
    default_output = sprintf('face_nonlinear_isaac_%s_results.mat', safe_roi);
    output_file = fullfile(output_dir, default_output);
else
    output_file = opts.output_file;
end

log_cfg(fluo_file, motion_file, neural_roi_file, opts);

%% 1. Load fluorescence metadata
[fluo_Y, fluo_X, fluo_T, sampling_rate] = load_dat_metadata(fluo_file);
fprintf('Fluorescence data: %d x %d x %d frames @ %.1f Hz\n', fluo_Y, fluo_X, fluo_T, sampling_rate);

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

%% 9. Build polynomial design matrix
fprintf('Building polynomial design matrix (X + X²)...\n');
X = [face_motion_z, face_motion_z.^2];
regressor_names = {'face_motion_linear', 'face_motion_squared'};
Y = neural_trace_z;

% Check for zero variance
if var(face_motion_z) < eps
    error('Face motion has zero variance. Cannot perform regression.');
end

%% 10. Check for multicollinearity
fprintf('Checking design matrix conditioning...\n');

% Correlation between X and X²
corr_X_X2 = corr(X(:,1), X(:,2));
fprintf('  Correlation(X, X²): %.3f\n', corr_X_X2);
if abs(corr_X_X2) > 0.7
    warning('High correlation (%.3f) between linear and quadratic terms.', corr_X_X2);
end

% Variance Inflation Factor (VIF)
R2_quad = corr_X_X2^2;
VIF = 1 / (1 - R2_quad);
fprintf('  VIF: %.2f\n', VIF);
if VIF > 10
    warning('VIF = %.2f suggests problematic multicollinearity (VIF > 10).', VIF);
elseif VIF > 5
    warning('VIF = %.2f suggests moderate multicollinearity (VIF > 5).', VIF);
end

% Condition number
condition_num = cond(X'*X);
fprintf('  Condition number: %.2f\n', condition_num);
if condition_num > 100
    warning('Condition number = %.2f. Design matrix is poorly conditioned.', condition_num);
end

%% 11. Polynomial OLS regression: Y = β₁X + β₂X² + intercept
fprintf('Performing polynomial OLS regression...\n');

% Solve for beta using mldivide
beta = X \ Y;

fprintf('  β₁ (linear term):    %.4f\n', beta(1));
fprintf('  β₂ (quadratic term): %.4f\n', beta(2));

% Calculate intercept (should be ~0 for z-scored data)
X_mean = [mean(face_motion), mean(face_motion.^2)];
Y_mean = mean(neural_trace);
intercept = Y_mean - X_mean * beta;

% Generate predictions
Y_pred = X * beta;  % Using z-scored data (mean already 0)

% Calculate R-squared
TSS = sum((Y - mean(Y)).^2);
RSS = sum((Y - Y_pred).^2);
R2_polynomial = max(0, 1 - RSS / TSS);

%% 12. Compare to linear-only model
fprintf('\nComparing to linear-only model...\n');
X_linear = face_motion_z;
beta_linear_only = X_linear \ Y;
Y_pred_linear = X_linear * beta_linear_only;

RSS_linear = sum((Y - Y_pred_linear).^2);
R2_linear = max(0, 1 - RSS_linear / TSS);

% Test if quadratic term significantly improves fit
delta_R2 = R2_polynomial - R2_linear;
fprintf('\nModel comparison:\n');
fprintf('  R² (linear only): %.4f (%.2f%%)\n', R2_linear, R2_linear * 100);
fprintf('  R² (polynomial):  %.4f (%.2f%%)\n', R2_polynomial, R2_polynomial * 100);
fprintf('  ΔR² improvement:  %.4f (%.2f%% points)\n', delta_R2, delta_R2 * 100);

if delta_R2 < 0.01
    warning('Quadratic term adds minimal explanatory power (ΔR² = %.4f). Linear model may be sufficient.', delta_R2);
end

%% 13. Save results
results = struct();
results.target_roi_name = opts.target_roi_name;
results.regressor_names = regressor_names;
results.beta = beta;
results.beta_linear = beta(1);
results.beta_quadratic = beta(2);
results.intercept = intercept;
results.R2_polynomial = R2_polynomial;
results.R2_linear_only = R2_linear;
results.delta_R2 = delta_R2;
results.Y_pred = Y_pred;
results.Y_pred_linear = Y_pred_linear;
results.neural_trace = neural_trace;
results.neural_trace_z = neural_trace_z;
results.face_motion = face_motion;
results.face_motion_z = face_motion_z;
results.sampling_rate = sampling_rate;
results.motion_sampling_rate = me_freq;
results.downsample_factor = downsample_factor;
results.n_timepoints = min_length;
results.diagnostics = struct('correlation_X_X2', corr_X_X2, 'VIF', VIF, 'condition_number', condition_num);
results.paths = struct('fluo_file', fluo_file, 'motion_file', motion_file, 'behav_roi_file', behav_roi_file, 'neural_roi_file', neural_roi_file);
results.timestamp = datestr(now);

save(output_file, 'results', '-v7.3');
fprintf('\nResults saved to %s\n', output_file);

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
    fprintf('=== Face-only polynomial OLS regression configuration ===\n');
    fprintf('  Fluorescence: %s\n', fluo_file);
    fprintf('  Motion:       %s\n', motion_file);
    fprintf('  Neural ROI:   %s (target = %s)\n', neural_roi_file, opts.target_roi_name);
    fprintf('  Model:        Y = β₁X + β₂X² + intercept\n');
    fprintf('==========================================================\n');
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
