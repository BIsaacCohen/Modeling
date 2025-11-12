function ridge_regression_events_only_mml(fluo_file, session_file, neural_roi_file, vascular_mask_file, brain_mask_file)
% Ridge regression (events only) using WNoiseLickWater discrete regressors with MML
%
%   Builds a design matrix composed solely of stimulus and lick-event
%   regressors (no motion energy terms) and fits per-ROI ridge regressions
%   using marginal maximum likelihood (ridgeMML) for regularization.
%
% Inputs:
%   fluo_file          - mov_aligned_dff.dat (fluorescence data)
%   session_file       - Bpod session .mat file with behavioral events
%   neural_roi_file    - fluoROIs.roimsk (AU_L, BC_L, etc.)
%   vascular_mask_file - VascularMask.roimsk
%   brain_mask_file    - Mask.roimsk
%
% Output:
%   Saves ridge regression results to ridge_events_only_mml_results.mat and
%   prints cross-validated R^2 / lambda statistics per ROI.

fprintf('=== Events-only Design Matrix Ridge Regression with MML ===\n');
fprintf('Protocol: WNoiseLickWater (21 stimulus + 224 lick = 245 regressors)\n');
fprintf('Optimization: Marginal Maximum Likelihood (per-ROI lambda)\n');
fprintf('Vascular masking: ENABLED (remove vascular pixels before brain mask)\n\n');

% Ensure ridgeMML is available
if ~exist('ridgeMML', 'file')
    addpath('C:\Users\shires\Downloads');
    addpath('H:\IsaacAndGarrettMatlabScripts\glm code\Puff_Dataset');
    fprintf('Added ridgeMML to path\n');
end

%% 1. Load fluorescence data with dual masking (vascular + brain)
fprintf('1. Loading fluorescence data with vascular + brain masking...\n');

[fluo_dir, fluo_name, ~] = fileparts(fluo_file);
fluo_mat = fullfile(fluo_dir, [fluo_name '.mat']);
fluo_meta = load(fluo_mat);
n_y = fluo_meta.datSize(1);
n_x = fluo_meta.datSize(2);
sampling_rate = fluo_meta.Freq;

fileInfo = dir(fluo_file);
fluo_T = fileInfo.bytes / (n_y * n_x * 4); % 4 bytes per single precision value

vascular_data = load(vascular_mask_file, '-mat');
brain_data = load(brain_mask_file, '-mat');
vascular_mask = vascular_data.ROI_info(1).Stats.ROI_binary_mask;
brain_mask = brain_data.ROI_info(1).Stats.ROI_binary_mask;

final_mask = brain_mask & ~vascular_mask;
final_mask_1d = final_mask(:);
n_brain_pixels = sum(final_mask_1d);

fprintf('  Fluorescence: %d x %d x %d frames @ %.1f Hz\n', n_y, n_x, fluo_T, sampling_rate);
fprintf('  Vascular pixels removed: %d\n', sum(vascular_mask(:)));
fprintf('  Brain pixels retained: %d (%.1f%% of frame)\n', n_brain_pixels, 100*n_brain_pixels/(n_y*n_x));

fprintf('  Loading masked fluorescence data...\n');
chunk_size = 500;
n_chunks = ceil(fluo_T / chunk_size);
fluo_masked = zeros(n_brain_pixels, fluo_T, 'single');

fid = fopen(fluo_file, 'r');
for chunk = 1:n_chunks
    start_idx = (chunk-1)*chunk_size + 1;
    end_idx = min(chunk*chunk_size, fluo_T);
    frames_in_chunk = end_idx - start_idx + 1;

    data_chunk = fread(fid, [n_y*n_x, frames_in_chunk], 'single');
    data_chunk = reshape(data_chunk, [n_y, n_x, frames_in_chunk]);

    for t = 1:frames_in_chunk
        frame = data_chunk(:,:,t);
        fluo_masked(:, start_idx + t - 1) = frame(final_mask_1d);
    end

    if mod(chunk, 10) == 0 || chunk == n_chunks
        fprintf('    Chunk %d/%d\n', chunk, n_chunks);
    end
end
fclose(fid);

fprintf('  Fluorescence data loaded: %d pixels x %d frames\n\n', size(fluo_masked));

%% 2. Extract neural ROI timeseries from masked fluorescence
fprintf('2. Extracting neural ROI timeseries...\n');

neural_data = load(neural_roi_file, '-mat');
n_neural = length(neural_data.ROI_info);
neural_timeseries = zeros(fluo_T, n_neural);
neural_names = cell(n_neural, 1);

for roi_idx = 1:n_neural
    mask_2d = neural_data.ROI_info(roi_idx).Stats.ROI_binary_mask;
    mask_1d = mask_2d(:);
    mask_masked = mask_1d(final_mask_1d);
    neural_timeseries(:, roi_idx) = mean(fluo_masked(mask_masked, :), 1)';
    neural_names{roi_idx} = neural_data.ROI_info(roi_idx).Name;
    fprintf('  %s: mean = %f, std = %f\n', neural_names{roi_idx}, ...
        mean(neural_timeseries(:, roi_idx)), std(neural_timeseries(:, roi_idx)));
end
fprintf('\n');

%% 3. Construct design matrix from session events (stimulus + licks only)
fprintf('3. Constructing events-only design matrix...\n');

min_length = fluo_T;
total_time_s = min_length / sampling_rate;
session_data = load(session_file);
if isfield(session_data, 'SessionData')
    [stimulus_events, lick_events] = extract_behavioral_events_session(session_data.SessionData, ...
        'WNoiseLickWater', total_time_s);
elseif all(isfield(session_data, {'eventID', 'timestamps', 'eventNameList', 'state'}))
    [stimulus_events, lick_events] = extract_behavioral_events_log(session_data, total_time_s);
else
    error('Session file %s does not contain SessionData or event log fields (eventID, timestamps, eventNameList).', ...
        session_file);
end

design_matrix = [];
regressor_names = {};

group_defs = struct( ...
    'label', {'Noise stimulus', 'Lick post-stimulus', 'Lick post-water (cued)', ...
              'Lick post-water (uncued)', 'Lick post-water (omission)'}, ...
    'prefix', {'noise_primary_', 'lick_post_stim_', 'lick_water_cued_', ...
               'lick_water_uncued_', 'lick_water_omission_'});
group_defs = group_defs(:);
n_groups = numel(group_defs);

% 3a. Noise stimulus regressors (21 columns, kernel 0:0.1:2 s)
stim_kernel = 0:0.1:2;
noise_regressors = create_stimulus_regressors( ...
    stimulus_events.noise_primary.times, ...
    stimulus_events.noise_primary.intensities, ...
    min_length, sampling_rate, stim_kernel);
design_matrix = [design_matrix, noise_regressors];
for i = 1:length(stim_kernel)
    regressor_names{end+1} = sprintf('noise_primary_t%.1fs', stim_kernel(i)); %#ok<AGROW>
end

% 3b. Lick regressors (4 categories x 56 columns, kernel -0.5:0.1:5 s)
lick_kernel = -0.5:0.1:5;

post_stim_regressors = create_lick_regressors( ...
    lick_events.post_stimulus, min_length, sampling_rate, lick_kernel);
design_matrix = [design_matrix, post_stim_regressors];
for i = 1:length(lick_kernel)
    regressor_names{end+1} = sprintf('lick_post_stim_t%.1fs', lick_kernel(i)); %#ok<AGROW>
end

cued_regressors = create_lick_regressors( ...
    lick_events.post_water_cued, min_length, sampling_rate, lick_kernel);
design_matrix = [design_matrix, cued_regressors];
for i = 1:length(lick_kernel)
    regressor_names{end+1} = sprintf('lick_water_cued_t%.1fs', lick_kernel(i)); %#ok<AGROW>
end

uncued_regressors = create_lick_regressors( ...
    lick_events.post_water_uncued, min_length, sampling_rate, lick_kernel);
design_matrix = [design_matrix, uncued_regressors];
for i = 1:length(lick_kernel)
    regressor_names{end+1} = sprintf('lick_water_uncued_t%.1fs', lick_kernel(i)); %#ok<AGROW>
end

omission_regressors = create_lick_regressors( ...
    lick_events.post_water_omission, min_length, sampling_rate, lick_kernel);
design_matrix = [design_matrix, omission_regressors];
for i = 1:length(lick_kernel)
    regressor_names{end+1} = sprintf('lick_water_omission_t%.1fs', lick_kernel(i)); %#ok<AGROW>
end

fprintf('  Design matrix constructed: %d timepoints x %d regressors\n\n', size(design_matrix));
n_regressors_original = numel(regressor_names);

% Align neural timeseries to design matrix length
neural_timeseries = neural_timeseries(1:min_length, :);

%% 4. Remove zero-variance regressors
fprintf('4. Removing zero-variance regressors...\n');
regressor_std = std(design_matrix);
zero_var_mask = (regressor_std < 1e-10);
n_zero = sum(zero_var_mask);

if n_zero > 0
    fprintf('  Removed %d zero-variance regressors\n', n_zero);
    design_matrix(:, zero_var_mask) = [];
    regressor_names(zero_var_mask) = [];
end
fprintf('  Final design matrix: %d x %d\n\n', size(design_matrix));

group_indices = cell(n_groups, 1);
group_lags = cell(n_groups, 1);
lag_pattern = 't([-\d\.]+)s';
for g = 1:n_groups
    prefix = group_defs(g).prefix;
    match_idx = find(startsWith(regressor_names, prefix));
    group_indices{g} = match_idx;

    lags = nan(numel(match_idx), 1);
    for k = 1:numel(match_idx)
        tok = regexp(regressor_names{match_idx(k)}, lag_pattern, 'tokens', 'once');
        if ~isempty(tok)
            lags(k) = str2double(tok{1});
        end
    end
    group_lags{g} = lags;
end

%% 5. Z-score normalize regressors
fprintf('5. Z-score normalizing regressors...\n');
design_matrix = zscore(design_matrix);

%% 6. Perform MML ridge regression for each neural ROI
fprintf('\n6. Performing MML ridge regression for each neural ROI...\n');
fprintf('WARNING: This may take a LONG time (tens of minutes per ROI)\n\n');

k_folds = 10;
results = struct();
rng('shuffle');

for roi_idx = 1:n_neural
    fprintf('\n=== Processing %s (ROI %d/%d) ===\n', neural_names{roi_idx}, roi_idx, n_neural);
    fprintf('Started at: %s\n', datestr(now));

    Y = neural_timeseries(:, roi_idx);
    X = design_matrix;

    allIdx = (1:size(X,1))';
    [trainIdx, testIdx] = kFoldSplit(allIdx, k_folds);

    R2_cv = zeros(k_folds, 1);
    lambdas_cv = cell(k_folds, 1);
    betas_cv = cell(k_folds, 1);
    group_single_R2 = nan(n_groups, k_folds);
    group_shuffle_R2 = nan(n_groups, k_folds);

    total_start = tic;
    for fold = 1:k_folds
        fprintf('\n  --- Fold %d/%d ---\n', fold, k_folds);
        fold_start = tic;

        X_train = X(trainIdx(:,fold),:);
        X_test = X(testIdx(:,fold),:);
        Y_train = Y(trainIdx(:,fold));
        Y_test = Y(testIdx(:,fold));

        fprintf('  Running ridgeMML (this may take 30+ minutes)...\n');
        [lambdas_cv{fold}, betas_cv{fold}, convergenceFailures] = ridgeMML(Y_train, X_train, 1);

        if convergenceFailures > 0
            fprintf('  WARNING: %d components failed to converge\n', convergenceFailures);
        end

        X_test_centered = X_test - mean(X_train);
        Y_pred = X_test_centered * betas_cv{fold};

        TSS = sum((Y_test - mean(Y_test)).^2);
        RSS = sum((Y_test - mean(Y_train) - Y_pred).^2);
        R2_cv(fold) = max(0, (1 - RSS/TSS) * 100);

        for g = 1:n_groups
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
                X_test_single_centered = X_test_single - mean(X_train_single, 1);
                Y_pred_single = X_test_single_centered * beta_single;
                TSS_single = sum((Y_test - mean(Y_test)).^2);
                RSS_single = sum((Y_test - mean(Y_train) - Y_pred_single).^2);
                group_single_R2(g, fold) = max(0, (1 - RSS_single / TSS_single) * 100);
            end

            X_train_shuff = X_train;
            X_test_shuff = X_test;
            perm_train = randperm(size(X_train_shuff, 1));
            perm_test = randperm(size(X_test_shuff, 1));
            X_train_shuff(:, idx) = X_train_shuff(perm_train, idx);
            X_test_shuff(:, idx) = X_test_shuff(perm_test, idx);

            [~, beta_shuff, ~] = ridgeMML(Y_train, X_train_shuff, 1);
            X_test_shuff_centered = X_test_shuff - mean(X_train_shuff, 1);
            Y_pred_shuff = X_test_shuff_centered * beta_shuff;
            RSS_shuff = sum((Y_test - mean(Y_train) - Y_pred_shuff).^2);
            TSS_shuff = sum((Y_test - mean(Y_test)).^2);
            group_shuffle_R2(g, fold) = max(0, (1 - RSS_shuff / TSS_shuff) * 100);
        end

        fold_elapsed = toc(fold_start);
        fprintf('  Fold complete: R^2 = %.2f%%, Time = %.1f min\n', R2_cv(fold), fold_elapsed/60);

        checkpoint = struct();
        checkpoint.roi_name = neural_names{roi_idx};
        checkpoint.roi_idx = roi_idx;
        checkpoint.fold = fold;
        checkpoint.R2_cv = R2_cv(1:fold);
        checkpoint.lambdas_cv = lambdas_cv(1:fold);
        checkpoint.betas_cv = betas_cv(1:fold);
        checkpoint.timestamp = datestr(now);
        save(sprintf('mml_checkpoint_events_%s_fold%d.mat', neural_names{roi_idx}, fold), 'checkpoint', '-v7.3');
    end

    total_elapsed = toc(total_start);
    R2_mean = mean(R2_cv);
    R2_std = std(R2_cv);

    fprintf('\n  %s COMPLETE:\n', neural_names{roi_idx});
    fprintf('  R^2: %.2f%% +/- %.2f%%\n', R2_mean, R2_std);
    fprintf('  Total time: %.1f min (%.2f hours)\n\n', total_elapsed/60, total_elapsed/3600);

    all_lambdas = [];
    for fold = 1:k_folds
        all_lambdas = [all_lambdas; lambdas_cv{fold}(:)]; %#ok<AGROW>
    end
    lambda_mean = mean(all_lambdas);
    lambda_std = std(all_lambdas);

    group_explained_mean = nan(n_groups, 1);
    group_explained_std = nan(n_groups, 1);
    group_unique_mean = nan(n_groups, 1);
    group_unique_std = nan(n_groups, 1);
    baseline_per_fold = R2_cv(:)';

    for g = 1:n_groups
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
            diff_vals = baseline_per_fold(valid_shuffle) - vals;
            diff_vals = max(diff_vals, 0);
            group_unique_mean(g) = mean(diff_vals);
            group_unique_std(g) = std(diff_vals);
        end
    end

    results.(neural_names{roi_idx}).R2_cv = R2_cv;
    results.(neural_names{roi_idx}).R2_mean = R2_mean;
    results.(neural_names{roi_idx}).R2_std = R2_std;
    results.(neural_names{roi_idx}).lambdas_cv = lambdas_cv;
    results.(neural_names{roi_idx}).lambda_mean = lambda_mean;
    results.(neural_names{roi_idx}).lambda_std = lambda_std;
    results.(neural_names{roi_idx}).betas_cv = betas_cv;
    results.(neural_names{roi_idx}).group_labels = {group_defs.label};
    results.(neural_names{roi_idx}).group_single_R2 = group_single_R2;
    results.(neural_names{roi_idx}).group_shuffle_R2 = group_shuffle_R2;
    results.(neural_names{roi_idx}).group_explained_mean = group_explained_mean;
    results.(neural_names{roi_idx}).group_explained_std = group_explained_std;
    results.(neural_names{roi_idx}).group_unique_mean = group_unique_mean;
    results.(neural_names{roi_idx}).group_unique_std = group_unique_std;
end

%% 7. Save final results
fprintf('\n7. Saving final results...\n');

results.metadata = struct();
timestamp_serial = now;
results.metadata.k_folds = k_folds;
results.metadata.sampling_rate = sampling_rate;
results.metadata.regressor_names = regressor_names;
results.metadata.neural_names = neural_names;
results.metadata.n_regressors_original = n_regressors_original;
results.metadata.n_regressors_final = size(design_matrix, 2);
results.metadata.n_zero_variance = n_zero;
results.metadata.group_labels = {group_defs.label};
results.metadata.group_prefixes = {group_defs.prefix};
results.metadata.group_lags = group_lags;
results.metadata.timestamp = datestr(timestamp_serial);
timestampTag = datestr(timestamp_serial, 'yyyymmdd_HHMMSS');
results.metadata.output_file = sprintf('ridge_events_only_mml_results_%s.mat', timestampTag);
output_file = fullfile(pwd, results.metadata.output_file);
save(output_file, 'results', '-v7.3');
output_info = dir(output_file);
fprintf('Results saved: %s (%.1f MB)\n\n', output_file, output_info.bytes/1e6);

%% 8. Print final summary
fprintf('=== FINAL SUMMARY ===\n');
fprintf('Design matrix: %d regressors\n', size(design_matrix, 2));
fprintf('  - Noise stimulus: 21\n');
fprintf('  - Lick events: 224 (4 categories - 56 lags)\n\n');

fprintf('Prediction performance (MML optimization):\n');
for roi_idx = 1:n_neural
    fprintf('  %s: R^2 = %.2f%% +/- %.2f%% (lambda = %.2f +/- %.2f)\n', ...
        neural_names{roi_idx}, ...
        results.(neural_names{roi_idx}).R2_mean, ...
        results.(neural_names{roi_idx}).R2_std, ...
        results.(neural_names{roi_idx}).lambda_mean, ...
        results.(neural_names{roi_idx}).lambda_std);
end

fprintf('\nAll processing complete at: %s\n', datestr(now));

end

%% Helper functions (copied/adapted from full design-matrix script)

function [stimulus_events, lick_events] = extract_behavioral_events_session(SessionData, protocol_type, total_time_s)
    % Extract behavioral events from Bpod SessionData (mirrors full-design helper)
    fprintf('    Extracting behavioral events from Bpod SessionData (%s)...\n', protocol_type);
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

    % Convert to column vectors
    stimulus_events.noise_primary.times = stimulus_events.noise_primary.times(:);
    stimulus_events.noise_primary.intensities = stimulus_events.noise_primary.intensities(:);
    lick_events.post_stimulus = lick_events.post_stimulus(:);
    lick_events.post_water_cued = lick_events.post_water_cued(:);
    lick_events.post_water_uncued = lick_events.post_water_uncued(:);
    lick_events.post_water_omission = lick_events.post_water_omission(:);

    % Trim events to recording duration
    cutoff_time = total_time_s;
    fields_to_trim = {'post_stimulus', 'post_water_cued', 'post_water_uncued', 'post_water_omission'};
    for f = fields_to_trim
        field_name = f{1};
        lick_events.(field_name) = lick_events.(field_name)(lick_events.(field_name) <= cutoff_time);
    end
    keep_idx = stimulus_events.noise_primary.times <= cutoff_time;
    stimulus_events.noise_primary.times = stimulus_events.noise_primary.times(keep_idx);
    stimulus_events.noise_primary.intensities = stimulus_events.noise_primary.intensities(keep_idx);

    fprintf('      Noise stimuli: %d\n', length(stimulus_events.noise_primary.times));
    fprintf('      Post-stimulus licks: %d\n', length(lick_events.post_stimulus));
    fprintf('      Post-water cued licks: %d\n', length(lick_events.post_water_cued));
    fprintf('      Post-water uncued licks: %d\n', length(lick_events.post_water_uncued));
    fprintf('      Post-water omission licks: %d\n', length(lick_events.post_water_omission));
end

function [stimulus_events, lick_events] = extract_behavioral_events_log(log_struct, total_time_s)
    % Extract events from imaging-aligned event log (timestamps + eventID)
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

    fprintf('      Noise/whisker stimuli: %d\n', length(stimulus_events.noise_primary.times));
    fprintf('      Post-stimulus licks: %d\n', length(lick_events.post_stimulus));
    fprintf('      Post-water (cued) licks: %d\n', length(lick_events.post_water_cued));
    fprintf('      Post-water uncued licks: %d\n', length(lick_events.post_water_uncued));
    fprintf('      Post-water omission licks: %d\n', length(lick_events.post_water_omission));
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

function [trainIdx, testIdx] = kFoldSplit(allIdx, kF)
    sigL = length(allIdx);
    trainPer = 1 - 1/kF;
    trainL = round(trainPer * sigL);
    testL = sigL - trainL;
    trainIdx = zeros(sigL, kF);

    trainIdx(1:trainL) = 1;
    for ii = 2:kF
        trainIdx(:, ii) = circshift(trainIdx(:, ii-1), testL);
    end

    testIdx = (trainIdx == 0);
    trainIdx = logical(trainIdx);
    testIdx = logical(testIdx);
end
