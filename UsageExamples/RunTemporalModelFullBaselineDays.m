%% Paths
mouse_root   = 'F:\2025Summer\Cohort2Analysis\AH1473';
session_root = fullfile(mouse_root, 'AH1473_07_30_25_WNoiseWater', 'CtxImg');

fluor_movie        = fullfile(session_root, 'mov_aligned.dat');          % fluorescence
motion_movie       = fullfile(session_root, 'motion_energy.dat');        % behavioral movie (Face ROI)
behavior_roi_file  = fullfile(session_root, 'behavROIs.roimsk');          % Face mask(s)
allen_roi_file     = fullfile(mouse_root, 'Allen.roimsk');               % global ROIs
vascular_mask_file = fullfile(mouse_root, 'VascularMask.roimsk');        % delete vasculature
outside_mask_file  = fullfile(mouse_root, 'Mask.roimsk');                % keep only in-brain pixels
roi_mat_file       = fullfile(session_root, 'ROI_AllenCtx.mat');

%% Generate ROI struct once per session
roi_opts = struct( ...
    'OutputFile', roi_mat_file, ...
    'SaveOutput', true, ...
    'ComputeDFF', true, ...
    'DFFParams', struct( ...
        'baseline_method', 'running_percentile', ...
        'baseline_window_seconds', 600, ...
        'baseline_percentile', 10, ...   % keep/change as needed
        'show_plot', false));            % avoid per-ROI figures

ROI = rois_to_mat(fluor_movie, motion_movie, behavior_roi_file, ...
    allen_roi_file, vascular_mask_file, outside_mask_file, roi_opts);

% If ROI already saved, load it instead of recomputing:
% tmp = load(roi_mat_file, 'ROI'); ROI = tmp.ROI;

%% Configure TemporalModelFull

opts = struct();
opts.target_neural_rois = {};        % empty => run all ROIs
opts.behavior_predictor = 'Face';
opts.min_lag_seconds = -2;           % adjust as needed
opts.max_lag_seconds =  4;
opts.cv_folds = 5;
opts.save_results = false;
opts.show_plots = true;              % set false if you want to skip figures
opts.poster_plots = true;
opts.no_category_plots = true;       % <-- NEW: Enable no-category version
opts.kernel_overlay_rois = {'AU_L', 'M2_L'};
opts.analysis_window_sec = 1;
opts.peak_metric = 'com';
opts.analysis_window_sec = [-.5,.5];
opts.kernel_mean_saturation = 0.8;      % lighter grey mean
opts.kernel_overlay_saturation = 0.4;   % soften both example colors



%% Run full temporal model
results = TemporalModelFull(ROI, opts);
