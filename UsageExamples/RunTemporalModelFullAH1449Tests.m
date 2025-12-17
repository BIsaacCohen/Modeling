%--- add analysis code to path --------------------------------------------
analysis_dir = 'C:/Users/shires/Documents/Isaac/Code/Modeling';
addpath(analysis_dir);   % ensures TemporalModelFull + dependencies resolve

%---Make ROI File ---------------------------------------------------------

addpath('C:/Users/shires/Documents/Isaac/Code/Modeling');

fluor_movie  = 'A:\UmiToolBoxAnalysis\AH1449\AH1449_06_12_25_WNoiseNoWater\CtxImg\mov_aligned.dat';
motion_movie = 'A:\UmiToolBoxAnalysis\AH1449\AH1449_06_12_25_WNoiseNoWater\CtxImg\motion_energy.dat';
behav_rois   = 'A:\UmiToolBoxAnalysis\AH1449\AH1449_06_12_25_WNoiseNoWater\CtxImg\behavROIs.roimsk';
fluo_rois    = 'A:/UmiToolBoxAnalysis/AH1449/Allen.roimsk';
vascular_mask = 'A:/UmiToolBoxAnalysis/AH1449/VascularMask.roimsk';
brain_mask    = 'A:/UmiToolBoxAnalysis/AH1449/Mask.roimsk';

opts = struct();
opts.DFFParams = struct( ...
    'baseline_method', 'running_percentile', ...
    'baseline_window_seconds', 600, ...
    'baseline_percentile', 10, ...
    'show_plot', false);

ROI = rois_to_mat(fluor_movie, motion_movie, behav_rois, fluo_rois, ...
                  vascular_mask, brain_mask, opts);


%--- load ROI struct from the sample dataset ------------------------------
roi_path = 'A:\UmiToolBoxAnalysis\AH1449\AH1449_06_12_25_WNoiseNoWater\CtxImg/ROI.mat';
roi_file = load(roi_path);
ROI = roi_file.ROI;

%--- configure model options ----------------------------------------------
opts = struct();
opts.target_neural_rois = {};        % empty => run all ROIs
opts.behavior_predictor = 'Face';
opts.min_lag_seconds = -5;           % adjust as needed
opts.max_lag_seconds =  10.5;
opts.cv_folds = 5;
opts.save_results = false;
opts.show_plots = true;              % set false if you want to skip figures
opts.kernel_overlay_rois = {'AU_L', 'M2_L'};
opts.analysis_window_sec = 1;
opts.peak_metric = 'com';
opts.analysis_window_sec = [-.5,.5];
opts.kernel_mean_saturation = 0.8;      % lighter grey mean
opts.kernel_overlay_saturation = 0.4;   % soften both example colors



%--- run the multi-ROI model ----------------------------------------------
results = TemporalModelFull(ROI, opts);

disp('TemporalModelFull finished. Results struct saved and available in workspace.');


%%

PlotPosterNoCategoryTemporalModelFull(results, opts)
