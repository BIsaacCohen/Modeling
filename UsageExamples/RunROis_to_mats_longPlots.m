addpath('C:/Users/shires/Documents/Isaac/Code/Modeling');

fluor_movie  = 'A:/UmiToolBoxAnalysis/AH1449/AH1449_06_23_25_WNoiseWater_5_to7buffer_EXPERT/CtxImg/hemoCorr_fluo.dat';
motion_movie = 'A:/UmiToolBoxAnalysis/AH1449/AH1449_06_23_25_WNoiseWater_5_to7buffer_EXPERT/CtxImg/motion_energy.dat';
behav_rois   = 'A:/UmiToolBoxAnalysis/AH1449/AH1449_06_23_25_WNoiseWater_5_to7buffer_EXPERT/CtxImg/behavROIs.roimsk';
fluo_rois    = 'A:/UmiToolBoxAnalysis/AH1449/fluoROIs.roimsk';
vascular_mask = 'A:/UmiToolBoxAnalysis/AH1449/VascularMask.roimsk';
brain_mask    = 'A:/UmiToolBoxAnalysis/AH1449/Mask.roimsk';

opts = struct();
opts.DFFParams = struct('cutoff_freq', 0.01, 'show_plot', true);

ROI = rois_to_mat(fluor_movie, motion_movie, behav_rois, fluo_rois, ...
                  vascular_mask, brain_mask, opts);

% Zoom every "dFF Phasic Validation" figure to the first 5 minutes (0–300 s)
plot_window = [0 300];
validation_figs = findall(0,'Type','figure','-regexp','Name','^dFF Phasic Validation');
arrayfun(@(fig) set(findall(fig,'Type','axes'),'XLim',plot_window), validation_figs);

save('A:/UmiToolBoxAnalysis/AH1449/AH1449_06_23_25_WNoiseWater_5_to7buffer_EXPERT/CtxImg/ROI.mat','ROI','-v7.3');
