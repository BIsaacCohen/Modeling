# Modeling (Temporal Model)

Minimal set for running the multi-ROI temporal model, no-category poster plots, and a single usage example.

## What’s here
- `TemporalModelFull.m` — Multi-ROI bidirectional lag ridge regression (shared lag design, CV, predictions).
- `PlotPosterNoCategoryTemporalModelFull.m` — Poster-quality plotting without predictive/reactive categories (kernel mean, heatmap, predictions, brain maps, CV summaries).
- `ridgeMML.m` — Ridge regression with marginal maximum likelihood lambda selection.
- `rois_to_mat.m` — Extracts fluorescence + behavior ROI traces into `ROI` struct (optionally runs `dFF_phasic`).
- `BatchRois_to_mat.m` — Batch wrapper to generate `ROI.mat` across session folders.
- `dFF_phasic.m` — Phasic dF/F computation with baseline removal.
- `ModelFitAnalysis/BoxWhiskerCOMandR2.m` — Box/whisker plots of CoM lags and CV R²/pearson r per region.
- `UsageExamples/RunTemporalModelFullAH1449Tests.m` — Example script wiring ROI extraction and model run.

Auxiliary:
- `.claude/`, `AGENTS.md`, `CLAUDE.md` — agent/config docs (do not delete).

## Typical workflow
1) Build ROI struct (once per session):
```matlab
ROI = rois_to_mat(fluor_movie, motion_movie, behav_rois, fluo_rois, vascular_mask, brain_mask, opts);
```
2) Run the temporal model and plot (always no-category):
```matlab
opts = struct();
opts.behavior_predictor = 'Face';
opts.min_lag_seconds = -5;
opts.max_lag_seconds = 10.5;
opts.cv_folds = 5;
opts.poster_plots = true;   % ignored; plotting is always no-category
results = TemporalModelFull(ROI, opts);
```
3) Optional analysis plot:
```matlab
BoxWhiskerCOMandR2(results, {'AU_L','M2_L'});
```

## Notes
- Plotting always uses `PlotPosterNoCategoryTemporalModelFull`; category mode is removed.
- Dependencies: Signal Processing (e.g., `resample`, `butter`, `filtfilt`) and Statistics (`zscore`, `movprctile`) toolboxes.
- Ensure `ridgeMML.m` is on the MATLAB path; `TemporalModelFull` adds local folder automatically.
