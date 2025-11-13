# TemporalModelFull.m - Multi-ROI Temporal Modeling

## Overview

`TemporalModelFull.m` extends the single-ROI temporal modeling approach to analyze **ALL neural ROIs simultaneously** in one unified ridge regression model. This provides ~10-20x speedup and enables direct cross-ROI comparisons of temporal dynamics.

## Key Innovation

Instead of running separate models for each ROI, **all ROIs are fitted together**:

```matlab
% ONE call to ridgeMML:
[lambdas, betas_all] = ridgeMML(Y_all, X, 1);

Where:
  Y_all = [n_valid × n_ROIs]    % All neural ROIs side-by-side
  X = [n_valid × n_lags]         % Shared lagged behavior design matrix
  betas_all = [n_lags × n_ROIs]  % Each column = one ROI's temporal kernel
  lambdas = [1 × n_ROIs]         % Optimal regularization per ROI
```

## Methodological Consistency

**Validation Result**: TemporalModelFull produces **identical** results to TemporalModel when analyzing the same ROI:
- R² values match exactly (difference < 0.000001)
- Beta weights correlation = 1.0000 (perfect match)
- Same CV methodology, same metrics, same interpretation

## Usage

### Basic Usage (Analyze All ROIs)

```matlab
load('ROI.mat');  % From rois_to_mat.m

opts = struct();
opts.behavior_predictor = 'Face';
opts.min_lag = -5;
opts.max_lag = 10;
opts.cv_folds = 5;

results = TemporalModelFull(ROI, opts);
```

### Advanced Usage (Analyze Subset of ROIs)

```matlab
opts = struct();
opts.target_neural_rois = {'M2_L', 'BC_L', 'RS_L'};  % Cell array of ROI names
opts.behavior_predictor = 'Face';
opts.min_lag_seconds = -0.5;  % Use seconds instead of frames
opts.max_lag_seconds = 1.0;
opts.cv_folds = 5;

results = TemporalModelFull(ROI, opts);
```

## Output Structure

### Per-ROI Results

Each ROI gets its own temporal kernel and performance metrics:

```matlab
results.temporal_kernels(i)     % [n_ROIs × 1] struct array
  .roi_name                      % 'AU_L', 'BC_L', etc.
  .beta_cv_mean                  % [n_lags × 1] - primary estimate
  .beta_cv_sem                   % Standard error across CV folds
  .peak_lag_sec                  % Peak response timing
  .peak_beta                     % Peak beta coefficient

results.performance(i)          % [n_ROIs × 1] struct array
  .roi_name
  .R2_cv_mean                    % Cross-validated R² (realistic)
  .R2_cv_sem                     % Standard error
  .R2_full_data                  % Full-data R² (optimistic)
  .lambda_cv_mean                % Regularization strength
```

### Cross-ROI Comparison

```matlab
results.comparison
  .roi_names                     % {1 × n_ROIs} cell array
  .beta_matrix_cv                % [n_lags × n_ROIs] all temporal kernels
  .R2_all_rois                   % [1 × n_ROIs] R² for each ROI
  .peak_lags_all_sec             % [1 × n_ROIs] peak timing for each
  .peak_betas_all                % [1 × n_ROIs] peak strength for each
```

### Predictions

```matlab
results.predictions
  .Y_pred                        % [n_valid × n_ROIs] predicted fluorescence
  .Y_actual                      % [n_valid × n_ROIs] actual (z-scored)
  .behavior_trace_z              % Z-scored behavior predictor
```

## Visualizations

Four publication-quality plots are automatically generated:

### 1. All Temporal Kernels Overlay
- All ROI temporal kernels on one plot
- Color-coded by ROI
- SEM envelopes for uncertainty
- R² values in legend

### 2. Temporal Kernel Heatmap
- ROIs (rows) × Lags (columns)
- Red-blue colormap (centered on zero)
- Identifies regions of strong positive/negative coupling
- Easy visualization of temporal structure across brain

### 3. Performance Comparison
- **Top panel**: R² for each ROI (bar plot with error bars)
- **Bottom panel**: Peak lag timing (color-coded: red=predictive, green=reactive)

### 4. Multi-ROI Predictions
- Top 4 ROIs by R²
- Actual vs predicted traces
- Behavioral predictor trace at bottom
- Time-aligned across all panels

## Test Results

Test with 4 neural ROIs (AU_L, BC_L, M2_L, RS_L):
- **Best ROI**: BC_L (R² = 69.01% ± 4.50%)
- **R² range**: [21.70%, 69.01%]
- **Peak lag range**: [-0.500s, +0.100s]
- **Processing time**: ~15 seconds for 4 ROIs × 38,514 timepoints

## Computational Advantages

| Approach | N ROIs | SVD Calls | Time (est) |
|----------|--------|-----------|------------|
| Loop TemporalModel | 10 | 50 (10 × 5 folds) | ~150s |
| **TemporalModelFull** | 10 | **5** (1 × 5 folds) | **~15s** |

**Speedup: ~10x** for typical analysis

## Scientific Applications

### 1. Regional Comparison
Identify which brain regions lead or lag behavior:
```matlab
% Find predictive regions (negative peak lag)
predictive_idx = find([results.comparison.peak_lags_all_sec] < 0);
predictive_rois = results.comparison.roi_names(predictive_idx);
```

### 2. Temporal Hierarchy
Order regions by response timing:
```matlab
[sorted_lags, order] = sort([results.comparison.peak_lags_all_sec]);
temporal_hierarchy = results.comparison.roi_names(order);
```

### 3. Coupling Strength
Compare how strongly different regions couple to behavior:
```matlab
coupling_strength = abs([results.comparison.peak_betas_all]);
[~, strongest_idx] = max(coupling_strength);
strongest_roi = results.comparison.roi_names{strongest_idx};
```

## Example Analysis Workflow

```matlab
%% 1. Load data
load('ROI.mat');

%% 2. Run full multi-ROI analysis
opts = struct();
opts.behavior_predictor = 'Face';
opts.min_lag_seconds = -0.5;
opts.max_lag_seconds = 1.0;
results = TemporalModelFull(ROI, opts);

%% 3. Identify motor planning regions (predictive)
predictive = find([results.comparison.peak_lags_all_sec] < 0);
fprintf('Predictive regions:\n');
for i = predictive
    fprintf('  %s: peak at %.3f s (R² = %.2f%%)\n', ...
        results.comparison.roi_names{i}, ...
        results.comparison.peak_lags_all_sec(i), ...
        results.comparison.R2_all_rois(i) * 100);
end

%% 4. Export temporal kernels for statistics
beta_matrix = results.comparison.beta_matrix_cv;  % [lags × ROIs]
lag_times = results.temporal_kernels(1).lag_times_sec;

% Save for external analysis (e.g., ANOVA, clustering)
save('temporal_kernels_for_stats.mat', 'beta_matrix', 'lag_times', ...
    'results.comparison.roi_names');
```

## Files Created

1. **TemporalModelFull.m** - Main function
2. **test_TemporalModelFull.m** - Test script with validation
3. **TemporalModelFull_test_all_rois.mat** - Example results (all ROIs)
4. **TemporalModelFull_test_subset.mat** - Example results (3 ROIs)

## NEW: TemporalModelEventsFull.m

TemporalModelEventsFull extends these same multi-ROI speedups to the combined
motion + event design used by TemporalModelEvents. It shares the same API and
outputs (temporal kernels, predictions, cross-ROI comparison), but each ROI is fit
against a design matrix that concatenates:

- Bidirectional face-motion lags (predictor from ROI.modalities.behavior)
- Stimulus kernels (e.g., 0:0.1:2 s) convolved with event onsets
- Lick kernels for post-stimulus / water / omission categories

Key additions:

- Supports `target_neural_rois`, `stim_kernel`, `lick_kernel`, `remove_initial_seconds`, etc.
- Builds a single design matrix and runs ridgeMML once per fold for all ROIs.
- Returns per-ROI motion kernels **and** event kernels, with CV means/SEMs.
- Produces multi-ROI plots (kernel overlay, heatmap, R^2 bars) and saves results to
  `TemporalModelEventsFull_*` MAT files.

Usage mirrors TemporalModelFull, now with the extra `session_file` argument:

```matlab
load('ROI.mat');
session_file = 'SessionData.mat';
opts = struct('behavior_predictor','Face','stim_kernel',0:0.1:2);
results = TemporalModelEventsFull(ROI, session_file, opts);
```

See `Modeling/TemporalModelEventsFull.m` for full option documentation.

## Comparison with TemporalModel.m

| Feature | TemporalModel | TemporalModelFull |
|---------|---------------|-------------------|
| ROIs analyzed | 1 | All (or subset) |
| Model fits | 1 | 1 (simultaneous) |
| Cross-ROI comparison | ❌ | ✅ |
| CV methodology | Identical | Identical |
| Computational cost | 1x | 0.1x per ROI |
| Use case | Single ROI deep dive | Regional comparison |

## Next Steps

- Use for analyzing regional differences in temporal dynamics
- Compare motor vs sensory cortex response timing
- Identify predictive vs reactive brain regions
- Statistical testing of cross-ROI differences
