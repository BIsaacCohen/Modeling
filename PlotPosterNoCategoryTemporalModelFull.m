function PlotPosterNoCategoryTemporalModelFull(results, opts)
% PlotPosterNoCategoryTemporalModelFull Generate poster-quality plots WITHOUT categorization.
%
%   PlotPosterNoCategoryTemporalModelFull(results)
%   PlotPosterNoCategoryTemporalModelFull(results, opts)
%
%   INPUTS:
%       results - Output struct returned by TemporalModelFull containing
%                 temporal kernels, performance metrics, predictions, and metadata.
%       opts    - Optional struct with fields:
%           .prediction_rois         - {1x2} ROI names for prediction plots
%                                      (default: {'M1_L', 'V1_L'})
%           .kernel_overlay_rois     - up to 2 ROI names to overlay on the
%                                      group-mean kernel plot (default: {})
%           .kernel_mean_saturation  - scalar in [0,1], desaturates/brightens
%                                      the group-mean line color (default: 0.45)
%           .kernel_overlay_saturation - scalar in [0,1] applied to overlay ROI
%                                      colors (default: 0.65)
%
%   This function generates 4 poster-quality figures WITHOUT predictive/reactive categorization:
%       1. Overall group mean temporal kernel (all ROIs combined)
%       2. Sorted temporal kernel heatmap (no category dividing line)
%       3. Model predictions for 2 selected ROIs + behavior trace
%       4. Brain maps (1x3 horizontal): Peak Lag | Beta | CV R²
%
%   Optimized for poster presentation with larger fonts and cleaner layouts.

if nargin < 1 || isempty(results)
    error('PlotPosterNoCategoryTemporalModelFull:MissingResults', ...
        'Results struct from TemporalModelFull is required.');
end

% Default options
if nargin < 2 || isempty(opts)
    opts = struct();
end

if ~isfield(opts, 'prediction_rois') || isempty(opts.prediction_rois)
    opts.prediction_rois = {'M1_L', 'V1_L'};
end
if ~isfield(opts, 'kernel_overlay_rois') || isempty(opts.kernel_overlay_rois)
    opts.kernel_overlay_rois = {};
elseif ischar(opts.kernel_overlay_rois)
    opts.kernel_overlay_rois = {opts.kernel_overlay_rois};
end
if ~isfield(opts, 'kernel_mean_saturation') || isempty(opts.kernel_mean_saturation)
    opts.kernel_mean_saturation = 0.45;
end
if ~isfield(opts, 'kernel_overlay_saturation') || isempty(opts.kernel_overlay_saturation)
    opts.kernel_overlay_saturation = 0.65;
end
opts.kernel_mean_saturation = clamp_unit_value(opts.kernel_mean_saturation);
opts.kernel_overlay_saturation = clamp_unit_value(opts.kernel_overlay_saturation);

if numel(opts.kernel_overlay_rois) > 2
    error('PlotPosterNoCategoryTemporalModelFull:TooManyKernelOverlays', ...
        'kernel_overlay_rois supports at most 2 entries (got %d).', ...
        numel(opts.kernel_overlay_rois));
end

opts.kernel_overlay_rois = opts.kernel_overlay_rois(:).';

fprintf('\nGenerating poster-quality plots (no categorization)...\n');
plot_all_temporal_kernels(results, opts.kernel_overlay_rois, ...
    opts.kernel_mean_saturation, opts.kernel_overlay_saturation);
plot_temporal_kernel_heatmap(results);
plot_multi_roi_predictions_poster(results, opts.prediction_rois);
plot_multi_roi_cv_predictions_poster(results, opts.prediction_rois);
plot_peak_beta_brainmaps_poster(results);
plot_cv_fold_progression(results);
plot_cv_fold_progression_corr(results);
plot_peak_beta_corr_brainmaps_poster(results);
fprintf('Poster plots generated.\n');

end

%% ================= Plotting Functions =================

function plot_all_temporal_kernels(results, overlay_roi_names, mean_sat, overlay_sat)
    % Plot overall group mean kernel with SEM shading (all ROIs combined)

    if nargin < 2 || isempty(overlay_roi_names)
        overlay_roi_names = {};
    elseif ischar(overlay_roi_names)
        overlay_roi_names = {overlay_roi_names};
    end
    if nargin < 3 || isempty(mean_sat)
        mean_sat = 1;
    end
    if nargin < 4 || isempty(overlay_sat)
        overlay_sat = 1;
    end
    mean_sat = clamp_unit_value(mean_sat);
    overlay_sat = clamp_unit_value(overlay_sat);

    tk = results.temporal_kernels;
    n_rois = numel(tk);
    lag_times = tk(1).lag_times_sec;
    n_lags = numel(lag_times);
    roi_names = cell(n_rois, 1);

    beta_matrix = zeros(n_lags, n_rois);
    for roi = 1:n_rois
        beta_matrix(:, roi) = tk(roi).beta_cv_mean;
        roi_names{roi} = tk(roi).roi_name;
    end

    % Compute overall group mean (all ROIs together)
    [overall_mean, overall_sem] = compute_group_curve(beta_matrix, true(n_rois, 1));

    % Determine metric for title
    metric_str = 'Peak';
    if isfield(results.metadata, 'peak_metric') && strcmp(results.metadata.peak_metric, 'com')
        metric_str = 'CoM';
    end

    fig_title = sprintf('Temporal Kernel (Group Mean): %s (%s)', ...
        results.metadata.behavior_predictor, metric_str);

    figure('Name', fig_title, 'Position', [100 500 900 600]);
    hold on;

    % Optionally overlay up to two specific ROI kernels
    overlay_roi_names = overlay_roi_names(~cellfun('isempty', overlay_roi_names));
    if numel(overlay_roi_names) > 2
        error('PlotPosterNoCategoryTemporalModelFull:TooManyKernelOverlays', ...
            'kernel_overlay_rois supports at most 2 entries.');
    end
    if ~isempty(overlay_roi_names)
        overlay_colors = [
            0.85 0.10 0.15;  % red tone
            0.00 0.35 0.80]; % blue tone
        for idx = 1:numel(overlay_roi_names)
            roi_idx = find(strcmpi(roi_names, overlay_roi_names{idx}), 1);
            if isempty(roi_idx)
                error('PlotPosterNoCategoryTemporalModelFull:OverlayROINotFound', ...
                    'Overlay ROI "%s" not found. Available ROIs: %s', ...
                    overlay_roi_names{idx}, strjoin(roi_names, ', '));
            end
            this_curve = tk(roi_idx).beta_cv_mean;
            adj_color = apply_saturation(overlay_colors(idx, :), overlay_sat);
            plot(lag_times, this_curve, 'Color', adj_color, ...
                'LineWidth', 2.5, 'DisplayName', sprintf('%s kernel', roi_names{roi_idx}));
        end
    end

    % Plot single overall curve with neutral color last so it stays on top
    neutral_color = apply_saturation([0 0 0], mean_sat);
    plot_group_curve(lag_times, overall_mean, overall_sem, neutral_color, ...
        sprintf('All ROIs (n=%d)', n_rois));

    yline(0, 'Color', [0 0 0], 'LineWidth', 1.5, ...
        'LineStyle', '--', 'HandleVisibility', 'off');
    yl = ylim;
    plot([0 0], yl, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

    hold off;

    xlabel('Lag time (seconds)', 'FontSize', 13);
    ylabel('Beta coefficient (z-scored)', 'FontSize', 13);
    title(fig_title, 'FontSize', 16, 'Interpreter', 'none');
    legend('Location', 'best', 'FontSize', 11);
    grid off;
end

function plot_temporal_kernel_heatmap(results)
    % Heatmap sorted by peak lag (no category separation)

    beta_matrix = results.comparison.beta_matrix_cv';  % [n_rois × n_lags]
    roi_names = results.comparison.roi_names(:);
    peak_lags = results.comparison.peak_lags_all_sec(:);
    lag_times = results.temporal_kernels(1).lag_times_sec;

    % Sort ROIs by peak lag so predictive (negative) are at top
    [sorted_peak_lags, sort_idx] = sort(peak_lags, 'ascend', 'MissingPlacement', 'last');
    beta_matrix = beta_matrix(sort_idx, :);
    roi_names = roi_names(sort_idx);

    fig_title = sprintf('Temporal Kernel Heatmap: %d ROIs vs %s', ...
        results.metadata.n_rois, results.metadata.behavior_predictor);

    figure('Name', fig_title, 'Position', [150 450 900 600]);

    imagesc(lag_times, 1:length(roi_names), beta_matrix);
    colormap(flipud(redbluecmap));
    colorbar;

    % Center colormap on zero
    clim_max = max(abs(beta_matrix(:)));
    caxis([-clim_max, clim_max]);

    % Determine metric label
    metric_label = 'peak lag';
    if isfield(results.metadata, 'peak_metric') && strcmp(results.metadata.peak_metric, 'com')
        metric_label = 'center of mass';
    end

    xlabel('Lag time (seconds)', 'FontSize', 13);
    ylabel(sprintf('Neural ROI (sorted by %s)', metric_label), 'FontSize', 13);
    title(fig_title, 'FontSize', 16, 'Interpreter', 'none');

    % Y-axis labels
    yticks(1:length(roi_names));
    yticklabels(roi_names);

    % Add vertical line at zero lag only (no category separation)
    hold on;
    plot([0, 0], ylim, 'k--', 'LineWidth', 1.5);
    hold off;

    % Add colorbar label
    cb = colorbar;
    cb.Label.String = 'Beta coefficient (z-scored)';
    cb.Label.FontSize = 12;
end

function plot_multi_roi_predictions_poster(results, target_roi_names)
    % Plot predictions for 2 user-specified ROIs + behavior trace
    % Optimized for poster presentation

    % Validate required fields
    if ~isfield(results.predictions, 'Y_pred') || isempty(results.predictions.Y_pred)
        error('PlotPosterNoCategoryTemporalModelFull:MissingYPred', ...
            'Full-data predictions (Y_pred) missing from results. Ensure TemporalModelFull ran successfully.');
    end
    if ~isfield(results.predictions, 'Y_actual') || isempty(results.predictions.Y_actual)
        error('PlotPosterNoCategoryTemporalModelFull:MissingYActual', ...
            'Actual neural data (Y_actual) missing from results. Ensure TemporalModelFull ran successfully.');
    end
    if ~isfield(results.metadata, 'fold_boundaries_sec')
        error('PlotPosterNoCategoryTemporalModelFull:MissingFoldBoundaries', ...
            'Fold boundaries metadata missing. Ensure TemporalModelFull was updated and re-run.');
    end

    % Validate and set defaults
    if nargin < 2 || isempty(target_roi_names)
        target_roi_names = {'M1_L', 'V1_L'};
        fprintf('  Using default ROIs for predictions: M1_L, V1_L\n');
    end

    if numel(target_roi_names) ~= 2
        error('PlotPosterNoCategoryTemporalModelFull:InvalidROICount', ...
            'Exactly 2 ROI names required for poster predictions (got %d)', ...
            numel(target_roi_names));
    end

    % Find ROI indices
    roi_names_all = results.comparison.roi_names;
    plot_indices = zeros(2, 1);

    for i = 1:2
        idx = find(strcmpi(roi_names_all, target_roi_names{i}), 1);
        if isempty(idx)
            error('PlotPosterNoCategoryTemporalModelFull:ROINotFound', ...
                'ROI "%s" not found in results.\nAvailable ROIs: %s', ...
                target_roi_names{i}, strjoin(roi_names_all, ', '));
        end
        plot_indices(i) = idx;
    end

    meta = results.metadata;
    pred = results.predictions;
    if isfield(pred, 'Y_actual_cv') && ~isempty(pred.Y_actual_cv)
        Y_actual_cv_plot = pred.Y_actual_cv;
    else
        Y_actual_cv_plot = pred.Y_actual;
    end

    % Time vector for truncated data
    n_valid = meta.n_timepoints_used;
    n_lost_start = meta.n_timepoints_lost_start;
    t_truncated = (n_lost_start:(n_lost_start + n_valid - 1)) / meta.sampling_rate;

    fig_title = sprintf('Model Predictions: %s & %s vs %s', ...
        target_roi_names{1}, target_roi_names{2}, meta.behavior_predictor);

    fig = figure('Name', fig_title, 'Position', [200 400 1000 700]);
    use_tiled = exist('tiledlayout', 'file') == 2;
    layout = [];
    if use_tiled
        layout = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    end
    add_super_title(fig, layout, fig_title);
    ax_handles = gobjects(3, 1);

    % Plot each ROI panel
    for i = 1:2
        roi_idx = plot_indices(i);
        roi_name = target_roi_names{i};

        if use_tiled
            ax = nexttile(layout, i);
        else
            ax = subplot(3, 1, i);
        end
        ax_handles(i) = ax;

        % Actual vs predicted
        plot(ax, t_truncated, Y_actual_cv_plot(:, roi_idx), 'Color', [21/255, 101/255, 192/255], ...
            'LineWidth', 2.0, 'DisplayName', sprintf('%s (actual)', roi_name));
        hold(ax, 'on');
        plot(ax, t_truncated, pred.Y_pred(:, roi_idx), 'Color', [216/255, 27/255, 96/255], ...
            'LineWidth', 2.0, 'DisplayName', 'Prediction');
        hold(ax, 'off');

        title(ax, roi_name, 'Interpreter', 'none', 'FontSize', 16, 'FontWeight', 'bold');
        ylabel(ax, sprintf('%s (z)', roi_name), 'Interpreter', 'none', 'FontSize', 13);
        legend(ax, 'Location', 'best', 'FontSize', 11);
        grid(ax, 'off');
        set(ax, 'Box', 'off');

        % Add R^2 annotation with enhanced visibility (CV fold range)
        R2_folds = results.performance(roi_idx).R2_cv_folds;
        R2_min = min(R2_folds) * 100;
        R2_max = max(R2_folds) * 100;
        text(ax, 0.02, 0.98, ...
            sprintf('R^2 (CV): %.2f%% [range: %.2f-%.2f%%]', ...
                results.performance(roi_idx).R2_cv_mean*100, R2_min, R2_max), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'FontSize', 11, 'FontWeight', 'bold', ...
            'BackgroundColor', [1 1 1 0.85], 'EdgeColor', 'k', 'LineWidth', 1.5);
    end

    % Bottom panel: Behavior trace
    if use_tiled
        ax_behav = nexttile(layout, 3);
    else
        ax_behav = subplot(3, 1, 3);
    end
    ax_handles(3) = ax_behav;

    t_behavior = (n_lost_start:(n_lost_start + n_valid - 1)) / meta.sampling_rate;
    plot(ax_behav, t_behavior, pred.behavior_trace_z, 'Color', [56/255 142/255 60/255], ...
        'LineWidth', 2.0);
    ylabel(ax_behav, sprintf('%s (z)', meta.behavior_predictor), ...
        'Interpreter', 'none', 'FontSize', 13);
    xlabel(ax_behav, 'Time (s)', 'FontSize', 13);
    title(ax_behav, 'Behavioral Predictor', 'FontSize', 16, 'FontWeight', 'bold');
    grid(ax_behav, 'off');
    set(ax_behav, 'Box', 'off');

    linkaxes(ax_handles, 'x');

    % Add fold boundary markers
    if isfield(meta, 'fold_boundaries_sec') && ~isempty(meta.fold_boundaries_sec)
        for fold_boundary = meta.fold_boundaries_sec(:)'
            for ax = ax_handles(:)'
                xline(ax, fold_boundary, 'Color', [0.7 0.7 0.7], 'LineStyle', '--', ...
                    'LineWidth', 1.5, 'HandleVisibility', 'off', 'Alpha', 0.6);
            end
        end
    end
end

function plot_multi_roi_cv_predictions_poster(results, target_roi_names)
    % Plot CV fold predictions for 2 user-specified ROIs + behavior trace
    % Shows stitched predictions from CV folds with fold boundaries marked
    % Missing regions (train indices) appear as gaps

    % Validate required fields
    if ~isfield(results.predictions, 'Y_pred_cv') || isempty(results.predictions.Y_pred_cv)
        error('PlotPosterNoCategoryTemporalModelFull:MissingYPredCV', ...
            'CV predictions (Y_pred_cv) missing from results. Ensure TemporalModelFull was updated and re-run.');
    end
    if ~isfield(results.predictions, 'Y_actual') || isempty(results.predictions.Y_actual)
        error('PlotPosterNoCategoryTemporalModelFull:MissingYActual', ...
            'Actual neural data (Y_actual) missing from results. Ensure TemporalModelFull ran successfully.');
    end
    if ~isfield(results.metadata, 'fold_boundaries_sec') || isempty(results.metadata.fold_boundaries_sec)
        error('PlotPosterNoCategoryTemporalModelFull:MissingFoldBoundaries', ...
            'Fold boundaries metadata missing. Ensure TemporalModelFull was updated and re-run.');
    end

    % Validate and set defaults
    if nargin < 2 || isempty(target_roi_names)
        target_roi_names = {'M1_L', 'V1_L'};
        fprintf('  Using default ROIs for CV predictions: M1_L, V1_L\n');
    end

    if numel(target_roi_names) ~= 2
        error('PlotPosterNoCategoryTemporalModelFull:InvalidROICount', ...
            'Exactly 2 ROI names required for poster predictions (got %d)', ...
            numel(target_roi_names));
    end

    % Find ROI indices
    roi_names_all = results.comparison.roi_names;
    plot_indices = zeros(2, 1);

    for i = 1:2
        idx = find(strcmpi(roi_names_all, target_roi_names{i}), 1);
        if isempty(idx)
            error('PlotPosterNoCategoryTemporalModelFull:ROINotFound', ...
                'ROI "%s" not found in results.\nAvailable ROIs: %s', ...
                target_roi_names{i}, strjoin(roi_names_all, ', '));
        end
        plot_indices(i) = idx;
    end

    meta = results.metadata;
    pred = results.predictions;

    % Time vector for truncated data
    n_valid = meta.n_timepoints_used;
    n_lost_start = meta.n_timepoints_lost_start;
    t_truncated = (n_lost_start:(n_lost_start + n_valid - 1)) / meta.sampling_rate;

    fig_title = sprintf('CV Predictions: %s & %s vs %s', ...
        target_roi_names{1}, target_roi_names{2}, meta.behavior_predictor);

    fig = figure('Name', fig_title, 'Position', [200 400 1000 700]);
    use_tiled = exist('tiledlayout', 'file') == 2;
    layout = [];
    if use_tiled
        layout = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    end
    add_super_title(fig, layout, fig_title);
    ax_handles = gobjects(3, 1);

    % Plot each ROI panel
    for i = 1:2
        roi_idx = plot_indices(i);
        roi_name = target_roi_names{i};

        if use_tiled
            ax = nexttile(layout, i);
        else
            ax = subplot(3, 1, i);
        end
        ax_handles(i) = ax;

        % Actual vs CV predicted
        plot(ax, t_truncated, pred.Y_actual(:, roi_idx), 'Color', [21/255, 101/255, 192/255], ...
            'LineWidth', 2.0, 'DisplayName', sprintf('%s (actual)', roi_name));
        hold(ax, 'on');
        plot(ax, t_truncated, pred.Y_pred_cv(:, roi_idx), 'Color', [216/255, 27/255, 96/255], ...
            'LineWidth', 2.0, 'DisplayName', 'CV Prediction');
        hold(ax, 'off');

        title(ax, roi_name, 'Interpreter', 'none', 'FontSize', 16, 'FontWeight', 'bold');
        ylabel(ax, sprintf('%s (z)', roi_name), 'Interpreter', 'none', 'FontSize', 13);
        legend(ax, 'Location', 'best', 'FontSize', 11);
        grid(ax, 'off');
        set(ax, 'Box', 'off');

        % Add R^2 annotation (CV mean, which matches these predictions)
        R2_mean = results.performance(roi_idx).R2_cv_mean;
        R2_folds = results.performance(roi_idx).R2_cv_folds;
        R2_min = min(R2_folds) * 100;
        R2_max = max(R2_folds) * 100;
        text(ax, 0.02, 0.98, ...
            sprintf('R^2 (CV): %.2f%% [range: %.2f-%.2f%%]', R2_mean*100, R2_min, R2_max), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'FontSize', 11, 'FontWeight', 'bold', ...
            'BackgroundColor', [1 1 1 0.85], 'EdgeColor', 'k', 'LineWidth', 1.5);
    end

    % Bottom panel: Behavior trace
    if use_tiled
        ax_behav = nexttile(layout, 3);
    else
        ax_behav = subplot(3, 1, 3);
    end
    ax_handles(3) = ax_behav;

    t_behavior = (n_lost_start:(n_lost_start + n_valid - 1)) / meta.sampling_rate;
    plot(ax_behav, t_behavior, pred.behavior_trace_z, 'Color', [56/255 142/255 60/255], ...
        'LineWidth', 2.0);
    ylabel(ax_behav, sprintf('%s (z)', meta.behavior_predictor), ...
        'Interpreter', 'none', 'FontSize', 13);
    xlabel(ax_behav, 'Time (s)', 'FontSize', 13);
    title(ax_behav, 'Behavioral Predictor', 'FontSize', 16, 'FontWeight', 'bold');
    grid(ax_behav, 'off');
    set(ax_behav, 'Box', 'off');

    linkaxes(ax_handles, 'x');

    % Add fold boundary markers
    if isfield(meta, 'fold_boundaries_sec') && ~isempty(meta.fold_boundaries_sec)
        for fold_boundary = meta.fold_boundaries_sec(:)'
            for ax = ax_handles(:)'
                xline(ax, fold_boundary, 'Color', [0.7 0.7 0.7], 'LineStyle', '--', ...
                    'LineWidth', 1.5, 'HandleVisibility', 'off', 'Alpha', 0.6);
            end
        end
    end
end

function plot_peak_beta_brainmaps_poster(results)
    % Plot peak lag/beta/R² summaries on cortex map (1x3 layout, no category labels)

    if ~isfield(results, 'comparison') || ~isfield(results.comparison, 'roi_names')
        warning('PlotPosterNoCategoryTemporalModelFull:NoComparison', ...
            'Comparison summaries missing; skipping brain maps.');
        return;
    end

    if ~isfield(results, 'metadata') || ...
            ~isfield(results.metadata, 'source_roi_file') || ...
            ~isstruct(results.metadata.source_roi_file)
        warning('PlotPosterNoCategoryTemporalModelFull:NoSpatialSource', ...
            'No ROI source metadata available; skipping brain maps.');
        return;
    end

    source = results.metadata.source_roi_file;
    neural_path = '';
    if isfield(source, 'neural_roi_file')
        neural_path = source.neural_roi_file;
    elseif isfield(source, 'neural_rois')
        neural_path = source.neural_rois;
    end

    if isempty(neural_path) || exist(neural_path, 'file') ~= 2
        warning('PlotPosterNoCategoryTemporalModelFull:MissingROIFile', ...
            'Neural ROI file not found (%s); skipping brain maps.', neural_path);
        return;
    end

    spatial = load(neural_path, '-mat');
    if ~isfield(spatial, 'ROI_info') || ~isfield(spatial, 'img_info')
        warning('PlotPosterNoCategoryTemporalModelFull:InvalidROIFile', ...
            'ROI file %s missing ROI_info/img_info; skipping brain maps.', neural_path);
        return;
    end

    img_info = spatial.img_info;
    roi_info = spatial.ROI_info;
    if ~isfield(img_info, 'imageData')
        warning('PlotPosterNoCategoryTemporalModelFull:NoImageData', ...
            'img_info.imageData missing in %s; skipping brain maps.', neural_path);
        return;
    end

    if isfield(roi_info(1), 'Stats') && isfield(roi_info(1).Stats, 'ROI_binary_mask')
        dims = size(roi_info(1).Stats.ROI_binary_mask);
    else
        dims = size(img_info.imageData);
    end
    roi_names_source = arrayfun(@(r)char(r.Name), roi_info, 'UniformOutput', false);

    target_names = results.comparison.roi_names;
    peak_lags = results.comparison.peak_lags_all_sec;

    % Use actual peak beta for brain map (not CoM beta if CoM metric enabled)
    if isfield(results.comparison, 'peak_method_betas_all')
        peak_betas = results.comparison.peak_method_betas_all;
    else
        peak_betas = results.comparison.peak_betas_all;  % Fallback for old results
    end

    cv_r2_all = [results.performance.R2_cv_mean];
    if numel(target_names) ~= numel(peak_lags) || numel(peak_lags) ~= numel(peak_betas)
        warning('PlotPosterNoCategoryTemporalModelFull:MismatchLength', ...
            'Mismatch in comparison arrays; skipping brain maps.');
        return;
    end

    lag_map = nan(dims);
    beta_map = nan(dims);
    r2_map = nan(dims);
    n_assigned = 0;
    n_skipped = 0;

    for i = 1:numel(target_names)
        roi_name = target_names{i};
        match_idx = find(strcmpi(roi_names_source, roi_name), 1);
        if isempty(match_idx)
            error('PlotPosterNoCategoryTemporalModelFull:ROIMaskMissing', ...
                'ROI "%s" not found in %s', roi_name, neural_path);
        end
        roi_struct = roi_info(match_idx);
        if ~isfield(roi_struct, 'Stats') || ...
                ~isfield(roi_struct.Stats, 'ROI_binary_mask')
            error('PlotPosterNoCategoryTemporalModelFull:MaskMissing', ...
                'ROI "%s" missing Stats.ROI_binary_mask in %s', roi_name, neural_path);
        end
        mask = roi_struct.Stats.ROI_binary_mask;
        if ~isequal(size(mask), dims)
            error('PlotPosterNoCategoryTemporalModelFull:MaskSizeMismatch', ...
                'ROI "%s" mask size mismatch (expected %dx%d).', roi_name, dims(1), dims(2));
        end

        lag_val = peak_lags(i);
        beta_val = peak_betas(i);
        r2_val = cv_r2_all(i) * 100;  % express as %
        if isnan(lag_val) || isnan(beta_val) || isnan(r2_val)
            n_skipped = n_skipped + 1;
            continue;
        end

        overlap = ~isnan(lag_map) & mask;
        if any(overlap(:))
            error('PlotPosterNoCategoryTemporalModelFull:OverlappingMasks', ...
                'ROI "%s" overlaps with another ROI in %s', roi_name, neural_path);
        end

        lag_map(mask) = lag_val;
        beta_map(mask) = beta_val;
        r2_map(mask) = r2_val;
        n_assigned = n_assigned + 1;
    end

    brain_mask = load_optional_mask(source, 'brain_mask_file', dims);
    if isempty(brain_mask) && isfield(img_info, 'logical_mask')
        brain_mask = logical(img_info.logical_mask);
    end
    if isempty(brain_mask)
        brain_mask = true(dims);
    end

    vascular_mask = load_optional_mask(source, 'vascular_mask_file', dims);
    if isempty(vascular_mask)
        vascular_mask = false(dims);
    end

    mask_shape = brain_mask & ~vascular_mask;
    lag_map(~mask_shape) = nan;
    beta_map(~mask_shape) = nan;
    r2_map(~mask_shape) = nan;

    base_rgb = build_mask_background(mask_shape);

    lag_span = max(abs(peak_lags(~isnan(peak_lags))));
    if isempty(lag_span) || lag_span == 0
        lag_span = max(abs([results.metadata.min_lag_seconds, ...
            results.metadata.max_lag_seconds]));
    end
    if isempty(lag_span) || lag_span == 0
        lag_span = 1;
    end
    lag_limits = [-lag_span, lag_span];

    beta_vals = beta_map(~isnan(beta_map));
    beta_max = max(beta_vals);
    if isempty(beta_vals) || beta_max == 0
        beta_max = 1;
    end
    beta_limits = [0, beta_max];
    r2_valid = r2_map(~isnan(r2_map));
    if isempty(r2_valid)
        r2_limits = [0 1];
    else
        r2_limits = [0 max(r2_valid)];
    end

    % Create 1x3 horizontal layout for poster
    fig = figure('Name', 'Temporal Peak Metrics Brain Maps (Poster, No Category)', ...
        'Position', [100 200 1800 500]);
    use_tiled = exist('tiledlayout', 'file') == 2;
    layout = [];
    if use_tiled
        layout = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    end

    % Determine metric for title
    metric_str = 'Peak';
    if isfield(results.metadata, 'peak_metric') && strcmp(results.metadata.peak_metric, 'com')
        metric_str = 'CoM';
    end

    title_str = sprintf('Temporal Kernel Spatial Summary (n=%d ROIs, %s)', n_assigned, metric_str);
    add_super_title(fig, layout, title_str);

    % Panel 1: Lag map (diverging, no category text)
    cmap_lag = redbluecmap(256);
    if use_tiled
        ax1 = nexttile(layout, 1);
    else
        ax1 = subplot(1, 3, 1);
    end
    plot_metric_map(ax1, base_rgb, lag_map, cmap_lag, lag_limits, ...
        sprintf('%s Lag (s)', metric_str), mask_shape);

    % Panel 2: Beta magnitude map (z-scored peak/CoM value)
    cmap_beta = parula(256);
    if use_tiled
        ax2 = nexttile(layout, 2);
    else
        ax2 = subplot(1, 3, 2);
    end
    plot_metric_map(ax2, base_rgb, beta_map, cmap_beta, beta_limits, ...
        'Peak Beta (z-scored)', mask_shape);

    % Panel 3: CV R^2 map
    if use_tiled
        ax3 = nexttile(layout, 3);
    else
        ax3 = subplot(1, 3, 3);
    end
    plot_metric_map(ax3, base_rgb, r2_map, parula(256), r2_limits, ...
        'CV R^2 (%)', mask_shape);
end

function plot_cv_fold_progression(results)
    % Plot all 5 CV fold R² values for each ROI with color indicating fold order
    % Y-axis: R² (%), X-axis: ROI names, Color: Fold sequence (1→5)

    roi_names = results.comparison.roi_names;
    n_rois = numel(roi_names);
    cv_folds = results.metadata.cv_folds;

    fig_title = 'CV Fold Progression';
    figure('Name', fig_title, 'Position', [300 300 1000 500]);

    hold on;

    % Sequential colormap for folds (fold 1 = light, fold 5 = dark)
    fold_colors = parula(cv_folds);

    for roi = 1:n_rois
        R2_folds = results.performance(roi).R2_cv_folds * 100;
        x_pos = repmat(roi, cv_folds, 1);

        % Plot points colored by fold order
        scatter(x_pos, R2_folds, 100, fold_colors, 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

        % Connect with light line to show progression through folds
        plot(x_pos, R2_folds, 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
    end

    hold off;

    xlabel('Neural ROI', 'FontSize', 13);
    ylabel('R² (%)', 'FontSize', 13);
    title(fig_title, 'FontSize', 16, 'FontWeight', 'bold');
    xticks(1:n_rois);
    xticklabels(roi_names);
    xtickangle(45);
    grid off;
    set(gca, 'Box', 'off');

    % Add colorbar legend for fold order
    cb = colorbar;
    cb.Label.String = 'Fold Number';
    cb.Label.FontSize = 12;
    caxis([1, cv_folds]);
end


function plot_cv_fold_progression_corr(results)
    % Additional figure: CV fold Pearson correlation values for each ROI

    roi_names = results.comparison.roi_names;
    n_rois = numel(roi_names);
    cv_folds = results.metadata.cv_folds;

    fig_title = 'CV Fold Progression (Correlation)';
    figure('Name', fig_title, 'Position', [300 250 1000 500]);

    hold on;
    fold_colors = parula(cv_folds);

    for roi = 1:n_rois
        corr_folds = results.performance(roi).Corr_cv_folds;
        x_pos = repmat(roi, cv_folds, 1);

        scatter(x_pos, corr_folds, 100, fold_colors, 'filled', ...
            'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        plot(x_pos, corr_folds, 'Color', [0.7 0.7 0.7], 'LineWidth', 1);
    end

    hold off;

    xlabel('Neural ROI', 'FontSize', 13);
    ylabel('Pearson r (CV)', 'FontSize', 13);
    title(fig_title, 'FontSize', 16, 'FontWeight', 'bold');
    xticks(1:n_rois);
    xticklabels(roi_names);
    xtickangle(45);
    ylim([0 1]);
    grid off;
    set(gca, 'Box', 'off');

    cb = colorbar;
    cb.Label.String = 'Fold Number';
    cb.Label.FontSize = 12;
    caxis([1, cv_folds]);
end

function plot_peak_beta_corr_brainmaps_poster(results)
    % Additional figure: Peak lag/beta with correlation map (poster layout, no categories)

    if ~isfield(results, 'comparison') || ~isfield(results.comparison, 'roi_names')
        warning('PlotPosterNoCategoryTemporalModelFull:NoComparison', ...
            'Comparison summaries missing; skipping correlation brain maps.');
        return;
    end

    if ~isfield(results.comparison, 'Corr_all_rois')
        warning('PlotPosterNoCategoryTemporalModelFull:NoCorrelation', ...
            'Correlation summaries missing; skipping correlation brain maps.');
        return;
    end

    if ~isfield(results, 'metadata') || ...
            ~isfield(results.metadata, 'source_roi_file') || ...
            ~isstruct(results.metadata.source_roi_file)
        warning('PlotPosterNoCategoryTemporalModelFull:NoSpatialSource', ...
            'No ROI source metadata available; skipping correlation brain maps.');
        return;
    end

    source = results.metadata.source_roi_file;
    neural_path = '';
    if isfield(source, 'neural_roi_file')
        neural_path = source.neural_roi_file;
    elseif isfield(source, 'neural_rois')
        neural_path = source.neural_rois;
    end

    if isempty(neural_path) || exist(neural_path, 'file') ~= 2
        warning('PlotPosterNoCategoryTemporalModelFull:MissingROIFile', ...
            'Neural ROI file not found (%s); skipping correlation brain maps.', neural_path);
        return;
    end

    spatial = load(neural_path, '-mat');
    if ~isfield(spatial, 'ROI_info') || ~isfield(spatial, 'img_info')
        warning('PlotPosterNoCategoryTemporalModelFull:InvalidROIFile', ...
            'ROI file %s missing ROI_info/img_info; skipping correlation brain maps.', neural_path);
        return;
    end

    img_info = spatial.img_info;
    roi_info = spatial.ROI_info;
    if ~isfield(img_info, 'imageData')
        warning('PlotPosterNoCategoryTemporalModelFull:NoImageData', ...
            'img_info.imageData missing in %s; skipping correlation brain maps.', neural_path);
        return;
    end

    if isfield(roi_info(1), 'Stats') && isfield(roi_info(1).Stats, 'ROI_binary_mask')
        dims = size(roi_info(1).Stats.ROI_binary_mask);
    else
        dims = size(img_info.imageData);
    end
    roi_names_source = arrayfun(@(r)char(r.Name), roi_info, 'UniformOutput', false);

    target_names = results.comparison.roi_names;
    peak_lags = results.comparison.peak_lags_all_sec;

    % Match beta values to the non-correlation map (always use actual peak betas when available)
    if isfield(results.comparison, 'peak_method_betas_all')
        peak_betas = results.comparison.peak_method_betas_all;
    else
        peak_betas = results.comparison.peak_betas_all;
    end

    corr_all = results.comparison.Corr_all_rois;
    if numel(target_names) ~= numel(peak_lags) || numel(peak_lags) ~= numel(peak_betas)
        warning('PlotPosterNoCategoryTemporalModelFull:MismatchLength', ...
            'Mismatch in comparison arrays; skipping correlation brain maps.');
        return;
    end

    lag_map = nan(dims);
    beta_map = nan(dims);
    corr_map = nan(dims);
    n_assigned = 0;
    n_skipped = 0;

    for i = 1:numel(target_names)
        roi_name = target_names{i};
        match_idx = find(strcmpi(roi_names_source, roi_name), 1);
        if isempty(match_idx)
            error('PlotPosterNoCategoryTemporalModelFull:ROIMaskMissing', ...
                'ROI "%s" not found in %s', roi_name, neural_path);
        end
        roi_struct = roi_info(match_idx);
        if ~isfield(roi_struct, 'Stats') || ...
                ~isfield(roi_struct.Stats, 'ROI_binary_mask')
            error('PlotPosterNoCategoryTemporalModelFull:MaskMissing', ...
                'ROI "%s" missing Stats.ROI_binary_mask in %s', roi_name, neural_path);
        end
        mask = roi_struct.Stats.ROI_binary_mask;
        if ~isequal(size(mask), dims)
            error('PlotPosterNoCategoryTemporalModelFull:MaskSizeMismatch', ...
                'ROI "%s" mask size mismatch (expected %dx%d).', roi_name, dims(1), dims(2));
        end

        lag_val = peak_lags(i);
        beta_val = peak_betas(i);
        corr_val = corr_all(i);
        if isnan(lag_val) || isnan(beta_val) || isnan(corr_val)
            n_skipped = n_skipped + 1;
            continue;
        end

        overlap = ~isnan(lag_map) & mask;
        if any(overlap(:))
            error('PlotPosterNoCategoryTemporalModelFull:OverlappingMasks', ...
                'ROI "%s" overlaps with another ROI in %s', roi_name, neural_path);
        end

        lag_map(mask) = lag_val;
        beta_map(mask) = beta_val;
        corr_map(mask) = corr_val;
        n_assigned = n_assigned + 1;
    end

    brain_mask = load_optional_mask(source, 'brain_mask_file', dims);
    if isempty(brain_mask) && isfield(img_info, 'logical_mask')
        brain_mask = logical(img_info.logical_mask);
    end
    if isempty(brain_mask)
        brain_mask = true(dims);
    end

    vascular_mask = load_optional_mask(source, 'vascular_mask_file', dims);
    if isempty(vascular_mask)
        vascular_mask = false(dims);
    end

    mask_shape = brain_mask & ~vascular_mask;
    lag_map(~mask_shape) = nan;
    beta_map(~mask_shape) = nan;
    corr_map(~mask_shape) = nan;

    base_rgb = build_mask_background(mask_shape);

    lag_span = max(abs(peak_lags(~isnan(peak_lags))));
    if isempty(lag_span) || lag_span == 0
        lag_span = max(abs([results.metadata.min_lag_seconds, ...
            results.metadata.max_lag_seconds]));
    end
    if isempty(lag_span) || lag_span == 0
        lag_span = 1;
    end
    lag_limits = [-lag_span, lag_span];

    beta_vals = beta_map(~isnan(beta_map));
    beta_max = max(beta_vals);
    if isempty(beta_vals) || beta_max == 0
        beta_max = 1;
    end
    beta_limits = [0, beta_max];
    corr_valid = corr_map(~isnan(corr_map));
    if isempty(corr_valid)
        corr_limits = [0 1];
    else
        corr_limits = [0 max(corr_valid)];
    end

    fig = figure('Name', 'Temporal Peak Metrics Brain Maps (Correlation, Poster)', ...
        'Position', [120 220 1800 500]);
    use_tiled = exist('tiledlayout', 'file') == 2;
    layout = [];
    if use_tiled
        layout = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    end

    metric_str = 'Peak';
    if isfield(results.metadata, 'peak_metric') && strcmp(results.metadata.peak_metric, 'com')
        metric_str = 'CoM';
    end

    title_str = sprintf('Temporal Kernel Spatial Summary (Correlation, n=%d, %s)', n_assigned, metric_str);
    add_super_title(fig, layout, title_str);

    % Lag map
    cmap_lag = redbluecmap(256);
    if use_tiled
        ax1 = nexttile(layout, 1);
    else
        ax1 = subplot(1, 3, 1);
    end
    plot_metric_map(ax1, base_rgb, lag_map, cmap_lag, lag_limits, ...
        sprintf('%s Lag (s)', metric_str), mask_shape);

    % Beta map
    cmap_beta = parula(256);
    if use_tiled
        ax2 = nexttile(layout, 2);
    else
        ax2 = subplot(1, 3, 2);
    end
    plot_metric_map(ax2, base_rgb, beta_map, cmap_beta, beta_limits, ...
        'Peak Beta (z-scored)', mask_shape);

    % Correlation map
    if use_tiled
        ax3 = nexttile(layout, 3);
    else
        ax3 = subplot(1, 3, 3);
    end
    plot_metric_map(ax3, base_rgb, corr_map, parula(256), corr_limits, ...
        'CV Pearson r', mask_shape);
end
%% ================= Helper Functions =================

function [mean_curve, sem_curve] = compute_group_curve(beta_matrix, mask)
    if nargin < 2 || isempty(mask) || all(~mask)
        mean_curve = [];
        sem_curve = [];
        return;
    end
    group_data = beta_matrix(:, mask);
    mean_curve = mean(group_data, 2, 'omitnan');
    n = sum(mask);
    if n > 1
        sem_curve = std(group_data, 0, 2, 'omitnan') ./ sqrt(n);
    else
        sem_curve = zeros(size(mean_curve));
    end
end

function plot_group_curve(lag_times, mean_curve, sem_curve, color_val, label_str)
    if isempty(mean_curve)
        return;
    end
    upper = mean_curve + sem_curve;
    lower = mean_curve - sem_curve;
    fill([lag_times; flipud(lag_times)], [upper; flipud(lower)], color_val, ...
        'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    plot(lag_times, mean_curve, 'Color', color_val, 'LineWidth', 3, ...
        'DisplayName', label_str);
end

function mask = load_optional_mask(source, field_name, dims)
    mask = [];
    if ~isfield(source, field_name)
        return;
    end
    path_str = source.(field_name);
    if isempty(path_str) || exist(path_str, 'file') ~= 2
        return;
    end
    data = load(path_str, '-mat');
    if isfield(data, 'ROI_info') && numel(data.ROI_info) >= 1 && ...
            isfield(data.ROI_info(1), 'Stats') && ...
            isfield(data.ROI_info(1).Stats, 'ROI_binary_mask')
        mask = logical(data.ROI_info(1).Stats.ROI_binary_mask);
    elseif isfield(data, 'mask')
        mask = logical(data.mask);
    else
        error('PlotPosterNoCategoryTemporalModelFull:InvalidMask', ...
            'Mask file %s missing ROI_info or mask variable.', path_str);
    end
    if ~isequal(size(mask), dims)
        error('PlotPosterNoCategoryTemporalModelFull:MaskDims', ...
            'Mask file %s size mismatch (expected %dx%d).', path_str, dims(1), dims(2));
    end
end

function plot_metric_map(ax, base_rgb, metric_map, cmap, clim, ttl, mask_shape, is_categorical, draw_mask_outline)
    if nargin < 8 || isempty(is_categorical)
        is_categorical = false;
    end
    if nargin < 9 || isempty(draw_mask_outline)
        draw_mask_outline = false;
    end
    axes(ax);
    cla(ax);
    image(ax, base_rgb);
    set(ax, 'YDir', 'reverse');
    axis(ax, 'image');
    axis(ax, 'off');
    hold(ax, 'on');
    alpha_mask = ~isnan(metric_map);
    if is_categorical
        overlay_data = double(metric_map);
        overlay_data(metric_map == 0) = nan;
        im = imagesc(ax, overlay_data, 'AlphaData', alpha_mask & metric_map ~= 0);
        set(im, 'CDataMapping', 'scaled');
        colormap(ax, cmap);
        caxis(ax, clim);
    else
        im = imagesc(ax, metric_map, 'AlphaData', alpha_mask);
        set(im, 'CDataMapping', 'scaled');
        colormap(ax, cmap);
        caxis(ax, clim);
        cb = colorbar(ax);
        cb.FontSize = 11;
    end
    if draw_mask_outline
        plot_mask_outline(ax, mask_shape);
    end
    hold(ax, 'off');
    title(ax, ttl, 'FontSize', 16, 'FontWeight', 'bold');
end

function base_rgb = build_mask_background(mask_shape)
    % Background stays black everywhere; mask outline supplies structure
    base_gray = zeros(size(mask_shape));
    base_rgb = repmat(base_gray, 1, 1, 3);
end

function plot_mask_outline(ax, mask_shape)
    contour(ax, mask_shape, [0.5 0.5], 'Color', [1 1 1], 'LineWidth', 1.2);
end

function add_super_title(fig_handle, layout_handle, title_str)
    if nargin >= 2 && ~isempty(layout_handle)
        title(layout_handle, title_str, 'FontSize', 16, 'Interpreter', 'none');
    elseif exist('sgtitle', 'file') == 2
        figure(fig_handle);
        sgtitle(title_str, 'Interpreter', 'none', 'FontSize', 16);
    else
        annotation(fig_handle, 'textbox', [0 0.95 1 0.04], ...
            'String', title_str, 'HorizontalAlignment', 'center', ...
            'EdgeColor', 'none', 'FontSize', 16, 'Interpreter', 'none');
    end
end

function cmap = redbluecmap(m)
    % Generate red-white-blue colormap centered on zero
    if nargin < 1
        m = 256;
    end

    % Red for negative, blue for positive, white at zero
    r = [ones(m/2, 1); linspace(1, 0, m/2)'];
    g = [linspace(0, 1, m/2)'; linspace(1, 0, m/2)'];
    b = [linspace(0, 1, m/2)'; ones(m/2, 1)];

    cmap = [r, g, b];
end

function clr = apply_saturation(color_vec, saturation)
    saturation = clamp_unit_value(saturation);
    base_color = reshape(color_vec, 1, 3);
    clr = saturation .* base_color + (1 - saturation) .* ones(1, 3);
    clr = min(max(clr, 0), 1);
end

function val = clamp_unit_value(val)
    if ~isnumeric(val) || ~isscalar(val) || ~isfinite(val)
        error('PlotPosterNoCategoryTemporalModelFull:InvalidSaturation', ...
            'Saturation values must be finite scalars in [0, 1].');
    end
    val = max(0, min(1, val));
end
