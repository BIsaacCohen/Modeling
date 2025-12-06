function PlotTemporalModelFull(results)
% PlotTemporalModelFull Generate diagnostic plots for TemporalModelFull results.
%
%   PlotTemporalModelFull(results)
%       results - Output struct returned by TemporalModelFull containing
%                 temporal kernels, performance metrics, predictions, and metadata.
%
%   This function replicates the plotting behavior that previously lived inside
%   TemporalModelFull.m so the modeling code stays focused on the regression
%   workflow while plot generation is isolated here.

if nargin < 1 || isempty(results)
    error('PlotTemporalModelFull:MissingResults', ...
        'Results struct from TemporalModelFull is required.');
end

fprintf('\nGenerating plots...\n');
plot_all_temporal_kernels(results);
plot_temporal_kernel_heatmap(results);
plot_performance_comparison(results);
plot_multi_roi_predictions(results);
plot_peak_beta_brainmaps(results);
fprintf('Plots generated.\n');

end

%% ================= Plotting Functions =================

function plot_all_temporal_kernels(results)
    % Plot predictive vs reactive group mean kernels with SEM shading

    tk = results.temporal_kernels;
    n_rois = numel(tk);
    lag_times = tk(1).lag_times_sec;
    n_lags = numel(lag_times);

    beta_matrix = zeros(n_lags, n_rois);
    for roi = 1:n_rois
        beta_matrix(:, roi) = tk(roi).beta_cv_mean;
    end

    peak_lags = results.comparison.peak_lags_all_sec(:);
    predictive_mask = peak_lags < 0;
    reactive_mask = peak_lags >= 0;

    [pred_mean, pred_sem] = compute_group_curve(beta_matrix, predictive_mask);
    [reac_mean, reac_sem] = compute_group_curve(beta_matrix, reactive_mask);

    % Determine metric for title
    metric_str = 'Peak';
    if isfield(results.metadata, 'peak_metric') && strcmp(results.metadata.peak_metric, 'com')
        metric_str = 'CoM';
    end

    fig_title = sprintf('Temporal Kernels (Group Means): Predictive vs Reactive (%s, %s)', ...
        results.metadata.behavior_predictor, metric_str);

    figure('Name', fig_title, 'Position', [100 100 900 600]);
    hold on;

    colors = struct('predictive', [0.2 0.45 0.9], 'reactive', [0.9 0.45 0.2]);

    plot_group_curve(lag_times, pred_mean, pred_sem, colors.predictive, ...
        sprintf('Predictive (n=%d)', sum(predictive_mask)));
    plot_group_curve(lag_times, reac_mean, reac_sem, colors.reactive, ...
        sprintf('Reactive (n=%d)', sum(reactive_mask)));

    plot(xlim, [0 0], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
    yl = ylim;
    plot([0 0], yl, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

    hold off;

    xlabel('Lag time (seconds)', 'FontSize', 13);
    ylabel('Beta coefficient (z-scored)', 'FontSize', 13);
    title(fig_title, 'FontSize', 15, 'Interpreter', 'none');
    legend('Location', 'best', 'FontSize', 10);
    grid on;
end

function plot_temporal_kernel_heatmap(results)
    % Heatmap of temporal kernels: ROIs (rows) × Lags (columns)

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

    figure('Name', fig_title, 'Position', [150 100 900 600]);

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

    xlabel('Lag time (seconds)', 'FontSize', 12);
    ylabel(sprintf('Neural ROI (sorted by %s)', metric_label), 'FontSize', 12);
    title(fig_title, 'FontSize', 14, 'Interpreter', 'none');

    % Y-axis labels
    yticks(1:length(roi_names));
    yticklabels(roi_names);

    % Add vertical line at zero lag and horizontal line at predictive/reactive boundary
    hold on;
    plot([0, 0], ylim, 'k--', 'LineWidth', 1.5);

    % Find boundary between predictive (negative peak lag) and reactive (positive peak lag)
    boundary_idx = find(sorted_peak_lags >= 0, 1);
    if ~isempty(boundary_idx) && boundary_idx > 1 && boundary_idx <= length(roi_names)
        plot(xlim, [boundary_idx - 0.5, boundary_idx - 0.5], 'k-', 'LineWidth', 2.5);

        % Add text labels for categories
        lag_range = max(lag_times) - min(lag_times);
        text_x = min(lag_times) + lag_range * 0.02;

        text(text_x, boundary_idx / 2, 'Predictive', ...
            'Color', [0.2 0.45 0.9], 'FontWeight', 'bold', 'FontSize', 11, ...
            'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'none');

        text(text_x, boundary_idx + (length(roi_names) - boundary_idx + 1) / 2, 'Reactive', ...
            'Color', [0.9 0.45 0.2], 'FontWeight', 'bold', 'FontSize', 11, ...
            'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'none');
    end
    hold off;

    % Add colorbar label
    cb = colorbar;
    cb.Label.String = 'Beta coefficient (z-scored)';
    cb.Label.FontSize = 11;
end

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
    plot(lag_times, mean_curve, 'Color', color_val, 'LineWidth', 2, ...
        'DisplayName', label_str);
end

function plot_performance_comparison(results)
    % Bar plot comparing R^2 across ROIs with error bars

    n_rois = results.metadata.n_rois;
    roi_names = results.comparison.roi_names;
    R2_cv = [results.performance.R2_cv_mean]';
    R2_ci95 = [results.performance.R2_cv_ci95]';
    R2_full = [results.performance.R2_full_data]';
    peak_lags = results.comparison.peak_lags_all_sec';

    fig_title = sprintf('Performance Comparison: %d ROIs vs %s', ...
        n_rois, results.metadata.behavior_predictor);

    fig = figure('Name', fig_title, 'Position', [200 100 1000 700]);
    use_tiled = exist('tiledlayout', 'file') == 2;
    layout = [];
    if use_tiled
        layout = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    end
    add_super_title(fig, layout, fig_title);

    % Top panel: R^2 comparison
    if use_tiled
        ax1 = nexttile(layout, 1);
    else
        ax1 = subplot(2, 1, 1);
    end
    x_pos = 1:n_rois;

    % Plot CV R^2 with 95% CI error bars
    bar(ax1, x_pos, R2_cv * 100, 'FaceColor', [0.3 0.5 0.8]);
    hold(ax1, 'on');
    errorbar(ax1, x_pos, R2_cv * 100, R2_ci95 * 100, 'k.', 'LineWidth', 1.5, ...
        'CapSize', 5);

    % Overlay full-data R^2 as dots
    plot(ax1, x_pos, R2_full * 100, 'ro', 'MarkerSize', 6, 'LineWidth', 1.5, ...
        'DisplayName', 'R^2 (full-data)');

    hold(ax1, 'off');

    ylabel(ax1, 'R^2 (%)', 'FontSize', 12);
    xticks(ax1, x_pos);
    xticklabels(ax1, roi_names);
    xtickangle(ax1, 45);
    legend(ax1, {'R^2 (CV mean, 95% CI)', 'R^2 (full-data)'}, 'Location', 'best');
    grid(ax1, 'on');
    title(ax1, 'Model Performance', 'FontSize', 12);

    % Bottom panel: Peak lag distribution
    if use_tiled
        ax2 = nexttile(layout, 2);
    else
        ax2 = subplot(2, 1, 2);
    end

    % Color bars by sign (predictive vs reactive)
    bar_colors = zeros(n_rois, 3);
    for i = 1:n_rois
        if peak_lags(i) < 0
            bar_colors(i, :) = [0.8 0.3 0.3];  % Red for predictive
        else
            bar_colors(i, :) = [0.3 0.7 0.3];  % Green for reactive
        end
    end

    b2 = bar(ax2, x_pos, peak_lags, 'FaceColor', 'flat');
    b2.CData = bar_colors;

    ylabel(ax2, 'Peak lag (seconds)', 'FontSize', 12);
    xlabel(ax2, 'Neural ROI', 'FontSize', 12);
    xticks(ax2, x_pos);
    xticklabels(ax2, roi_names);
    xtickangle(ax2, 45);

    % Add reference line at zero
    hold(ax2, 'on');
    plot(ax2, [0.5, n_rois+0.5], [0, 0], 'k--', 'LineWidth', 1.5);
    hold(ax2, 'off');

    grid(ax2, 'on');
    title(ax2, 'Peak Response Timing', 'FontSize', 12);

    % Add legend for colors
    legend(ax2, {'Predictive (leads)', 'Reactive (lags)'}, 'Location', 'best');
end

function plot_multi_roi_predictions(results)
    % Plot best vs worst ROI predictions (top 2 vs bottom 2 by R^2)

    meta = results.metadata;
    pred = results.predictions;

    R2_all = [results.performance.R2_cv_mean];
    [~, desc_idx] = sort(R2_all, 'descend');
    [~, asc_idx] = sort(R2_all, 'ascend');

    n_best = min(2, meta.n_rois);
    n_worst = min(2, max(meta.n_rois - n_best, 0));

    best_idx = desc_idx(1:n_best);
    worst_candidates = setdiff(asc_idx, best_idx, 'stable');
    worst_idx = worst_candidates(1:min(n_worst, numel(worst_candidates)));

    plot_indices = [best_idx(:); worst_idx(:)];
    group_flags = [repmat({'Best'}, numel(best_idx), 1); ...
                   repmat({'Worst'}, numel(worst_idx), 1)];
    n_plot = numel(plot_indices);
    if n_plot == 0
        warning('plot_multi_roi_predictions:NoROIs', ...
            'Unable to select ROIs for prediction plot.');
        return;
    end

    % Time vector for truncated data
    n_valid = meta.n_timepoints_used;
    n_lost_start = meta.n_timepoints_lost_start;
    t_truncated = (n_lost_start:(n_lost_start + n_valid - 1)) / meta.sampling_rate;

    fig_title = sprintf('Model Predictions: Best/ Worst ROIs vs %s', ...
        meta.behavior_predictor);

    fig = figure('Name', fig_title, 'Position', [250 50 1200 800]);
    use_tiled = exist('tiledlayout', 'file') == 2;
    layout = [];
    if use_tiled
        layout = tiledlayout(n_plot + 1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    end
    add_super_title(fig, layout, fig_title);
    ax_handles = gobjects(n_plot + 1, 1);

    % Plot each ROI
    for i = 1:n_plot
        roi_idx = plot_indices(i);
        roi_tag = group_flags{i};
        roi_name = meta.roi_names{roi_idx};

        if use_tiled
            ax = nexttile(layout, i);
        else
            ax = subplot(n_plot + 1, 1, i);
        end
        ax_handles(i) = ax;

        % Actual vs predicted
        plot(ax, t_truncated, pred.Y_actual(:, roi_idx), 'Color', [0.2 0.2 0.8], ...
            'DisplayName', sprintf('%s (actual)', roi_name));
        hold(ax, 'on');
        plot(ax, t_truncated, pred.Y_pred(:, roi_idx), 'Color', [0.85 0.33 0.1], ...
            'LineWidth', 1.25, 'DisplayName', 'Prediction');
        hold(ax, 'off');

        title(ax, sprintf('%s ROI: %s', roi_tag, roi_name), 'Interpreter', 'none', ...
            'FontSize', 11);
        ylabel(ax, sprintf('%s (z)', roi_name), 'Interpreter', 'none', 'FontSize', 10);
        legend(ax, 'Location', 'best', 'FontSize', 8);
        grid(ax, 'on');

        % Add R^2 annotation with 95% CI
        text(ax, 0.02, 0.98, ...
            sprintf('%s R^2 (CV): %.2f%% [95%% CI: %.2f-%.2f%%]', roi_tag, ...
                results.performance(roi_idx).R2_cv_mean*100, ...
                (results.performance(roi_idx).R2_cv_mean - results.performance(roi_idx).R2_cv_ci95)*100, ...
                (results.performance(roi_idx).R2_cv_mean + results.performance(roi_idx).R2_cv_ci95)*100), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'FontSize', 8, 'BackgroundColor', 'w', 'EdgeColor', 'k');
    end

    % Bottom panel: Behavior trace
    if use_tiled
        ax_behav = nexttile(layout, n_plot + 1);
    else
        ax_behav = subplot(n_plot + 1, 1, n_plot + 1);
    end
    ax_handles(end) = ax_behav;
    t_behavior = t_truncated;
    plot(ax_behav, t_behavior, pred.behavior_trace_z, 'Color', [0.13 0.55 0.13]);
    ylabel(ax_behav, sprintf('%s (z)', meta.behavior_predictor), 'Interpreter', 'none');
    xlabel(ax_behav, 'Time (s)');
    grid(ax_behav, 'on');

    linkaxes(ax_handles, 'x');
end

function plot_peak_beta_brainmaps(results)
    % Plot peak lag/beta summaries back onto the cortex map using ROI masks

    if ~isfield(results, 'comparison') || ~isfield(results.comparison, 'roi_names')
        warning('TemporalModelFull:NoComparison', ...
            'Comparison summaries missing; skipping brain maps.');
        return;
    end

    if ~isfield(results, 'metadata') || ...
            ~isfield(results.metadata, 'source_roi_file') || ...
            ~isstruct(results.metadata.source_roi_file)
        warning('TemporalModelFull:NoSpatialSource', ...
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
        warning('TemporalModelFull:MissingROIFile', ...
            'Neural ROI file not found (%s); skipping brain maps.', neural_path);
        return;
    end

    spatial = load(neural_path, '-mat');
    if ~isfield(spatial, 'ROI_info') || ~isfield(spatial, 'img_info')
        warning('TemporalModelFull:InvalidROIFile', ...
            'ROI file %s missing ROI_info/img_info; skipping brain maps.', neural_path);
        return;
    end

    img_info = spatial.img_info;
    roi_info = spatial.ROI_info;
    if ~isfield(img_info, 'imageData')
        warning('TemporalModelFull:NoImageData', ...
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
    peak_betas = results.comparison.peak_betas_all;
    cv_r2_all = [results.performance.R2_cv_mean];
    if numel(target_names) ~= numel(peak_lags) || numel(peak_lags) ~= numel(peak_betas)
        warning('TemporalModelFull:MismatchLength', ...
            'Mismatch in comparison arrays; skipping brain maps.');
        return;
    end

    lag_map = nan(dims);
    beta_map = nan(dims);
    r2_map = nan(dims);
    cat_map = zeros(dims, 'uint8');  % 0 = background
    n_assigned = 0;
    n_skipped = 0;

    for i = 1:numel(target_names)
        roi_name = target_names{i};
        match_idx = find(strcmpi(roi_names_source, roi_name), 1);
        if isempty(match_idx)
            error('TemporalModelFull:ROIMaskMissing', ...
                'ROI "%s" not found in %s', roi_name, neural_path);
        end
        roi_struct = roi_info(match_idx);
        if ~isfield(roi_struct, 'Stats') || ...
                ~isfield(roi_struct.Stats, 'ROI_binary_mask')
            error('TemporalModelFull:MaskMissing', ...
                'ROI "%s" missing Stats.ROI_binary_mask in %s', roi_name, neural_path);
        end
        mask = roi_struct.Stats.ROI_binary_mask;
        if ~isequal(size(mask), dims)
            error('TemporalModelFull:MaskSizeMismatch', ...
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
            error('TemporalModelFull:OverlappingMasks', ...
                'ROI "%s" overlaps with another ROI in %s', roi_name, neural_path);
        end

        lag_map(mask) = lag_val;
        beta_map(mask) = beta_val;
        r2_map(mask) = r2_val;
        if lag_val < 0 && beta_val >= 0
            cat_map(mask) = 1;  % predictive facilitatory
        elseif lag_val < 0 && beta_val < 0
            cat_map(mask) = 2;  % predictive suppressive
        elseif lag_val >= 0 && beta_val >= 0
            cat_map(mask) = 3;  % reactive facilitatory
        else
            cat_map(mask) = 4;  % reactive suppressive
        end
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
    cat_map(~mask_shape) = 0;

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

    fig = figure('Name', 'Temporal Peak Metrics Brain Maps', ...
        'Position', [200 100 1500 800]);
    use_tiled = exist('tiledlayout', 'file') == 2;
    layout = [];
    if use_tiled
        layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    end

    % Determine metric for title
    metric_str = 'Peak';
    if isfield(results.metadata, 'peak_metric') && strcmp(results.metadata.peak_metric, 'com')
        metric_str = 'CoM';
    end

    title_str = sprintf('%s Lag/Beta/R^2 Spatial Maps (n=%d, skipped=%d)', ...
        metric_str, n_assigned, n_skipped);
    add_super_title(fig, layout, title_str);

    % Lag map (diverging)
    cmap_lag = redbluecmap(256);
    if use_tiled
        ax1 = nexttile(layout, 1);
    else
        ax1 = subplot(2, 2, 1);
    end
    plot_metric_map(ax1, base_rgb, lag_map, cmap_lag, lag_limits, ...
        sprintf('%s Lag (s)\nPredictive < 0, Reactive > 0', metric_str), mask_shape);

    % Beta magnitude map
    cmap_beta = parula(256);
    if use_tiled
        ax2 = nexttile(layout, 2);
    else
        ax2 = subplot(2, 2, 2);
    end
    plot_metric_map(ax2, base_rgb, beta_map, cmap_beta, beta_limits, ...
        'Peak Beta (z-scored)', mask_shape);

    % Categorical map
    if use_tiled
        ax3 = nexttile(layout, 3);
    else
        ax3 = subplot(2, 2, 3);
    end
    category_colors = [0.35 0.65 1.0; 0.0 0.45 0.9; ...
        1.0 0.6 0.3; 0.8 0.2 0.2];
    plot_metric_map(ax3, base_rgb, cat_map, category_colors, [0.5 4.5], ...
        'Lag/Beta Quadrants', mask_shape, true);

    add_category_legend(ax3);

    % CV R^2 map
    if use_tiled
        ax4 = nexttile(layout, 4);
    else
        ax4 = subplot(2, 2, 4);
    end
    plot_metric_map(ax4, base_rgb, r2_map, parula(256), r2_limits, ...
        'CV R^2 (%)', mask_shape);
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
        error('TemporalModelFull:InvalidMask', ...
            'Mask file %s missing ROI_info or mask variable.', path_str);
    end
    if ~isequal(size(mask), dims)
        error('TemporalModelFull:MaskDims', ...
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
        colorbar(ax);
    end
    if draw_mask_outline
        plot_mask_outline(ax, mask_shape);
    end
    hold(ax, 'off');
    title(ax, ttl, 'FontSize', 12);
end

function add_category_legend(ax)
    labels = { ...
        'Predictive (+beta)', ...
        'Predictive (-beta)', ...
        'Reactive (+beta)', ...
        'Reactive (-beta)'};
    colors = [0.35 0.65 1.0; 0.0 0.45 0.9; 1.0 0.6 0.3; 0.8 0.2 0.2];
    hold(ax, 'on');
    for i = 1:4
        plot(ax, NaN, NaN, 's', 'MarkerFaceColor', colors(i, :), ...
            'MarkerEdgeColor', 'k', 'DisplayName', labels{i});
    end
    legend(ax, 'Location', 'southoutside', 'Orientation', 'horizontal');
    hold(ax, 'off');
end

function base_rgb = build_mask_background(mask_shape)
    % Render all non-data areas pure black so missing ROIs don't leave gray halos
    base_gray = zeros(size(mask_shape));
    base_rgb = repmat(base_gray, 1, 1, 3);
end

function plot_mask_outline(ax, mask_shape)
    contour(ax, mask_shape, [0.5 0.5], 'Color', [1 1 1], 'LineWidth', 1.2);
end

function add_super_title(fig_handle, layout_handle, title_str)
    if nargin >= 2 && ~isempty(layout_handle)
        title(layout_handle, title_str, 'FontSize', 14, 'Interpreter', 'none');
    elseif exist('sgtitle', 'file') == 2
        figure(fig_handle);
        sgtitle(title_str, 'Interpreter', 'none', 'FontSize', 14);
    else
        annotation(fig_handle, 'textbox', [0 0.95 1 0.04], ...
            'String', title_str, 'HorizontalAlignment', 'center', ...
            'EdgeColor', 'none', 'FontSize', 14, 'Interpreter', 'none');
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













