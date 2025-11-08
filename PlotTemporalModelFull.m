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
    % Plot all temporal kernels overlaid with color-coding

    n_rois = results.metadata.n_rois;
    roi_names = results.metadata.roi_names;

    fig_title = sprintf('All Temporal Kernels: %d ROIs vs %s', ...
        n_rois, results.metadata.behavior_predictor);

    figure('Name', fig_title, 'Position', [100 100 1000 700]);

    hold on;

    % Generate distinct colors for each ROI
    colors = lines(n_rois);

    % Plot each ROI's temporal kernel
    for roi = 1:n_rois
        tk = results.temporal_kernels(roi);

        % SEM envelope (semi-transparent)
        sem_upper = tk.beta_cv_mean + tk.beta_cv_sem;
        sem_lower = tk.beta_cv_mean - tk.beta_cv_sem;

        fill([tk.lag_times_sec; flipud(tk.lag_times_sec)], ...
             [sem_upper; flipud(sem_lower)], colors(roi, :), ...
             'FaceAlpha', 0.1, 'EdgeColor', 'none', ...
             'HandleVisibility', 'off');

        % Mean line
        plot(tk.lag_times_sec, tk.beta_cv_mean, ...
            'LineWidth', 1.8, 'Color', colors(roi, :), ...
            'DisplayName', sprintf('%s (R┬▓=%.2f%%)', roi_names{roi}, ...
                results.performance(roi).R2_cv_mean*100));
    end

    % Reference lines
    yl = ylim;
    plot([min(tk.lag_times_sec), max(tk.lag_times_sec)], [0 0], ...
        'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
    plot([0, 0], yl, 'k:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

    hold off;

    xlabel('Lag time (seconds)', 'FontSize', 13);
    ylabel('Beta coefficient (z-scored)', 'FontSize', 13);
    title(fig_title, 'FontSize', 15, 'Interpreter', 'none');
    legend('Location', 'bestoutside', 'FontSize', 9);
    grid on;

    % Add region labels
    text(min(tk.lag_times_sec)*0.8, yl(2)*0.95, 'Predictive', ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', [0.3 0.2 0.2]);
    text(max(tk.lag_times_sec)*0.8, yl(2)*0.95, 'Reactive', ...
        'HorizontalAlignment', 'center', 'FontSize', 12, 'Color', [0.2 0.2 0.2]);
end

function plot_temporal_kernel_heatmap(results)
    % Heatmap of temporal kernels: ROIs (rows) ├ù Lags (columns)

    beta_matrix = results.comparison.beta_matrix_cv';  % [n_rois ├ù n_lags]
    roi_names = results.comparison.roi_names;
    lag_times = results.temporal_kernels(1).lag_times_sec;

    fig_title = sprintf('Temporal Kernel Heatmap: %d ROIs vs %s', ...
        results.metadata.n_rois, results.metadata.behavior_predictor);

    figure('Name', fig_title, 'Position', [150 100 900 600]);

    imagesc(lag_times, 1:length(roi_names), beta_matrix);
    colormap(redbluecmap);
    colorbar;

    % Center colormap on zero
    clim_max = max(abs(beta_matrix(:)));
    clim([-clim_max, clim_max]);

    xlabel('Lag time (seconds)', 'FontSize', 12);
    ylabel('Neural ROI', 'FontSize', 12);
    title(fig_title, 'FontSize', 14, 'Interpreter', 'none');

    % Y-axis labels
    yticks(1:length(roi_names));
    yticklabels(roi_names);

    % Add vertical line at zero lag
    hold on;
    plot([0, 0], ylim, 'k--', 'LineWidth', 1.5);
    hold off;

    % Add colorbar label
    cb = colorbar;
    cb.Label.String = 'Beta coefficient (z-scored)';
    cb.Label.FontSize = 11;
end

function plot_performance_comparison(results)
    % Bar plot comparing R┬▓ across ROIs with error bars

    n_rois = results.metadata.n_rois;
    roi_names = results.comparison.roi_names;
    R2_cv = [results.performance.R2_cv_mean]';
    R2_sem = [results.performance.R2_cv_sem]';
    R2_full = [results.performance.R2_full_data]';
    peak_lags = results.comparison.peak_lags_all_sec';

    fig_title = sprintf('Performance Comparison: %d ROIs vs %s', ...
        n_rois, results.metadata.behavior_predictor);

    figure('Name', fig_title, 'Position', [200 100 1000 700]);
    tiled = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tiled, fig_title, 'Interpreter', 'none', 'FontSize', 14);

    % Top panel: R┬▓ comparison
    ax1 = nexttile(tiled);
    x_pos = 1:n_rois;

    % Plot CV R┬▓ with error bars
    b1 = bar(ax1, x_pos, R2_cv * 100, 'FaceColor', [0.3 0.5 0.8]);
    hold(ax1, 'on');
    errorbar(ax1, x_pos, R2_cv * 100, R2_sem * 100, 'k.', 'LineWidth', 1.5, ...
        'CapSize', 5);

    % Overlay full-data R┬▓ as dots
    plot(ax1, x_pos, R2_full * 100, 'ro', 'MarkerSize', 6, 'LineWidth', 1.5, ...
        'DisplayName', 'R┬▓ (full-data)');

    hold(ax1, 'off');

    ylabel(ax1, 'R┬▓ (%)', 'FontSize', 12);
    xticks(ax1, x_pos);
    xticklabels(ax1, roi_names);
    xtickangle(ax1, 45);
    legend(ax1, {'R┬▓ (CV mean ┬▒ SEM)', '', 'R┬▓ (full-data)'}, 'Location', 'best');
    grid(ax1, 'on');
    title(ax1, 'Model Performance', 'FontSize', 12);

    % Bottom panel: Peak lag distribution
    ax2 = nexttile(tiled);

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
    % Plot predictions for a subset of ROIs (top 4 by R┬▓)

    meta = results.metadata;
    pred = results.predictions;

    % Select top 4 ROIs by R┬▓
    R2_all = [results.performance.R2_cv_mean];
    [~, sorted_idx] = sort(R2_all, 'descend');
    n_plot = min(4, meta.n_rois);
    plot_indices = sorted_idx(1:n_plot);

    % Time vector for truncated data
    n_valid = meta.n_timepoints_used;
    n_lost_start = meta.n_timepoints_lost_start;
    t_truncated = (n_lost_start:(n_lost_start + n_valid - 1)) / meta.sampling_rate;

    fig_title = sprintf('Model Predictions: Top %d ROIs vs %s', ...
        n_plot, meta.behavior_predictor);

    figure('Name', fig_title, 'Position', [250 50 1200 800]);
    tiled = tiledlayout(n_plot + 1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tiled, fig_title, 'Interpreter', 'none', 'FontSize', 14);

    % Plot each ROI
    for i = 1:n_plot
        roi_idx = plot_indices(i);
        roi_name = meta.roi_names{roi_idx};

        ax = nexttile(tiled);

        % Actual vs predicted
        plot(ax, t_truncated, pred.Y_actual(:, roi_idx), 'Color', [0.2 0.2 0.8], ...
            'DisplayName', sprintf('%s (actual)', roi_name));
        hold(ax, 'on');
        plot(ax, t_truncated, pred.Y_pred(:, roi_idx), 'Color', [0.85 0.33 0.1], ...
            'LineWidth', 1.25, 'DisplayName', 'Prediction');
        hold(ax, 'off');

        ylabel(ax, sprintf('%s (z)', roi_name), 'Interpreter', 'none', 'FontSize', 10);
        legend(ax, 'Location', 'best', 'FontSize', 8);
        grid(ax, 'on');

        % Add R┬▓ annotation
        text(ax, 0.02, 0.98, ...
            sprintf('R┬▓(CV): %.2f%% ┬▒ %.2f%%', ...
                results.performance(roi_idx).R2_cv_mean*100, ...
                results.performance(roi_idx).R2_cv_sem*100), ...
            'Units', 'normalized', 'VerticalAlignment', 'top', ...
            'FontSize', 8, 'BackgroundColor', 'w', 'EdgeColor', 'k');
    end

    % Bottom panel: Behavior trace
    ax_behav = nexttile(tiled);
    t_full = (0:(length(pred.behavior_trace_z)-1)) / meta.sampling_rate;
    plot(ax_behav, t_full, pred.behavior_trace_z, 'Color', [0.13 0.55 0.13]);
    ylabel(ax_behav, sprintf('%s (z)', meta.behavior_predictor), 'Interpreter', 'none');
    xlabel(ax_behav, 'Time (s)');
    grid(ax_behav, 'on');

    linkaxes([tiled.Children], 'x');
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
    if numel(target_names) ~= numel(peak_lags) || numel(peak_lags) ~= numel(peak_betas)
        warning('TemporalModelFull:MismatchLength', ...
            'Mismatch in comparison arrays; skipping brain maps.');
        return;
    end

    lag_map = nan(dims);
    beta_map = nan(dims);
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
        if isnan(lag_val) || isnan(beta_val)
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

    beta_abs = max(abs(peak_betas(~isnan(peak_betas))));
    if isempty(beta_abs) || beta_abs == 0
        beta_abs = 1;
    end

    figure('Name', 'Temporal Peak Metrics Brain Maps', ...
        'Position', [200 100 1500 550]);
    tiled = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
    title_str = sprintf('Peak Lag/Beta Spatial Maps (n=%d, skipped=%d)', ...
        n_assigned, n_skipped);
    title(tiled, title_str, 'FontSize', 14);

    % Lag map (diverging)
    cmap_lag = redbluecmap(256);
    ax1 = nexttile(tiled);
    plot_metric_map(ax1, base_rgb, lag_map, cmap_lag, lag_limits, ...
        sprintf('Peak Lag (s)\nPredictive < 0, Reactive > 0'), mask_shape);

    % |beta| map
    abs_beta_map = abs(beta_map);
    ax2 = nexttile(tiled);
    plot_metric_map(ax2, base_rgb, abs_beta_map, parula(256), [0, beta_abs], ...
        'Peak |Beta| (a.u.)', mask_shape);

    % Categorical map
    ax3 = nexttile(tiled);
    category_colors = [0.35 0.65 1.0; 0.0 0.45 0.9; ...
        1.0 0.6 0.3; 0.8 0.2 0.2];
    plot_metric_map(ax3, base_rgb, cat_map, category_colors, [0.5 4.5], ...
        'Lag/Beta Quadrants', mask_shape, true);

    add_category_legend(ax3);
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

function plot_metric_map(ax, base_rgb, metric_map, cmap, clim, ttl, mask_shape, is_categorical)
    if nargin < 8
        is_categorical = false;
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
    plot_mask_outline(ax, mask_shape);
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
    base_gray = 0.08 * ones(size(mask_shape));
    base_gray(mask_shape) = 0.25;
    base_rgb = repmat(base_gray, 1, 1, 3);
end

function plot_mask_outline(ax, mask_shape)
    contour(ax, mask_shape, [0.5 0.5], 'Color', [1 1 1], 'LineWidth', 1.2);
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

