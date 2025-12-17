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
plot_peak_beta_brainmaps_poster(results);
plot_variable_cvr2_brainmaps(results);
plot_variable_beta_peak_brainmaps(results);
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

    % Time vector for truncated data (fallback for multi-ROI events results)
    if isfield(pred, 'time_vector') && ~isempty(pred.time_vector)
        t_truncated = pred.time_vector(:)';
        n_valid = numel(t_truncated);
        n_lost_start = 0;
    else
        n_valid = size(pred.Y_actual, 1);
        n_lost_start = derive_lost_start(meta);
        t_truncated = (n_lost_start:(n_lost_start + n_valid - 1)) / meta.sampling_rate;
    end

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
        plot(ax, t_truncated, pred.Y_actual(:, roi_idx), 'Color', [21/255, 101/255, 192/255], ...
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

        % Add R^2 annotation with enhanced visibility
        text(ax, 0.02, 0.98, ...
            sprintf('R^2 (CV): %.2f%% ± %.2f%%', ...
                results.performance(roi_idx).R2_cv_mean*100, ...
                results.performance(roi_idx).R2_cv_sem*100), ...
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

    if isfield(pred, 'time_vector') && ~isempty(pred.time_vector)
        t_behavior = pred.time_vector(:)';
    else
        t_behavior = (n_lost_start:(n_lost_start + n_valid - 1)) / meta.sampling_rate;
    end

    if isfield(pred, 'truncated_behavior') && ~isempty(pred.truncated_behavior)
        behavior_trace = pred.truncated_behavior(:)';
        t_behavior = t_behavior(1:numel(behavior_trace));
    else
        behavior_trace = pred.behavior_trace_z(:)';
        min_len = min(numel(behavior_trace), numel(t_behavior));
        behavior_trace = behavior_trace(1:min_len);
        t_behavior = t_behavior(1:min_len);
    end

    plot(ax_behav, t_behavior, behavior_trace, 'Color', [56/255 142/255 60/255], ...
        'LineWidth', 2.0, 'DisplayName', sprintf('%s (z)', meta.behavior_predictor));
    hold(ax_behav, 'on');
    overlay_behavior_event_markers(ax_behav, results, t_behavior);
    hold(ax_behav, 'off');
    legend(ax_behav, 'Location', 'best', 'FontSize', 11);
    ylabel(ax_behav, sprintf('%s (z)', meta.behavior_predictor), ...
        'Interpreter', 'none', 'FontSize', 13);
    xlabel(ax_behav, 'Time (s)', 'FontSize', 13);
    title(ax_behav, 'Behavioral Predictor', 'FontSize', 16, 'FontWeight', 'bold');
    grid(ax_behav, 'off');
    set(ax_behav, 'Box', 'off');

    linkaxes(ax_handles, 'x');
end

function overlay_behavior_event_markers(ax, results, time_vec)
    if nargin < 3 || isempty(time_vec)
        time_range = get(ax, 'XLim');
    else
        time_range = [min(time_vec), max(time_vec)];
    end

    event_data = struct();
    if isfield(results, 'events') && ~isempty(results.events)
        event_data = results.events;
    elseif isfield(results, 'design_matrix') && isfield(results.design_matrix, 'events')
        event_data = results.design_matrix.events;
    end

    if isempty(event_data) || ~isstruct(event_data) || isempty(fieldnames(event_data))
        return;
    end

    event_specs = {
        'noise_onsets',        [0.35 0.35 0.35], 'Noise stimulus start';
        'lick_post_stimulus',  [0.80 0.30 0.30], 'First lick post stimulus';
        'lick_post_water_all', [0.20 0.55 0.75], 'First lick post water'};

    for es = 1:size(event_specs, 1)
        field_name = event_specs{es, 1};
        color = event_specs{es, 2};
        label = event_specs{es, 3};

        if ~isfield(event_data, field_name) || isempty(event_data.(field_name))
            continue;
        end

        times = double(event_data.(field_name)(:));
        times = times(isfinite(times));
        times = times(times >= time_range(1) & times <= time_range(2));
        if isempty(times)
            continue;
        end

        first_marker = true;
        for t_evt = times(:)'
            h = xline(ax, t_evt, '--', 'Color', color, 'LineWidth', 1.2);
            if first_marker
                set(h, 'DisplayName', label);
                first_marker = false;
            else
                set(h, 'HandleVisibility', 'off');
            end
        end
    end
end

function plot_peak_beta_brainmaps_poster(results)
    % Plot peak lag/beta/R^2 summaries on cortex map (1x3 layout, no category labels)

    ctx = prepare_roi_brain_context(results);
    if isempty(ctx)
        return;
    end

    target_names = ctx.target_names;
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

    lag_map = nan(ctx.dims);
    beta_map = nan(ctx.dims);
    r2_map = nan(ctx.dims);
    n_assigned = 0;

    for i = 1:numel(target_names)
        mask = ctx.roi_masks{i};
        if isempty(mask)
            continue;
        end

        lag_val = peak_lags(i);
        beta_val = peak_betas(i);
        r2_val = cv_r2_all(i) * 100;  % express as %
        if isnan(lag_val) || isnan(beta_val) || isnan(r2_val)
            continue;
        end

        overlap = ~isnan(lag_map) & mask;
        if any(overlap(:))
            error('PlotPosterNoCategoryTemporalModelFull:OverlappingMasks', ...
                'ROI "%s" overlaps with another ROI.', target_names{i});
        end

        lag_map(mask) = lag_val;
        beta_map(mask) = beta_val;
        r2_map(mask) = r2_val;
        n_assigned = n_assigned + 1;
    end

    lag_map(~ctx.mask_shape) = nan;
    beta_map(~ctx.mask_shape) = nan;
    r2_map(~ctx.mask_shape) = nan;

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
    plot_metric_map(ax1, ctx.base_rgb, lag_map, cmap_lag, lag_limits, ...
        sprintf('%s Lag (s)', metric_str), ctx.mask_shape, false, false, ctx.roi_masks);

    % Panel 2: Beta magnitude map (z-scored peak/CoM value)
    cmap_beta = parula(256);
    if use_tiled
        ax2 = nexttile(layout, 2);
    else
        ax2 = subplot(1, 3, 2);
    end
    plot_metric_map(ax2, ctx.base_rgb, beta_map, cmap_beta, beta_limits, ...
        'Peak Beta (z-scored)', ctx.mask_shape, false, false, ctx.roi_masks);

    % Panel 3: CV R^2 map
    if use_tiled
        ax3 = nexttile(layout, 3);
    else
        ax3 = subplot(1, 3, 3);
    end
    plot_metric_map(ax3, ctx.base_rgb, r2_map, parula(256), r2_limits, ...
        'CV R^2 (%)', ctx.mask_shape, false, false, ctx.roi_masks);
end


function plot_variable_cvr2_brainmaps(results)
    ctx = prepare_roi_brain_context(results);
    if isempty(ctx)
        return;
    end

    if ~isfield(results, 'contributions')
        warning('PlotPosterNoCategoryTemporalModelFull:NoContributions', ...
            'Results struct lacks contributions; skipping per-variable CVR^2 maps.');
        return;
    end

    contr = results.contributions;
    required = {'group_single_R2', 'group_labels'};
    if any(~isfield(contr, required))
        warning('PlotPosterNoCategoryTemporalModelFull:IncompleteContributions', ...
            'Contributions struct missing required fields; skipping per-variable CVR^2 maps.');
        return;
    end

    group_labels = contr.group_labels(:);
    group_r2 = contr.group_single_R2;
    if ndims(group_r2) ~= 3
        warning('PlotPosterNoCategoryTemporalModelFull:InvalidContributionDims', ...
            'group_single_R2 must be G x R x folds; skipping per-variable CVR^2 maps.');
        return;
    end

    group_mean = mean(group_r2, 3, 'omitnan');
    if size(group_mean, 2) ~= numel(ctx.target_names)
        warning('PlotPosterNoCategoryTemporalModelFull:RoiMismatch', ...
            'Mismatch between ROI counts in contributions and spatial metadata; skipping per-variable CVR^2 maps.');
        return;
    end

    n_groups = size(group_mean, 1);
    if n_groups == 0
        return;
    end

    clim = [min(group_mean(:), [], 'omitnan'), max(group_mean(:), [], 'omitnan')];
    if any(isnan(clim))
        clim = [0, 1];
    elseif clim(1) == clim(2)
        if clim(2) == 0 || isnan(clim(2))
            clim = [0, 1];
        else
            clim = [0, clim(2)];
        end
    end

    fig = figure('Name', 'Variable CVR^2 Brain Maps', 'Position', [100 100 1600 600]);
    n_cols = min(4, n_groups);
    n_rows = ceil(n_groups / n_cols);
    use_tiled = exist('tiledlayout', 'file') == 2;
    layout = [];
    if use_tiled
        layout = tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    end

    for g = 1:n_groups
        if use_tiled
            ax = nexttile(layout, g);
        else
            ax = subplot(n_rows, n_cols, g);
        end
        value_map = nan(ctx.dims);
        for r = 1:numel(ctx.roi_masks)
            mask = ctx.roi_masks{r};
            if isempty(mask)
                continue;
            end
            val = group_mean(g, r);
            if isnan(val)
                continue;
            end
            value_map(mask) = val;
        end
        ttl = sprintf('%s CV R^2 (%%)', group_labels{g});
        plot_metric_map(ax, ctx.base_rgb, value_map, parula(256), clim, ...
            ttl, ctx.mask_shape, false, false, ctx.roi_masks);
    end

    add_super_title(fig, layout, 'Single-variable CVR^2 brain maps');
end

function plot_variable_beta_peak_brainmaps(results)
    ctx = prepare_roi_brain_context(results);
    if isempty(ctx)
        return;
    end

    if ~isfield(results, 'event_kernels') || isempty(results.event_kernels)
        warning('PlotPosterNoCategoryTemporalModelFull:NoEventKernels', ...
            'Event kernel summaries missing; skipping per-variable beta maps.');
        return;
    end

    event_kernels = results.event_kernels;
    template_idx = find(~cellfun(@isempty, event_kernels), 1, 'first');
    if isempty(template_idx)
        warning('PlotPosterNoCategoryTemporalModelFull:EmptyEventKernels', ...
            'Event kernel entries are empty; skipping per-variable beta maps.');
        return;
    end

    template = event_kernels{template_idx};
    if isempty(template)
        warning('PlotPosterNoCategoryTemporalModelFull:TemplateKernelMissing', ...
            'Template event kernel missing; skipping per-variable beta maps.');
        return;
    end

    n_rois = numel(ctx.target_names);
    lag_counts = arrayfun(@(k) numel(k.lag_times_sec), template);
    valid_groups = find(lag_counts > 0);

    label_list = {};
    beta_rows = [];
    peak_times = [];
    motion_snapshot = [];
    motion_label_entry = '';
    motion_peak_time = NaN;

    for idx = valid_groups(:)'
        lag_times = template(idx).lag_times_sec(:)';
        n_lags = numel(lag_times);
        if n_lags == 0
            continue;
        end
        roi_curves = nan(n_rois, n_lags);
        for r = 1:n_rois
            roi_kernel = event_kernels{r};
            if isempty(roi_kernel) || numel(roi_kernel) < idx
                continue;
            end
            curve = roi_kernel(idx).beta_cv_mean(:)';
            if isempty(curve)
                continue;
            end
            len = min(numel(curve), n_lags);
            roi_curves(r, 1:len) = curve(1:len);
        end
        [snapshot, peak_time] = extract_peak_snapshot(roi_curves, lag_times);
        if isempty(snapshot)
            continue;
        end
        label_list{end+1} = assign_default_label(template(idx).label, idx); %#ok<AGROW>
        peak_times(end+1, 1) = peak_time; %#ok<AGROW>
        beta_rows = [beta_rows; snapshot']; %#ok<AGROW>
    end

    % Add motion predictor as its own group, if available
    if isfield(results, 'temporal_kernels') && ~isempty(results.temporal_kernels)
        motion_lag = results.temporal_kernels(1).lag_times_sec(:)';
        if ~isempty(motion_lag)
            n_motion_lags = numel(motion_lag);
            roi_curves = nan(n_rois, n_motion_lags);
            for r = 1:min(numel(results.temporal_kernels), n_rois)
                tk = results.temporal_kernels(r);
                if isempty(tk) || isempty(tk.beta_cv_mean)
                    continue;
                end
                curve = tk.beta_cv_mean(:)';
                len = min(numel(curve), n_motion_lags);
                roi_curves(r, 1:len) = curve(1:len);
            end
            [snapshot, peak_time] = extract_peak_snapshot(roi_curves, motion_lag);
            if ~isempty(snapshot)
                if isfield(results, 'metadata') && isfield(results.metadata, 'behavior_predictor') ...
                        && ~isempty(results.metadata.behavior_predictor)
                    motion_label_str = sprintf('%s motion', results.metadata.behavior_predictor);
                else
                    motion_label_str = 'Motion predictor';
                end
                motion_label_entry = motion_label_str;
                motion_peak_time = peak_time;
                motion_snapshot = snapshot';
            end
        end
    end

    if ~isempty(motion_snapshot)
        beta_rows = [motion_snapshot; beta_rows];
        label_list = [{motion_label_entry}; label_list];
        peak_times = [motion_peak_time; peak_times];
    end

    if isempty(beta_rows)
        warning('PlotPosterNoCategoryTemporalModelFull:NoLagInfo', ...
            'No predictor groups had valid lag metadata; skipping beta peak brain maps.');
        return;
    end

    n_groups = size(beta_rows, 1);
    beta_values = beta_rows;

    beta_span = max(abs(beta_values(:)), [], 'omitnan');
    if isempty(beta_span) || beta_span == 0
        beta_span = 1;
    end
    clim = [-beta_span, beta_span];

    fig = figure('Name', 'Variable Beta Snapshots (Peak Avg Time)', ...
        'Position', [150 150 1600 600]);
    n_cols = min(4, n_groups);
    n_rows = ceil(n_groups / n_cols);
    use_tiled = exist('tiledlayout', 'file') == 2;
    layout = [];
    if use_tiled
        layout = tiledlayout(n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    end

    for gi = 1:n_groups
        if use_tiled
            ax = nexttile(layout, gi);
        else
            ax = subplot(n_rows, n_cols, gi);
        end
        value_map = nan(ctx.dims);
        for r = 1:numel(ctx.roi_masks)
            mask = ctx.roi_masks{r};
            if isempty(mask)
                continue;
            end
            val = beta_values(gi, r);
            if isnan(val)
                continue;
            end
            value_map(mask) = val;
        end
        if isnan(peak_times(gi))
            time_str = 't = N/A';
        else
            time_str = sprintf('t = %.2f s', peak_times(gi));
        end
        ttl = sprintf('%s\n(%s)', label_list{gi}, time_str);
        plot_metric_map(ax, ctx.base_rgb, value_map, redbluecmap(256), clim, ...
            ttl, ctx.mask_shape, false, false, ctx.roi_masks);
    end

    add_super_title(fig, layout, 'Variable beta weights at peak average time');
end

function [snapshot, peak_time] = extract_peak_snapshot(roi_curves, lag_times)
    snapshot = [];
    peak_time = NaN;
    if isempty(roi_curves) || isempty(lag_times)
        return;
    end
    avg_curve = mean(roi_curves, 1, 'omitnan');
    avg_abs = mean(abs(roi_curves), 1, 'omitnan');
    if all(isnan(avg_abs))
        avg_abs = abs(avg_curve);
    end
    if all(isnan(avg_abs))
        return;
    end
    [~, peak_idx] = max(avg_abs);
    snapshot = roi_curves(:, peak_idx);
    peak_time = lag_times(peak_idx);
end

function ctx = prepare_roi_brain_context(results)
    ctx = [];

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

    target_names = results.comparison.roi_names;
    roi_names_source = arrayfun(@(r) char(r.Name), roi_info, 'UniformOutput', false);
    roi_masks = cell(numel(target_names), 1);
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
        roi_masks{i} = mask;
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
    base_rgb = build_mask_background(mask_shape);

    ctx = struct();
    ctx.target_names = target_names;
    ctx.roi_masks = roi_masks;
    ctx.mask_shape = mask_shape;
    ctx.base_rgb = base_rgb;
    ctx.dims = dims;
end

function name_out = assign_default_label(name_in, idx)
    if nargin < 1 || isempty(name_in)
        name_out = sprintf('Group %d', idx);
    else
        name_out = strtrim(name_in);
        if isempty(name_out)
            name_out = sprintf('Group %d', idx);
        end
    end
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

function plot_metric_map(ax, base_rgb, metric_map, cmap, clim, ttl, mask_shape, is_categorical, draw_mask_outline, roi_masks)
    if nargin < 8 || isempty(is_categorical)
        is_categorical = false;
    end
    if nargin < 9 || isempty(draw_mask_outline)
        draw_mask_outline = false;
    end
    if nargin < 10 || isempty(roi_masks)
        roi_masks = {};
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
    if ~isempty(roi_masks)
        for idx = 1:numel(roi_masks)
            mask = roi_masks{idx};
            if isempty(mask) || ~any(mask(:))
                continue;
            end
            contour(ax, mask, [0.5 0.5], 'Color', [0 0 0], 'LineWidth', 0.8);
        end
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
function n_lost_start = derive_lost_start(meta)
    if isfield(meta, 'n_timepoints_lost_start')
        n_lost_start = meta.n_timepoints_lost_start;
    elseif isfield(meta, 'frames_lost_start')
        n_lost_start = meta.frames_lost_start;
    else
        n_lost_start = 0;
    end
end
