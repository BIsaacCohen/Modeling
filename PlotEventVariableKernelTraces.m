function PlotEventVariableKernelTraces(results_input, opts)
% PlotEventVariableKernelTraces  Plot event predictor beta kernels for each group.
%
%   PlotEventVariableKernelTraces(results_input)
%   PlotEventVariableKernelTraces(results_input, opts)
%
%   results_input : MAT-file path produced by TemporalModel*Contributions or
%                   a results struct already in the workspace.
%   opts.overlay_rois : optional cell array (max 2 entries) of ROI names whose
%                       event kernels should be overlaid on every subplot.
%   opts.show_fig     : true (default) to display the generated figure.
%
%   This script aggregates the beta weight traces (cv-means + SEM) for each
%   event predictor across all ROIs and plots them in a grid. Up to two ROIs
%   can be highlighted with their individual kernels, mirroring the overlay
%   behavior from PlotPosterNoCategoryTemporalModelFull.

if nargin < 1 || isempty(results_input)
    error('Please provide a TemporalModel* results struct or MAT file path.');
end

if nargin < 2 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'overlay_rois') || isempty(opts.overlay_rois)
    opts.overlay_rois = {};
elseif ischar(opts.overlay_rois) || isstring(opts.overlay_rois)
    opts.overlay_rois = cellstr(opts.overlay_rois);
end
if numel(opts.overlay_rois) > 2
    error('opts.overlay_rois supports at most 2 ROI names.');
end
if ~isfield(opts, 'show_fig') || isempty(opts.show_fig)
    opts.show_fig = true;
end
if ~isfield(opts, 'plot_peak_maps') || isempty(opts.plot_peak_maps)
    opts.plot_peak_maps = true;
end

results = load_results_struct(results_input);
if ~isfield(results, 'event_kernels') || isempty(results.event_kernels)
    error('Results struct missing event_kernels; rerun TemporalModel with events.');
end

event_cells = results.event_kernels;
n_rois = numel(event_cells);
template_idx = find(~cellfun(@isempty, event_cells), 1, 'first');
if isempty(template_idx)
    error('All event kernel entries are empty.');
end
template = event_cells{template_idx};
n_groups = numel(template);
if n_groups == 0 && (~isfield(results, 'temporal_kernels') || isempty(results.temporal_kernels))
    error('No event predictor groups or motion kernels available.');
end

roi_names = derive_roi_names(results, n_rois);
overlay_idx = map_overlay_indices(opts.overlay_rois, roi_names);

motion_info = build_motion_group_info(results, n_rois);
total_groups = n_groups + double(motion_info.available);

visible_flag = ternary(opts.show_fig, 'on', 'off');
fig = figure('Name', 'Event Variable Beta Kernels', ...
    'Visible', visible_flag, 'Position', [100 100 1200 700]);

n_cols = min(3, total_groups);
n_rows = ceil(total_groups / n_cols);
use_tiled = exist('tiledlayout', 'file') == 2;
if use_tiled
    layout = tiledlayout(fig, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
else
    layout = [];
end

mean_color = [0.2 0.4 0.8];
sem_color = mean_color + (1 - mean_color) * 0.6;
overlay_colors = lines(max(2, numel(overlay_idx)));

for plot_idx = 1:total_groups
    if use_tiled
        ax = nexttile(layout, plot_idx);
    else
        ax = subplot(n_rows, n_cols, plot_idx);
    end
    hold(ax, 'on');

    lag_times = [];
    mean_trace = [];
    sem_trace = [];
    roi_traces = {};
    title_label = '';
    if motion_info.available && plot_idx == 1
        lag_times = motion_info.lag_times;
        mean_trace = motion_info.mean_trace;
        sem_trace = motion_info.sem_trace;
        roi_traces = motion_info.roi_traces;
        title_label = motion_info.label;
    else
        g = plot_idx - motion_info.available;
        if g < 1 || g > n_groups
            hold(ax, 'off');
            continue;
        end
        lag_times = template(g).lag_times_sec(:);
        [mean_trace, sem_trace, roi_traces] = compute_group_statistics(event_cells, g, lag_times);
        title_label = assign_default_label(template(g).label, g);
    end

    if isempty(mean_trace)
        title(ax, sprintf('%s (no data)', title_label), 'Interpreter', 'none');
        hold(ax, 'off');
        continue;
    end

    fill_x = [lag_times; flipud(lag_times)];
    fill_y = [mean_trace - sem_trace; flipud(mean_trace + sem_trace)];
    patch(ax, fill_x, fill_y, sem_color, 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    plot(ax, lag_times, mean_trace, 'Color', mean_color, 'LineWidth', 2, ...
        'DisplayName', 'All ROIs mean');

    legend_entries = {'All ROIs'};
    for oi = 1:numel(overlay_idx)
        roi_idx = overlay_idx(oi);
        if roi_idx < 1 || roi_idx > numel(roi_traces) || isempty(roi_traces{roi_idx})
            continue;
        end
        trace = roi_traces{roi_idx};
        plot(ax, lag_times, trace, 'LineWidth', 1.5, ...
            'Color', overlay_colors(oi, :), 'DisplayName', roi_names{roi_idx});
        legend_entries{end+1} = roi_names{roi_idx}; %#ok<AGROW>
    end

    title(ax, title_label, 'Interpreter', 'none');
    xlabel(ax, 'Lag (s)');
    ylabel(ax, '\beta');
    grid(ax, 'on');
    set(ax, 'Box', 'off');
    legend(ax, 'Location', 'best');
    hold(ax, 'off');
end

if ~isempty(layout)
    title(layout, 'Event Predictor Beta Kernels', 'FontSize', 16, 'Interpreter', 'none');
else
    sgtitle('Event Predictor Beta Kernels', 'Interpreter', 'none', 'FontSize', 16);
end

if ~opts.show_fig
    close(fig);
end

if opts.plot_peak_maps
    try
        plot_event_peak_beta_maps(results);
    catch ME
        warning('PlotEventVariableKernelTraces:PeakMapFailure', ...
            'Unable to plot beta-weight maps: %s', ME.message);
    end
end
end

function results = load_results_struct(input)
if ischar(input) || isstring(input)
    file_path = char(input);
    if exist(file_path, 'file') ~= 2
        error('Results file "%s" not found.', file_path);
    end
    data = load(file_path, 'results');
    if ~isfield(data, 'results')
        error('File %s does not contain a "results" struct.', file_path);
    end
    results = data.results;
elseif isstruct(input)
    if isfield(input, 'event_kernels')
        results = input;
    elseif isfield(input, 'results')
        results = input.results;
    else
        error('Provided struct does not contain an event_kernels field.');
    end
else
    error('Unsupported results_input type: %s', class(input));
end
end

function roi_names = derive_roi_names(results, n_rois)
roi_names = arrayfun(@(idx) sprintf('ROI_%d', idx), 1:n_rois, 'UniformOutput', false);
if isfield(results, 'temporal_kernels') && ~isempty(results.temporal_kernels)
    tk = results.temporal_kernels;
    for i = 1:min(numel(tk), n_rois)
        if isfield(tk(i), 'roi_name') && ~isempty(tk(i).roi_name)
            roi_names{i} = tk(i).roi_name;
        end
    end
elseif isfield(results, 'metadata') && isfield(results.metadata, 'target_neural_rois')
    target_rois = results.metadata.target_neural_rois;
    for i = 1:min(numel(target_rois), n_rois)
        roi_names{i} = target_rois{i};
    end
end
end

function overlay_idx = map_overlay_indices(overlay_rois, roi_names)
overlay_idx = [];
if isempty(overlay_rois)
    return;
end
roi_lower = lower(roi_names);
for i = 1:numel(overlay_rois)
    target = lower(overlay_rois{i});
    match = find(strcmpi(target, roi_names), 1);
    if isempty(match)
        match = find(strcmp(roi_lower, target), 1);
    end
    if isempty(match)
        warning('Overlay ROI "%s" not recognized; skipping overlay.', overlay_rois{i});
        continue;
    end
    overlay_idx(end+1) = match; %#ok<AGROW>
end
overlay_idx = unique(overlay_idx, 'stable');
end

function [mean_trace, sem_trace, roi_traces] = compute_group_statistics(event_cells, group_idx, lag_times)
n_rois = numel(event_cells);
roi_traces = cell(n_rois, 1);
trace_matrix = nan(numel(lag_times), n_rois);

for r = 1:n_rois
    kernels = event_cells{r};
    if isempty(kernels) || numel(kernels) < group_idx
        continue;
    end
    vec = kernels(group_idx).beta_cv_mean(:);
    if isempty(vec)
        continue;
    end
    len = min(numel(vec), numel(lag_times));
    padded = nan(numel(lag_times), 1);
    padded(1:len) = vec(1:len);
    roi_traces{r} = padded;
    trace_matrix(:, r) = padded;
end

mean_trace = mean(trace_matrix, 2, 'omitnan');
counts = sum(~isnan(trace_matrix), 2);
sem_trace = std(trace_matrix, 0, 2, 'omitnan') ./ sqrt(max(counts, 1));
if all(isnan(mean_trace))
    mean_trace = [];
    sem_trace = [];
end
end

function info = build_motion_group_info(results, n_rois_target)
info = struct('available', false, 'label', '', 'lag_times', [], ...
    'mean_trace', [], 'sem_trace', [], 'roi_traces', [], 'trace_matrix', []);
if ~isfield(results, 'temporal_kernels') || isempty(results.temporal_kernels)
    return;
end
tk = results.temporal_kernels;
lag_times = tk(1).lag_times_sec(:);
if isempty(lag_times)
    return;
end
n_rois = min(numel(tk), n_rois_target);
if n_rois == 0
    return;
end
n_lags = numel(lag_times);
trace_matrix = nan(n_lags, n_rois);
roi_traces = cell(n_rois, 1);
for r = 1:n_rois
    curve = tk(r).beta_cv_mean(:);
    if isempty(curve)
        continue;
    end
    len = min(numel(curve), n_lags);
    padded = nan(n_lags, 1);
    padded(1:len) = curve(1:len);
    trace_matrix(:, r) = padded;
    roi_traces{r} = padded;
end
mean_trace = mean(trace_matrix, 2, 'omitnan');
counts = sum(~isnan(trace_matrix), 2);
sem_trace = std(trace_matrix, 0, 2, 'omitnan') ./ sqrt(max(counts, 1));
if all(isnan(mean_trace))
    return;
end
info.available = true;
if isfield(results, 'metadata') && isfield(results.metadata, 'behavior_predictor') ...
        && ~isempty(results.metadata.behavior_predictor)
    info.label = sprintf('%s motion', results.metadata.behavior_predictor);
else
    info.label = 'Motion predictor';
end
info.lag_times = lag_times;
info.mean_trace = mean_trace;
info.sem_trace = sem_trace;
info.roi_traces = roi_traces;
info.trace_matrix = trace_matrix;
end

function plot_event_peak_beta_maps(results)
ctx = prepare_roi_brain_context(results);
if isempty(ctx)
    warning('PlotEventVariableKernelTraces:NoSpatialContext', ...
        'Unable to locate ROI spatial metadata; skipping beta weight maps.');
    return;
end
event_cells = results.event_kernels;
n_rois = min(numel(event_cells), numel(ctx.roi_masks));
template_idx = find(~cellfun(@isempty, event_cells), 1, 'first');
if isempty(template_idx)
    warning('PlotEventVariableKernelTraces:EmptyEventKernels', ...
        'Event kernel entries are empty; skipping maps.');
    return;
end
template = event_cells{template_idx};
n_groups = numel(template);
if n_groups == 0 && (~isfield(results, 'temporal_kernels') || isempty(results.temporal_kernels))
    warning('PlotEventVariableKernelTraces:NoGroups', ...
        'No event predictor groups available for mapping.');
    return;
end

snapshots = cell(n_groups, 1);
peak_times = nan(n_groups, 1);
label_list = cell(n_groups, 1);
for gi = 1:n_groups
    lag_times = template(gi).lag_times_sec(:)';
    n_lags = numel(lag_times);
    if n_lags == 0
        continue;
    end
    roi_curves = nan(n_rois, n_lags);
    for r = 1:n_rois
        kernels = event_cells{r};
        if isempty(kernels) || numel(kernels) < gi
            continue;
        end
        vec = kernels(gi).beta_cv_mean(:)';
        if isempty(vec)
            continue;
        end
        len = min(numel(vec), n_lags);
        roi_curves(r, 1:len) = vec(1:len);
    end
    [snapshot, peak_time] = extract_peak_snapshot(roi_curves, lag_times);
    snapshots{gi} = snapshot;
    peak_times(gi) = peak_time;
    label_list{gi} = assign_default_label(template(gi).label, gi);
end

valid_snapshots = snapshots(~cellfun(@isempty, snapshots));
if isempty(valid_snapshots)
    warning('PlotEventVariableKernelTraces:NoSnapshotValues', ...
        'Could not derive peak beta values for any event group.');
    return;
end
all_vals = cell2mat(cellfun(@(v) v(:), valid_snapshots, 'UniformOutput', false));
beta_span = max(abs(all_vals), [], 'omitnan');
if isempty(beta_span) || beta_span == 0
    beta_span = 1;
end
clim = [-beta_span, beta_span];

motion_info = build_motion_group_info(results, n_rois);
if motion_info.available
    roi_curves_motion = motion_info.trace_matrix';
    [snapshot_motion, peak_motion] = extract_peak_snapshot(roi_curves_motion, motion_info.lag_times(:)');
    if ~isempty(snapshot_motion)
        snapshots = [{snapshot_motion}; snapshots];
        peak_times = [peak_motion; peak_times];
        label_list = [{motion_info.label}; label_list];
    end
end

total_groups = numel(snapshots);
n_cols = min(4, total_groups);
n_rows = ceil(total_groups / n_cols);
use_tiled = exist('tiledlayout', 'file') == 2;
fig = figure('Name', 'Event Beta Weights at Peak Mean', ...
    'Position', [150 150 1500 600]);
if use_tiled
    layout = tiledlayout(fig, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
else
    layout = [];
end

for gi = 1:total_groups
    snapshot = snapshots{gi};
    if isempty(snapshot)
        continue;
    end
    if use_tiled
        ax = nexttile(layout, gi);
    else
        ax = subplot(n_rows, n_cols, gi);
    end
    value_map = nan(ctx.dims);
    for r = 1:min(numel(snapshot), numel(ctx.roi_masks))
        mask = ctx.roi_masks{r};
        if isempty(mask)
            continue;
        end
        val = snapshot(r);
        if isnan(val)
            continue;
        end
        value_map(mask) = val;
    end
    if isnan(peak_times(gi))
        time_str = 'Peak t = N/A';
    else
        time_str = sprintf('Peak t = %.2f s', peak_times(gi));
    end
    ttl = sprintf('%s\n(%s)', label_list{gi}, time_str);
    plot_metric_map(ax, ctx.base_rgb, value_map, redbluecmap(256), clim, ...
        ttl, ctx.mask_shape, false, false, ctx.roi_masks);
end

add_super_title(fig, layout, 'Event predictor beta weights at peak mean lag');
end

function [snapshot, peak_time] = extract_peak_snapshot(roi_curves, lag_times)
snapshot = [];
peak_time = NaN;
if isempty(roi_curves) || isempty(lag_times)
    return;
end
avg_curve = mean(roi_curves, 1, 'omitnan');
if all(isnan(avg_curve))
    return;
end
[~, peak_idx] = max(avg_curve);
snapshot = roi_curves(:, peak_idx);
peak_time = lag_times(peak_idx);
end

function ctx = prepare_roi_brain_context(results)
ctx = [];
if ~isfield(results, 'comparison') || ~isfield(results.comparison, 'roi_names')
    warning('PlotEventVariableKernelTraces:NoComparisonInfo', ...
        'Comparison summaries missing; cannot build brain maps.');
    return;
end
if ~isfield(results, 'metadata') || ...
        ~isfield(results.metadata, 'source_roi_file') || ...
        ~isstruct(results.metadata.source_roi_file)
    warning('PlotEventVariableKernelTraces:NoSpatialSource', ...
        'No ROI source metadata available; cannot build brain maps.');
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
    warning('PlotEventVariableKernelTraces:MissingROIFile', ...
        'Neural ROI file not found (%s).', neural_path);
    return;
end

spatial = load(neural_path, '-mat');
if ~isfield(spatial, 'ROI_info') || ~isfield(spatial, 'img_info')
    warning('PlotEventVariableKernelTraces:InvalidROIFile', ...
        'ROI file %s missing ROI_info/img_info.', neural_path);
    return;
end

img_info = spatial.img_info;
roi_info = spatial.ROI_info;
if ~isfield(img_info, 'imageData')
    warning('PlotEventVariableKernelTraces:NoImageData', ...
        'img_info.imageData missing in %s.', neural_path);
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
        warning('PlotEventVariableKernelTraces:ROIMaskMissing', ...
            'ROI "%s" not found in %s', roi_name, neural_path);
        return;
    end
    roi_struct = roi_info(match_idx);
    if ~isfield(roi_struct, 'Stats') || ...
            ~isfield(roi_struct.Stats, 'ROI_binary_mask')
        warning('PlotEventVariableKernelTraces:MaskMissing', ...
            'ROI "%s" missing Stats.ROI_binary_mask in %s', roi_name, neural_path);
        return;
    end
    mask = roi_struct.Stats.ROI_binary_mask;
    if ~isequal(size(mask), dims)
        warning('PlotEventVariableKernelTraces:MaskSizeMismatch', ...
            'ROI "%s" mask size mismatch (expected %dx%d).', roi_name, dims(1), dims(2));
        return;
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
    error('PlotEventVariableKernelTraces:InvalidMask', ...
        'Mask file %s missing ROI_info or mask variable.', path_str);
end
if ~isequal(size(mask), dims)
    error('PlotEventVariableKernelTraces:MaskDims', ...
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
title(ax, ttl, 'FontSize', 14, 'FontWeight', 'bold');
end

function base_rgb = build_mask_background(mask_shape)
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
if nargin < 1
    m = 256;
end
r = [linspace(0, 1, m/2)'; ones(m/2, 1)];
g = [linspace(0, 1, m/2)'; linspace(1, 0, m/2)'];
b = [ones(m/2, 1); linspace(1, 0, m/2)'];
cmap = [r, g, b];
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

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end
