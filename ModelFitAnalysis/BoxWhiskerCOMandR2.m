function figs = BoxWhiskerCOMandR2(results, Regions, opts)
% BoxWhiskerCOMandR2 - Box/whisker plots of CoM lags and CV R² (per-fold) with overlaid points
%
%   figs = BoxWhiskerCOMandR2(results)
%   figs = BoxWhiskerCOMandR2(results, Regions)
%   figs = BoxWhiskerCOMandR2(results, Regions, opts_or_markerSize)
%
% INPUTS
%   results   - Output struct from TemporalModelFull
%   Regions   - Cell array of ROI names to include (default: {'AU_L','M2_L'})
%   opts      - Optional struct; supports .marker_size (point size, default 48).
%               Alternatively, pass a scalar as the third argument to set marker size directly.
%
% OUTPUT
%   figs      - Struct of figure handles:
%                  .com_sec      - CoM lag (seconds)
%                  .r2_cv        - CV R²
%                  .corr_cv      - CV Pearson correlation

if nargin < 2 || isempty(Regions)
    Regions = {'AU_L', 'M2_L'};
end
if nargin < 3
    opts = struct();
else
    % Allow scalar marker size as shorthand
    if isnumeric(opts) && isscalar(opts)
        opts = struct('marker_size', opts);
    elseif ~isstruct(opts)
        error('BoxWhiskerCOMandR2:InvalidOpts', ...
            'Third argument must be a struct or a scalar marker size.');
    end
end

if ischar(Regions)
    Regions = {Regions};
end

if ~iscellstr(Regions)
    error('BoxWhiskerCOMandR2:InvalidRegions', ...
        'Regions must be a cell array of character vectors.');
end

if ~isstruct(results) || ~isfield(results, 'temporal_kernels') || ~isfield(results, 'performance')
    error('BoxWhiskerCOMandR2:InvalidResults', ...
        'results must be a TemporalModelFull output struct with temporal_kernels and performance fields.');
end

roi_names = results.metadata.roi_names(:);
n_regions = numel(Regions);
roi_idx = zeros(n_regions, 1);

for i = 1:n_regions
    match_idx = find(strcmpi(roi_names, Regions{i}), 1);
    if isempty(match_idx)
        error('BoxWhiskerCOMandR2:ROINotFound', ...
            'Requested region "%s" not found in results.metadata.roi_names.', Regions{i});
    end
    roi_idx(i) = match_idx;
end

% Lag times and analysis window
if ~isfield(results.temporal_kernels(1), 'lag_times_sec')
    error('BoxWhiskerCOMandR2:MissingLagTimes', 'results.temporal_kernels.lag_times_sec is required.');
end
lag_times = results.temporal_kernels(1).lag_times_sec(:);
sampling_rate = results.metadata.sampling_rate;

if isfield(results.metadata, 'analysis_window_sec') && numel(results.metadata.analysis_window_sec) == 2
    win_bounds = results.metadata.analysis_window_sec;
    window_mask = (lag_times >= win_bounds(1)) & (lag_times <= win_bounds(2));
else
    window_mask = true(size(lag_times));
end
lag_times_window = lag_times(window_mask);

% Gather per-fold metrics
com_sec_cells = cell(n_regions, 1);
r2_cells = cell(n_regions, 1);
corr_cells = cell(n_regions, 1);

% Prefer block-wise resampling CoM if present; otherwise fall back to CV train-fit kernels
if isfield(results, 'resampling') && isfield(results.resampling, 'com_lag_sec_blocks') ...
        && ~isempty(results.resampling.com_lag_sec_blocks)
    com_matrix_sec = results.resampling.com_lag_sec_blocks;
    n_folds = size(com_matrix_sec, 2);
    for i = 1:n_regions
        com_sec_cells{i} = double(com_matrix_sec(roi_idx(i), :)).';
    end
else
    for i = 1:n_regions
        idx = roi_idx(i);

        % CoM per fold from stored beta_cv_folds
        beta_folds = results.temporal_kernels(idx).beta_cv_folds;  % [n_lags x n_folds]
        if size(beta_folds, 1) ~= numel(lag_times)
            error('BoxWhiskerCOMandR2:LagMismatch', ...
                'lag_times_sec length (%d) does not match beta_cv_folds rows (%d).', ...
                numel(lag_times), size(beta_folds, 1));
        end

        n_folds = size(beta_folds, 2);
        com_sec = zeros(n_folds, 1);

        for f = 1:n_folds
            beta_win = beta_folds(window_mask, f);
            positive_beta = max(beta_win, 0);
            total_wt = sum(positive_beta);
            if total_wt > 0
                com_val_sec = sum(lag_times_window .* positive_beta) / total_wt;
            else
                com_val_sec = 0;
            end
            com_sec(f) = com_val_sec;
        end

        com_sec_cells{i} = com_sec;
    end
end

for i = 1:n_regions
    if ~isfield(results.performance, 'R2_cv_folds')
        error('BoxWhiskerCOMandR2:MissingR2Folds', 'results.performance.R2_cv_folds is missing.');
    end
    r2_cells{i} = results.performance(roi_idx(i)).R2_cv_folds(:);
    if ~isfield(results.performance, 'Corr_cv_folds')
        error('BoxWhiskerCOMandR2:MissingCorrFolds', 'results.performance.Corr_cv_folds is missing.');
    end
    corr_cells{i} = results.performance(roi_idx(i)).Corr_cv_folds(:);
end

% Build plots
figs = struct();
figs.com_sec = make_box(com_sec_cells, Regions, 'CoM Lag (s)', ...
    'Center-of-Mass Lag (seconds)', numel(com_sec_cells{1}), get_marker_size(opts));
figs.r2_cv = make_box(r2_cells, Regions, 'Cross-validated R^2', ...
    'Model Fit (CV R^2)', numel(r2_cells{1}), get_marker_size(opts));
figs.corr_cv = make_box(corr_cells, Regions, 'Pearson r (CV)', ...
    'Model Fit (CV Pearson r)', numel(corr_cells{1}), get_marker_size(opts));

fprintf('BoxWhiskerCOMandR2: plotted %d regions using %d folds (CoM in seconds and CV R^2).\n', ...
    n_regions, numel(com_sec_cells{1}));

end

%% Helper: box/whisker with overlaid points
function fig = make_box(values_cell, region_labels, y_label, title_str, n_folds, marker_size)
    fig = figure('Name', title_str, 'NumberTitle', 'off');
    hold on;

    region_labels = region_labels(:);
    n_groups = numel(region_labels);

    % Flatten values and labels for plotting
    vals_all = [];
    labels_all = {};
    for i = 1:n_groups
        v = values_cell{i};
        if iscell(v)
            v = cell2mat(v(:));
        end
        if ~isnumeric(v)
            error('BoxWhiskerCOMandR2:NonNumericValues', ...
                'Values for region %s are class %s (expected numeric).', ...
                region_labels{i}, class(v));
        end
        v = double(v(:));
        v = v(isfinite(v));

        vals_all = [vals_all; v];
        labels_all = [labels_all; repmat(region_labels(i), numel(v), 1)];
    end

    if isempty(vals_all)
        warning('BoxWhiskerCOMandR2:NoData', 'No finite data to plot.');
        hold off;
        return;
    end

    labels_all = cellstr(labels_all);
    cats = categorical(labels_all, region_labels, 'Ordinal', true);
    group_codes = double(categorical(region_labels, region_labels, 'Ordinal', true));

    positions = 1:n_groups;

    % Overlay individual points with jitter (draw first so boxplot sits on top)
    for i = 1:n_groups
        vals_i = vals_all(strcmp(labels_all, region_labels{i}));
        if isempty(vals_i)
            continue;
        end
        x_pos = positions(i);
        for f = 1:numel(vals_i)
            scatter(x_pos + randn * 0.05, vals_i(f), marker_size, f, 'filled', ...
                'MarkerEdgeColor', 'none', 'MarkerFaceAlpha', 0.8);
        end
    end

    % Use classic boxplot with whiskers set to min/max (Whisker=Inf)
    boxplot(vals_all, labels_all, 'Whisker', Inf, 'Symbol', '');

    yline(0, 'k:', 'LineWidth', 1);
    grid off;
    ylabel(y_label, 'FontSize', 12);
    title(title_str, 'FontSize', 13, 'Interpreter', 'none');
    set(gca, 'FontSize', 10);

    colormap(parula(max(n_folds, 2)));
    caxis([1, n_folds]);
    cb = colorbar;
    cb.Label.String = 'Fold Number';
    cb.Label.FontSize = 10;

    hold off;
end

function ms = get_marker_size(opts)
    if nargin >= 1 && isstruct(opts) && isfield(opts, 'marker_size') && ~isempty(opts.marker_size)
        ms = opts.marker_size;
    else
        ms = 48;  % default larger size
    end
end
