function PlotCVR2bySlope(results, opts)
% PlotCVR2bySlope Brain maps of CV R² during rising vs falling neural activity
%
%   Analyzes how well the temporal model predicts fluorescence during
%   periods of increasing neural activity (positive slope) versus decreasing
%   activity (negative slope). This reveals whether the model captures
%   excitation and inhibition dynamics equally well.
%
%   PlotCVR2bySlope(results, opts)
%
%   REQUIRED INPUTS:
%       results - Results struct from TemporalModelFull containing:
%                 .predictions.Y_actual [n_valid x n_rois] (z-scored fluorescence)
%                 .predictions.Y_pred [n_valid x n_rois] (model predictions)
%                 .metadata.roi_names {1 x n_rois}
%                 .metadata.source_roi_file (struct with neural_roi_file path)
%
%   OPTIONAL INPUTS (opts struct):
%       smooth_window       - Moving average window for smoothing (frames, default: 5)
%       slope_threshold     - Minimum |slope| to classify as rising/falling (default: 0.1)
%       min_timepoints      - Minimum timepoints required per ROI per phase (default: 20)
%       save_figure         - Save figure to file (default: false)
%       output_file         - Path for saved figure (default: auto-generated PNG)
%       show_figure         - Display figure (default: true)
%       figure_title        - Custom title (default: auto-generated)
%
%   OUTPUT:
%       Generates a 1x3 brain map figure:
%           Panel 1: CV R² during rising/excitation phases
%           Panel 2: CV R² during falling/inhibition phases
%           Panel 3: Difference map (Rising R² - Falling R²)
%       First two panels share the same color scale for direct comparison.
%       Difference panel uses red-blue diverging scale centered on zero.
%
%   ALGORITHM:
%       1. For each ROI independently:
%          - Smooth Y_actual with movmean (smooth_window frames)
%          - Compute derivative: slope = diff(Y_smoothed)
%          - Classify timepoints: rising (slope > threshold), falling (slope < -threshold)
%          - Calculate R² separately for rising and falling timepoints
%       2. Map R² values onto cortical ROI masks
%       3. Plot side-by-side brain maps with unified color scale
%
%   NOTES:
%       - Edge handling: diff() reduces length by 1; we exclude first timepoint
%       - ROIs with insufficient rising or falling timepoints show NaN (background)
%       - Z-scored data means slope threshold is in standard deviation units

%% ================= INPUT VALIDATION & SETUP =================

if nargin < 1 || isempty(results)
    error('PlotCVR2bySlope:MissingResults', ...
        'Results struct from TemporalModelFull is required.');
end

% Validate results structure
required_fields = {'predictions', 'metadata'};
for i = 1:numel(required_fields)
    if ~isfield(results, required_fields{i})
        error('PlotCVR2bySlope:MissingField', ...
            'results.%s is required', required_fields{i});
    end
end

% Validate predictions subfields
if ~isfield(results.predictions, 'Y_actual') || ...
   ~isfield(results.predictions, 'Y_pred')
    error('PlotCVR2bySlope:MissingPredictions', ...
        'results.predictions must contain Y_actual and Y_pred');
end

% Validate metadata
if ~isfield(results.metadata, 'roi_names') || ...
   ~isfield(results.metadata, 'source_roi_file')
    error('PlotCVR2bySlope:MissingMetadata', ...
        'results.metadata must contain roi_names and source_roi_file');
end

% Extract core data
Y_actual = results.predictions.Y_actual;  % [n_valid x n_rois]
Y_pred = results.predictions.Y_pred;      % [n_valid x n_rois]
roi_names = results.metadata.roi_names;   % {1 x n_rois}

[n_valid, n_rois] = size(Y_actual);

% Validate dimensions
if ~isequal(size(Y_pred), [n_valid, n_rois])
    error('PlotCVR2bySlope:DimensionMismatch', ...
        'Y_pred and Y_actual must have same dimensions');
end

if numel(roi_names) ~= n_rois
    error('PlotCVR2bySlope:ROICountMismatch', ...
        'Number of ROI names (%d) does not match data columns (%d)', ...
        numel(roi_names), n_rois);
end

% Populate defaults
if nargin < 2 || isempty(opts)
    opts = struct();
end

defaults = struct(...
    'smooth_window', 5, ...
    'slope_threshold', 0.1, ...
    'min_timepoints', 20, ...
    'save_figure', false, ...
    'output_file', '', ...
    'show_figure', true, ...
    'figure_title', '');

opts = populate_defaults(opts, defaults);

fprintf('\nGenerating CV R² slope analysis...\n');

%% ================= PER-ROI ANALYSIS =================

% Preallocate output arrays
R2_rising = nan(n_rois, 1);
R2_falling = nan(n_rois, 1);
n_rising = zeros(n_rois, 1);
n_falling = zeros(n_rois, 1);
n_excluded = zeros(n_rois, 1);

fprintf('Analyzing slope-dependent R² for %d ROIs...\n', n_rois);

for roi = 1:n_rois
    % Extract data for this ROI
    y_actual_roi = Y_actual(:, roi);  % [n_valid x 1]
    y_pred_roi = Y_pred(:, roi);      % [n_valid x 1]

    % Smooth the actual fluorescence trace
    y_smoothed = movmean(y_actual_roi, opts.smooth_window, 'Endpoints', 'shrink');

    % Calculate derivative (slope)
    % Note: diff() reduces length from n_valid to (n_valid-1)
    slopes = diff(y_smoothed);  % [n_valid-1 x 1]

    % Align Y_actual and Y_pred with slopes by excluding first timepoint
    y_actual_aligned = y_actual_roi(2:end);  % [n_valid-1 x 1]
    y_pred_aligned = y_pred_roi(2:end);      % [n_valid-1 x 1]

    % Classify timepoints by slope
    rising_mask = slopes > opts.slope_threshold;
    falling_mask = slopes < -opts.slope_threshold;

    n_rising(roi) = sum(rising_mask);
    n_falling(roi) = sum(falling_mask);
    n_excluded(roi) = sum(~rising_mask & ~falling_mask);

    % Calculate R² for rising phase
    if n_rising(roi) >= opts.min_timepoints
        y_act_rising = y_actual_aligned(rising_mask);
        y_pred_rising = y_pred_aligned(rising_mask);
        R2_rising(roi) = calculate_r2(y_act_rising, y_pred_rising);
    else
        % Insufficient data; leave as NaN
        R2_rising(roi) = nan;
    end

    % Calculate R² for falling phase
    if n_falling(roi) >= opts.min_timepoints
        y_act_falling = y_actual_aligned(falling_mask);
        y_pred_falling = y_pred_aligned(falling_mask);
        R2_falling(roi) = calculate_r2(y_act_falling, y_pred_falling);
    else
        R2_falling(roi) = nan;
    end
end

fprintf('  Rising phase: mean R² = %.3f (valid ROIs: %d/%d)\n', ...
    nanmean(R2_rising), sum(~isnan(R2_rising)), n_rois);
fprintf('  Falling phase: mean R² = %.3f (valid ROIs: %d/%d)\n', ...
    nanmean(R2_falling), sum(~isnan(R2_falling)), n_rois);

%% ================= LOAD ROI SPATIAL DATA =================

source = results.metadata.source_roi_file;
neural_path = '';
if isfield(source, 'neural_roi_file')
    neural_path = source.neural_roi_file;
elseif isfield(source, 'neural_rois')
    neural_path = source.neural_rois;
end

if isempty(neural_path) || exist(neural_path, 'file') ~= 2
    error('PlotCVR2bySlope:MissingROIFile', ...
        'Neural ROI file not found: %s', neural_path);
end

% Load spatial data
spatial = load(neural_path, '-mat');
if ~isfield(spatial, 'ROI_info') || ~isfield(spatial, 'img_info')
    error('PlotCVR2bySlope:InvalidROIFile', ...
        'ROI file %s missing ROI_info/img_info fields', neural_path);
end

img_info = spatial.img_info;
roi_info = spatial.ROI_info;

if ~isfield(img_info, 'imageData')
    error('PlotCVR2bySlope:NoImageData', ...
        'img_info.imageData missing in %s', neural_path);
end

% Determine image dimensions
if isfield(roi_info(1), 'Stats') && isfield(roi_info(1).Stats, 'ROI_binary_mask')
    dims = size(roi_info(1).Stats.ROI_binary_mask);
else
    dims = size(img_info.imageData);
end

% Extract ROI names from spatial file
roi_names_source = arrayfun(@(r)char(r.Name), roi_info, 'UniformOutput', false);

%% ================= BUILD BRAIN MAPS =================

% Initialize brain maps
rising_map = nan(dims);
falling_map = nan(dims);

% Convert R² to percentage
R2_rising_pct = R2_rising * 100;
R2_falling_pct = R2_falling * 100;

n_assigned = 0;
n_skipped = 0;

for roi = 1:n_rois
    roi_name = roi_names{roi};

    % Find matching ROI in spatial data
    match_idx = find(strcmpi(roi_names_source, roi_name), 1);
    if isempty(match_idx)
        error('PlotCVR2bySlope:ROIMaskMissing', ...
            'ROI "%s" not found in %s', roi_name, neural_path);
    end

    roi_struct = roi_info(match_idx);
    if ~isfield(roi_struct, 'Stats') || ...
       ~isfield(roi_struct.Stats, 'ROI_binary_mask')
        error('PlotCVR2bySlope:MaskMissing', ...
            'ROI "%s" missing Stats.ROI_binary_mask', roi_name);
    end

    mask = roi_struct.Stats.ROI_binary_mask;
    if ~isequal(size(mask), dims)
        error('PlotCVR2bySlope:MaskSizeMismatch', ...
            'ROI "%s" mask size mismatch (expected %dx%d)', ...
            roi_name, dims(1), dims(2));
    end

    % Check for overlap
    overlap_rising = ~isnan(rising_map) & mask;
    overlap_falling = ~isnan(falling_map) & mask;
    if any(overlap_rising(:)) || any(overlap_falling(:))
        error('PlotCVR2bySlope:OverlappingMasks', ...
            'ROI "%s" overlaps with another ROI', roi_name);
    end

    % Assign values to maps (NaN values stay as NaN)
    rising_val = R2_rising_pct(roi);
    falling_val = R2_falling_pct(roi);

    if ~isnan(rising_val)
        rising_map(mask) = rising_val;
    end

    if ~isnan(falling_val)
        falling_map(mask) = falling_val;
    end

    if ~isnan(rising_val) || ~isnan(falling_val)
        n_assigned = n_assigned + 1;
    else
        n_skipped = n_skipped + 1;
    end
end

fprintf('  Brain maps: %d ROIs assigned, %d skipped (insufficient data)\n', ...
    n_assigned, n_skipped);

%% ================= APPLY MASKS =================

% Load optional masks
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

% Combine into display mask
mask_shape = brain_mask & ~vascular_mask;

% Apply display mask
rising_map(~mask_shape) = nan;
falling_map(~mask_shape) = nan;

% Create difference map (rising - falling)
diff_map = rising_map - falling_map;

% Build black background RGB image
base_rgb = build_mask_background(mask_shape);

% Determine unified color scale for rising/falling (CRITICAL for comparison)
all_r2_values = [R2_rising_pct(~isnan(R2_rising_pct)); ...
                 R2_falling_pct(~isnan(R2_falling_pct))];

if isempty(all_r2_values)
    r2_limits = [0 1];  % Default if no valid data
    warning('PlotCVR2bySlope:NoValidData', ...
        'No valid R² values found; using default color scale [0, 1]');
else
    r2_max = max(all_r2_values);
    r2_limits = [0, r2_max];
end

% Determine symmetric color scale for difference (centered on zero)
diff_values = diff_map(~isnan(diff_map));
if isempty(diff_values)
    diff_limits = [-1 1];
    warning('PlotCVR2bySlope:NoDiffData', ...
        'No valid difference values found; using default color scale [-1, 1]');
else
    diff_span = max(abs(diff_values));
    diff_limits = [-diff_span, diff_span];
end

fprintf('  Unified R² color scale: [%.1f, %.1f]%%\n', r2_limits(1), r2_limits(2));
fprintf('  Difference color scale: [%.1f, %.1f]%%\n', diff_limits(1), diff_limits(2));

%% ================= GENERATE FIGURE =================

% Create figure
if isempty(opts.figure_title)
    fig_title = sprintf('CV R^2 by Slope: Rising vs Falling (slope threshold=%.2f, smooth=%d frames)', ...
        opts.slope_threshold, opts.smooth_window);
else
    fig_title = opts.figure_title;
end

fig = figure('Name', 'CV R2 by Slope Analysis', ...
    'Position', [100 200 1800 500]);

% Use tiledlayout if available (R2019b+), otherwise subplot
use_tiled = exist('tiledlayout', 'file') == 2;
layout = [];
if use_tiled
    layout = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
end

% Colormap (parula for consistency with other R² plots)
cmap_r2 = parula(256);

% Panel 1: Rising/Excitation phase
if use_tiled
    ax1 = nexttile(layout, 1);
else
    ax1 = subplot(1, 3, 1);
end
plot_metric_map(ax1, base_rgb, rising_map, cmap_r2, r2_limits, ...
    'CV R^2 - Rising/Excitation', mask_shape);

% Panel 2: Falling/Inhibition phase
if use_tiled
    ax2 = nexttile(layout, 2);
else
    ax2 = subplot(1, 3, 2);
end
plot_metric_map(ax2, base_rgb, falling_map, cmap_r2, r2_limits, ...
    'CV R^2 - Falling/Inhibition', mask_shape);

% Panel 3: Difference (Rising - Falling)
if use_tiled
    ax3 = nexttile(layout, 3);
else
    ax3 = subplot(1, 3, 3);
end
cmap_diff = redbluecmap(256);  % Red-white-blue diverging colormap
plot_metric_map(ax3, base_rgb, diff_map, cmap_diff, diff_limits, ...
    'R^2 Difference (Rising - Falling)', mask_shape);

% Add super title
add_super_title(fig, layout, fig_title);

% Display or hide
if ~opts.show_figure
    set(fig, 'Visible', 'off');
end

% Save figure
if opts.save_figure
    if isempty(opts.output_file)
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        output_file = fullfile(pwd, sprintf('CVR2bySlope_%s.png', timestamp));
    else
        output_file = opts.output_file;
    end

    fprintf('  Saving figure to: %s\n', output_file);
    print(fig, output_file, '-dpng', '-r300');
end

fprintf('PlotCVR2bySlope complete.\n\n');

end

%% ================= HELPER FUNCTIONS =================

function R2 = calculate_r2(y_actual, y_pred)
% calculate_r2 Compute coefficient of determination (R²)
%
%   R2 = calculate_r2(y_actual, y_pred)
%
%   Inputs:
%       y_actual - Actual values [N x 1]
%       y_pred   - Predicted values [N x 1]
%
%   Output:
%       R2 - Coefficient of determination, bounded at 0 (no negative R²)

    % Total Sum of Squares
    TSS = sum((y_actual - mean(y_actual)).^2);

    % Residual Sum of Squares
    RSS = sum((y_actual - y_pred).^2);

    % R² = 1 - RSS/TSS, bounded at 0
    R2 = max(0, 1 - RSS / TSS);
end

function opts_out = populate_defaults(opts_in, defaults)
% Populate missing options with default values
    opts_out = defaults;
    if isempty(opts_in)
        return;
    end

    fn = fieldnames(opts_in);
    for i = 1:numel(fn)
        if isfield(defaults, fn{i})
            opts_out.(fn{i}) = opts_in.(fn{i});
        end
    end
end

function mask = load_optional_mask(source, field_name, dims)
% Load optional mask from file (copied from PlotPosterNoCategoryTemporalModelFull.m)
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
        error('PlotCVR2bySlope:InvalidMask', ...
            'Mask file %s missing ROI_info or mask variable.', path_str);
    end
    if ~isequal(size(mask), dims)
        error('PlotCVR2bySlope:MaskDims', ...
            'Mask file %s size mismatch (expected %dx%d).', path_str, dims(1), dims(2));
    end
end

function base_rgb = build_mask_background(mask_shape)
% Background stays black everywhere; mask outline supplies structure
% (copied from PlotPosterNoCategoryTemporalModelFull.m)
    base_gray = zeros(size(mask_shape));
    base_rgb = repmat(base_gray, 1, 1, 3);
end

function plot_metric_map(ax, base_rgb, metric_map, cmap, clim, ttl, mask_shape, is_categorical, draw_mask_outline)
% Plot metric map on brain background with colorbar
% (adapted from PlotPosterNoCategoryTemporalModelFull.m)
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

function plot_mask_outline(ax, mask_shape)
% Plot outline of brain mask (copied from PlotPosterNoCategoryTemporalModelFull.m)
    contour(ax, mask_shape, [0.5 0.5], 'Color', [1 1 1], 'LineWidth', 1.2);
end

function add_super_title(fig_handle, layout_handle, title_str)
% Add super title to figure (copied from PlotPosterNoCategoryTemporalModelFull.m)
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
% Used for diverging maps where negative=red, zero=white, positive=blue
    if nargin < 1
        m = 256;
    end

    % Red for negative, blue for positive, white at zero
    r = [ones(m/2, 1); linspace(1, 0, m/2)'];
    g = [linspace(0, 1, m/2)'; linspace(1, 0, m/2)'];
    b = [linspace(0, 1, m/2)'; ones(m/2, 1)];

    cmap = [r, g, b];
end
