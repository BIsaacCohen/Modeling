function PlotCVR2bySlope_DiagnosticsROI(results, roi_name, opts)
% PlotCVR2bySlope_DiagnosticsROI Diagnostic plot of slope classification
%
%   Shows the original fluorescence trace and smoothed derivative for a
%   specific ROI, with regions colored by slope classification (rising/falling).
%   Used to validate the slope detection parameters.
%
%   PlotCVR2bySlope_DiagnosticsROI(results, roi_name, opts)
%
%   INPUTS:
%       results - Results struct from TemporalModelFull
%       roi_name - Name of ROI to visualize (string)
%       opts - Optional struct with fields:
%           .time_start_sec     - Start time in seconds (default: 0)
%           .time_window_sec    - Duration to display in seconds (default: 500)
%           .smooth_window      - Smoothing window in frames (default: 5)
%           .slope_threshold    - Slope threshold for classification (default: 0.1)
%           .sampling_rate      - Sampling rate in Hz (default: from metadata)

if nargin < 2 || isempty(roi_name)
    error('PlotCVR2bySlope_DiagnosticsROI:MissingROI', ...
        'ROI name is required');
end

if nargin < 3 || isempty(opts)
    opts = struct();
end

% Validate results
if ~isfield(results, 'predictions') || ~isfield(results, 'metadata')
    error('PlotCVR2bySlope_DiagnosticsROI:InvalidResults', ...
        'Results structure is incomplete');
end

% Set defaults
if ~isfield(opts, 'time_start_sec')
    opts.time_start_sec = 0;
end
if ~isfield(opts, 'time_window_sec')
    opts.time_window_sec = 500;
end
if ~isfield(opts, 'smooth_window')
    opts.smooth_window = 5;
end
if ~isfield(opts, 'slope_threshold')
    opts.slope_threshold = 0.1;
end
if ~isfield(opts, 'sampling_rate')
    if isfield(results.metadata, 'sampling_rate')
        opts.sampling_rate = results.metadata.sampling_rate;
    else
        opts.sampling_rate = 30;  % default fallback
    end
end

% Extract data
Y_actual = results.predictions.Y_actual;
roi_names = results.metadata.roi_names;

% Find ROI index
roi_idx = find(strcmpi(roi_names, roi_name), 1);
if isempty(roi_idx)
    error('PlotCVR2bySlope_DiagnosticsROI:ROINotFound', ...
        'ROI "%s" not found. Available ROIs: %s', ...
        roi_name, strjoin(roi_names, ', '));
end

% Extract data for this ROI
y_actual = Y_actual(:, roi_idx);

% Time conversion
start_frame = max(1, round(opts.time_start_sec * opts.sampling_rate) + 1);
n_frames = round(opts.time_window_sec * opts.sampling_rate);
end_frame = min(numel(y_actual), start_frame + n_frames - 1);
frame_indices = start_frame:end_frame;
actual_frames = numel(frame_indices);

% Extract time window
y_actual_window = y_actual(frame_indices);
time_vec = (frame_indices - 1) / opts.sampling_rate;

% Smooth
y_smoothed = movmean(y_actual_window, opts.smooth_window, 'Endpoints', 'shrink');

% Calculate slope
slopes = diff(y_smoothed);  % Reduced by 1
time_slopes = time_vec(1:end-1);

% Classify
rising_mask = slopes > opts.slope_threshold;
falling_mask = slopes < -opts.slope_threshold;
flat_mask = ~rising_mask & ~falling_mask;

fprintf('Slope classification for %s (time %.1f-%.1f sec):\n', ...
    roi_name, time_vec(1), time_vec(end));
fprintf('  Rising frames: %d (%.1f%%)\n', sum(rising_mask), 100*sum(rising_mask)/numel(slopes));
fprintf('  Falling frames: %d (%.1f%%)\n', sum(falling_mask), 100*sum(falling_mask)/numel(slopes));
fprintf('  Flat frames: %d (%.1f%%)\n', sum(flat_mask), 100*sum(flat_mask)/numel(slopes));

% Create figure
fig = figure('Name', sprintf('Slope Diagnostics: %s', roi_name), ...
    'Position', [100 100 1200 700]);

% Top panel: Original trace
ax1 = subplot(2, 1, 1);
hold(ax1, 'on');
plot(ax1, time_vec, y_actual_window, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Actual');
plot(ax1, time_vec, y_smoothed, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('Smoothed (window=%d)', opts.smooth_window));

% Shade rising regions
for i = 1:numel(slopes)
    if rising_mask(i)
        patch(ax1, [time_slopes(i), time_slopes(i)+1/opts.sampling_rate, ...
                    time_slopes(i)+1/opts.sampling_rate, time_slopes(i)], ...
              [ylim(ax1) fliplr(ylim(ax1))], [0.2 0.8 0.2], ...
              'EdgeColor', 'none', 'FaceAlpha', 0.15, 'HandleVisibility', 'off');
    end
end

% Shade falling regions
for i = 1:numel(slopes)
    if falling_mask(i)
        patch(ax1, [time_slopes(i), time_slopes(i)+1/opts.sampling_rate, ...
                    time_slopes(i)+1/opts.sampling_rate, time_slopes(i)], ...
              [ylim(ax1) fliplr(ylim(ax1))], [0.8 0.2 0.2], ...
              'EdgeColor', 'none', 'FaceAlpha', 0.15, 'HandleVisibility', 'off');
    end
end

hold(ax1, 'off');
ylabel(ax1, sprintf('%s (z-scored)', roi_name), 'FontSize', 12, 'Interpreter', 'none');
title(ax1, sprintf('Fluorescence Trace: %s', roi_name), 'FontSize', 14, 'FontWeight', 'bold');
legend(ax1, 'Location', 'best', 'FontSize', 10);
grid(ax1, 'on');
set(ax1, 'Box', 'off');
ax1.GridAlpha = 0.3;

% Bottom panel: Derivative (slope)
ax2 = subplot(2, 1, 2);
hold(ax2, 'on');

% Plot baseline
axline = yline(ax2, 0, 'k--', 'LineWidth', 1);
axline.HandleVisibility = 'off';
axline_high = yline(ax2, opts.slope_threshold, 'g--', 'LineWidth', 1);
axline_high.HandleVisibility = 'off';
axline_low = yline(ax2, -opts.slope_threshold, 'r--', 'LineWidth', 1);
axline_low.HandleVisibility = 'off';

% Plot slopes
plot(ax2, time_slopes, slopes, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Slope (derivative)');

% Shade regions
for i = 1:numel(slopes)
    if rising_mask(i)
        patch(ax2, [time_slopes(i), time_slopes(i)+1/opts.sampling_rate, ...
                    time_slopes(i)+1/opts.sampling_rate, time_slopes(i)], ...
              [min(slopes)-0.01, min(slopes)-0.01, max(slopes)+0.01, max(slopes)+0.01], ...
              [0.2 0.8 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.15, 'HandleVisibility', 'off');
    end
end

for i = 1:numel(slopes)
    if falling_mask(i)
        patch(ax2, [time_slopes(i), time_slopes(i)+1/opts.sampling_rate, ...
                    time_slopes(i)+1/opts.sampling_rate, time_slopes(i)], ...
              [min(slopes)-0.01, min(slopes)-0.01, max(slopes)+0.01, max(slopes)+0.01], ...
              [0.8 0.2 0.2], 'EdgeColor', 'none', 'FaceAlpha', 0.15, 'HandleVisibility', 'off');
    end
end

hold(ax2, 'off');
xlabel(ax2, 'Time (seconds)', 'FontSize', 12);
ylabel(ax2, 'Slope (z/frame)', 'FontSize', 12);
title(ax2, sprintf('Smoothed Derivative (threshold=±%.2f)', opts.slope_threshold), ...
    'FontSize', 14, 'FontWeight', 'bold');
legend(ax2, 'Location', 'best', 'FontSize', 10);
grid(ax2, 'on');
set(ax2, 'Box', 'off');
ax2.GridAlpha = 0.3;

% Link axes for zooming
linkaxes([ax1, ax2], 'x');

% Add super title
if exist('sgtitle', 'file') == 2
    sgtitle(sprintf('Slope Classification Diagnostics: %s (threshold=%.2f, smooth=%d)', ...
        roi_name, opts.slope_threshold, opts.smooth_window), ...
        'FontSize', 14, 'Interpreter', 'none');
end

end
