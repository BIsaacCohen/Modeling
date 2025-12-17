function [dff_phasic, diagnostics] = dFF_phasic(trace, sampling_rate, opts)
% dFF_phasic Compute phasic df/f by removing slow baseline components
%
%   Removes slow drift and baseline fluctuations by low-pass filtering the
%   raw trace to estimate a time-varying baseline F₀(t), then computes:
%       df/f_phasic = (F(t) - F₀(t)) / F₀(t)
%
%   This isolates phasic activity (fast transients) while ignoring slow
%   trends like photobleaching.
%
%   [dff_phasic, diagnostics] = dFF_phasic(trace, sampling_rate, opts)
%
%   REQUIRED INPUTS:
%       trace           - 1D fluorescence timeseries (already ROI-averaged)
%       sampling_rate   - Sampling frequency in Hz
%
%   OPTIONAL INPUTS (opts struct):
%       cutoff_freq     - Low-pass cutoff frequency in Hz (default 0.1 Hz = 10s)
%       filter_order    - Butterworth filter order (default 4)
%       show_plot       - Generate validation plot (default true)
%       roi_name        - ROI name for plot title (default 'ROI')
%       min_baseline    - Minimum baseline value to prevent division issues (default 1e-6)
%       zoom_seconds    - If provided, x-axis zoom window [0, zoom_seconds]; empty shows full trace
%
%   OUTPUTS:
%       dff_phasic      - Phasic df/f timeseries (same length as input)
%       diagnostics     - Struct containing:
%           .baseline_F0       - Low-passed baseline F₀(t)
%           .cutoff_freq       - Cutoff frequency used
%           .filter_order      - Filter order used
%           .nyquist_freq      - Nyquist frequency
%           .filter_valid      - Whether filter design succeeded
%
%   EXAMPLE:
%       % After extracting ROI trace
%       neural_trace = extract_roi_trace(...);
%       [dff_phasic, diag] = dFF_phasic(neural_trace, 16.3, ...
%           struct('cutoff_freq', 0.1, 'show_plot', true, 'roi_name', 'AU_L'));
%
%   METHOD:
%       1. Design Butterworth low-pass filter at cutoff_freq
%       2. Apply zero-phase filtering (filtfilt) to extract baseline F₀(t)
%       3. Compute phasic df/f = (F - F₀) / F₀
%       4. Generate validation plot if requested
%
%   NOTES:
%       - Uses zero-phase filtering to avoid temporal shifts
%       - Handles edge effects automatically via filtfilt padding
%       - Validates filter stability (cutoff must be < Nyquist frequency)
%       - Clamps baseline to minimum value to prevent division by zero

% Parse inputs
if nargin < 3 || isempty(opts)
    opts = struct();
end

defaults = struct(...
    'cutoff_freq', 0.1, ...
    'filter_order', 4, ...
    'show_plot', true, ...
    'roi_name', 'ROI', ...
    'min_baseline', 1e-6, ...
    'baseline_method', 'butterworth', ...
    'baseline_window_seconds', 600, ...
    'baseline_percentile', 10, ...
    'zoom_seconds', []);

opts = populate_defaults(opts, defaults);

% Validate inputs
assert(isvector(trace), 'trace must be a 1D vector');
assert(sampling_rate > 0, 'sampling_rate must be positive');
assert(opts.cutoff_freq > 0, 'cutoff_freq must be positive');
assert(opts.filter_order > 0 && mod(opts.filter_order, 1) == 0, ...
    'filter_order must be a positive integer');

% Ensure column vector
trace = trace(:);
n_samples = length(trace);

fprintf('=== dFF_phasic: Phasic df/f computation ===\n');
fprintf('  Input trace: %d samples (%.2f s @ %.2f Hz)\n', ...
    n_samples, n_samples / sampling_rate, sampling_rate);
fprintf('  Baseline method: %s\n', opts.baseline_method);
if strcmpi(opts.baseline_method, 'butterworth')
    fprintf('  Cutoff frequency: %.3f Hz (period = %.1f s)\n', ...
        opts.cutoff_freq, 1 / opts.cutoff_freq);
    fprintf('  Filter order: %d\n', opts.filter_order);
end

baseline_method = lower(opts.baseline_method);
nyquist_freq = sampling_rate / 2;
filter_valid = true;

switch baseline_method
    case 'butterworth'
        if opts.cutoff_freq >= nyquist_freq
            error('Cutoff frequency (%.3f Hz) must be less than Nyquist frequency (%.3f Hz)', ...
                opts.cutoff_freq, nyquist_freq);
        end
        try
            Wn = opts.cutoff_freq / nyquist_freq;
            [b, a] = butter(opts.filter_order, Wn, 'low');
            if max(abs(roots(a))) >= 1
                warning('Filter design may be unstable. Reducing order.');
                [b, a] = butter(2, Wn, 'low');
            end
            baseline_F0 = filtfilt(b, a, trace);
            fprintf('  Baseline extraction: filtfilt (zero-phase)\n');
        catch ME
            warning('Filter design failed: %s. Using moving average fallback.', ME.message);
            filter_valid = false;
            window_samples = max(5, round(sampling_rate / max(opts.cutoff_freq, eps)));
            baseline_F0 = movmean(trace, window_samples, 'Endpoints', 'shrink');
            fprintf('  Baseline extraction: movmean (fallback, window=%d)\n', window_samples);
        end
    case 'running_percentile'
        window_seconds = opts.baseline_window_seconds;
        percentile = opts.baseline_percentile;
        fprintf('  Baseline extraction: running percentile (window %.1f s, %.1f%%)\n', ...
            window_seconds, percentile);
        baseline_F0 = running_percentile_baseline(trace, sampling_rate, window_seconds, percentile);
    otherwise
        error('Unknown baseline_method: %s', opts.baseline_method);
end

% Clamp baseline to prevent division by zero or negative values
baseline_F0 = max(baseline_F0, opts.min_baseline);

%% Compute phasic df/f
% df/f = (F - F₀) / F₀
dff_phasic = (trace - baseline_F0) ./ baseline_F0;

fprintf('  Phasic df/f range: [%.4f, %.4f]\n', min(dff_phasic), max(dff_phasic));
fprintf('  Mean phasic df/f: %.4f (should be ~0)\n', mean(dff_phasic));
fprintf('  Std phasic df/f: %.4f\n', std(dff_phasic));

%% Store diagnostics
diagnostics = struct();
diagnostics.baseline_F0 = baseline_F0;
diagnostics.cutoff_freq = opts.cutoff_freq;
diagnostics.filter_order = opts.filter_order;
diagnostics.baseline_method = opts.baseline_method;
diagnostics.nyquist_freq = nyquist_freq;
diagnostics.filter_valid = filter_valid;
diagnostics.n_samples = n_samples;
diagnostics.sampling_rate = sampling_rate;

%% Generate validation plot
if opts.show_plot
    fprintf('  Generating validation plot...\n');
    plot_dff_phasic_validation(trace, baseline_F0, dff_phasic, sampling_rate, opts.roi_name, opts.zoom_seconds);
end

fprintf('=== dFF_phasic: Complete ===\n\n');

end

%% ================= Helper Functions =================

function opts = populate_defaults(opts, defaults)
    fields = fieldnames(defaults);
    for i = 1:numel(fields)
        name = fields{i};
        if ~isfield(opts, name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
    end
end

function plot_dff_phasic_validation(raw_trace, baseline, dff_phasic, fs, roi_name, zoom_seconds)
% plot_dff_phasic_validation Generate 2-panel validation plot
%
%   Top panel: Raw trace (blue) + Baseline F₀ (red)
%   Bottom panel: Phasic df/f output
%   Linked x-axes

    n_samples = length(raw_trace);
    time_vec = (0:n_samples-1) / fs;

    fig_title = sprintf('dFF Phasic Validation: %s', roi_name);
    fig = figure('Name', fig_title, 'Position', [100, 100, 1200, 600]);

    tiled = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tiled, fig_title, 'Interpreter', 'none', 'FontSize', 13, 'FontWeight', 'bold');

    %% Top panel: Raw + Baseline
    ax1 = nexttile(tiled);

    % Plot raw trace
    plot(ax1, time_vec, raw_trace, '-', 'Color', [0.2 0.4 0.8], ...
        'LineWidth', 1.2, 'DisplayName', 'Raw fluorescence');

    hold(ax1, 'on');

    % Plot baseline
    plot(ax1, time_vec, baseline, '-', 'Color', [0.85 0.2 0.2], ...
        'LineWidth', 2, 'DisplayName', 'Baseline F₀ (low-pass)');

    hold(ax1, 'off');

    ylabel(ax1, 'Fluorescence (a.u.)', 'FontSize', 11);
    legend(ax1, 'Location', 'best', 'FontSize', 10);
    grid(ax1, 'on');
    title(ax1, 'Raw Signal and Slow Baseline', 'FontSize', 11);

    % Add stats text
    text(ax1, 0.02, 0.98, sprintf('Raw: μ=%.2f, σ=%.2f\nBaseline: μ=%.2f, σ=%.2f', ...
        mean(raw_trace), std(raw_trace), mean(baseline), std(baseline)), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 9, 'BackgroundColor', 'w', 'EdgeColor', 'k');

    %% Bottom panel: Phasic df/f
    ax2 = nexttile(tiled);

    plot(ax2, time_vec, dff_phasic, '-', 'Color', [0.13 0.55 0.13], ...
        'LineWidth', 1.2, 'DisplayName', 'Phasic df/f');

    hold(ax2, 'on');

    % Zero line
    plot(ax2, [time_vec(1), time_vec(end)], [0, 0], 'k--', ...
        'LineWidth', 1, 'HandleVisibility', 'off');

    hold(ax2, 'off');

    xlabel(ax2, 'Time (s)', 'FontSize', 11);
    ylabel(ax2, 'df/f (phasic)', 'FontSize', 11);
    legend(ax2, 'Location', 'best', 'FontSize', 10);
    grid(ax2, 'on');
    title(ax2, 'Phasic df/f (Slow Components Removed)', 'FontSize', 11);

    % Add stats text
    text(ax2, 0.02, 0.98, sprintf('Phasic df/f: μ=%.4f, σ=%.4f\nRange: [%.4f, %.4f]', ...
        mean(dff_phasic), std(dff_phasic), min(dff_phasic), max(dff_phasic)), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', 9, 'BackgroundColor', 'w', 'EdgeColor', 'k');

    % Link x-axes
    linkaxes([ax1, ax2], 'x');

    % Zoom window if requested; otherwise show full trace
    if ~isempty(zoom_seconds) && zoom_seconds > 0
        xlim(ax1, [0, min(zoom_seconds, time_vec(end))]);
    else
        xlim(ax1, [0, time_vec(end)]);
    end
end

function baseline = running_percentile_baseline(trace, sampling_rate, window_seconds, percentile)
if nargin < 4 || isempty(percentile)
    percentile = 10;
end
if nargin < 3 || isempty(window_seconds)
    window_seconds = 60;
end
window_samples = max(3, round(window_seconds * sampling_rate));
window_samples = min(window_samples, numel(trace));
if mod(window_samples, 2) == 0
    window_samples = window_samples + 1;
end
% Use movmedian when percentile == 50 as an optimization
try
    baseline = movprctile(trace, window_samples, percentile, 'Endpoints', 'shrink');
catch
    % Fallback: use movmedian for 50th percentile
    if percentile == 50
        baseline = movmedian(trace, window_samples, 'Endpoints', 'shrink');
    else
        half = floor(window_samples / 2);
        pad_front = repmat(trace(1), half, 1);
        pad_back = repmat(trace(end), half, 1);
        padded = [pad_front; trace; pad_back];
        baseline = zeros(size(trace));
        for idx = 1:numel(trace)
            segment = padded(idx : idx + window_samples - 1);
            baseline(idx) = prctile(segment, percentile);
        end
    end
end
end
