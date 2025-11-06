function Isaac_simple_Prediction_plot(results, opts)
% Isaac_simple_Prediction_plot Plot neural data, prediction, and face motion
%
%   Isaac_simple_Prediction_plot(results)
%   Isaac_simple_Prediction_plot(results, opts)
%
%   results : struct produced by face_only_simple_regression_isaac
%             Required fields: neural_trace_z, Y_pred, face_motion_z,
%             sampling_rate (Hz), n_timepoints.
%   opts    : optional struct with fields:
%             .title_suffix - char/string appended to figure title.
%
%   The function creates a two-panel plot with linked x-axes:
%       Top:  neural_trace_z (observed) and Y_pred (predicted)
%       Bottom: face_motion_z

if nargin < 2 || isempty(opts)
    opts = struct();
end

required_fields = {'neural_trace_z', 'Y_pred', 'face_motion_z', ...
    'sampling_rate', 'n_timepoints'};
missing = setdiff(required_fields, fieldnames(results));
if ~isempty(missing)
    error('Results struct is missing required fields: %s', strjoin(missing, ', '));
end

n_t = results.n_timepoints;
if numel(results.neural_trace_z) ~= n_t || numel(results.Y_pred) ~= n_t ...
        || numel(results.face_motion_z) ~= n_t
    error('Length mismatch: expected %d samples in neural_trace_z, Y_pred, and face_motion_z.', n_t);
end

t = (0:n_t-1) ./ results.sampling_rate;

fig_title = 'Face-only Prediction';
if isfield(opts, 'title_suffix') && ~isempty(opts.title_suffix)
    fig_title = sprintf('%s - %s', fig_title, opts.title_suffix);
elseif isfield(results, 'target_roi_name')
    fig_title = sprintf('%s (%s)', fig_title, results.target_roi_name);
end

figure('Name', fig_title);
tiled = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tiled, fig_title);

ax1 = nexttile(tiled);
plot(t, results.neural_trace_z, 'Color', [0.2 0.2 0.8], 'DisplayName', 'Neural (z-score)');
hold(ax1, 'on');
plot(t, results.Y_pred, 'Color', [0.85 0.33 0.1], 'LineWidth', 1.25, 'DisplayName', 'Prediction (z-score)');
hold(ax1, 'off');
ylabel(ax1, 'Fluorescence (z-score)');
legend(ax1, 'Location', 'best');
grid(ax1, 'on');

ax2 = nexttile(tiled);
plot(t, results.face_motion_z, 'Color', [0.13 0.55 0.13], 'DisplayName', 'Face motion (z-score)');
ylabel(ax2, 'Face motion (z-score)');
xlabel(ax2, 'Time (s)');
grid(ax2, 'on');

linkaxes([ax1, ax2], 'x');

if isfield(results, 'downsample_factor')
    dim_info = sprintf('Sampling: %g Hz (motion %g Hz)', ...
        results.sampling_rate, results.motion_sampling_rate);
else
    dim_info = sprintf('Sampling: %g Hz', results.sampling_rate);
end

annotation(gcf, 'textbox', [0.01 0.01 0.3 0.05], 'String', dim_info, ...
    'EdgeColor', 'none', 'Interpreter', 'none');
end

