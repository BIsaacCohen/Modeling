function Isaac_nonlinear_Prediction_plot(results, opts)
% Isaac_nonlinear_Prediction_plot Visualize polynomial face-only regression outputs
%
%   Isaac_nonlinear_Prediction_plot(results)
%   Isaac_nonlinear_Prediction_plot(results, opts)
%
%   results : struct returned by face_only_nonlinear_isaac.
%             Required fields:
%               neural_trace_z, Y_pred, Y_pred_linear, face_motion_z,
%               sampling_rate, n_timepoints.
%   opts    : optional struct with fields:
%               .title_suffix  - extra text appended to the figure title.
%
%   Layout:
%       Tile 1: neural (z) vs polynomial prediction (z)
%       Tile 2: neural (z) vs linear-only prediction (z)
%       Tile 3: face motion (z)

if nargin < 2 || isempty(opts)
    opts = struct();
end

required = {'neural_trace_z', 'Y_pred', 'Y_pred_linear', ...
    'face_motion_z', 'sampling_rate', 'n_timepoints'};
missing = setdiff(required, fieldnames(results));
if ~isempty(missing)
    error('Results struct missing required fields: %s', strjoin(missing, ', '));
end

n_t = results.n_timepoints;
if any([numel(results.neural_trace_z), numel(results.Y_pred), ...
        numel(results.Y_pred_linear), numel(results.face_motion_z)] ~= n_t)
    error('Mismatch in result vector lengths; expected %d samples each.', n_t);
end

t = (0:n_t-1) ./ results.sampling_rate;

fig_title = 'Face-only Polynomial Prediction';
if isfield(opts, 'title_suffix') && ~isempty(opts.title_suffix)
    fig_title = sprintf('%s - %s', fig_title, opts.title_suffix);
elseif isfield(results, 'target_roi_name')
    fig_title = sprintf('%s (%s)', fig_title, results.target_roi_name);
end

figure('Name', fig_title);
tiled = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
title(tiled, fig_title);

ax1 = nexttile(tiled);
plot(t, results.neural_trace_z, 'Color', [0.2 0.2 0.8], 'DisplayName', 'Neural (z)');
hold(ax1, 'on');
plot(t, results.Y_pred, 'Color', [0.85 0.33 0.1], 'LineWidth', 1.25, ...
    'DisplayName', 'Poly prediction (z)');
hold(ax1, 'off');
ylabel(ax1, 'Fluorescence (z)');
legend(ax1, 'Location', 'best');
grid(ax1, 'on');

ax2 = nexttile(tiled);
plot(t, results.neural_trace_z, 'Color', [0.2 0.2 0.8], 'DisplayName', 'Neural (z)');
hold(ax2, 'on');
plot(t, results.Y_pred_linear, 'Color', [0.47 0.67 0.19], 'LineWidth', 1, ...
    'DisplayName', 'Linear prediction (z)');
hold(ax2, 'off');
ylabel(ax2, 'Fluorescence (z)');
legend(ax2, 'Location', 'best');
grid(ax2, 'on');

ax3 = nexttile(tiled);
plot(t, results.face_motion_z, 'Color', [0.13 0.55 0.13], 'DisplayName', 'Face motion (z)');
ylabel(ax3, 'Face motion (z)');
xlabel(ax3, 'Time (s)');
grid(ax3, 'on');

linkaxes([ax1, ax2, ax3], 'x');

if isfield(results, 'R2_polynomial') && isfield(results, 'R2_linear_only')
    delta_r2 = results.R2_polynomial - results.R2_linear_only;
    subtitle_text = sprintf('R^2 poly = %.3f | R^2 linear = %.3f | ΔR^2 = %.3f', ...
        results.R2_polynomial, results.R2_linear_only, delta_r2);
    subtitle(tiled, subtitle_text);
end

if isfield(results, 'motion_sampling_rate')
    dim_info = sprintf('Fluo %.1f Hz | Motion %.1f Hz', ...
        results.sampling_rate, results.motion_sampling_rate);
else
    dim_info = sprintf('Fluo %.1f Hz', results.sampling_rate);
end

annotation(gcf, 'textbox', [0.01 0.01 0.35 0.05], 'String', dim_info, ...
    'EdgeColor', 'none', 'Interpreter', 'none');
end

