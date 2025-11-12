function plot_events_only_variable_contributions(results_file, opts)
% plot_events_only_variable_contributions Plot explained and unique contributions.
%
%   plot_events_only_variable_contributions()
%   plot_events_only_variable_contributions(results_file)
%   plot_events_only_variable_contributions(results_file, opts)
%
%   results_file : path to ridge_events_only_mml_results_TIMESTAMP.mat
%   opts.save_dir: optional output directory (default: same as results_file)
%   opts.show_fig: true/false to display figures (default: true)
%
%   Each ROI figure shows paired violin plots for:
%       - Explained variance (cvR^2) using only that regressor group
%       - Unique contribution (Delta R^2) computed via shuffle control
%
%   Requires results produced by ridge_regression_events_only_mml.m

if nargin < 1 || isempty(results_file)
    results_file = 'ridge_events_only_mml_results.mat';
end

if nargin < 2 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'show_fig'); opts.show_fig = true; end

fprintf('=== Plotting events-only variable contributions ===\n');
fprintf('Loading results: %s\n', results_file);

data = load(results_file, 'results');
if ~isfield(data, 'results')
    error('File %s does not contain a ''results'' struct.', results_file);
end
results = data.results;

if ~isfield(results, 'metadata') || ~isfield(results.metadata, 'group_labels')
    error('Results struct missing metadata.group_labels (run updated ridge script).');
end

group_labels = results.metadata.group_labels(:);
n_groups = numel(group_labels);
roi_names = results.metadata.neural_names(:);

default_colors = [ ...
    0.30 0.45 0.75;  % Noise
    0.85 0.33 0.10;  % Lick post-stimulus
    0.47 0.67 0.19;  % Lick post-water (cued)
    0.49 0.18 0.56;  % Lick post-water (uncued)
    0.25 0.25 0.25]; % Lick post-water (omission)

if size(default_colors, 1) < n_groups
    default_colors = lines(n_groups);
end

[res_dir, ~] = fileparts(results_file);
if isempty(res_dir); res_dir = pwd; end
if ~isfield(opts, 'save_dir') || isempty(opts.save_dir)
    save_dir = res_dir;
else
    save_dir = opts.save_dir;
end
if exist(save_dir, 'dir') ~= 7
    mkdir(save_dir);
end

for r = 1:numel(roi_names)
    roi_name = roi_names{r};
    if ~isfield(results, roi_name)
        warning('ROI %s missing from results struct. Skipping.', roi_name);
        continue;
    end
    roi_data = results.(roi_name);

    required_fields = {'group_explained_mean', 'group_unique_mean', ...
        'group_single_R2', 'group_shuffle_R2', 'R2_cv'};
    if any(~isfield(roi_data, required_fields))
        warning('ROI %s lacks required contribution details. Re-run ridge script with contribution analysis.', roi_name);
        continue;
    end

    explained_mean = roi_data.group_explained_mean(:);
    unique_mean = roi_data.group_unique_mean(:);
    explained_std = roi_data.group_explained_std(:);
    unique_std = roi_data.group_unique_std(:);

    valid = ~(isnan(explained_mean) & isnan(unique_mean));
    if ~any(valid)
        warning('ROI %s has no valid contribution data. Skipping.', roi_name);
        continue;
    end

    labels = group_labels(valid);
    ngroups = numel(labels);
    colors = default_colors(valid, :);

    explained_vals = roi_data.group_single_R2(valid, :);
    baseline_folds = roi_data.R2_cv(:)';
    if isempty(baseline_folds)
        warning('ROI %s missing cv R^2 per fold. Skipping.', roi_name);
        continue;
    end
    shuffle_vals = roi_data.group_shuffle_R2(valid, :);
    unique_vals = bsxfun(@minus, baseline_folds, shuffle_vals);

    fig = figure('Name', sprintf('Events-only contributions - %s', roi_name), ...
        'Visible', ternary(opts.show_fig, 'on', 'off'));
    ax = axes(fig);
    hold(ax, 'on');

    offset = 0.18;
    width = 0.18;
    legend_handles = gobjects(2, 1);
    legend_labels = {'Explained (cvR^2)', 'Unique (Delta R^2)'};

    for g = 1:ngroups
        x_center = g;
        exp_data = explained_vals(g, :);
        uniq_data = unique_vals(g, :);
        exp_color = colors(g, :);
        uniq_color = lighten_color(exp_color, 0.35);

        h_exp = draw_violin(ax, exp_data, x_center - offset, exp_color, width);
        h_uniq = draw_violin(ax, uniq_data, x_center + offset, uniq_color, width);

        if g == 1
            legend_handles(1) = h_exp;
            legend_handles(2) = h_uniq;
        end
    end

    xticks(ax, 1:ngroups);
    xticklabels(ax, labels);
    ax.XTickLabelRotation = 30;
    ylabel(ax, 'Percent R^2');
    legend(ax, legend_handles, legend_labels, 'Location', 'northwest');
    grid(ax, 'off');

    baseline = roi_data.R2_mean;
    yline(ax, baseline, '--', sprintf('Baseline R^2 = %.2f%%', baseline), 'LabelVerticalAlignment', 'bottom');

    title(ax, sprintf('Events-only contributions (%s)', roi_name), 'Interpreter', 'none');
    all_vals = [explained_vals(:); unique_vals(:); baseline];
    ymax = max(all_vals, [], 'omitnan');
    if isempty(ymax) || isnan(ymax)
        ymax = baseline;
    end
    ymin = min(all_vals, [], 'omitnan');
    if isempty(ymin) || isnan(ymin)
        ymin = 0;
    end
    upper = max(5, 1.2 * ymax);
    lower = min(0, 1.2 * ymin);
    ylim(ax, [lower, upper]);

    out_name = sprintf('%s_events_only_contributions.png', roi_name);
    out_path = fullfile(save_dir, out_name);
    exportgraphics(fig, out_path, 'Resolution', 300);
    fprintf('  Saved figure: %s\n', out_path);

    if ~opts.show_fig
        close(fig);
    end
end

fprintf('All figures generated.\n');
end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end

function h = draw_violin(ax, data, x_pos, color, width)
data = data(~isnan(data));
if isempty(data)
    h = plot(ax, x_pos, NaN);
    return;
end

if numel(data) < 2 || std(data) < eps
    y = mean(data);
    h = plot(ax, x_pos, y, 'o', 'MarkerFaceColor', color, 'MarkerEdgeColor', color);
    return;
end

[density, value] = ksdensity(data);
if all(density == 0)
    density = ones(size(density));
end
density = density / max(density);
density = density * width;

patch_x = [x_pos - density, fliplr(x_pos + density)];
patch_y = [value, fliplr(value)];

h = patch(ax, patch_x, patch_y, color, 'FaceAlpha', 0.3, 'EdgeColor', color, 'LineWidth', 1);

median_val = median(data);
plot(ax, [x_pos - width/2, x_pos + width/2], [median_val, median_val], 'Color', color, 'LineWidth', 1.2);
plot(ax, x_pos, mean(data), 'o', 'MarkerEdgeColor', color, 'MarkerFaceColor', 'w');
end

function color_out = lighten_color(color_in, factor)
factor = max(0, min(1, factor));
color_out = color_in + (1 - color_in) * factor;
color_out = min(max(color_out, 0), 1);
end
