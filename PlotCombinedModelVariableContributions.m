function PlotCombinedModelVariableContributions(results_file, opts)
% PlotCombinedModelVariableContributions  Visualize single-regressor and unique contributions.
%
%   PlotCombinedModelVariableContributions(results_file)
%   PlotCombinedModelVariableContributions(results_file, opts)
%
%   results_file : MAT file containing a struct named "results" produced by
%       the combined TemporalModelEvents contribution workflow
%   opts.show_fig : true/false to display the generated figures (default true)
%   opts.save_dir : optional directory for exported PNG figures (default: alongside
%       the results file)
%
%   The function expects each ROI entry in "results" to contain:
%       - group_single_R2 : cvR^2 (%) per group when only that group is fit
%       - group_shuffle_R2: cvR^2 (%) per group after shuffling that group's columns
%       - R2_cv           : baseline cvR^2 (%) per fold for the full model
%       - R2_mean         : mean baseline cvR^2 (%) used for the reference line
%
%   These quantities should be computed with all time-shifted columns for the group
%   included in both the single-regressor fits and the shuffle-based delta-R^2 tests
%   (matching the TemporalModelEvents combined-model assumptions).

if nargin < 1 || isempty(results_file)
    error('Please provide a TemporalModelEvents combined-model results file.');
end

if nargin < 2 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'show_fig'); opts.show_fig = true; end

fprintf('=== Plotting combined-model variable contributions ===\n');
fprintf('Loading results: %s\n', results_file);

data = load(results_file, 'results');
if ~isfield(data, 'results')
    error('File %s does not contain a ''results'' struct.', results_file);
end
results = data.results;

if ~isfield(results, 'metadata') || ~isfield(results.metadata, 'group_labels')
    error('Results struct missing metadata.group_labels. Ensure the contribution analysis script was run.');
end

group_labels = results.metadata.group_labels(:);
n_groups = numel(group_labels);
roi_names = infer_roi_names(results);

color_map = builtin_color_map();
group_colors = zeros(n_groups, 3);
for g = 1:n_groups
    group_colors(g, :) = pick_color(group_labels{g}, color_map, g);
end

[res_dir, ~, ~] = fileparts(results_file);
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

    required_fields = {'group_single_R2', 'group_shuffle_R2', 'R2_cv', 'R2_mean'};
    if any(~isfield(roi_data, required_fields))
        warning('ROI %s lacks required contribution fields. Skipping.', roi_name);
        continue;
    end

    explained_vals = roi_data.group_single_R2;
    shuffle_vals = roi_data.group_shuffle_R2;
    baseline_folds = roi_data.R2_cv(:)';

    if size(explained_vals, 1) ~= n_groups
        warning('ROI %s has %d group entries but metadata lists %d groups. Skipping.', ...
            roi_name, size(explained_vals, 1), n_groups);
        continue;
    end

    unique_vals = bsxfun(@minus, baseline_folds, shuffle_vals);
    unique_vals(unique_vals < 0) = 0;

    visible_flag = ternary(opts.show_fig, 'on', 'off');
    fig = figure('Name', sprintf('Combined-model contributions - %s', roi_name), ...
        'Visible', visible_flag);
    ax = axes(fig);
    hold(ax, 'on');

    offset = 0.18;
    width = 0.18;
    legend_handles = gobjects(2, 1);
    legend_labels = {'Explained (cvR^2)', 'Unique (Delta R^2)'};

    for g = 1:n_groups
        x_center = g;
        exp_color = group_colors(g, :);
        uniq_color = lighten_color(exp_color, 0.35);

        h_exp = draw_violin(ax, explained_vals(g, :), x_center - offset, exp_color, width);
        h_uniq = draw_violin(ax, unique_vals(g, :), x_center + offset, uniq_color, width);

        if g == 1
            legend_handles(1) = h_exp;
            legend_handles(2) = h_uniq;
        end
    end

    xticks(ax, 1:n_groups);
    xticklabels(ax, group_labels);
    ax.XTickLabelRotation = 30;
    ylabel(ax, 'Percent R^2');
    legend(ax, legend_handles, legend_labels, 'Location', 'northwest');
    grid(ax, 'off');

    baseline = roi_data.R2_mean;
    yline(ax, baseline, '--', sprintf('Baseline R^2 = %.2f%%', baseline), ...
        'LabelVerticalAlignment', 'bottom');

    title(ax, sprintf('Combined-model contributions (%s)', roi_name), 'Interpreter', 'none');

    all_vals = [explained_vals(:); unique_vals(:); baseline];
    ymax = max(all_vals, [], 'omitnan');
    if isempty(ymax) || isnan(ymax); ymax = baseline; end
    ymin = min(all_vals, [], 'omitnan');
    if isempty(ymin) || isnan(ymin); ymin = 0; end
    upper = max(5, 1.2 * ymax);
    lower = min(0, 1.2 * ymin);
    ylim(ax, [lower, upper]);

    out_name = sprintf('%s_combined_contributions.png', sanitize_filename(roi_name));
    out_path = fullfile(save_dir, out_name);
    exportgraphics(fig, out_path, 'Resolution', 300);
    fprintf('  Saved figure: %s\n', out_path);

    if ~opts.show_fig
        close(fig);
    end
end

fprintf('All combined-model contribution figures generated.\n');
end

function names = infer_roi_names(results)
if isfield(results, 'metadata') && isfield(results.metadata, 'neural_names')
    names = results.metadata.neural_names(:);
else
    cand = fieldnames(results);
    cand(strcmpi(cand, 'metadata')) = [];
    names = cand(:);
end
end

function cmap = builtin_color_map()
cmap = struct();
cmap.motion = [0.15 0.35 0.80];
cmap.noise = [0.30 0.45 0.75];
cmap.lick_stim = [0.85 0.33 0.10];
cmap.lick_water_cued = [0.47 0.67 0.19];
cmap.lick_water_uncued = [0.49 0.18 0.56];
cmap.lick_water_omission = [0.25 0.25 0.25];
cmap.default = lines(8);
end

function color = pick_color(label, cmap, idx)
lbl = lower(label);
if contains(lbl, 'motion')
    color = cmap.motion;
elseif contains(lbl, 'noise')
    color = cmap.noise;
elseif contains(lbl, 'post-stim')
    color = cmap.lick_stim;
elseif contains(lbl, 'post-water') && contains(lbl, 'cued')
    color = cmap.lick_water_cued;
elseif contains(lbl, 'post-water') && contains(lbl, 'uncued')
    color = cmap.lick_water_uncued;
elseif contains(lbl, 'omission')
    color = cmap.lick_water_omission;
else
    palette_idx = mod(idx-1, size(cmap.default, 1)) + 1;
    color = cmap.default(palette_idx, :);
end
end

function safe = sanitize_filename(name)
safe = regexprep(name, '[^a-zA-Z0-9_-]', '_');
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

h = patch(ax, patch_x, patch_y, color, ...
    'FaceAlpha', 0.3, 'EdgeColor', color, 'LineWidth', 1);

median_val = median(data);
plot(ax, [x_pos - width/2, x_pos + width/2], [median_val, median_val], ...
    'Color', color, 'LineWidth', 1.2);
plot(ax, x_pos, mean(data), 'o', 'MarkerEdgeColor', color, 'MarkerFaceColor', 'w');
end

function color_out = lighten_color(color_in, factor)
factor = max(0, min(1, factor));
color_out = color_in + (1 - color_in) * factor;
color_out = min(max(color_out, 0), 1);
end
