function PlotCombinedModelVariableContributions_Full(results_input, opts)
% PlotCombinedModelVariableContributions_Full  Aggregate ROI contribution violins.
%
%   PlotCombinedModelVariableContributions_Full(results_input)
%   PlotCombinedModelVariableContributions_Full(results_input, opts)
%
%   Builds a single figure summarizing the single-variable and unique
%   (shuffle-based) contributions for every regressor group across ALL ROIs
%   produced by TemporalModelEventsFull_Contributions. Each violin contains
%   all folds from all ROIs (e.g., 48 ROIs * 5 folds -> 240 points).
%
%   results_input : MAT-file path containing "results" or the struct itself.
%   opts.show_fig : true/false to display the generated figure (default true)
%   opts.save_dir : optional directory for the exported PNG (defaults next
%                   to the results file)

if nargin < 1 || isempty(results_input)
    error('Please provide a TemporalModelEventsFull_Contributions results file or struct.');
end

if nargin < 2 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'show_fig'); opts.show_fig = true; end

fprintf('=== Plotting FULL combined-model variable contributions ===\n');
results_file = '';
if ischar(results_input) || isstring(results_input)
    results_file = char(results_input);
    if exist(results_file, 'file') ~= 2
        error('Results file "%s" not found.', results_file);
    end
    fprintf('Loading results: %s\n', results_file);
    data = load(results_file, 'results');
    if ~isfield(data, 'results')
        error('File %s does not contain a ''results'' struct.', results_file);
    end
    results = data.results;
elseif isstruct(results_input)
    fprintf('Using provided results struct input.\n');
    results = results_input;
else
    error('results_input must be a MAT file path or a results struct.');
end

if ~isfield(results, 'contributions')
    error('Results struct missing contributions field. Did you run TemporalModelEventsFull_Contributions?');
end

contr = results.contributions;
required = {'group_labels','group_single_R2','group_shuffle_R2','R2_cv_percent','R2_mean_percent'};
missing = required(~isfield(contr, required));
if ~isempty(missing)
    error('Contributions struct missing fields: %s', strjoin(missing, ', '));
end

group_labels = contr.group_labels(:);
n_groups = numel(group_labels);
group_single = contr.group_single_R2;
group_shuffle = contr.group_shuffle_R2;
baseline_folds = contr.R2_cv_percent;

if ndims(group_single) ~= 3
    error('group_single_R2 must be G x R x F (groups x ROIs x folds).');
end

[~, n_rois, cv_folds] = size(group_single);

color_map = builtin_color_map();
group_colors = zeros(n_groups, 3);
for g = 1:n_groups
    group_colors(g, :) = pick_color(group_labels{g}, color_map, g);
end

if ~isempty(results_file)
    [res_dir, res_name, ~] = fileparts(results_file);
else
    res_dir = pwd;
    res_name = 'results_struct';
end
if isempty(res_dir); res_dir = pwd; end
if ~isfield(opts, 'save_dir') || isempty(opts.save_dir)
    save_dir = res_dir;
else
    save_dir = opts.save_dir;
end
if exist(save_dir, 'dir') ~= 7
    mkdir(save_dir);
end

explained_values = cell(n_groups, 1);
unique_values = cell(n_groups, 1);

if ndims(baseline_folds) == 2
    baseline_matrix = baseline_folds;
else
    baseline_matrix = reshape(baseline_folds, n_rois, []);
end
if size(baseline_matrix, 2) ~= cv_folds
    error('R2_cv_percent has %d folds but group_single_R2 reports %d.', size(baseline_matrix, 2), cv_folds);
end

for g = 1:n_groups
    single_matrix = reshape(group_single(g, :, :), n_rois, cv_folds);
    shuffle_matrix = reshape(group_shuffle(g, :, :), n_rois, cv_folds);

    explained_values{g} = single_matrix(:);
    unique_matrix = bsxfun(@minus, baseline_matrix, shuffle_matrix);
    unique_matrix(unique_matrix < 0) = 0;
    unique_values{g} = unique_matrix(:);
end

visible_flag = ternary(opts.show_fig, 'on', 'off');
fig = figure('Name', 'Combined-model contributions (all ROIs)', ...
    'Visible', visible_flag);
ax = axes(fig);
hold(ax, 'on');

offset = 0.18;
width = 0.18;
legend_handles = gobjects(2, 1);
legend_labels = {'Explained (cvR^2)', 'Unique (Delta R^2)'};

all_vals = [];
for g = 1:n_groups
    x_center = g;
    exp_color = group_colors(g, :);
    uniq_color = lighten_color(exp_color, 0.35);

    exp_vals = explained_values{g};
    uniq_vals = unique_values{g};

    h_exp = draw_violin(ax, exp_vals, x_center - offset, exp_color, width);
    h_uniq = draw_violin(ax, uniq_vals, x_center + offset, uniq_color, width);

    all_vals = [all_vals; exp_vals(:); uniq_vals(:)]; %#ok<AGROW>

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

full_model_mean = mean(contr.R2_mean_percent(:), 'omitnan');
if isempty(full_model_mean) || isnan(full_model_mean)
    full_model_mean = mean(all_vals, 'omitnan');
end
yline(ax, full_model_mean, '--', sprintf('Full Model Mean R^2 = %.2f%%', full_model_mean), ...
    'LabelVerticalAlignment', 'bottom');

title(ax, sprintf('Combined-model contributions (All ROIs, n=%d, folds=%d)', n_rois, cv_folds), ...
    'Interpreter', 'none');

valid_vals = all_vals(~isnan(all_vals));
if isempty(valid_vals)
    valid_vals = full_model_mean;
end
ymax = max(valid_vals, [], 'omitnan');
ymin = min(valid_vals, [], 'omitnan');
if isempty(ymax) || isnan(ymax); ymax = full_model_mean; end
if isempty(ymin) || isnan(ymin); ymin = 0; end
upper = max(5, 1.2 * ymax);
lower = min(0, 1.2 * ymin);
ylim(ax, [lower, upper]);

out_name = sprintf('%s_combined_contributions_full.png', sanitize_filename(res_name));
out_path = fullfile(save_dir, out_name);
exportgraphics(fig, out_path, 'Resolution', 300);
fprintf('  Saved aggregated figure: %s\n', out_path);

if ~opts.show_fig
    close(fig);
end

fprintf('Full ROI combined-model contribution figure generated.\n');
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
if nargin < 5
    width = 0.2;
end
if nargin < 4
    color = [0 0 0];
end
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
