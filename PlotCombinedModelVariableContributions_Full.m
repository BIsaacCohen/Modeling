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
[results, default_name, default_dir] = resolve_results_input(results_input);

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

res_dir = default_dir;
if isempty(res_dir) || ~isfolder(res_dir)
    res_dir = pwd;
end
res_name = default_name;
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

    perform_one_sample_ttest(group_labels{g}, exp_vals, 'Explained');
    perform_one_sample_ttest(group_labels{g}, uniq_vals, 'Unique');
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

function perform_one_sample_ttest(group_label, values, metric_label)
if nargin < 3 || isempty(metric_label)
    metric_label = 'Metric';
end
vals = values(:);
vals = vals(~isnan(vals));
if numel(vals) < 2
    fprintf('  [ttest] %s (%s): not enough samples (n=%d).\n', group_label, metric_label, numel(vals));
    return;
end
[~, p_val, ~, stats] = ttest(vals, 0, 'Tail', 'right');
t_stat = stats.tstat;
df = stats.df;
mean_val = mean(vals);
fprintf('  [ttest] %s (%s): mean=%.3f%%, t(%d)=%.3f, p=%.3g\n', ...
    group_label, metric_label, mean_val, df, t_stat, p_val);
end

function [results, res_name, res_dir] = resolve_results_input(input_value)
results = [];
res_name = 'results_struct';
res_dir = pwd;
if iscell(input_value)
    [results, res_name] = combine_results_for_plot(input_value);
elseif isstruct(input_value) && numel(input_value) > 1
    [results, res_name] = combine_results_for_plot(num2cell(input_value));
elseif ischar(input_value) || isstring(input_value)
    file_path = char(input_value);
    if exist(file_path, 'file') ~= 2
        error('Results file "%s" not found.', file_path);
    end
    fprintf('Loading results: %s\n', file_path);
    data = load(file_path, 'results');
    if ~isfield(data, 'results')
        error('File %s does not contain a ''results'' struct.', file_path);
    end
    results = data.results;
    [res_dir, res_name, ~] = fileparts(file_path);
elseif isstruct(input_value)
    fprintf('Using provided results struct input.\n');
    results = input_value;
else
    error('results_input must be a MAT file path, struct, or cell array of inputs.');
end
end

function [combined_results, combined_name] = combine_results_for_plot(input_list)
if isempty(input_list)
    error('No result entries provided for combination.');
end
results_list = cell(numel(input_list), 1);
for i = 1:numel(input_list)
    entry = input_list{i};
    if isempty(entry)
        error('Entry %d in the input list is empty.', i);
    end
    if ischar(entry) || isstring(entry)
        file_path = char(entry);
        if exist(file_path, 'file') ~= 2
            error('Results file "%s" not found.', file_path);
        end
        fprintf('  Loading results %d: %s\n', i, file_path);
        data = load(file_path, 'results');
        if ~isfield(data, 'results')
            error('File %s does not contain a ''results'' struct.', file_path);
        end
        results_list{i} = data.results;
    elseif isstruct(entry)
        results_list{i} = entry;
    else
        error('Unsupported entry type in results list (index %d).', i);
    end
end
fprintf('Combining %d result inputs for aggregated plotting.\n', numel(results_list));
combined_contr = merge_contribution_structs(results_list);
combined_results = struct('contributions', combined_contr);
combined_name = sprintf('combined_%d_sessions', numel(results_list));
end

function combined = merge_contribution_structs(results_list)
first = results_list{1};
if ~isfield(first, 'contributions')
    error('Result entry 1 missing contributions field.');
end
combined = first.contributions;
required = {'group_labels','group_single_R2','group_shuffle_R2','R2_cv_percent','R2_mean_percent'};
missing = required(~isfield(combined, required));
if ~isempty(missing)
    error('First contributions struct missing fields: %s', strjoin(missing, ', '));
end
for idx = 2:numel(results_list)
    res = results_list{idx};
    if ~isfield(res, 'contributions')
        error('Result entry %d missing contributions field.', idx);
    end
    contr = res.contributions;
    if numel(contr.group_labels) ~= numel(combined.group_labels) || ...
            any(~strcmp(contr.group_labels(:), combined.group_labels(:)))
        error('Group labels do not match between result entries 1 and %d.', idx);
    end
    combined.group_single_R2 = cat(2, combined.group_single_R2, contr.group_single_R2);
    combined.group_shuffle_R2 = cat(2, combined.group_shuffle_R2, contr.group_shuffle_R2);
    combined.group_explained_mean = cat(2, combined.group_explained_mean, contr.group_explained_mean);
    combined.group_explained_std = cat(2, combined.group_explained_std, contr.group_explained_std);
    combined.group_unique_mean = cat(2, combined.group_unique_mean, contr.group_unique_mean);
    combined.group_unique_std = cat(2, combined.group_unique_std, contr.group_unique_std);
    combined.R2_cv_percent = cat(1, combined.R2_cv_percent, contr.R2_cv_percent);
    combined.R2_mean_percent = cat(1, combined.R2_mean_percent, contr.R2_mean_percent(:));
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
