function CompareCVR2_EarlyVsExpert(early_inputs, expert_inputs, opts)
% CompareCVR2_EarlyVsExpert  Contrast cvR^2 distributions between two stages.
%
%   CompareCVR2_EarlyVsExpert(early_inputs, expert_inputs)
%   CompareCVR2_EarlyVsExpert(..., opts)
%
%   early_inputs / expert_inputs can be MAT-file paths (containing a "results"
%   struct) or the structs themselves. Cell arrays and mixed inputs are allowed.
%   The function aggregates all ROIs and CV folds for each condition, then
%   generates violin plots comparing the cvR^2 distributions for
%       1) Full model (baseline cvR^2)
%       2) Each predictor group (single-variable cvR^2)
%
%   Options (all optional):
%       opts.show_fig        - true/false to display figures (default true)
%       opts.save_dir        - directory to export PNGs (default pwd)
%       opts.condition_labels- {1x2} labels for the two conditions
%                              (default {'Early','Expert'})
%       opts.print_ttests    - true to print right-tailed one-sample t-tests
%                              against 0 for each condition (default true)
%       opts.print_between_tests - true to print between-condition tests (default true)
%       opts.between_test    - 'paired', 'ttest2' (Welch), or 'ranksum'
%                              (default 'paired')
%       opts.confidence_level- confidence level for reported CIs (default 0.95)
%       opts.normality_alpha  - alpha for normality tests (default 0.05)

if nargin < 2
    error('CompareCVR2_EarlyVsExpert:MissingInputs', ...
        'Provide inputs for both early and expert conditions.');
end
if nargin < 3 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'show_fig') || isempty(opts.show_fig)
    opts.show_fig = true;
end
if ~isfield(opts, 'save_dir') || isempty(opts.save_dir)
    opts.save_dir = pwd;
end
if ~isfield(opts, 'condition_labels') || numel(opts.condition_labels) ~= 2
    opts.condition_labels = {'Early', 'Expert'};
end
if ~isfield(opts, 'print_ttests') || isempty(opts.print_ttests)
    opts.print_ttests = true;
end
if ~isfield(opts, 'print_between_tests') || isempty(opts.print_between_tests)
    opts.print_between_tests = true;
end
if ~isfield(opts, 'between_test') || isempty(opts.between_test)
    opts.between_test = 'paired'; % options: 'paired','ttest2','ranksum'
end
if ~isfield(opts, 'confidence_level') || isempty(opts.confidence_level)
    opts.confidence_level = 0.95;
end
if ~isfield(opts, 'normality_alpha') || isempty(opts.normality_alpha)
    opts.normality_alpha = 0.05;
end

fprintf('=== Aggregating cvR^2 for Early condition ===\n');
early = aggregate_condition_data(early_inputs);
fprintf('=== Aggregating cvR^2 for Expert condition ===\n');
expert = aggregate_condition_data(expert_inputs);

if numel(early.group_labels) ~= numel(expert.group_labels) || ...
        any(~strcmp(early.group_labels, expert.group_labels))
    error('CompareCVR2_EarlyVsExpert:GroupMismatch', ...
        'Predictor group labels differ between conditions.');
end

ensure_dir(opts.save_dir);

plot_condition_violin(early.full_values, expert.full_values, ...
    opts.condition_labels, 'Full model cvR^2 (%)', 'full_model', opts);

for g = 1:numel(early.group_labels)
    ttl = sprintf('%s cvR^2 (%%)', early.group_labels{g});
    filename = sprintf('cvr2_%s', sanitize_filename(early.group_labels{g}));
    plot_condition_violin(early.explained_values{g}, expert.explained_values{g}, ...
        opts.condition_labels, ttl, filename, opts);
end

fprintf('cvR^2 comparison plots complete.\n');
end

%% ---------------- Aggregation helpers ----------------
function agg = aggregate_condition_data(inputs)
entries = normalize_input_list(inputs);
if isempty(entries)
    error('Condition input list is empty.');
end
agg = struct();
agg.group_labels = [];
agg.explained_values = {};
agg.full_values = [];
for i = 1:numel(entries)
    res = load_results(entries{i});
    if ~isfield(res, 'contributions')
        error('Result entry %d missing contributions field.', i);
    end
    contr = res.contributions;
    if isempty(agg.group_labels)
        agg.group_labels = contr.group_labels(:);
        agg.explained_values = repmat({[]}, numel(agg.group_labels), 1);
    else
        if numel(agg.group_labels) ~= numel(contr.group_labels) || ...
                any(~strcmp(agg.group_labels, contr.group_labels(:)))
            error('Group labels differ between entry 1 and entry %d.', i);
        end
    end
    baseline_mat = extract_baseline_matrix(contr.R2_cv_percent);
    agg.full_values = [agg.full_values; baseline_mat(:)]; %#ok<AGROW>
    group_single = contr.group_single_R2;
    if ndims(group_single) ~= 3
        error('group_single_R2 must be G x R x F.');
    end
    [~, n_rois, cv_folds] = size(group_single);
    for g = 1:numel(agg.group_labels)
        single_matrix = reshape(group_single(g, :, :), n_rois, cv_folds);
        agg.explained_values{g} = [agg.explained_values{g}; single_matrix(:)]; %#ok<AGROW>
    end
end
end

function lst = normalize_input_list(inputs)
if iscell(inputs)
    lst = inputs;
elseif isstruct(inputs) && numel(inputs) > 1
    lst = num2cell(inputs);
else
    lst = {inputs};
end
end

function results = load_results(entry)
if ischar(entry) || isstring(entry)
    file_path = char(entry);
    if exist(file_path, 'file') ~= 2
        error('Results file "%s" not found.', file_path);
    end
    data = load(file_path, 'results');
    if ~isfield(data, 'results')
        error('File %s does not contain a "results" struct.', file_path);
    end
    results = data.results;
elseif isstruct(entry)
    if isfield(entry, 'contributions')
        results = entry;
    elseif isfield(entry, 'results')
        results = entry.results;
    else
        error('Struct entry does not contain contributions/results fields.');
    end
else
    error('Unsupported input type: %s', class(entry));
end
end

function baseline_matrix = extract_baseline_matrix(R2_vals)
if ismatrix(R2_vals)
    baseline_matrix = R2_vals;
else
    n_rois = size(R2_vals, 1);
    baseline_matrix = reshape(R2_vals, n_rois, []);
end
end

%% ---------------- Plotting helpers ----------------
function plot_condition_violin(vals_a, vals_b, cond_labels, title_str, filename, opts)
vals_a = vals_a(:);
vals_b = vals_b(:);
vals_a = vals_a(~isnan(vals_a));
vals_b = vals_b(~isnan(vals_b));
visible_flag = ternary(opts.show_fig, 'on', 'off');
fig = figure('Name', title_str, 'Visible', visible_flag);
ax = axes(fig);
hold(ax, 'on');
width = 0.25;
color_a = [0.20 0.45 0.80];
color_b = [0.85 0.33 0.10];
hA = draw_violin(ax, vals_a, 0.75, color_a, width);
hB = draw_violin(ax, vals_b, 1.25, color_b, width);
plot(ax, [0.5 1.5], [0 0], 'k--', 'HandleVisibility', 'off');
xticks(ax, [0.75 1.25]);
xticklabels(ax, cond_labels);
ylabel(ax, 'cvR^2 (%)');
title(ax, title_str, 'Interpreter', 'none');
legend(ax, [hA hB], cond_labels, 'Location', 'best');
grid(ax, 'off');
hold(ax, 'off');
ax.XLim = [0.4 1.6];

save_path = fullfile(opts.save_dir, sprintf('%s.png', filename));
exportgraphics(fig, save_path, 'Resolution', 300);
fprintf('  Saved comparison plot: %s\n', save_path);

if opts.print_ttests
    print_ttest(cond_labels{1}, vals_a, opts.confidence_level);
    print_ttest(cond_labels{2}, vals_b, opts.confidence_level);
end
if opts.print_between_tests
    print_between_condition_test(cond_labels, vals_a, vals_b, opts.between_test, opts.confidence_level, opts.normality_alpha);
end

if ~opts.show_fig
    close(fig);
end
end

function handle = draw_violin(ax, data, x_pos, color_val, width)
data = data(~isnan(data));
if isempty(data)
    handle = plot(ax, x_pos, NaN, 'o', 'Color', color_val);
    return;
end
if numel(data) < 5 || std(data, 0, 'omitnan') < eps
    handle = plot(ax, x_pos, mean(data), 'o', ...
        'MarkerFaceColor', color_val, 'MarkerEdgeColor', color_val, 'MarkerSize', 6);
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
patch(ax, patch_x, patch_y, color_val, ...
    'FaceAlpha', 0.25, 'EdgeColor', color_val, 'LineWidth', 1.2);
plot(ax, [x_pos - width/2, x_pos + width/2], [median(data), median(data)], ...
    'Color', color_val, 'LineWidth', 1.2);
handle = plot(ax, x_pos, mean(data), 'o', 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', color_val, 'MarkerSize', 5);
end

function ensure_dir(path_str)
if exist(path_str, 'dir') ~= 7
    mkdir(path_str);
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

function print_ttest(label, values, conf_level)
if numel(values) < 2
    fprintf('  [ttest] %s: not enough samples (n=%d)\n', label, numel(values));
    return;
end
if nargin < 3 || isempty(conf_level)
    conf_level = 0.95;
end
alpha = 1 - conf_level;
[~, p_val, ci, stats] = ttest(values, 0, 'Tail', 'right', 'Alpha', alpha);
fprintf('  [ttest] %s: mean=%.3f%%, t(%d)=%.3f, p=%.3g, %g%% CI=[%.3f, %.3f]%%\n', ...
    label, mean(values), stats.df, stats.tstat, p_val, conf_level*100, ci(1), ci(2));
end

function print_between_condition_test(cond_labels, vals_a, vals_b, method, conf_level, normality_alpha)
vals_a = vals_a(:);
vals_b = vals_b(:);
vals_a = vals_a(~isnan(vals_a));
vals_b = vals_b(~isnan(vals_b));
if numel(vals_a) < 2 || numel(vals_b) < 2
    fprintf('  [between-test] %s vs %s: insufficient samples (%d vs %d)\n', ...
        cond_labels{1}, cond_labels{2}, numel(vals_a), numel(vals_b));
    return;
end
if nargin < 5 || isempty(conf_level)
    conf_level = 0.95;
end
if nargin < 6 || isempty(normality_alpha)
    normality_alpha = 0.05;
end
alpha = 1 - conf_level;
switch lower(method)
    case {'paired','ttestpaired','pairedttest'}
        if numel(vals_a) ~= numel(vals_b)
            warning('Paired test requested but sample sizes differ (%d vs %d); skipping paired test.', ...
                numel(vals_a), numel(vals_b));
            return;
        end
        diffs = vals_a - vals_b;
        [normal_flag, normal_p] = lillietest(diffs, 'Alpha', normality_alpha);
        [p_val,~,stats] = signrank(vals_a, vals_b);
        median_diff = median(diffs, 'omitnan');
        fprintf('  [Wilcoxon signed-rank] %s vs %s: median diff=%.3f%%, z=%.3f, p=%.3g (Normality p=%.3g -> %s)\n', ...
            cond_labels{1}, cond_labels{2}, median_diff, stats.zval, p_val, normal_p, ...
            ternary(normal_flag==0, 'normality plausible', 'non-normal diffs'));
    case {'ttest2','ttest'}
        [~, p_val, ci, stats] = ttest2(vals_a, vals_b, 'Vartype', 'unequal', 'Alpha', alpha);
        mean_diff = mean(vals_a) - mean(vals_b);
        fprintf('  [ttest2] %s vs %s: mean=%.3f%% vs %.3f%%, Δ=%.3f%%, t(%g)=%.3f, p=%.3g, %g%% CI=[%.3f, %.3f]%%\n', ...
            cond_labels{1}, cond_labels{2}, mean(vals_a), mean(vals_b), mean_diff, stats.df, stats.tstat, p_val, ...
            conf_level*100, ci(1), ci(2));
    case {'ranksum','wilcoxon'}
        p_val = ranksum(vals_a, vals_b);
        fprintf('  [ranksum] %s vs %s: median=%.3f%% vs %.3f%%, p=%.3g (CI N/A)\n', ...
            cond_labels{1}, cond_labels{2}, median(vals_a), median(vals_b), p_val);
    otherwise
        warning('Unknown between-test method "%s". Supported: ttest2, ranksum', method);
end
end

