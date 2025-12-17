function combined_results = CombineTemporalModelResults(varargin)
% CombineTemporalModelResults  Merge multiple TemporalModel results into one struct.
%
%   combined = CombineTemporalModelResults({fileA, fileB})
%   combined = CombineTemporalModelResults(results_struct_A, results_struct_B, ...)
%   combined = CombineTemporalModelResults(..., opts)
%
%   result_inputs may be MAT-file paths (containing a "results" struct) or the
%   structs themselves. Inputs can be provided as a cell array or as separate
%   arguments. This utility concatenates all ROI-level fields (temporal kernels,
%   event kernels, performance, contribution matrices, etc.) so downstream
%   plotting functions can treat data from multiple animals as a single dataset.
%
%   Options (all optional):
%       opts.append_session_label (default true) :
%           Prefix ROI names with their session label to keep them unique.
%       opts.roi_name_format (default '%s__%s') :
%           sprintf format for prefixing ROI names (session, roi).
%       opts.session_labels :
%           Cell array of custom labels (same length as result_inputs).
%       opts.show_summary (default true) :
%           Print a short summary of the combined composition.
%
%   IMPORTANT:
%       Spatial brain-map plots expect all ROIs to share the same anatomical
%       mask. Because this script mixes sessions/animals, metadata.source_roi_file
%       is cleared and the per-session paths are stored under
%       metadata.source_roi_files instead. Spatial plots will politely skip in
%       this scenario.

if nargin < 1
    error('CombineTemporalModelResults:MissingInputs', ...
        'Provide at least one results struct or MAT-file.');
end
[result_inputs, opts] = parse_inputs(varargin{:});
if isempty(result_inputs)
    error('CombineTemporalModelResults:NoValidInputs', ...
        'Result input list is empty after parsing.');
end

if ~isfield(opts, 'append_session_label') || isempty(opts.append_session_label)
    opts.append_session_label = true;
end
if ~isfield(opts, 'roi_name_format') || isempty(opts.roi_name_format)
    opts.roi_name_format = '%s__%s';
end
if ~isfield(opts, 'session_labels')
    opts.session_labels = {};
end
if ~isfield(opts, 'show_summary') || isempty(opts.show_summary)
    opts.show_summary = true;
end

results_list = cell(numel(result_inputs), 1);
for i = 1:numel(result_inputs)
    results_list{i} = load_result_struct(result_inputs{i});
end

n_sessions = numel(results_list);
if n_sessions < 2
    warning('CombineTemporalModelResults:SingleInput', ...
        'Only one session supplied; returning original results.');
    combined_results = results_list{1};
    return;
end

session_labels = derive_session_labels(results_list, opts.session_labels);

base = results_list{1};
temporal_fields = collect_union_fieldnames(results_list, @(res) res.temporal_kernels);
performance_fields = collect_union_fieldnames(results_list, @(res) res.performance);
temporal_template = empty_struct_with_fields(temporal_fields);
performance_template = empty_struct_with_fields(performance_fields);
combined_temporal = repmat(temporal_template, 0, 1);
combined_event = cell(0, 1);
combined_performance = repmat(performance_template, 0, 1);
combined_predictions = cell(n_sessions, 1);
combined_roi_names = {};
combined_session_refs = {};
combined_contributions = init_combined_contributions(base);

lag_template = get_motion_lag_template(base);
group_labels_ref = base.contributions.group_labels(:);

roi_counter = 0;
for s = 1:n_sessions
    res = results_list{s};
    ensure_group_compatibility(group_labels_ref, res.contributions.group_labels);
    ensure_motion_lag_match(lag_template, get_motion_lag_template(res));

    session_label = session_labels{s};
    combined_predictions{s} = res.predictions;

    rois_current = numel(res.temporal_kernels);
    for r = 1:rois_current
        roi_counter = roi_counter + 1;
        tk = align_struct_fields(res.temporal_kernels(r), temporal_fields, temporal_template);
        perf = align_struct_fields(res.performance(r), performance_fields, performance_template);
        roi_name_out = tk.roi_name;
        if opts.append_session_label
            roi_name_out = sprintf(opts.roi_name_format, sanitize_label(session_label), tk.roi_name);
        end
        tk.roi_name = roi_name_out;
        perf.roi_name = roi_name_out;
        combined_temporal(roi_counter, 1) = tk;
        combined_performance(roi_counter, 1) = perf;
        combined_event{roi_counter, 1} = res.event_kernels{r};
        combined_roi_names{roi_counter, 1} = roi_name_out;
        combined_session_refs{roi_counter, 1} = session_label;
    end

    combined_contributions = append_contribution_fields(combined_contributions, res.contributions);
end

comparison = build_combined_comparison(combined_temporal, combined_performance, combined_roi_names);

combined_results = struct();
combined_results.temporal_kernels = combined_temporal;
combined_results.event_kernels = combined_event;
combined_results.performance = combined_performance;
combined_results.predictions = struct('sessions', {session_labels}, ...
    'per_session', {combined_predictions});
combined_results.comparison = comparison;
combined_results.design_matrix = base.design_matrix;
combined_results.contributions = combined_contributions;

meta = base.metadata;
meta.target_neural_rois = combined_roi_names;
meta.n_rois = numel(combined_roi_names);
meta.session_files_combined = get_session_files(results_list);
meta.session_labels_combined = session_labels;
meta.source_roi_files = get_source_roi_paths(results_list);
meta.source_roi_file = []; % Disable single-brain assumptions
combined_results.metadata = meta;

if opts.show_summary
    fprintf('Combined %d sessions -> %d ROIs total.\n', n_sessions, numel(combined_roi_names));
    for s = 1:n_sessions
        fprintf('  %s : %d ROIs\n', session_labels{s}, numel(results_list{s}.temporal_kernels));
    end
end
end

%% ---------------- Helper functions ----------------

function [inputs, opts] = parse_inputs(varargin)
if isempty(varargin)
    inputs = {};
    opts = struct();
    return;
end

maybe_opts = varargin{end};
if isstruct(maybe_opts) && (isfield(maybe_opts, 'append_session_label') || ...
        isfield(maybe_opts, 'roi_name_format') || isfield(maybe_opts, 'session_labels') || ...
        isfield(maybe_opts, 'show_summary'))
    opts = maybe_opts;
    varargin(end) = [];
else
    opts = struct();
end

if numel(varargin) == 1 && iscell(varargin{1})
    inputs = varargin{1};
else
    inputs = varargin;
end
end

function names = collect_union_fieldnames(results_list, accessor)
names = {};
for i = 1:numel(results_list)
    res = results_list{i};
    if isempty(res) || ~isa(res, 'struct')
        continue;
    end
    arr = accessor(res);
    if isempty(arr)
        continue;
    end
    fields = fieldnames(arr);
    names = merge_field_lists(names, fields);
end
end

function merged = merge_field_lists(existing, new_fields)
merged = existing;
for i = 1:numel(new_fields)
    fld = new_fields{i};
    if ~any(strcmp(merged, fld))
        merged{end+1} = fld; %#ok<AGROW>
    end
end
end

function template = empty_struct_with_fields(field_list)
if isempty(field_list)
    template = struct();
    return;
end
values = cell(numel(field_list), 1);
for i = 1:numel(values)
    values{i} = [];
end
template = cell2struct(values, field_list, 1);
end

function out_struct = align_struct_fields(s, field_order, template)
if nargin < 3 || isempty(template)
    template = empty_struct_with_fields(field_order);
end
out_struct = template;
if isempty(s)
    return;
end
current_fields = fieldnames(s);
for i = 1:numel(current_fields)
    fld = current_fields{i};
    out_struct.(fld) = s.(fld);
end
end

function res = load_result_struct(entry)
if ischar(entry) || isstring(entry)
    file_path = char(entry);
    if exist(file_path, 'file') ~= 2
        error('CombineTemporalModelResults:FileNotFound', ...
            'Results file "%s" not found.', file_path);
    end
    data = load(file_path, 'results');
    if ~isfield(data, 'results')
        error('CombineTemporalModelResults:MissingResultsStruct', ...
            'File %s does not contain a "results" struct.', file_path);
    end
    res = data.results;
elseif isstruct(entry)
    if isfield(entry, 'event_kernels') && isfield(entry, 'temporal_kernels')
        res = entry;
    elseif isfield(entry, 'results')
        res = entry.results;
    else
        error('CombineTemporalModelResults:InvalidStruct', ...
            'Provided struct does not look like a TemporalModel results struct.');
    end
else
    error('CombineTemporalModelResults:UnsupportedInput', ...
        'Unsupported input type: %s', class(entry));
end
end

function labels = derive_session_labels(results_list, custom_labels)
n = numel(results_list);
labels = cell(n, 1);
if ~isempty(custom_labels)
    if numel(custom_labels) ~= n
        error('CombineTemporalModelResults:LabelCountMismatch', ...
            'opts.session_labels must match the number of inputs.');
    end
    labels = custom_labels(:);
    return;
end
for i = 1:n
    res = results_list{i};
    meta = struct();
    if isfield(res, 'metadata')
        meta = res.metadata;
    end
    parts = {};
    if isfield(meta, 'session_mouse_label') && ~isempty(meta.session_mouse_label)
        parts{end+1} = char(meta.session_mouse_label); %#ok<AGROW>
    end
    if isfield(meta, 'session_recording_label') && ~isempty(meta.session_recording_label)
        parts{end+1} = char(meta.session_recording_label); %#ok<AGROW>
    end
    if isempty(parts)
        if isfield(meta, 'session_file') && ~isempty(meta.session_file)
            [~, base, ~] = fileparts(char(meta.session_file));
            parts = {base};
        else
            parts = {sprintf('session_%02d', i)};
        end
    end
    labels{i} = strjoin(parts, '_');
end
end

function lag_template = get_motion_lag_template(results_struct)
if isempty(results_struct.temporal_kernels)
    error('CombineTemporalModelResults:MissingTemporalKernels', ...
        'Results struct lacks temporal kernels.');
end
lag_template = results_struct.temporal_kernels(1).lag_times_sec(:);
end

function ensure_motion_lag_match(template, candidate)
if numel(template) ~= numel(candidate) || max(abs(template - candidate)) > 1e-9
    error('CombineTemporalModelResults:LagMismatch', ...
        'Motion kernel lag configuration differs between sessions.');
end
end

function ensure_group_compatibility(ref_labels, new_labels)
ref = ref_labels(:);
new = new_labels(:);
if numel(ref) ~= numel(new) || any(~strcmp(ref, new))
    error('CombineTemporalModelResults:GroupLabelMismatch', ...
        'Event group labels do not match across sessions.');
end
end

function contributions = init_combined_contributions(base)
contributions = base.contributions;
% zero out ROI-dependent arrays to prepare concatenation
contributions.group_single_R2 = [];
contributions.group_shuffle_R2 = [];
contributions.group_explained_mean = [];
contributions.group_explained_std = [];
contributions.group_unique_mean = [];
contributions.group_unique_std = [];
contributions.R2_cv_percent = [];
contributions.R2_mean_percent = [];
end

function combined = append_contribution_fields(combined, new_data)
combined.group_single_R2 = cat(2, combined.group_single_R2, new_data.group_single_R2);
combined.group_shuffle_R2 = cat(2, combined.group_shuffle_R2, new_data.group_shuffle_R2);
combined.group_explained_mean = cat(2, combined.group_explained_mean, new_data.group_explained_mean);
combined.group_explained_std = cat(2, combined.group_explained_std, new_data.group_explained_std);
combined.group_unique_mean = cat(2, combined.group_unique_mean, new_data.group_unique_mean);
combined.group_unique_std = cat(2, combined.group_unique_std, new_data.group_unique_std);
combined.R2_cv_percent = cat(1, combined.R2_cv_percent, new_data.R2_cv_percent);
combined.R2_mean_percent = cat(1, combined.R2_mean_percent, new_data.R2_mean_percent(:));
end

function comparison = build_combined_comparison(temp_kernels, perf, roi_names)
n_rois = numel(temp_kernels);
if n_rois == 0
    comparison = struct('roi_names', {{}}, 'beta_matrix_cv', [], ...
        'R2_all_rois', [], 'peak_lags_all_sec', [], 'peak_betas_all', []);
    return;
end
n_lags = numel(temp_kernels(1).beta_cv_mean);
beta_matrix = nan(n_lags, n_rois);
for r = 1:n_rois
    beta_curve = temp_kernels(r).beta_cv_mean(:);
    if numel(beta_curve) ~= n_lags
        error('CombineTemporalModelResults:BetaLengthMismatch', ...
            'ROI %d beta length mismatch across sessions.', r);
    end
    beta_matrix(:, r) = beta_curve;
end
comparison = struct();
comparison.roi_names = roi_names(:);
comparison.beta_matrix_cv = beta_matrix;
comparison.R2_all_rois = arrayfun(@(p) p.R2_cv_mean, perf);
comparison.peak_lags_all_sec = arrayfun(@(tk) tk.peak_lag_sec, temp_kernels);
comparison.peak_betas_all = arrayfun(@(tk) tk.peak_beta, temp_kernels);
end

function session_files = get_session_files(results_list)
n = numel(results_list);
session_files = cell(n, 1);
for i = 1:n
    session_files{i} = '';
    if isfield(results_list{i}, 'metadata') && isfield(results_list{i}.metadata, 'session_file')
        session_files{i} = results_list{i}.metadata.session_file;
    end
end
end

function roi_paths = get_source_roi_paths(results_list)
n = numel(results_list);
roi_paths = cell(n, 1);
for i = 1:n
    roi_paths{i} = [];
    if isfield(results_list{i}, 'metadata') && ...
            isfield(results_list{i}.metadata, 'source_roi_file')
        roi_paths{i} = results_list{i}.metadata.source_roi_file;
    end
end
end

function label = sanitize_label(str_in)
label = strrep(char(str_in), ' ', '');
label = regexprep(label, '[^a-zA-Z0-9_-]', '_');
end
