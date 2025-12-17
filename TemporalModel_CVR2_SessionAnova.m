function anova_results = TemporalModel_CVR2_SessionAnova(varargin)
% TemporalModel_CVR2_SessionAnova  Compare per-variable cvR^2 across sessions.
%
%   results = TemporalModel_CVR2_SessionAnova(result_inputs)
%   results = TemporalModel_CVR2_SessionAnova(result_inputs, opts)
%   results = TemporalModel_CVR2_SessionAnova(result1, result2, ..., opts)
%
%   Loads multiple TemporalModelSoundLevelFull_Contributions result files and runs a
%   separate one-way ANOVA for each predictor group (variable), comparing the
%   distribution of single-variable cvR^2 values across sessions.
%
%   result_inputs : Outputs from TemporalModelSoundLevelFull_Contributions provided either
%                   as MAT file paths/patterns/directories or as in-memory results structs.
%                   Cell arrays can mix structs and paths. You can also pass multiple inputs
%                   directly without wrapping them in a cell. If omitted, a picker opens.
%
%   opts.show_anova_figures    : true to display the ANOVA mean plots (default false).
%   opts.min_points_per_group  : minimum number of observations required per session
%                                before it is included in the ANOVA (default 2).
%   opts.summary_csv           : optional path to write the per-variable summary table.
%   opts.session_csv           : optional path to write per-variable/session breakdowns.
%
%   The returned struct contains:
%       .summary           - table of p-values and ANOVA statistics per variable
%       .session_details   - table of per-session sample counts and summary stats
%       .session_names     - cell array of derived session labels
%       .input_sources     - descriptive labels (file paths or struct tags) for sessions
%       .input_files       - alias for input_sources (for backwards compatibility)
%       .source_entries    - struct array describing each loaded session source
%       .variable_data     - struct array holding the raw cvR^2 vectors per session

if nargin == 0
    result_inputs = prompt_for_files();
    opts = struct();
else
    [result_inputs, opts] = parse_inputs(varargin{:});
    if isempty(result_inputs)
        result_inputs = prompt_for_files();
    end
end

if ~isfield(opts, 'show_anova_figures') || isempty(opts.show_anova_figures)
    opts.show_anova_figures = false;
end
if ~isfield(opts, 'min_points_per_group') || isempty(opts.min_points_per_group)
    opts.min_points_per_group = 2;
end
if ~isfield(opts, 'summary_csv')
    opts.summary_csv = '';
end
if ~isfield(opts, 'session_csv')
    opts.session_csv = '';
end

sources = gather_result_sources(result_inputs);
if isempty(sources)
    error('TemporalModel_CVR2_SessionAnova:MissingFiles', ...
        'No valid MAT files or results structs were provided.');
end

[sessions, session_sources] = collect_session_distributions(sources);
if isempty(sessions)
    error('TemporalModel_CVR2_SessionAnova:NoValidSessions', ...
        'None of the provided files contained usable contributions data.');
end

session_names = {sessions.name};
all_variables = gather_all_variables(sessions);
if isempty(all_variables)
    error('TemporalModel_CVR2_SessionAnova:NoVariables', ...
        'No predictor groups were found across the provided sessions.');
end

display_mode = ternary(opts.show_anova_figures, 'on', 'off');
summary_rows = cell(numel(all_variables), 5);
detail_rows = {};
variable_data = repmat(struct('name', '', 'session_values', []), numel(all_variables), 1);

fprintf('=== Running cvR^2 ANOVAs across %d session(s) ===\n', numel(sessions));
for vi = 1:numel(all_variables)
    var_name = all_variables{vi};
    [values_per_session, session_stats] = extract_variable_values(sessions, var_name, opts.min_points_per_group);
    variable_data(vi).name = var_name;
    variable_data(vi).session_values = values_per_session;
    detail_rows = [detail_rows; session_stats]; %#ok<AGROW>

    valid_sessions = find(~cellfun(@isempty, values_per_session));
    if numel(valid_sessions) < 2
        warning('TemporalModel_CVR2_SessionAnova:InsufficientGroups', ...
            'Variable "%s" has <2 sessions with data; ANOVA skipped.', var_name);
        summary_rows(vi, :) = {var_name, NaN, NaN, NaN, NaN};
        continue;
    end

    data_vec = [];
    group_vec = {};
    for si = valid_sessions(:)'
        vals = values_per_session{si};
        data_vec = [data_vec; vals(:)];
        group_vec = [group_vec; repmat(session_names(si), numel(vals), 1)]; %#ok<AGROW>
    end

    if numel(data_vec) < numel(valid_sessions)
        warning('TemporalModel_CVR2_SessionAnova:InsufficientReplicates', ...
            'Variable "%s" does not have enough replicates; ANOVA skipped.', var_name);
        summary_rows(vi, :) = {var_name, NaN, NaN, NaN, NaN};
        continue;
    end

    try
        [p_val, tbl] = anova1(data_vec, group_vec, display_mode);
    catch ME
        warning('TemporalModel_CVR2_SessionAnova:AnovaFailure', ...
            'ANOVA failed for "%s": %s', var_name, ME.message);
        p_val = NaN;
        tbl = {};
    end

    if opts.show_anova_figures
        title(sprintf('%s cvR^2 by session', var_name), 'Interpreter', 'none');
        ylabel('cvR^2 (%)');
    end

    [F_stat, df_between, df_within] = extract_anova_numbers(tbl);
    summary_rows(vi, :) = {var_name, p_val, F_stat, df_between, df_within};

    fprintf('  %s: p = %.4g, F(%s,%s) = %.3f\n', var_name, p_val, ...
        format_df(df_between), format_df(df_within), F_stat);
end

summary_tbl = cell2table(summary_rows, ...
    'VariableNames', {'Variable', 'pValue', 'FStatistic', 'dfBetween', 'dfWithin'});
detail_tbl = cell2table(detail_rows, ...
    'VariableNames', {'Variable', 'Session', 'NumSamples', 'MeanCVR2', 'StdCVR2'});

anova_results = struct();
anova_results.summary = summary_tbl;
anova_results.session_details = detail_tbl;
anova_results.session_names = session_names;
anova_results.input_sources = session_sources;
anova_results.input_files = session_sources;
anova_results.source_entries = sessions;
anova_results.variable_data = variable_data;

if ~isempty(opts.summary_csv)
    writetable(summary_tbl, opts.summary_csv);
    fprintf('  Saved summary table -> %s\n', opts.summary_csv);
end
if ~isempty(opts.session_csv)
    writetable(detail_tbl, opts.session_csv);
    fprintf('  Saved session breakdown -> %s\n', opts.session_csv);
end
fprintf('=== cvR^2 ANOVA analysis complete. ===\n');
end

function files = prompt_for_files()
[picked, picked_path] = uigetfile('*.mat', ...
    'Select TemporalModelSoundLevelFull_Contributions results', ...
    'MultiSelect', 'on');
if isequal(picked, 0)
    error('TemporalModel_CVR2_SessionAnova:NoFiles', ...
        'No result files were selected.');
end
if iscell(picked)
    files = fullfile(picked_path, picked);
else
    files = {fullfile(picked_path, picked)};
end
end

function [result_inputs, opts] = parse_inputs(varargin)
if isempty(varargin)
    result_inputs = {};
    opts = struct();
    return;
end

maybe_opts = varargin{end};
if is_options_struct(maybe_opts)
    opts = maybe_opts;
    varargin(end) = [];
else
    opts = struct();
end

if numel(varargin) == 1 && iscell(varargin{1})
    result_inputs = varargin{1};
else
    result_inputs = varargin;
end

if isempty(result_inputs)
    result_inputs = {};
elseif ~iscell(result_inputs)
    result_inputs = {result_inputs};
end
end

function tf = is_options_struct(candidate)
tf = false;
if ~isstruct(candidate) || isempty(candidate)
    return;
end
opt_fields = {'show_anova_figures','min_points_per_group','summary_csv','session_csv'};
tf = any(isfield(candidate, opt_fields));
end

function sources = gather_result_sources(inputs)
sources = struct('type', {}, 'value', {}, 'label', {});
if nargin < 1 || isempty(inputs)
    return;
end

if ~iscell(inputs)
    inputs = {inputs};
end

struct_counter = 0;
for idx = 1:numel(inputs)
    cur = inputs{idx};
    if isempty(cur)
        continue;
    end

    if isstruct(cur)
        [resolved, is_result] = resolve_results_struct(cur);
        if is_result
            struct_counter = struct_counter + 1;
            label = sprintf('workspace_struct_%d', struct_counter);
            sources(end+1) = struct('type', 'struct', 'value', resolved, 'label', label); %#ok<AGROW>
            continue;
        end
        if isfield(cur, 'name')
            if isfield(cur, 'isdir') && cur.isdir
                continue;
            end
            folder = '';
            if isfield(cur, 'folder')
                folder = cur.folder;
            end
            file_path = fullfile(folder, cur.name);
            file_sources = expand_result_file_list(file_path);
            sources = append_file_sources(sources, file_sources);
            continue;
        end
        warning('TemporalModel_CVR2_SessionAnova:UnsupportedStructInput', ...
            'Struct entry at index %d is not a results struct; skipping.', idx);
        continue;
    end

    if ischar(cur) || isstring(cur)
        file_sources = expand_result_file_list(cur);
        sources = append_file_sources(sources, file_sources);
        continue;
    end

    if isa(cur, 'matlab.io.MatFile')
        try
            file_path = cur.Properties.Source;
        catch
            file_path = '';
        end
        if isempty(file_path)
            warning('TemporalModel_CVR2_SessionAnova:MatFileNoSource', ...
                'matlab.io.MatFile entry at index %d lacks a backing file.', idx);
            continue;
        end
        sources(end+1) = struct('type', 'file', 'value', char(file_path), 'label', char(file_path)); %#ok<AGROW>
        continue;
    end

    warning('TemporalModel_CVR2_SessionAnova:UnsupportedInputType', ...
        'Entry %d of type %s is not supported; skipping.', idx, class(cur));
end
end

function sources = append_file_sources(existing_sources, file_paths)
sources = existing_sources;
if isempty(file_paths)
    return;
end
for i = 1:numel(file_paths)
    path = file_paths{i};
    sources(end+1) = struct('type', 'file', 'value', path, 'label', path); %#ok<AGROW>
end
end

function file_list = expand_result_file_list(inputs)
if ischar(inputs) || isstring(inputs)
    inputs = cellstr(inputs);
elseif ~iscell(inputs)
    error('result_inputs must be a string, cell array, or directory path.');
end

file_list = {};
for i = 1:numel(inputs)
    cur = inputs{i};
    if isfolder(cur)
        listing = dir(fullfile(cur, '*.mat'));
    else
        listing = dir(cur);
    end
    for j = 1:numel(listing)
        if listing(j).isdir
            continue;
        end
        file_list{end+1, 1} = fullfile(listing(j).folder, listing(j).name); %#ok<AGROW>
    end
end

if isempty(file_list)
    return;
end
file_list = unique(file_list, 'stable');
end

function [sessions, source_labels] = collect_session_distributions(sources)
sessions = struct('name', {}, 'source_label', {}, 'group_labels', {}, 'values', {});
source_labels = {};
for i = 1:numel(sources)
    src = sources(i);
    res = [];
    switch src.type
        case 'file'
            file_path = src.value;
            data = load(file_path, 'results');
            if ~isfield(data, 'results')
                warning('TemporalModel_CVR2_SessionAnova:MissingResultsVar', ...
                    'File "%s" does not contain a results struct.', file_path);
                continue;
            end
            res = data.results;
        case 'struct'
            res = src.value;
        otherwise
            warning('TemporalModel_CVR2_SessionAnova:UnknownSource', ...
                'Unsupported source type "%s"; skipping.', src.type);
            continue;
    end

    if ~isfield(res, 'contributions') || ~isfield(res.contributions, 'group_single_R2') ...
            || ~isfield(res.contributions, 'group_labels')
        warning('TemporalModel_CVR2_SessionAnova:MissingContributionFields', ...
            'Source "%s" lacks contribution fields; skipping.', src.label);
        continue;
    end

    group_labels = res.contributions.group_labels(:);
    group_values = cell(numel(group_labels), 1);
    for g = 1:numel(group_labels)
        vals = squeeze(res.contributions.group_single_R2(g, :, :));
        vals = vals(:);
        vals = vals(~isnan(vals));
        group_values{g} = vals;
    end

    session_label = derive_session_label(res, src.label);

    sessions(end+1).name = session_label; %#ok<AGROW>
    sessions(end).source_label = src.label;
    sessions(end).group_labels = group_labels;
    sessions(end).values = group_values;
    source_labels{end+1, 1} = src.label; %#ok<AGROW>
end
end

function session_label = derive_session_label(results_struct, source_tag)
session_label = '';
if nargin < 2 || isempty(source_tag)
    source_tag = 'session';
end
if isfield(results_struct, 'metadata')
    meta = results_struct.metadata;
    labels = {};
    if isfield(meta, 'session_mouse_label') && ~isempty(meta.session_mouse_label)
        labels{end+1} = char(meta.session_mouse_label); %#ok<AGROW>
    end
    if isfield(meta, 'session_recording_label') && ~isempty(meta.session_recording_label)
        labels{end+1} = char(meta.session_recording_label); %#ok<AGROW>
    end
    if isempty(labels) && isfield(meta, 'timestamp')
        labels = {char(meta.timestamp)};
    end
    if ~isempty(labels)
        session_label = strjoin(labels, '_');
    end
end

if isempty(session_label)
    [~, base_name, ~] = fileparts(source_tag);
    if isempty(base_name)
        base_name = source_tag;
    end
    session_label = base_name;
end
end

function [res, is_valid] = resolve_results_struct(candidate)
res = [];
is_valid = false;
if ~isstruct(candidate)
    return;
end
if isfield(candidate, 'contributions') && isfield(candidate.contributions, 'group_single_R2')
    res = candidate;
    is_valid = true;
    return;
end
if isfield(candidate, 'results')
    nested = candidate.results;
    if isstruct(nested) && isfield(nested, 'contributions') ...
            && isfield(nested.contributions, 'group_single_R2')
        res = nested;
        is_valid = true;
    end
end
end

function variables = gather_all_variables(sessions)
variables = {};
for i = 1:numel(sessions)
    lbls = sessions(i).group_labels;
    for g = 1:numel(lbls)
        label = lbls{g};
        if ~any(strcmpi(label, variables))
            variables{end+1} = label; %#ok<AGROW>
        end
    end
end
end

function [values_per_session, stats_rows] = extract_variable_values(sessions, target_label, min_points)
values_per_session = cell(numel(sessions), 1);
stats_rows = cell(0, 5);
for si = 1:numel(sessions)
    cur_idx = find(strcmpi(target_label, sessions(si).group_labels), 1);
    if isempty(cur_idx)
        stats_rows(end+1, :) = {target_label, sessions(si).name, 0, NaN, NaN}; %#ok<AGROW>
        continue;
    end
    vals = sessions(si).values{cur_idx};
    if numel(vals) < min_points
        stats_rows(end+1, :) = {target_label, sessions(si).name, numel(vals), NaN, NaN}; %#ok<AGROW>
        continue;
    end
    vals = vals(:);
    vals = vals(~isnan(vals));
    values_per_session{si} = vals;
    stats_rows(end+1, :) = {target_label, sessions(si).name, numel(vals), ...
        mean(vals, 'omitnan'), std(vals, 0, 'omitnan')}; %#ok<AGROW>
end
end

function [F_stat, df_between, df_within] = extract_anova_numbers(tbl)
F_stat = NaN;
df_between = NaN;
df_within = NaN;
if isempty(tbl) || size(tbl, 1) < 3 || size(tbl, 2) < 5
    return;
end
F_stat = to_double(tbl{2, 5});
df_between = to_double(tbl{2, 3});
df_within = to_double(tbl{3, 3});
end

function val = to_double(entry)
if isnumeric(entry)
    val = double(entry);
elseif ischar(entry) || isstring(entry)
    val = str2double(entry);
else
    val = NaN;
end
end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end

function txt = format_df(df_val)
if isnan(df_val)
    txt = 'NaN';
else
    txt = sprintf('%g', df_val);
end
end
