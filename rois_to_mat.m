function ROI = rois_to_mat(fluo_file, motion_file, behav_roi_file, neural_roi_file, vascular_mask_file, brain_mask_file, opts)
% rois_to_mat Extract fluorescence + behavior ROI traces and save ROI.mat
%
%   ROI = rois_to_mat(fluo_file, motion_file, behav_roi_file, neural_roi_file, ...)
%
%   REQUIRED INPUTS:
%       fluo_file          - Hemodynamic-corrected fluorescence movie (.dat)
%       motion_file        - Motion-energy movie (.dat) for behavioral ROIs (use [] to skip)
%       behav_roi_file     - Behavioral ROI masks (.roimsk or .mat with ROI_info)
%       neural_roi_file    - Fluorescence ROI masks (.roimsk or neural SessionROIData)
%       vascular_mask_file - Optional vascular exclusion mask (.roimsk) [] to skip
%       brain_mask_file    - Optional brain/anatomical mask (.roimsk) [] to skip
%       opts               - (optional) struct with Name-Value overrides
%
%   KEY OPTIONS (opts fields):
%       chunk_size_frames  - Frames per IO chunk when reading .dat (default 500)
%       ComputeDFF         - If true (default), fluorescence traces converted to dF/F via dFF_phasic
%       DFFParams          - Struct of parameters passed to dFF_phasic (default struct())
%       Fs                 - Sampling rate override for dFF (default: fluorescence metadata, fallback 10 Hz)
%       OutputFile         - Path for ROI.mat (default: <fluorescence folder>/ROI.mat)
%       SaveOutput         - Save ROI struct to disk (default true)
%
%   OUTPUT (ROI struct):
%       .metadata          - provenance, timestamps, source paths
%       .modalities.fluorescence
%           .data          - [T x N] fluorescence dF/F (or raw if ComputeDFF==false)
%           .raw_data      - [T x N] original fluorescence (when dF/F computed)
%           .labels        - ROI names from fluoROIs.roimsk
%           .time          - time vector (seconds)
%           .sampling_rate - Hz from metadata (or opts.Fs)
%           .traces_table  - table with labeled columns (dF/F)
%       .modalities.behavior (present when motion + behav ROI inputs supplied)
%           .data          - [Tb x Nb] motion-energy ROI means (raw)
%           .labels        - ROI names (e.g., 'Face','Licking','Forelimbs')
%           .time          - time vector (seconds)
%           .sampling_rate - Hz from motion metadata
%           .traces_table  - table with labeled columns
%       Top-level convenience aliases (.data/.labels/...) mirror fluorescence modality
%
%   The function never re-estimates ROIs; it simply loads masks from the
%   provided .roimsk/.mat files, averages the specified .dat movies within
%   each ROI, (optionally) runs dFF_phasic on fluorescence traces assuming
%   10 Hz unless metadata overrides, and saves ROI.mat.

    if nargin < 7 || isempty(opts)
        opts = struct();
    end

    defaults = struct('chunk_size_frames', 500, 'ComputeDFF', true, ...
        'DFFParams', struct(), 'Fs', [], 'OutputFile', '', 'SaveOutput', true);
    opts = populate_defaults(opts, defaults);

    if nargin < 2 || isempty(motion_file)
        motion_file = '';
    end
    if nargin < 3 || isempty(behav_roi_file)
        behav_roi_file = '';
    end
    if nargin < 5 || isempty(vascular_mask_file)
        vascular_mask_file = '';
    end
    if nargin < 6 || isempty(brain_mask_file)
        brain_mask_file = '';
    end

    assert_file_exists(fluo_file, 'fluo_file');
    assert_file_exists(neural_roi_file, 'neural_roi_file');
    if ~isempty(motion_file)
        assert_file_exists(motion_file, 'motion_file');
    end
    if ~isempty(behav_roi_file)
        assert_file_exists(behav_roi_file, 'behav_roi_file');
    end
    if ~isempty(vascular_mask_file)
        assert_file_exists(vascular_mask_file, 'vascular_mask_file');
    end
    if ~isempty(brain_mask_file)
        assert_file_exists(brain_mask_file, 'brain_mask_file');
    end

    if isempty(opts.OutputFile)
        opts.OutputFile = fullfile(fileparts(fluo_file), 'ROI.mat');
    end

    fprintf('=== rois_to_mat ===\n');
    fprintf('  Fluorescence movie : %s\n', fluo_file);
    if ~isempty(motion_file)
        fprintf('  Motion movie       : %s\n', motion_file);
    end
    fprintf('  Neural ROI file    : %s\n', neural_roi_file);
    if ~isempty(behav_roi_file)
        fprintf('  Behavior ROI file  : %s\n', behav_roi_file);
    end
    fprintf('  Output             : %s\n', opts.OutputFile);

    %% Fluorescence ROI extraction
    [fluo_Y, fluo_X, fluo_T, fluo_freq] = load_dat_metadata(fluo_file);
    fprintf('Fluorescence metadata: %d x %d x %d frames @ %.2f Hz\n', fluo_Y, fluo_X, fluo_T, fluo_freq);

    final_mask = true(fluo_Y, fluo_X);
    if ~isempty(vascular_mask_file)
        vascular_mask = load_single_roi_mask(vascular_mask_file, [fluo_Y, fluo_X]);
        final_mask = final_mask & ~vascular_mask;
        fprintf('  Applied vascular mask (%d pixels removed)\n', sum(vascular_mask(:)));
    end
    if ~isempty(brain_mask_file)
        brain_mask = load_single_roi_mask(brain_mask_file, [fluo_Y, fluo_X]);
        final_mask = final_mask & brain_mask;
        fprintf('  Applied brain mask (%d pixels kept)\n', sum(brain_mask(:)));
    end

    [roi_info, roi_source_path] = load_neural_roi_info(neural_roi_file);
    [fluo_idx, fluo_labels, fluo_pixels] = prepare_roi_indices(roi_info, final_mask, [fluo_Y, fluo_X]);
    fprintf('  Fluorescence ROIs loaded: %s\n', strjoin(fluo_labels, ', '));

    fluo_data_raw = extract_roi_traces_multi(fluo_file, [fluo_Y, fluo_X], fluo_T, fluo_idx, opts.chunk_size_frames, 'fluorescence');
    fprintf('  Fluorescence traces extracted (%d samples x %d ROIs)\n', size(fluo_data_raw,1), size(fluo_data_raw,2));

    Fs = opts.Fs;
    if isempty(Fs)
        Fs = fluo_freq;
    end
    if isempty(Fs) || Fs <= 0
        Fs = 10;
    end

    Fluo = struct();
    Fluo.labels = fluo_labels;
    Fluo.pixel_counts = fluo_pixels;
    Fluo.sample_rate = Fs;
    Fluo.time = build_time_vector(size(fluo_data_raw,1), Fs);

    if opts.ComputeDFF
        if exist('dFF_phasic', 'file') ~= 2
            error('dFF_phasic.m not found on MATLAB path.');
        end
        Fluo.raw_data = fluo_data_raw;
        fluo_dff = compute_dff_matrix(fluo_data_raw, opts.DFFParams, Fs, Fluo.labels);
        Fluo.data = fluo_dff;
        Fluo.type = 'dFF_phasic';
        Fluo.dff_params = pack_params(opts.DFFParams, Fs);
    else
        Fluo.data = fluo_data_raw;
        Fluo.type = 'raw';
        Fluo.raw_data = [];
        Fluo.dff_params = struct();
    end

    Fluo.n_samples = size(Fluo.data,1);
    Fluo.n_rois = size(Fluo.data,2);
    Fluo.traces_table = build_table(Fluo.data, Fluo.labels);

    %% Behavior ROI extraction (optional)
    Beh = [];
    if ~isempty(motion_file) && ~isempty(behav_roi_file)
        [me_Y, me_X, me_T, me_freq] = load_dat_metadata(motion_file);
        fprintf('Behavior metadata: %d x %d x %d frames @ %.2f Hz\n', me_Y, me_X, me_T, me_freq);
        behav_data = load(behav_roi_file, '-mat');
        if ~isfield(behav_data, 'ROI_info')
            error('Behavior ROI file %s does not contain ROI_info.', behav_roi_file);
        end
        [beh_idx, beh_labels, beh_pixels] = prepare_roi_indices(behav_data.ROI_info, true(me_Y, me_X), [me_Y, me_X]);
        fprintf('  Behavior ROIs loaded: %s\n', strjoin(beh_labels, ', '));
        beh_data = extract_roi_traces_multi(motion_file, [me_Y, me_X], me_T, beh_idx, opts.chunk_size_frames, 'behavior');
        Beh = struct();
        Beh.data = beh_data;
        Beh.labels = beh_labels;
        Beh.pixel_counts = beh_pixels;
        Beh.type = 'motion_energy';
        Beh.sample_rate = me_freq;
        Beh.time = build_time_vector(size(beh_data,1), me_freq);
        Beh.n_samples = size(beh_data,1);
        Beh.n_rois = size(beh_data,2);
        Beh.traces_table = build_table(beh_data, beh_labels);
    end

    %% Assemble ROI struct
    ROI = struct();
    ROI.metadata = struct();
    ROI.metadata.created_by = mfilename;
    ROI.metadata.created_at = datestr(now, 31);
    ROI.metadata.version = '3.0.0';
    ROI.metadata.source = struct('fluorescence_movie', fluo_file, ...
        'motion_movie', motion_file, 'neural_roi_file', roi_source_path, ...
        'behavior_roi_file', behav_roi_file, 'vascular_mask_file', vascular_mask_file, ...
        'brain_mask_file', brain_mask_file);
    mods = {'fluorescence'};
    if ~isempty(Beh)
        mods{end+1} = 'behavior';
    end
    ROI.metadata.modalities = mods;

    ROI.modalities = struct();
    ROI.modalities.fluorescence = Fluo;
    if ~isempty(Beh)
        ROI.modalities.behavior = Beh;
    end

    ROI.time = Fluo.time;
    ROI.data = Fluo.data;
    ROI.labels = Fluo.labels;
    ROI.type = Fluo.type;
    ROI.raw_data = Fluo.raw_data;
    ROI.dff_params = Fluo.dff_params;
    ROI.sample_rate = Fluo.sample_rate;
    ROI.n_samples = Fluo.n_samples;
    ROI.n_rois = Fluo.n_rois;
    ROI.traces_table = Fluo.traces_table;
    ROI.paths = ROI.metadata.source;

    if opts.SaveOutput
        save(opts.OutputFile, 'ROI', '-v7.3');
        fprintf('ROI struct saved to %s\n', opts.OutputFile);
    else
        fprintf('SaveOutput == false; skipping save.\n');
    end
end

%% ================= Helper functions =================

function opts = populate_defaults(opts, defaults)
    fields = fieldnames(defaults);
    for i = 1:numel(fields)
        name = fields{i};
        if ~isfield(opts, name) || isempty(opts.(name))
            opts.(name) = defaults.(name);
        end
    end
end

function assert_file_exists(path_str, label)
    if isempty(path_str)
        return;
    end
    if exist(path_str, 'file') ~= 2
        error('Required %s not found: %s', label, char(path_str));
    end
end

function [dimY, dimX, n_frames, freq] = load_dat_metadata(dat_path)
    [dat_dir, dat_name, ~] = fileparts(dat_path);
    meta_file = fullfile(dat_dir, [dat_name '.mat']);
    if exist(meta_file, 'file') ~= 2
        error('Metadata file %s not found for %s', meta_file, dat_path);
    end
    meta = load(meta_file);
    dimY = meta.datSize(1);
    dimX = meta.datSize(2);
    freq = meta.Freq;
    info = dir(dat_path);
    pixels_per_frame = dimY * dimX;
    n_frames = info.bytes / (pixels_per_frame * 4);
    if abs(n_frames - round(n_frames)) > 1e-3
        error('Non-integer frame count detected for %s', dat_path);
    end
    n_frames = round(n_frames);
end

function mask = load_single_roi_mask(mask_file, dims)
    data = load(mask_file, '-mat');
    if ~isfield(data, 'ROI_info')
        error('Mask file %s does not contain ROI_info', mask_file);
    end
    mask = logical(data.ROI_info(1).Stats.ROI_binary_mask);
    if ~isequal(size(mask), dims)
        error('Mask dimension mismatch for %s', mask_file);
    end
end

function [roi_info, roi_source_path] = load_neural_roi_info(neural_roi_file)
    data = load(neural_roi_file, '-mat');
    if isfield(data, 'ROI_info')
        roi_info = data.ROI_info;
        roi_source_path = neural_roi_file;
        return;
    end
    if isfield(data, 'SessionROIData')
        session_roi = data.SessionROIData;
        if isstruct(session_roi) && isfield(session_roi, 'AnalysisOptions') && ...
                isfield(session_roi.AnalysisOptions, 'FluoROI_filename')
            roi_reference = session_roi.AnalysisOptions.FluoROI_filename;
            roi_source_path = resolve_referenced_roi_file(neural_roi_file, roi_reference);
            roi_data = load(roi_source_path, '-mat');
            if ~isfield(roi_data, 'ROI_info')
                error('Referenced ROI file %s lacks ROI_info.', roi_source_path);
            end
            roi_info = roi_data.ROI_info;
            return;
        end
        error('SessionROIData in %s missing AnalysisOptions.FluoROI_filename.', neural_roi_file);
    end
    error('Neural ROI file %s lacks ROI_info/SessionROIData.', neural_roi_file);
end

function roi_path = resolve_referenced_roi_file(base_file, roi_reference)
    if isempty(roi_reference)
        error('Empty fluorescence ROI filename referenced from %s.', base_file);
    end
    if exist(roi_reference, 'file') == 2
        roi_path = roi_reference;
        return;
    end
    base_dir = fileparts(base_file);
    candidate = fullfile(base_dir, roi_reference);
    if exist(candidate, 'file') == 2
        roi_path = candidate;
        return;
    end
    current_dir = base_dir;
    max_levels = 5;
    for depth = 1:max_levels
        parent_dir = fileparts(current_dir);
        if strcmp(parent_dir, current_dir)
            break;
        end
        candidate = fullfile(parent_dir, roi_reference);
        if exist(candidate, 'file') == 2
            roi_path = candidate;
            return;
        end
        current_dir = parent_dir;
    end
    error('Unable to resolve fluorescence ROI file "%s" referenced from %s.', roi_reference, base_file);
end

function [roi_idx, labels, pix_counts] = prepare_roi_indices(roi_info, mask_limit, dims)
    n = numel(roi_info);
    roi_idx = cell(1, n);
    labels = cell(1, n);
    pix_counts = zeros(1, n);
    keep = true(1, n);
    for i = 1:n
        labels{i} = char(roi_info(i).Name);
        mask = logical(roi_info(i).Stats.ROI_binary_mask);
        if ~isequal(size(mask), dims)
            error('ROI %s has mismatched dimensions.', labels{i});
        end
        mask = mask & mask_limit;
        idx = find(mask(:));
        pix_counts(i) = numel(idx);
        if isempty(idx)
            warning('ROI %s has 0 pixels after masking and will be dropped.', labels{i});
            keep(i) = false;
        else
            roi_idx{i} = idx;
        end
    end
    roi_idx = roi_idx(keep);
    labels = labels(keep);
    pix_counts = pix_counts(keep);
end

function traces = extract_roi_traces_multi(dat_file, dims, n_frames, roi_idx, chunk_size, label)
    n_rois = numel(roi_idx);
    traces = zeros(n_frames, n_rois);
    pixels_per_frame = prod(dims);
    fid = fopen(dat_file, 'r');
    if fid < 0
        error('Unable to open %s', dat_file);
    end
    cleanup = onCleanup(@() fclose(fid));
    n_chunks = ceil(n_frames / chunk_size);
    for chunk = 1:n_chunks
        start_idx = (chunk - 1) * chunk_size + 1;
        end_idx = min(chunk * chunk_size, n_frames);
        frames_this_chunk = end_idx - start_idx + 1;
        raw = fread(fid, pixels_per_frame * frames_this_chunk, 'single=>double');
        if numel(raw) < pixels_per_frame * frames_this_chunk
            error('Unexpected EOF while reading %s', dat_file);
        end
        raw = reshape(raw, pixels_per_frame, frames_this_chunk);
        for r = 1:n_rois
            idx = roi_idx{r};
            if isempty(idx)
                continue;
            end
            traces(start_idx:end_idx, r) = mean(raw(idx, :), 1)';
        end
        if mod(chunk, 20) == 0 || chunk == n_chunks
            fprintf('  %s chunk %d/%d processed\n', label, chunk, n_chunks);
        end
    end
end

function tv = build_time_vector(n_samples, Fs)
    if isempty(Fs) || Fs <= 0
        tv = (0:n_samples-1)';
    else
        tv = (0:n_samples-1)' ./ Fs;
    end
end

function tbl = build_table(data, labels)
    try
        tbl = array2table(data, 'VariableNames', sanitize_varnames(labels));
    catch
        tbl = array2table(data);
        tbl.Properties.VariableNames = generate_default_labels(size(data,2));
    end
end

function Y = compute_dff_matrix(X, params, Fs, labels)
    T = size(X,1);
    N = size(X,2);
    Y = zeros(T, N);
    for c = 1:N
        if nargin < 4 || isempty(labels)
            roi_label = sprintf('ROI_%02d', c);
        else
            roi_label = labels{c};
        end
        Y(:,c) = call_dFF_phasic(X(:,c), Fs, params, roi_label);
    end
end

function y = call_dFF_phasic(x, Fs, params, roi_label)
    x = ensure_column(x);
    if nargin < 3 || ~isstruct(params)
        opts_local = struct();
    else
        opts_local = params;
    end
    if ~isfield(opts_local, 'roi_name') && nargin >= 4 && ~isempty(roi_label)
        opts_local.roi_name = roi_label;
    end
    if ~isfield(opts_local, 'show_plot')
        opts_local.show_plot = true;
    end
    try
        y = dFF_phasic(x, Fs, opts_local);
    catch ME
        error('dFF_phasic call failed for ROI "%s": %s', roi_label, ME.message);
    end
    y = ensure_column(y);
end

function y = ensure_column(y)
    if isrow(y)
        y = y';
    end
end

function vn = sanitize_varnames(names)
    if isstring(names)
        names = cellstr(names);
    end
    vn = matlab.lang.makeUniqueStrings(matlab.lang.makeValidName(names));
end

function names = generate_default_labels(N)
    names = arrayfun(@(k)sprintf('ROI_%03d', k), 1:N, 'UniformOutput', false);
end

function p = pack_params(params, Fs)
    if ~isstruct(params)
        params = struct();
    end
    params.Fs = Fs;
    p = params;
end
