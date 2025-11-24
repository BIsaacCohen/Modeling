function report = BatchRois_to_mat(folderPath)
% BatchRois_to_mat Generate ROI.mat files for every session under folderPath.
%
%   report = BatchRois_to_mat(folderPath) walks each immediate subfolder of
%   folderPath, looks for the standard GRAB-ACh processing files, and calls
%   rois_to_mat using the hard coded naming convention:
%       CtxImg/mov_aligned.dat      - fluorescence movie
%       CtxImg/motion_energy.dat    - behavioral motion movie
%       CtxImg/behavROIs.roimsk     - behavioral ROI masks
%       Allen.roimsk                - fluorescence ROI mask (inside folderPath)
%       VascularMask.roimsk         - vascular exclusion mask (inside folderPath)
%       Mask.roimsk                 - brain mask (inside folderPath)
%
%   ROI.mat is written inside each CtxImg directory. Sessions missing any of
%   the required input files are skipped. Existing ROI.mat files are trusted
%   and left untouched.

    if nargin < 1 || isempty(folderPath)
        error('BatchRois_to_mat:MissingInput', 'folderPath is required.');
    end
    folderPath = convertStringsToChars(folderPath);
    if ~isfolder(folderPath)
        error('BatchRois_to_mat:InvalidFolder', 'folderPath "%s" does not exist.', folderPath);
    end

    shared_inputs = struct( ...
        'neural_rois', fullfile(folderPath, 'Allen.roimsk'), ...
        'vascular_mask', fullfile(folderPath, 'VascularMask.roimsk'), ...
        'brain_mask', fullfile(folderPath, 'Mask.roimsk'));

    shared_names = fieldnames(shared_inputs);
    for k = 1:numel(shared_names)
        this_file = shared_inputs.(shared_names{k});
        if ~isfile(this_file)
            error('BatchRois_to_mat:MissingMask', 'Required file "%s" is missing.', this_file);
        end
    end

    session_dirs = dir(folderPath);
    session_dirs = session_dirs([session_dirs.isdir]);
    session_dirs = session_dirs(~ismember({session_dirs.name}, {'.', '..'}));

    report = repmat(struct('session', '', 'status', '', 'message', '', 'outputFile', ''), numel(session_dirs), 1);

    fprintf('=== BatchRois_to_mat ===\nRoot: %s\nSessions found: %d\n', folderPath, numel(session_dirs));

    for idx = 1:numel(session_dirs)
        session_name = session_dirs(idx).name;
        session_path = fullfile(folderPath, session_name);
        ctx_dir = fullfile(session_path, 'CtxImg');

        report(idx).session = session_name;
        report(idx).outputFile = fullfile(ctx_dir, 'ROI.mat');

        if ~isfolder(ctx_dir)
            msg = 'CtxImg folder not found';
            fprintf(' - %s: %s\n', session_name, msg);
            report(idx).status = 'skipped';
            report(idx).message = msg;
            continue;
        end

        required = struct( ...
            'fluor_movie', fullfile(ctx_dir, 'mov_aligned.dat'), ...
            'motion_movie', fullfile(ctx_dir, 'motion_energy.dat'), ...
            'behav_rois', fullfile(ctx_dir, 'behavROIs.roimsk'));

        required_names = fieldnames(required);
        required_cells = struct2cell(required);
        missing = required_names(~cellfun(@isfile, required_cells));

        if ~isempty(missing)
            msg = sprintf('Missing required files: %s', strjoin(missing, ', '));
            fprintf(' - %s: %s\n', session_name, msg);
            report(idx).status = 'skipped';
            report(idx).message = msg;
            continue;
        end

        if isfile(report(idx).outputFile)
            msg = 'ROI.mat already exists (skipping)';
            fprintf(' - %s: %s\n', session_name, msg);
            report(idx).status = 'skipped';
            report(idx).message = msg;
            continue;
        end

        fprintf(' - %s: processing...\n', session_name);
        try
            opts = struct('OutputFile', report(idx).outputFile, 'SaveOutput', true);
            opts.DFFParams = struct( ...
                'baseline_method', 'running_percentile', ...
                'baseline_window_seconds', 600, ...
                'baseline_percentile', 10, ...
                'show_plot', false);
            rois_to_mat(required.fluor_movie, required.motion_movie, required.behav_rois, ...
                shared_inputs.neural_rois, shared_inputs.vascular_mask, shared_inputs.brain_mask, opts);
            report(idx).status = 'completed';
            report(idx).message = 'ROI.mat generated';
        catch ME
            warning('BatchRois_to_mat:FailedSession', '%s failed: %s', session_name, ME.message);
            report(idx).status = 'failed';
            report(idx).message = ME.message;
        end
    end

    completed = sum(strcmp({report.status}, 'completed'));
    skipped = sum(strcmp({report.status}, 'skipped'));
    failed = sum(strcmp({report.status}, 'failed'));
    fprintf('=== Done ===\nCompleted: %d | Skipped: %d | Failed: %d\n', completed, skipped, failed);

    if nargout == 0
        clear report;
    end
end
