function outFile = compute_dff_fir_baseline(RawFolder, SaveFolder, varargin)
% COMPUTE_DFF_FIR_BASELINE Compute dF/F using an FIR baseline in umIT style.
%   OUTFILE = COMPUTE_DFF_FIR_BASELINE(RAWFOLDER, SAVEFOLDER, OPTS) maps a
%   fluorescence .dat file located in RAWFOLDER (or at a user supplied path),
%   extracts a slowly varying baseline with an FIR low-pass filter, computes
%   dF/F, and stores the processed data and metadata in SAVEFOLDER.
%
%   Required inputs:
%     RawFolder  - Path to raw data folder (umIT requirement).
%     SaveFolder - Path to output folder (umIT requirement).
%
%   Optional struct input `opts` fields:
%     InputFile      - Relative or absolute path to the fluorescence .dat file.
%                      Default: 'fluo.dat'
%     InputMeta      - Relative or absolute path to metadata .mat file. Empty
%                      uses <InputFile>.mat.
%     OutputPrefix   - Prefix used for generated files. Default reuses the
%                      input file name.
%     FilterOrder    - FIR filter order (even values recommended). Default: 100
%     CutoffHz       - Low-pass cutoff frequency in Hz. Default: 1e-3
%     ChunkSize      - Number of pixels processed per iteration. Default: 1000
%     Epsilon        - Small offset to avoid dividing by zero. Default: 1e-6
%     GenerateFigure - Flag to create comparison plots. Default: true
%     SavePng        - Save PNG visualization when GenerateFigure is true.
%                      Default: true
%     SaveFigFile    - Save MATLAB figure when GenerateFigure is true.
%                      Default: false
%     SamplePixel    - Pixel index or [row col] to visualize. Default: 'random'
%
%   The function returns a cell array `outFile` listing the generated files.
%
%   Example:
%     outFile = compute_dff_fir_baseline(rawRoot, saveRoot, ...
%                struct('InputFile','session01_fluo.dat','FilterOrder',150));
%
% See also FIR1, FILTFILT, MEMMAPFILE.

default_Output = {'*_dff.dat','*_dff.mat','*_dff_comparison.png','*_dff_comparison.fig'}; %#ok
default_opts = struct('InputFile', 'fluo.dat', 'InputMeta', '', 'OutputPrefix', '', 'FilterOrder', 100, 'CutoffHz', 1e-3, 'ChunkSize', 1000, 'Epsilon', 1e-6, 'GenerateFigure', true, 'SavePng', true, 'SaveFigFile', false, 'SamplePixel', 'random');
opts_values = struct('FilterOrder', 50:50:400, 'CutoffHz', {0.0005:0.0005:0.005}, 'ChunkSize', [250, 500, 1000, 2000, 5000]); %#ok Used by PipelineManager for UI hints.

p = inputParser;
addRequired(p, 'RawFolder', @isfolder);
addRequired(p, 'SaveFolder', @isfolder);
addOptional(p, 'opts', default_opts, @(x) isstruct(x) && ~isempty(x));
parse(p, RawFolder, SaveFolder, varargin{:});
RawFolder = p.Results.RawFolder;
SaveFolder = p.Results.SaveFolder;
opts = p.Results.opts;
clear p

% Merge user supplied options with defaults.
defNames = fieldnames(default_opts);
for kk = 1:numel(defNames)
    if ~isfield(opts, defNames{kk}) || isempty(opts.(defNames{kk}))
        opts.(defNames{kk}) = default_opts.(defNames{kk});
    end
end

if ~exist(SaveFolder, 'dir')
    mkdir(SaveFolder);
end

% Resolve input file paths.
if isfile(opts.InputFile)
    inputDatFile = opts.InputFile;
else
    inputDatFile = fullfile(RawFolder, opts.InputFile);
end

if isempty(opts.InputMeta)
    [inputDatDir, inputDatName] = fileparts(inputDatFile);
    metaCandidates = { ...
        fullfile(inputDatDir, [inputDatName '.mat']), ...
        fullfile(RawFolder, [inputDatName '.mat'])};
    metaFile = '';
    for kk = 1:numel(metaCandidates)
        if isfile(metaCandidates{kk})
            metaFile = metaCandidates{kk};
            break
        end
    end
else
    if isfile(opts.InputMeta)
        metaFile = opts.InputMeta;
    else
        metaFile = fullfile(RawFolder, opts.InputMeta);
    end
end

if ~isfile(inputDatFile)
    error('compute_dff_fir_baseline:MissingData', ...
        'Fluorescence file not found: %s', inputDatFile);
end

if ~isfile(metaFile)
    error('compute_dff_fir_baseline:MissingMeta', ...
        'Metadata file not found: %s', metaFile);
end

[~, inputBaseName] = fileparts(inputDatFile);
if isempty(opts.OutputPrefix)
    outputPrefix = inputBaseName;
else
    outputPrefix = opts.OutputPrefix;
end

fprintf('=== Computing dF/F with FIR baseline ===\n\n');
fprintf('Raw folder : %s\n', RawFolder);
fprintf('Save folder: %s\n', SaveFolder);
fprintf('Input file : %s\n', inputDatFile);
fprintf('Metadata   : %s\n\n', metaFile);

%% Load metadata and determine data dimensions.
metaStruct = load(metaFile);
requiredFields = {'datSize', 'Freq', 'Datatype'};
for kk = 1:numel(requiredFields)
    if ~isfield(metaStruct, requiredFields{kk})
        error('compute_dff_fir_baseline:MissingField', ...
            'Metadata is missing required field "%s".', requiredFields{kk});
    end
end

Y = metaStruct.datSize(1);
X = metaStruct.datSize(2);
fs = metaStruct.Freq;
datatype = metaStruct.Datatype;

fileInfo = dir(inputDatFile);
if isempty(fileInfo)
    error('compute_dff_fir_baseline:FileInfo', ...
        'Unable to read file info for %s.', inputDatFile);
end

switch lower(datatype)
    case {'single', 'float32'}
        bytesPerValue = 4;
        matlabClass = 'single';
    case {'double', 'float64'}
        bytesPerValue = 8;
        matlabClass = 'double';
    otherwise
        error('compute_dff_fir_baseline:UnsupportedType', ...
            'Datatype "%s" is not supported.', datatype);
end

bytesPerFrame = Y * X * bytesPerValue;
T = fileInfo.bytes / bytesPerFrame;
if abs(T - round(T)) > eps
    error('compute_dff_fir_baseline:SizeMismatch', ...
        'File size is not divisible by frame size. Check metadata.');
end
T = round(T);

fprintf('Data size: %d x %d x %d frames (%.1f s, %.1f Hz)\n', ...
    Y, X, T, T / fs, fs);

%% Design FIR baseline filter.
filterOrder = opts.FilterOrder;
if mod(filterOrder, 2) ~= 0
    warning('compute_dff_fir_baseline:FilterOrder', ...
        'Filter order should be even; rounding %d to %d.', ...
        filterOrder, filterOrder + 1);
    filterOrder = filterOrder + 1;
end

cutoffHz = opts.CutoffHz;
nyquist = fs / 2;
normalizedCutoff = cutoffHz / nyquist;
if normalizedCutoff <= 0 || normalizedCutoff >= 1
    error('compute_dff_fir_baseline:Cutoff', ...
        'Cutoff frequency must be between 0 and Nyquist.');
end

fprintf('\nDesigning FIR baseline filter...\n');
fprintf('  Order    : %d\n', filterOrder);
fprintf('  Cutoff   : %.6f Hz (normalized: %.6e)\n', cutoffHz, normalizedCutoff);

b = fir1(filterOrder, normalizedCutoff, 'low');
fprintf('  Created FIR filter with %d taps.\n', numel(b));

%% Memory-map fluorescence data and prepare output arrays.
fprintf('\nMapping fluorescence data...\n');
mapObj = memmapfile(inputDatFile, 'Format', {matlabClass, [Y, X, T], 'data'});
fluoData = mapObj.Data.data;
fluoReshaped = reshape(fluoData, [Y * X, T]);

fprintf('Processing %d pixels (chunk size: %d)...\n', Y * X, opts.ChunkSize);

nPixels = Y * X;
chunkSize = max(1, min(opts.ChunkSize, nPixels));
dffData = zeros(nPixels, T, 'single');
epsilon = opts.Epsilon;

tic;
chunkStarts = 1:chunkSize:nPixels;
for idx = 1:numel(chunkStarts)
    chunkStart = chunkStarts(idx);
    chunkEnd = min(chunkStart + chunkSize - 1, nPixels);
    pixelIdx = chunkStart:chunkEnd;

    for pix = pixelIdx
        F = double(fluoReshaped(pix, :));
        F0 = filtfilt(b, 1, F);
        dffData(pix, :) = single((F - F0) ./ (F0 + epsilon));
    end

    if idx == 1 || idx == numel(chunkStarts) || mod(idx, max(1, floor(numel(chunkStarts) / 10))) == 0
        fprintf('  Processed %d / %d pixels\n', chunkEnd, nPixels);
    end
end
processingTime = toc;

fprintf('dF/F computation finished in %.1f minutes.\n', processingTime / 60);

dffData = reshape(dffData, [Y, X, T]);

%% Derive basic statistics for logging and visualization.
fprintf('\nComputing summary statistics...\n');
globalMin = min(fluoReshaped(:));
globalMax = max(fluoReshaped(:));
dffMin = min(dffData(:));
dffMax = max(dffData(:));
dffMean = mean(dffData(:));
dffStd = std(dffData(:));
fprintf('  Raw fluorescence range: [%.2f, %.2f]\n', globalMin, globalMax);
fprintf('  dF/F range           : [%.4f, %.4f]\n', dffMin, dffMax);
fprintf('  dF/F mean / std      : %.4f / %.4f\n', dffMean, dffStd);

samplePixelIdx = resolveSamplePixel(opts.SamplePixel, Y, X, nPixels);
[sampleY, sampleX] = ind2sub([Y, X], samplePixelIdx);
rawTrace = double(fluoData(sampleY, sampleX, :));
rawTrace = rawTrace(:)';
baselineTrace = filtfilt(b, 1, rawTrace);
dffTrace = double(dffData(sampleY, sampleX, :));
dffTrace = dffTrace(:)';

%% Create visualization when requested.
pngFile = '';
figFile = '';
if opts.GenerateFigure
    fprintf('\nGenerating visualization for pixel [%d, %d]...\n', sampleY, sampleX);

    timeVec = (0:T-1) / fs;
    avgRaw = squeeze(mean(fluoReshaped, 1));
    avgDff = squeeze(mean(reshape(dffData, [nPixels, T]), 1));

    figHandle = figure('Name', 'dF/F FIR Baseline', 'Position', [100, 100, 1400, 900]);

    subplot(3, 2, 1);
    plot(timeVec, rawTrace, 'b-', 'LineWidth', 1);
    hold on;
    plot(timeVec, baselineTrace, 'r-', 'LineWidth', 1.2);
    hold off;
    xlabel('Time (s)');
    ylabel('Fluorescence (a.u.)');
    title(sprintf('Pixel [%d, %d] Raw vs Baseline', sampleY, sampleX));
    legend({'Raw', 'FIR Baseline'}, 'Location', 'best');
    grid on;

    subplot(3, 2, 2);
    plot(timeVec, dffTrace, 'g-', 'LineWidth', 1);
    xlabel('Time (s)');
    ylabel('dF/F');
    title(sprintf('Pixel [%d, %d] dF/F', sampleY, sampleX));
    grid on;
    yline(0, 'k--', 'Alpha', 0.3);

    subplot(3, 2, 3);
    plot(timeVec, avgRaw, 'b-', 'LineWidth', 1.2);
    xlabel('Time (s)');
    ylabel('Mean Fluorescence (a.u.)');
    title('Population Mean (Raw)');
    grid on;

    subplot(3, 2, 4);
    plot(timeVec, avgDff, 'g-', 'LineWidth', 1.2);
    xlabel('Time (s)');
    ylabel('Mean dF/F');
    title('Population Mean (dF/F)');
    grid on;
    yline(0, 'k--', 'Alpha', 0.3);

    subplot(3, 2, 5);
    histogram(fluoReshaped(:), 100, 'FaceColor', [0.1 0.4 0.8], 'EdgeColor', 'none');
    xlabel('Fluorescence (a.u.)');
    ylabel('Count');
    title('Distribution (Raw)');
    grid on;

    subplot(3, 2, 6);
    histogram(dffData(:), 100, 'FaceColor', [0.2 0.6 0.2], 'EdgeColor', 'none');
    xlabel('dF/F');
    ylabel('Count');
    title('Distribution (dF/F)');
    grid on;
    xline(0, 'k--', 'LineWidth', 1.2);

    sgtitle(sprintf('dF/F FIR Baseline | Order %d | Cutoff %.4f Hz', filterOrder, cutoffHz), ...
        'FontWeight', 'bold');

    if opts.SavePng
        pngFile = fullfile(SaveFolder, sprintf('%s_dff_comparison.png', outputPrefix));
        saveas(figHandle, pngFile);
        fprintf('  Saved PNG: %s\n', pngFile);
    end

    if opts.SaveFigFile
        figFile = fullfile(SaveFolder, sprintf('%s_dff_comparison.fig', outputPrefix));
        saveas(figHandle, figFile);
        fprintf('  Saved FIG: %s\n', figFile);
    end

    close(figHandle);
end

%% Persist processed data and metadata.
fprintf('\nSaving dF/F data...\n');
outputDatFile = fullfile(SaveFolder, sprintf('%s_dff.dat', outputPrefix));
outputMatFile = fullfile(SaveFolder, sprintf('%s_dff.mat', outputPrefix));

fid = fopen(outputDatFile, 'w');
if fid == -1
    error('compute_dff_fir_baseline:WriteError', ...
        'Unable to open %s for writing.', outputDatFile);
end
fwrite(fid, dffData, 'single');
fclose(fid);
fprintf('  Saved dF/F data: %s (%.2f MB)\n', outputDatFile, dir(outputDatFile).bytes / (1024^2));

dffMetadata = metaStruct;
dffMetadata.datFile = outputDatFile;
dffMetadata.datName = sprintf('%s_dff', outputPrefix);
dffMetadata.datSize = [Y, X, T];
dffMetadata.Datatype = 'single';
dffMetadata.preprocessing_info = struct( ...
    'method', 'FIR_filter_baseline', ...
    'filter_order', filterOrder, ...
    'cutoff_hz', cutoffHz, ...
    'epsilon', epsilon, ...
    'formula', '(F - F0) / (F0 + epsilon)', ...
    'timestamp', datestr(now));

save(outputMatFile, '-struct', 'dffMetadata');
fprintf('  Saved metadata : %s\n', outputMatFile);

%% Collect outputs for umIT pipeline manager.
outFile = {outputDatFile, outputMatFile};
if ~isempty(pngFile)
    outFile{end+1} = pngFile; %#ok<AGROW>
end
if ~isempty(figFile)
    outFile{end+1} = figFile; %#ok<AGROW>
end

%% Final summary.
fprintf('\n=== dF/F FIR Baseline Complete ===\n');
fprintf('Processing time : %.1f minutes\n', processingTime / 60);
fprintf('Output files    :\n');
for kk = 1:numel(outFile)
    fprintf('  %s\n', outFile{kk});
end
fprintf('Sample pixel    : [%d, %d]\n', sampleY, sampleX);
fprintf('dF/F range      : [%.4f, %.4f]\n', dffMin, dffMax);
fprintf('dF/F mean / std : %.4f / %.4f\n', dffMean, dffStd);
fprintf('===================================\n');

end

function idx = resolveSamplePixel(sampleOpt, Y, X, nPixels)
%RESOLVESAMPLEPIXEL Convert user selection into linear pixel index.
if isnumeric(sampleOpt)
    sampleOpt = sampleOpt(:);
    if isempty(sampleOpt)
        idx = randi(nPixels);
        return;
    end

    if numel(sampleOpt) == 1
        idx = min(max(1, round(sampleOpt)), nPixels);
        return;
    elseif numel(sampleOpt) == 2
        row = min(max(1, round(sampleOpt(1))), Y);
        col = min(max(1, round(sampleOpt(2))), X);
        idx = sub2ind([Y, X], row, col);
        return;
    end
end

if ischar(sampleOpt) || (isstring(sampleOpt) && isscalar(sampleOpt))
    token = lower(string(sampleOpt));
    switch token
        case "center"
            idx = sub2ind([Y, X], ceil(Y / 2), ceil(X / 2));
            return;
        case "first"
            idx = 1;
            return;
        case "last"
            idx = nPixels;
            return;
        otherwise
            % fallthrough to random
    end
end

idx = randi(nPixels);
end
