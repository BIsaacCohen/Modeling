function results = TemporalModelEvents_NoHabituation(ROI, session_file, opts)
% TemporalModelEvents_NoHabituation  Wrapper that trims the first 3 minutes.
%
%   results = TemporalModelEvents_NoHabituation(ROI, session_file)
%   results = TemporalModelEvents_NoHabituation(ROI, session_file, opts)
%
%   This convenience wrapper calls TemporalModelEvents after discarding the
%   first 180 seconds (3 minutes) of neural, behavioral, and event data to
%   remove habituation periods while keeping all lagged regressors aligned.

if nargin < 3 || isempty(opts)
    opts = struct();
end

opts.remove_initial_seconds = 180;
results = TemporalModelEvents(ROI, session_file, opts);
end
