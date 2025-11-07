# Plan: rois_to_mat ROI Exporter

## Goals
- Read fluorescence ROIs from luoROIs.roimsk (or referenced via SessionROIData) and extract every ROI trace from the fluorescence movie (*.dat).
- Convert fluorescence traces to dF/F (phasic) using dFF_phasic, defaulting to metadata sampling rate (10 Hz for the AH1463 example) with fallback to 10 Hz.
- Read behavior ROIs from ehavROIs.roimsk, extract all motion-energy traces from motion_energy.dat, and keep them raw (no dF/F, “just don’t process them”).
- Store both modalities in a single ROI struct with clear labeling and save it as ROI.mat next to the fluorescence movie.

## Scope & Assumptions
- The fluorescence .dat file has an accompanying .mat metadata file containing datSize and Freq (e.g., mov_aligned.dat / mov_aligned.mat).
- Behavioral motion energy .dat also has metadata (e.g., motion_energy.mat).
- ROI definitions live in .roimsk files that contain ROI_info (confirmed in sample data: luoROIs.roimsk -> {'AU_L','BC_L','M2_L','RS_L'}, ehavROIs.roimsk -> {'Face','Licking','Forelimbs'}).
- Optional vascular/brain masks should be honored when extracting fluorescence ROIs.
- dFF_phasic.m exists on the MATLAB path; if Fs is omitted it assumes 10 Hz, but we pass the detected rate.

## Data Structure (ROI.mat)
`
ROI.mat
+- ROI
   +- metadata
   ¦  +- created_by / created_at / version
   ¦  +- source
   ¦  ¦  +- fluorescence_movie / motion_movie
   ¦  ¦  +- neural_roi_file / behavior_roi_file
   ¦  ¦  +- vascular_mask_file / brain_mask_file
   ¦  +- modalities: {'fluorescence', ['behavior']}
   +- modalities
   ¦  +- fluorescence
   ¦  ¦  +- data [T x N] (dF/F or raw)
   ¦  ¦  +- raw_data [T x N] (when dF/F applied)
   ¦  ¦  +- labels {1xN}
   ¦  ¦  +- time [T x 1]
   ¦  ¦  +- sample_rate (Hz)
   ¦  ¦  +- dff_params (struct, includes Fs)
   ¦  ¦  +- traces_table (table, columns = sanitized labels)
   ¦  +- behavior (when provided)
   ¦     +- data [Tb x Nb] (raw motion energy)
   ¦     +- labels {1xNb}
   ¦     +- time [Tb x 1]
   ¦     +- sample_rate (Hz)
   ¦     +- traces_table (table)
   +- data / labels / time / type / raw_data … (aliases to fluorescence)
   +- paths (copy of metadata.source)
`

## Indexing Examples
- Fluorescence AU_L trace: ROI.data(:, strcmp(ROI.labels,'AU_L'))
- Raw AU_L trace (if dF/F computed): ROI.raw_data(:,1) (assuming AU_L sorted first)
- Behavior “Face” trace: ROI.modalities.behavior.data(:, strcmp(ROI.modalities.behavior.labels,'Face'))
- All fluorescence labels: ROI.labels

## Implementation Steps
1. **Inputs & Options**
   - Function signature mirrors modeling scripts: (fluo_file, motion_file, behav_roi_file, neural_roi_file, vascular_mask_file, brain_mask_file, opts).
   - Options: chunk_size_frames, ComputeDFF (default true), DFFParams, Fs override, OutputFile, SaveOutput.
2. **Fluorescence Pass**
   - Load metadata via load_dat_metadata (reused helper from modeling code).
   - Build inal_mask by intersecting vascular/brain masks when provided.
   - Load ROI_info via load_neural_roi_info (supports referenced files via SessionROIData).
   - Convert each ROI mask -> linear indices; drop empty ROIs with warnings.
   - Stream the .dat file chunk-wise; average pixels in each ROI per chunk (new helper extract_roi_traces_multi).
   - Run dFF_phasic per ROI column when ComputeDFF == true; default Fs = metadata.Freq (10 Hz sample data).
3. **Behavior Pass (optional)**
   - If both motion_file and ehav_roi_file are supplied, repeat the extraction with behavior ROIs, but skip dF/F.
   - Save as modality 	ype = 'motion_energy'.
4. **Struct Assembly**
   - Populate ROI.modalities.fluorescence / .behavior, copy fluorescence modality to top-level aliases, and attach metadata + file paths.
   - Always save ROI.mat (unless SaveOutput == false).

## Testing Plan (AH1463 sample data)
- Fluorescence movie: F:\SampleData\Analysis\AH1463\AH1463_08_08_25_WNoiseWater_AutoTrain\CtxImg\mov_aligned.dat
- Motion movie: F:\SampleData\Analysis\AH1463\AH1463_08_08_25_WNoiseWater_AutoTrain\CtxImg\motion_energy.dat
- Neural ROI file: F:\SampleData\Analysis\AH1463\fluoROIs.roimsk
- Behavior ROI file: F:\SampleData\Analysis\AH1463\AH1463_08_08_25_WNoiseWater_AutoTrain\CtxImg\behavROIs.roimsk
- Masks: F:\SampleData\Analysis\AH1463\VascularMask.roimsk, F:\SampleData\Analysis\AH1463\Mask.roimsk

MATLAB (run via PowerShell):
`powershell
matlab -batch "addpath('C:/Users/shires/Documents/Isaac/Code/Modeling'); \
ROI = rois_to_mat('F:/SampleData/Analysis/AH1463/AH1463_08_08_25_WNoiseWater_AutoTrain/CtxImg/mov_aligned.dat', ...
                  'F:/SampleData/Analysis/AH1463/AH1463_08_08_25_WNoiseWater_AutoTrain/CtxImg/motion_energy.dat', ...
                  'F:/SampleData/Analysis/AH1463/AH1463_08_08_25_WNoiseWater_AutoTrain/CtxImg/behavROIs.roimsk', ...
                  'F:/SampleData/Analysis/AH1463/fluoROIs.roimsk', ...
                  'F:/SampleData/Analysis/AH1463/VascularMask.roimsk', ...
                  'F:/SampleData/Analysis/AH1463/Mask.roimsk', struct()); \
whos ROI"
`
- Verify: ROI.labels == {'AU_L','BC_L','M2_L','RS_L'}; ROI.modalities.behavior.labels == {'Face','Licking','Forelimbs'}.
- Inspect sample counts: ROI.n_samples matches metadata frames (should equal fluorescence frame count).
- Confirm ROI.traces_table.Face works and ROI.mat saved in fluorescence folder.
- Plot sanity checks (optional): plot(ROI.time, ROI.data(:,1)) etc.

## Files
- ois_to_mat.m
- Plans/plan.md (this document)
