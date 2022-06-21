This repository should contain all of the code and data to recreate the analysis used in our study.



# Data: 
This folder contains intermeidary datasets including:
Decoding_xTimeRoi_All.csv  - Output of decoding analysis for all trials, ROIs, and subjects. 
Decoding_N_1_xTimeRoi_All.csv - Same as above but training on identity of the N-1 trial.
Decoding_N_plus1_xTimeRoi_All.csv - Same as above but training on identity of the N+1 trial (control analysis, should be no info).
CD_data_clean.csv - Behavioral data from behavior only cohort

# DataRaw:
File structure generated for each subject, session.
Contains within each session folder:
'DAT_CD_TASK.pickle' which contains for each session the TR x voxel matrix for each retinotopically defined ROI
'INFO_CD_TASK.pickle' which is a table containing the identity and timestamps of all events during the session.

# DataDeconvolution:
Includes intermediary weights estimated through response deconvolution.
*subj*_

# EpochedData:
Contains ROIxVoxelxTrialxTime data for all subjects. May be uploaded seperately to save space.

# Analysis:
This folder contains all scripts needed to analyze processed behavioral and BOLD data used in study.
Analysis/Behavioral_mainAnalysis.ipynb | behavioral only cohort analysis. Fig 1. Figs S1-S2 (need to modify flag to subset random sequences)
Analysis/fMRI_mainAnalysis.ipynb | *Primary empirical analysis on fMRI cohort.* Probably what most people would be interested in.
                        | This script starts with loading output of classifiers (Decoding_xTimeRoi_All.csv). Figs 2, 3C-F,
                        | S3A-F, S4C-E, S6 (need to modify flag to subset random sequences)
Analysis/fMRI_decodeN-1.ipynb    | loads data for decoder on identity of trial N-1 Fig 2F; and N+1 Fig S3F
Analysis/dimensionallity_analysis.ipynb | Performs PCA on BOLD data Fig S5
Analysis/fMRI_analyzeDeconvolvedActivity.ipynb | Performs decoding and analysis on already deconovolved traces. Gen Figure 3B
Analysis/Demo_adaptationModel.ipynb | lightweight version of spiking model for demonstration. Figure 4.
Analysis/fMRI_runModeling.ipynb  | Fits unaware, aware, and Bayesian models to behavioral and neural data. 
                        | Model fitting is very computationally expensive and is setup to run using multiprocessing.
Analysis/fMRI_AnalyzeModeling.ipynb | Loads output of fMRI_runModeling and creates figure 5, S8, table 1

# Processing:
This folder contains more backend analyses that create datasets loaded out by analyses functions.
Processing/DeconvolveTaskRespSubj-2Gamma_single_cell_subject.ipynb | performs voxelwise and ROI wise deconvolution and saves out to 
                                                                   | "DataDeconvolution". Also generates Figure 3A.
Processing/Sim_task_deconvolve_all_v1.ipynb    | Runs simulation for figure S7
Processing/run_subj_decode_wrapper.ipynb       | wrapper function for running decoding on epoched data. Saves out to /IterativeFits                                                             
Processing/DecodeNoiseCorrelation.py           | Python implentation of van Bergen et al. "Prince" model. Also allows simple IEM fitting
                                               | as this is part of fitting procedure. 


