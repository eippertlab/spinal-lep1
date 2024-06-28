# Perform Signal Space Projection
# Mainly working from tutorial https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html
# SSP uses singular value decomposition to create the projection matrix
# Apply SSP to anteriorly rereferenced data

import os
import mne
import numpy as np
import pandas as pd


def apply_SSP_anterior(subject, task, sampling_rate, n_p, input_path, save_path):
    ###########################################################################################
    # Load
    ###########################################################################################
    # load imported ESG data
    fname = f'noStimart_sr{sampling_rate}_{task}.fif'

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    raw.set_eeg_reference(ref_channels=['CA1'])

    ##########################################################################################
    # SSP
    ##########################################################################################
    projs, events = mne.preprocessing.compute_proj_ecg(raw, n_eeg=n_p, reject=None,
                                                       n_jobs=len(raw.ch_names), ch_name='ECG')

    # Apply projections (clean data)
    clean_raw = raw.copy().add_proj(projs)
    clean_raw = clean_raw.apply_proj()

    ##############################################################################################
    # Save
    ##############################################################################################
    # Save the SSP cleaned data for future comparison
    clean_raw.save(f"{save_path}ssp{n_p}_cleaned_{task}.fif", fmt='double', overwrite=True)
