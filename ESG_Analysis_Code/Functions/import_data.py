##########################################################################################
#                               This Script
# 1) imports the blocks based on the condition name in EEGLAB form from the BIDS directory
# 2) removes the stimulus artifact iv: -1.5 to 6 ms, for ESG use -7 to 7ms - linear interpolation
# 3) downsample the signal to 5000 Hz
# 4) Append mne raws of the same condition
# 5) Add qrs events as annotations
# 6) saves the new raw structure
# Emma Bailey, October 2022
# .matfile with R-peak locations is at 1000Hz - will still give rough idea (annotations not used)
##########################################################################################

# Import necessary packages
import mne
from mne_bids import BIDSPath, read_raw_bids
from Functions.get_channels import *
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


def import_data(subject_id, task, sampling_rate, bids_root, input_path, save_path):
    fname_save = f'noStimart_sr{sampling_rate}_{task}.fif'

    sampling_rate_og = 10000

    # Set interpolation window (different for eeg and esg data, both in seconds)
    if task == 'pain':
        tstart_esg = -0.013
        tmax_esg = 0.013
    else:
        tstart_esg = -0.007
        tmax_esg = 0.007

    # Find out which channels are which, include ECG, exclude EOG
    esg_chans = get_channels(includesEcg=True)

    raw = []
    event_files = []

    # Get file names that match pattern
    search = input_path + '*' + task + '*.eeg'

    cond_files = glob.glob(search)
    cond_files = sorted(cond_files)  # Arrange in order from lowest to highest value
    nblocks = len(cond_files)

    ####################################################################
    # Extract the raw data for each block, remove stimulus artefact, down-sample, concatenate, detect ecg,
    # and then save
    ####################################################################
    # Only dealing with one condition at a time, loop through however many blocks of said task
    for iblock in range(nblocks):
        run = iblock + 1
        bids_path = BIDSPath(subject=subject_id, run=run, task=task, root=bids_root)
        raw = read_raw_bids(bids_path=bids_path, extra_params={'preload': True})

        event_file = pd.read_csv(os.path.join(bids_root, 'sub-{}'.format(subject_id), 'eeg',
                                              'sub-{}_task-{}_run-{:02d}_events.tsv'.format(subject_id, task, run)),
                                 sep='\t')
        event_files.append(event_file)

        # If you only want to look at esg channels, drop the rest
        raw.pick_channels(esg_chans, ordered=True)

        # Interpolate required channels
        events, event_dict = mne.events_from_annotations(raw)

        trigger_name = set(raw.annotations.description)
        # trigger_name = set([task])

        # Acts in place to edit raw via linear interpolation to remove stimulus artefact
        # Need to loop as can be 2 trigger names and event_ids at play
        for j in trigger_name:
            mne.preprocessing.fix_stim_artifact(raw, events=events, event_id=event_dict[j], tmin=tstart_esg,
                                                tmax=tmax_esg, mode='linear', stim_channel=None)
            # # Need to get indices of events linked to this trigger
            # trigger_points = events[:, np.where(event_dict[j])]
            # trigger_points = trigger_points.reshape(-1).reshape(-1)
            # interpol_window = [tstart_esg, tmax_esg]
            # PCHIP_kwargs = dict(
            #     debug_mode=False, interpol_window_sec=interpol_window,
            #     trigger_indices=trigger_points, fs=sampling_rate_og
            # )
            # raw.apply_function(PCHIP_interpolation, picks=esg_chans, **PCHIP_kwargs,
            #                    n_jobs=len(esg_chans))

        # Downsample the data
        raw.resample(sampling_rate)  # resamples to desired

        fig, axes = plt.subplots(figsize=(12, 8))
        fig.suptitle(f"Power spectral density for data sub-{subject_id}")
        fig.tight_layout(pad=3.0)
        if 'TH6' in raw.ch_names:  # Can't use zero value in spectrum for channel
            raw.copy().drop_channels('TH6').compute_psd(fmax=300).plot(axes=axes, show=False)
        else:
            raw.compute_psd(fmax=300).plot(axes=axes, show=False)
        axes.set_ylim([-80, 50])
        plt.savefig(save_path + f'psd_{task}_{iblock}.png')

        # Append blocks of the same condition
        if iblock == 0:
            raw_concat = raw
        else:
            mne.concatenate_raws([raw_concat, raw])

    ##############################################################################################
    # Reference and Remove Powerline Noise
    # High pass filter at 1Hz
    ##############################################################################################
    # make sure recording reference is included
    mne.add_reference_channels(raw_concat, ref_channels=['TH6'], copy=False)  # Modifying in place

    raw_concat.filter(l_freq=1, h_freq=None, method='iir')
    raw_concat.notch_filter(freqs=50, method='iir')
    # iir filters cannot use multiple frequencies:
    raw_concat.notch_filter(freqs=100, method='iir')
    raw_concat.notch_filter(freqs=150, method='iir')
    raw_concat.notch_filter(freqs=200, method='iir')
    # raw_concat.notch_filter(freqs=[notch_low, notch_high], n_jobs=len(raw_concat.ch_names), method='fir', phase='zero')
    # raw_concat.filter(l_freq=1, h_freq=None, n_jobs=len(raw.ch_names), method='iir', iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    # Save data without stim artefact and downsampled to 1000
    raw_concat.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)

