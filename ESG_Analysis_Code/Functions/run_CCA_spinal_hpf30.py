# Script to actually run CCA on the data
# Using the meet package https://github.com/neurophysics/meet.git to run the CCA


import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from Functions.get_channels import get_channels
from Functions.IsopotentialFunctions_CbarLabel import mrmr_esg_isopotentialplot
import matplotlib.pyplot as plt
from Functions.transform import transform
import matplotlib as mpl
import pandas as pd
import pickle


def run_CCA_hpf30(subject_id, condition, trigger, sampling_rate, input_path, save_path, folder):
    plot_graphs = True
    fname = f'ssp6_cleaned_{condition}.fif'
    figure_path_spatial = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30/ComponentIsopotentialPlots/sub-{subject_id}/'
    figure_path_time = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30/ComponentTimePlots/sub-{subject_id}/'
    figure_path_st = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30/ComponentSinglePlots/sub-{subject_id}/'
    figure_path = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30/ComponentPlots/sub-{subject_id}/'
    os.makedirs(figure_path_spatial, exist_ok=True)
    os.makedirs(figure_path_time, exist_ok=True)
    os.makedirs(figure_path_st, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    iv_baseline = [-0.1, -0.01]
    iv_epoch = [-0.1, 0.3]

    esg_chans = get_channels(includesEcg=False)

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    if condition in ['pain', 'cold']:
        raw.filter(l_freq=30, h_freq=150, n_jobs=len(raw.ch_names), method='iir', iir_params={'order': 4, 'ftype': 'butter'},
                   phase='zero')
    else:
        raw.filter(l_freq=30, h_freq=None, n_jobs=len(raw.ch_names), method='iir',
                   iir_params={'order': 4, 'ftype': 'butter'},
                   phase='zero')

    # now create epochs based on the trigger names
    events, ids = mne.events_from_annotations(raw)
    epochs_all = mne.Epochs(raw, events, ids, tmin=iv_epoch[0], tmax=iv_epoch[1], baseline=tuple(iv_baseline), preload=True)
    epochs = epochs_all[trigger]

    # cca window size
    epochs = epochs.pick_channels(esg_chans, ordered=True)
    if condition == 'pain':
        # Don't know where SEP latency is yet so not adding it here
        window_times = [45/1000, 90/1000]  # Testing for laser
    elif condition == 'somatosensory':
        sep_latency = 0.013
        window_times = [7/1000, 37/1000]  # Python_Cardiac timings
    elif condition == 'cold':
        window_times = [60/1000, 120/1000]  # Testing for cold

    # Crop the epochs
    window = epochs.time_as_index(window_times)
    epo_cca = epochs.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)

    # Prepare matrices for cca
    ##### Average matrix
    epo_av = epo_cca.copy().average().data.T
    # Now want channels x observations matrix #np.shape()[0] gets number of trials
    # Epo av is no_times x no_channels (10x40)
    # Want to repeat this to form an array thats no. observations x no.channels (20000x40)
    # Need to repeat the array, no_trials/times amount along the y axis
    avg_matrix = np.tile(epo_av, (int((np.shape(epochs.get_data())[0])), 1))
    avg_matrix = avg_matrix.T  # Need to transpose for correct form for function - channels x observations

    ##### Single trial matrix
    epo_cca_data = epo_cca.get_data(picks=esg_chans)
    epo_data = epochs.get_data(picks=esg_chans)

    # 0 to access number of epochs, 1 to access number of channels
    # channels x observations
    no_times = int(window[1] - window[0])
    # Need to transpose to get it in the form CCA wants
    st_matrix = np.swapaxes(epo_cca_data, 1, 2).reshape(-1, epo_cca_data.shape[1]).T
    st_matrix_long = np.swapaxes(epo_data, 1, 2).reshape(-1, epo_data.shape[1]).T

    # Run CCA
    W_avg, W_st, r = spatfilt.CCA_data(avg_matrix, st_matrix)

    all_components = len(r)

    # Apply obtained weights to the long dataset (dimensions 40x9) - matrix multiplication
    CCA_concat = st_matrix_long.T @ W_st[:, 0:all_components]
    CCA_concat = CCA_concat.T

    # Spatial Patterns
    A_st = np.cov(st_matrix) @ W_st

    # Reshape - (900, 2000, 9)
    no_times_long = np.shape(epochs.get_data())[2]
    no_epochs = np.shape(epochs.get_data())[0]

    # Perform reshape
    CCA_comps = np.reshape(CCA_concat, (all_components, no_times_long, no_epochs), order='F')

    # Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
    CCA_comps = np.swapaxes(CCA_comps, 0, 2)
    CCA_comps = np.swapaxes(CCA_comps, 1, 2)
    selected_components = all_components  # Just keeping all for now to avoid rerunning

    ################################ Check if it needs inverting ###########################
    is_inverted = [False, False, False, False]
    if condition == 'somatosensory':  # Only check this for non-pain stim for now as we don't know where pain sep is
        # sep_latency is in ms
        # Get the data in this time window for all components
        # Find the peak in a 5ms window on either side
        check_window = epochs.time_as_index([(sep_latency - 5/ 1000), (sep_latency + 5/ 1000)])
        for icomp in np.arange(0, 4):
            check_data = CCA_comps[:, icomp, check_window[0]:check_window[1]]
            check_average = np.mean(np.mean(check_data, axis=0), axis=0)
            check_edges = np.mean(check_data, axis=0)
            min = np.min(check_edges)
            max = np.max(check_edges)

            # if check_average > 0:
            if np.abs(max) > np.abs(min):
                is_inverted[icomp] = True
                CCA_comps[:, icomp, :] *= -1
                # For manual correction - noticed some that should've/shouldn't be inverted - correct here

    #######################  Epoch data class to store the information ####################
    data = CCA_comps[:, 0:selected_components, :]
    events = epochs.events
    event_id = epochs.event_id
    tmin = iv_epoch[0]
    sfreq = sampling_rate

    ch_names = []
    ch_types = []
    for i in np.arange(0, all_components):
        ch_names.append(f'Cor{i+1}')
        ch_types.append('eeg')

    # Initialize an info structure
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq
    )

    # Create and save
    cca_epochs = mne.EpochsArray(data, info, events, tmin, event_id)
    cca_epochs = cca_epochs.apply_baseline(baseline=tuple(iv_baseline))
    fname = f'{condition}.fif'
    cca_epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    ################################ Save Spatial Pattern an weights #################################
    afile = open(save_path + f'A_st_{condition}.pkl', 'wb')
    pickle.dump(A_st, afile)
    afile.close()

    afile = open(save_path + f'W_st_{condition}.pkl', 'wb')
    pickle.dump(W_st, afile)
    afile.close()

    ################################ Plotting Graphs #######################################
    os.makedirs(figure_path_spatial, exist_ok=True)

    if plot_graphs:
        ####### Spinal Isopotential Plots for the first 4 components ########
        # fig, axes = plt.figure()
        fig, axes = plt.subplots(2, 2)
        axes_unflat = axes
        axes = axes.flatten()
        for icomp in np.arange(0, 4):  # Plot for each of four components
            # plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
            colorbar_axes = [-1.5, 1.5]
            chan_labels = epochs.ch_names
            colorbar = True
            time = 0.0
            mrmr_esg_isopotentialplot([1], A_st[:, icomp], colorbar_axes, chan_labels,
                                      colorbar, time, axes[icomp], 'Amplitude (AU)')
            axes[icomp].set_title(f'Component {icomp + 1}')
            axes[icomp].set_yticklabels([])
            axes[icomp].set_ylabel(None)
            axes[icomp].set_xticklabels([])
            axes[icomp].set_xlabel(None)

        plt.savefig(figure_path_spatial + f'{condition}.png')
        plt.close(fig)

        ############ Time Course of First 4 components ###############
        # cca_epochs and cca_epochs_d both already baseline corrected before this point
        os.makedirs(figure_path_time, exist_ok=True)

        fig = plt.figure()
        for icomp in np.arange(0, 4):
            if is_inverted[icomp] is True:
                plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}, inv, r={r[icomp]:.3f}')
            else:
                plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}, r={r[icomp]:.3f}')
            # Want to plot Cor1 - Cor4
            # Plot for the mixed nerve data
            # get_data returns (n_epochs, n_channels, n_times)
            data = cca_epochs.get_data(picks=[f'Cor{icomp + 1}'])
            to_plot = np.mean(data[:, 0, :], axis=0)
            plt.plot(cca_epochs.times, to_plot)
            if condition == 'somatosensory':
                plt.xlim([-0.025, 0.065])
                line_label = f"{sep_latency}s"
                plt.axvline(x=sep_latency, color='r', linewidth='0.6', label=line_label)
                plt.legend()
            elif condition in ['pain', 'cold']:
                plt.xlim([-0.025, 0.2])
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [A.U.]')
            plt.tight_layout()
        plt.savefig(figure_path_time + f'{condition}.png')
        # plt.show()
        plt.close(fig)

        ############################ Combine to one Image ##########################
        spatial = plt.imread(figure_path_spatial + f'{condition}.png')
        time = plt.imread(figure_path_time + f'{condition}.png')

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].imshow(time)
        axes[0].axis('off')
        axes[1].imshow(spatial)
        axes[1].axis('off')

        plt.subplots_adjust(top=0.95, wspace=0, hspace=0)

        plt.suptitle(f'Subject {subject_id}, {condition}')
        plt.savefig(figure_path + f'{condition}.png')
        plt.close(fig)
