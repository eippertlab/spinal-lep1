#################################################################################################
# Generate plots of comp after CCA - time and spatial
# Also make GA of same
#################################################################################################

import os
import mne
import numpy as np
import pandas as pd
from Functions.invert import invert
import pickle
from scipy.stats import sem
import matplotlib.pyplot as plt
from Functions.get_channels import get_channels
from Functions.IsopotentialFunctions_CbarLabel import *
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places


if __name__ == '__main__':

    # Testing laser stuff
    folders = ['main', 'piloting']  # Can be main or piloting
    sampling_rate = 1000  # Frequency to downsample to from original of 10kHz

    conditions = ['pain']

    # Dictionary of component to pick and whether to flip
    # [component, flip_space], flipping space WILL flip time course also

    erp_list = []
    spatial_list = []
    for condition in conditions:
        subs_flipped = []
        for folder in folders:
            if folder == 'piloting':
                subjects = np.arange(1, 3)  # Only first 2 being tested
                subject_ids = [f'esgpilot{str(subject).zfill(2)}' for subject in subjects]
                chosen = {'esgpilot01': [1, False],
                          'esgpilot02': [1, False]}
            else:
                subjects = np.arange(1, 6)  # Only first 5 subjects have pain
                subject_ids = [f'esg{str(subject).zfill(2)}' for subject in subjects]
                chosen = {'esg01': [1, False],
                          'esg02': [1, False],
                          'esg03': [1, False],
                          'esg04': [1, False],
                          'esg05': [1, False]}
            if condition != 'pain':
                raise RuntimeError('Condition must be pain')

            for subject_id in subject_ids:
                input_path = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30/sub-{subject_id}/'
                fname = f'{condition}.fif'
                figure_path = f'/data/pt_02889/main&pilot/Time_Spatial_Combined_TestFlip/'

                # Load epochs
                os.makedirs(figure_path, exist_ok=True)
                channel_no = chosen[subject_id][0]
                flip = chosen[subject_id][1]
                if flip is True:
                    subs_flipped.append(subject_id)
                channel = f'Cor{channel_no}'
                epochs = mne.read_epochs(input_path + fname, preload=True)
                erp = epochs.pick(channel).average()
                if flip is True:
                    data = erp.apply_function(invert).get_data()
                else:
                    data = erp.get_data()

                erp_list.append(data)

                # Load spatial pattern
                fname = f"A_st_{condition}.pkl"
                with open(f'{input_path}{fname}', 'rb') as f:
                    A_st = pickle.load(f)

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                # Plot time
                if flip is True:
                    erp = erp.apply_function(invert)
                axes[0].plot(erp.times, data.reshape(-1))
                axes[0].set_xlim([-0.01, 0.15])
                axes[0].set_ylabel('Amplitude')
                axes[0].set_xlabel('Time (s)')
                axes[0].set_title(f'Time Course')

                # Plot Spatial
                icomp = channel_no-1
                if flip:
                    A_st *= -1
                spatial_list.append(A_st[:, icomp])

                chan_labels = get_channels(False)
                colorbar_axes = [-1.5, 1.5]
                subjects_4grid = np.arange(1, 6)
                # you can also base the grid on an several subjects
                # then the function takes the average over the channel positions of all those subjects
                time = 0.0
                colorbar = True
                mrmr_esg_isopotentialplot(subjects_4grid, A_st[:, icomp], colorbar_axes, chan_labels,
                                          colorbar, time, axes[1], colorbar_label='Amplitude (AU)')
                axes[1].set_yticklabels([])
                axes[1].set_ylabel(None)
                axes[1].set_xticklabels([])
                axes[1].set_xlabel(None)
                axes[1].set_title(f'Spatial Topography')
                if flip:
                    plt.suptitle(f'{subject_id}, {condition}, Flipped Time and Spatial\n')
                    plt.savefig(figure_path+f'{subject_id}_{condition}_flip.png')
                else:
                    plt.suptitle(f'{subject_id}, {condition}, No flipping\n')
                    plt.savefig(figure_path+f'{subject_id}_{condition}_noflip.png')
                plt.close()
                # plt.show()

        # GA after CCA
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # Plot time
        ga_time = np.mean(erp_list, axis=0)
        print(np.shape(ga_time))
        print(np.argmin(ga_time))
        print(erp.times[np.argmin(ga_time)])
        error = sem(erp_list, axis=0)
        upper = (ga_time + error).reshape(-1)
        lower = (ga_time - error).reshape(-1)
        axes[0].plot(erp.times, ga_time[0, :], color='blue')
        axes[0].fill_between(erp.times, lower, upper, color='blue', alpha=0.3)
        axes[0].set_xlim([-0.01, 0.15])
        axes[0].set_ylim([-0.25, 0.2])
        axes[0].set_ylabel('Amplitude')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_title(f'Time Course')

        # Plot Spatial
        chan_labels = get_channels(False)
        colorbar_axes = [-1, 1]
        subjects_4grid = np.arange(1, 6)
        # you can also base the grid on an several subjects
        # then the function takes the average over the channel positions of all those subjects
        time = 0.0
        colorbar = True
        mrmr_esg_isopotentialplot(subjects_4grid, np.mean(spatial_list, axis=0), colorbar_axes, chan_labels,
                                  colorbar, time, axes[1], colorbar_label='Amplitude (AU)')
        axes[1].set_yticklabels([])
        axes[1].set_ylabel(None)
        axes[1].set_xticklabels([])
        axes[1].set_xlabel(None)
        axes[1].set_title(f'Spatial Topography')
        plt.suptitle(f'GA, {condition}, n={len(erp_list)}\n')
        plt.savefig(figure_path + f'GA_{condition}_flipped:{subs_flipped}.png')
        plt.savefig(figure_path + f'GA_{condition}_flipped:{subs_flipped}.pdf',
                    bbox_inches='tight', format="pdf")
        plt.show()
