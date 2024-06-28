#################################################################################################
# Generate plots of phase synchrony between even-odd trials after CCA
# https://jinhyuncheong.com/jekyll/update/2017/12/10/Timeseries_synchrony_tutorial_and_simulations.html
#################################################################################################

import os
import mne
from scipy.signal import hilbert
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))  # 2 decimal places


if __name__ == '__main__':
    # Testing laser stuff
    folders = ['main', 'piloting']  # Can be main or piloting
    sampling_rate = 1000  # Frequency to downsample to from original of 10kHz

    conditions = ['pain']
    hpf = 30  # Can be 30 or 1

    for condition in conditions:
        even_list_1 = []
        odd_list_1 = []
        even_list_2 = []
        odd_list_2 = []

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

            for subject_id in subject_ids:
                if hpf == 30:
                    input_path = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30/sub-{subject_id}/'
                    fname = f'{condition}.fif'
                    figure_path = f'/data/pt_02889/main&pilot/EveryFourth_GA/CCA_hpf30/'
                    os.makedirs(figure_path, exist_ok=True)
                else:
                    input_path = f'/data/pt_02889/{folder}/esg_analysis/cca/sub-{subject_id}/'
                    fname = f'{condition}.fif'
                    figure_path = f'/data/pt_02889/main&pilot/EveryFourth/CCA/'
                    os.makedirs(figure_path, exist_ok=True)

                epochs = mne.read_epochs(input_path + fname, preload=True)

                iv_baseline = [-0.1, -0.01]
                iv_epoch = [-0.1, 0.3]

                epochs_stim_even_1 = epochs[0::4]
                epochs_stim_odd_1 = epochs[1::4]
                epochs_stim_even_2 = epochs[2::4]
                epochs_stim_odd_2 = epochs[3::4]

                pick = 'Cor1'
                stim_erps_even_1 = epochs_stim_even_1.pick_channels([pick]).average().get_data().reshape(-1)
                stim_erps_odd_1 = epochs_stim_odd_1.pick_channels([pick]).average().get_data().reshape(-1)
                stim_erps_even_2 = epochs_stim_even_2.pick_channels([pick]).average().get_data().reshape(-1)
                stim_erps_odd_2 = epochs_stim_odd_2.pick_channels([pick]).average().get_data().reshape(-1)

                even_list_1.append(stim_erps_even_1)
                odd_list_1.append(stim_erps_odd_1)
                even_list_2.append(stim_erps_even_2)
                odd_list_2.append(stim_erps_odd_2)

        grand_average_even_1 = np.mean(even_list_1, axis=0)
        grand_average_odd_1 = np.mean(odd_list_1, axis=0)
        grand_average_even_2 = np.mean(even_list_2, axis=0)
        grand_average_odd_2 = np.mean(odd_list_2, axis=0)
        fig, axis = plt.subplots(1, 1)
        axis.plot(epochs_stim_even_1.times, grand_average_even_1)
        axis.plot(epochs_stim_odd_1.times, grand_average_odd_1)
        axis.plot(epochs_stim_even_2.times, grand_average_even_2)
        axis.plot(epochs_stim_odd_2.times, grand_average_odd_2)
        axis.set_ylabel('Amplitude (AU)')
        axis.set_xlabel('Time (s)')
        axis.set_title(f'{pick}')
        # plt.legend()
        if condition == 'somatosensory':
            axis.set_xlim([-0.01, 0.065])
        else:
            axis.set_xlim([-0.01, 0.15])
            axis.set_ylim([-0.25, 0.2])
        plt.suptitle(f'GA, Component 1, {condition}\n')
        plt.tight_layout()
        plt.savefig(figure_path+f'GA_comp1_{condition}.png')
        plt.savefig(figure_path+f'GA_comp1_{condition}.pdf',
                    bbox_inches='tight', format="pdf")
        # plt.show()
