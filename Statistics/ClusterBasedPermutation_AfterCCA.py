##############################################################################################
# Testing if the evoked response across subjects is significant
##############################################################################################

import matplotlib.pyplot as plt
import mne
import numpy as np
import os
from scipy.stats import sem
from mne import io
from mne.datasets import sample
from mne.stats import permutation_cluster_1samp_test


if __name__ == '__main__':
    # Testing laser stuff
    folders = ['main', 'piloting']  # Can be main or piloting
    sampling_rate = 1000  # Frequency to downsample to from original of 10kHz

    conditions = ['pain']

    epochs_list = []
    erp_list = []
    for condition in conditions:
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
                figure_path = f'/data/pt_02889/main&pilot/ClusterPermutation_AfterCCA/'

                # Load epochs
                os.makedirs(figure_path, exist_ok=True)
                channel_no = chosen[subject_id][0]
                flip = chosen[subject_id][1]
                channel = f'Cor{channel_no}'
                epochs = mne.read_epochs(input_path + fname, preload=True).pick(channel)
                data = epochs.get_data(copy=False)[:, 0, :]
                epochs_list.append(data)
                erp_list.append(data.mean(axis=0))  # Average across trials
                # exit()

        print(np.shape(np.array(erp_list)))

        #################################################
        # Erps based
        #################################################
        t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
            np.array(erp_list),
            n_permutations=1000,
            threshold=None,
            tail=0,
            n_jobs=None,
            out_type="mask",
        )
        # Plot time if clusters found
        if H0.size>0:
            times = epochs.times
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
            ax.set_title("Channel : " + channel)
            for condition in erp_list:
                ax.plot(
                    times,
                    condition,
                    label="ERPs",
                )
            ax.set_ylabel("Amplitude (AU)")
            ax.legend()

            for i_c, c in enumerate(clusters):
                c = c[0]
                if cluster_p_values[i_c] <= 0.05:
                    h = ax2.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                    ax2.legend((h,), ("cluster p-value < 0.05",))
                else:
                    ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

            hf = plt.plot(times, t_obs, "g")
            ax2.set_xlabel("time (s)")
            ax2.set_ylabel("t-values")

            plt.show()