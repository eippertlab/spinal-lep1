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
    folder = 'piloting'  # Can be main or piloting
    hpf = 30  # If 30, CCA was performed after hpf at 30Hz
    sampling_rate = 1000  # Frequency to downsample to from original of 10kHz
    if folder == 'piloting':
        subjects = np.arange(1, 3)
        subject_ids = [f'esgpilot{str(subject).zfill(2)}' for subject in subjects]
        # Dictionary of component to pick and whether to flip
        # [component, flip_space], flipping space WILL NOT flip time course also
        chosen = {'esgpilot01': [1, False],
                  'esgpilot02': [1, False]}
    else:
        subjects = np.arange(1, 7)
        subject_ids = [f'esg{str(subject).zfill(2)}' for subject in subjects]
        # Dictionary of component to pick and whether to flip
        # [component, flip_space], flipping space WILL NOT flip time course also
        chosen = {'esg01': [1, False],
                  'esg02': [1, False],
                  'esg03': [1, False],
                  'esg04': [1, False],
                  'esg05': [1, False]}

    conditions = ['pain']

    for condition in conditions:
        if condition != 'pain':
            raise RuntimeError('Condition must be pain')

        for subject_id in subject_ids:
            if hpf == 30:
                input_path = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30/sub-{subject_id}/'
                fname = f'{condition}.fif'
                figure_path = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30/PhaseSynchrony_EvenOdd/sub-{subject_id}/'
                os.makedirs(figure_path, exist_ok=True)
            else:
                input_path = f'/data/pt_02889/{folder}/esg_analysis/cca/sub-{subject_id}/'
                fname = f'{condition}.fif'
                figure_path = f'/data/pt_02889/{folder}/esg_analysis/cca/PhaseSynchrony_EvenOdd/sub-{subject_id}/'
                os.makedirs(figure_path, exist_ok=True)

            # Load epochs
            os.makedirs(figure_path, exist_ok=True)
            channel_no = chosen[subject_id][0]
            flip = chosen[subject_id][1]
            channel = f'Cor{channel_no}'
            epochs = mne.read_epochs(input_path + fname, preload=True).pick(channel)
            data = epochs.get_data(copy=False)[:, 0, :]

            # Can't just pass in epochs, each epoch needs to be a separate observations
            # Given we have a single subject here, and no separate groups within a subject

            #################################################
            # Epochs based
            # Within subject, across trials test
            #################################################
            T_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                data,
                n_permutations=1000,
                threshold=None,
                tail=0,
                n_jobs=None,
                out_type="mask",
            )
            print(np.shape(T_obs))
            print(T_obs)
            print(H0)
            print(H0.size)
            # H0 will be of non-zero size if clusters are found
            if H0.size > 0:
                # Plot time
                times = epochs.times
                fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
                ax.set_title("Channel : " + channel)
                ax.plot(times, data.mean(axis=0), label=f"{subject_id}")
                ax.set_ylabel("Amplitude (AU)")
                ax.legend()

                for i_c, c in enumerate(clusters):
                    c = c[0]
                    if cluster_p_values[i_c] <= 0.05:
                        h = ax2.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                        ax2.legend((h,), ("cluster p-value < 0.05",))
                    else:
                        ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

                hf = plt.plot(times, T_obs, "g")
                ax2.set_xlabel("time (ms)")
                ax2.set_ylabel("f-values")
                plt.show()
