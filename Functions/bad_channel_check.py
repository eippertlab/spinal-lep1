#################################################################################################
# Generate plots to identify bad channels to mark for removal
#################################################################################################

import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bad_channel_check(subject, task, sampling_rate):
    input_path = f'/data/pt_02835/OmissionEEG/pilot/esg/ssp_cleaned/{subject}/'
    save_path = f'/data/pt_02835/OmissionEEG/pilot/esg/bad_channels_esg/{subject}/'
    fname = f'ssp6_cleaned_{task}.fif'
    os.makedirs(save_path, exist_ok=True)

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    iv_baseline = [-0.1, -0.01]
    iv_epoch = [-0.1, 0.3]

    ##########################################################################################
    # Generates psd - can click on plot to find bad channel name
    ##########################################################################################
    fig, axes = plt.subplots(figsize=(12, 8))
    fig.suptitle(f"Power spectral density for data sub-{subject}")
    fig.tight_layout(pad=3.0)
    if 'TH6' in raw.ch_names:  # Can't use zero value in spectrum for channel
        raw.copy().drop_channels('TH6').compute_psd(fmax=500).plot(axes=axes, show=False)
    else:
        raw.compute_psd(fmax=2000).plot(axes=axes, show=False)
    axes.set_ylim([-80, 50])
    plt.savefig(save_path + f'psd_{task}.png')

    ###########################################################################################
    # Squared log means of each channel
    ###########################################################################################
    events, event_ids = mne.events_from_annotations(raw)
    trigger_name = set(raw.annotations.description)
    event_id_dict = {key: value for key, value in event_ids.items() if key in trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=tuple(iv_baseline), preload=True)
    fig, axes = plt.subplots(figsize=(12, 8))
    fig.suptitle(f"Squared log means per epoch sub-{subject}")
    table = epochs.to_data_frame()
    # I don't separate by condition for this trial check
    table = table.drop(columns=["time", "ECG", "TH6", "condition"])
    table = pd.concat([table.iloc[:, :2], np.square(table.iloc[:, 2:])], axis=1)
    table = pd.concat([table.iloc[:, :2], np.log(table.iloc[:, 2:])], axis=1)
    means = table.groupby(['epoch']).mean().T  # average
    ax_i = axes.matshow(means, aspect='auto')  # plots mean values by colorscale
    plt.colorbar(ax_i, ax=axes)
    axes.set_yticks(np.arange(0, len(list(means.index))), list(means.index))  # Don't hardcode 41
    axes.tick_params(labelbottom=True)
    plt.savefig(save_path + f'meanlog_{task}.png')
    plt.show()

    bad_chans = list(map(str, input("Enter bad channels (separated by a space, press enter if none): ").split()))
    filename = save_path + f'bad_channels_{task}.txt'
    with open(filename, mode="w") as outfile:
        for s in bad_chans:
            outfile.write("%s\n" % s)

    if bad_chans:  # If there's something in here append it to the bads
        raw.info['bads'].extend(bad_chans)

        # Save them with the bad channels now marked
        raw.save(f"{input_path}{fname}", fmt='double', overwrite=True)
