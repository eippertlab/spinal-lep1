import mne
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
import os
import numpy as np
import matplotlib.pyplot as plt

def readFilter(subject, data_path, l_freq, sr_new, save_path, n_runs=10, task='pain'):
    esg_list = ['CS1', 'C1z', 'C3z', 'C5z', 'C7z', 'C9z', 'C11z', 'C13z', 'C73', 'C81', 'C82', 'C93', 'C94', 'C97',
                'C74', 'C98', 'CA1', 'CA2', 'CA3', 'nose-Th6', 'C21', 'C101', 'C58', 'C57', 'C61', 'C62', 'C42', 'C34',
                'C53', 'C54', 'C33', 'C41', 'C121', 'C22', 'C102', 'C113', 'C114', 'C122', 'Biceps', 'Iz']
    raw = []
    event_files = []
    for i in range(n_runs):
        run = i + 1   #get correct run number
        bids_path = BIDSPath(subject=subject, run=run, task=task, root=data_path)
        data = read_raw_bids(bids_path=bids_path, extra_params={'preload': True})
        if subject == 'esgpilot02' and i in [3, 4, 5, 6, 7]:
            data.drop_channels(['STI 015'])
        data.drop_channels(esg_list)
        raw.append(data)

        event_file = pd.read_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{0}_task-{1}_run-{2:02d}_events.tsv'.format(subject, task, run)), sep='\t')
        event_files.append(event_file)

    mne.concatenate_raws(raw)
    raw = raw[0]
    events_tsv = pd.concat(event_files)
    events_tsv.reset_index(drop=True, inplace=True)

    raw.resample(sr_new)

    raw.filter(l_freq = l_freq, h_freq=None, method='iir')  #per default 4th order butterworth filter_type
    for freq in np.arange(50, 250, 50):
        raw.notch_filter(freqs=freq, method='iir')

    events, ids = mne.events_from_annotations(raw)
    epochs = mne.Epochs(raw, events, ids, tmin=-0.3, tmax=1, baseline=None)

    events_tsv['dropped'] = 0
    events_tsv['drop reason'] = 'n/a'

    # delete trials in which laser aborted
    epochs_subset = epochs['pain']
    drop_idx = [x for x, y in enumerate(epochs_subset.drop_log) if y == ('IGNORED',)]
    events_tsv.loc[drop_idx, 'dropped'] = 1
    events_tsv.loc[drop_idx, 'drop reason'] = 'LASER ABORT'

    events_tsv.to_csv(os.path.join(save_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_events.tsv'.format(subject)),
                      sep='\t', index=False, na_rep='n/a')
    epochs_subset.save(os.path.join(save_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_epo.fif'.format(subject)))

# plots single epochs, user can manually reject an epoch by clicking on it or reject a channel by clicking on the name
# to open interactive plots it could be necessary to run the folllowing:
# import matplotlib
# matplotlib.use('TkAgg')
def manRejectPre(subject, data_path):
    epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_epo.fif'.format(subject)), preload=True)
    events_tsv = pd.read_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_events.tsv'.format(subject)), sep='\t')

    if not len(events_tsv) == len(epochs.drop_log):
        raise Exception("Lengths of events file and drop log don't match!")

    epochs.plot(n_epochs=1, n_channels=32, block=True)
    drop_idx = [x for x, y in enumerate(epochs.drop_log) if y == ('USER',)]
    events_tsv.loc[drop_idx, 'dropped'] = 1
    events_tsv.loc[drop_idx, 'drop reason'] = 'MAN REJECT PRE ICA'

    if not len(events_tsv) == len(epochs.drop_log):
        raise Exception("Lengths of events file and drop log don't match!")

    events_tsv.to_csv(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_manReject_events.tsv'.format(subject)),
                      sep='\t', index=False, na_rep='n/a')
    epochs.save(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg', 'sub-{}_woAborts_hpfilter_manReject_epo.fif'.format(subject)))

def quickAverage(subject, data_path):
    epochs = mne.read_epochs(os.path.join(data_path, 'sub-{}'.format(subject), 'eeg',
                                          'sub-{}_woAborts_hpfilter_manReject_muscleCleaned_epo.fif'.format(subject)),
                             preload=True)

    epochs.filter(l_freq=None, h_freq=30, method='iir')
    evo = epochs.average()
    evo_avg = evo.copy().set_eeg_reference('average')
    evo_Fz = evo.copy().set_eeg_reference(['Fz'])

    _, lat_N1, amp_N1 = evo_Fz.copy().pick(['T8']).get_peak(tmin=0.1, tmax=0.265, mode='neg', return_amplitude=True)
    _, lat_N2, amp_N2 = evo_avg.copy().pick(['Cz']).get_peak(tmin=0.2, tmax=0.4, mode='neg', return_amplitude=True)
    _, lat_P2, amp_P2 = evo_avg.copy().pick(['Cz']).get_peak(tmin=0.3, tmax=0.6, mode='pos', return_amplitude=True)

    fig, ax = plt.subplots()
    mne.viz.plot_compare_evokeds(evo_avg, picks=['Cz'], title='{}, Cz-avg'.format(subject), axes=ax, show=False)
    plt.text((lat_N2 + 0.05), (amp_N2*1e6 + 0.5), "N2:\n{:.3} ms\n{:.3} V".format(lat_N2, amp_N2), axes=ax)
    plt.text((lat_P2 + 0.05), (amp_P2*1e6 - 1.2), "P2:\n{:.3} ms\n{:.3} V".format(lat_P2, amp_P2), axes=ax)
    plt.show()

    fig, ax = plt.subplots()
    mne.viz.plot_compare_evokeds(evo_Fz, picks=['T8'], title='{}, T8-Fz'.format(subject), axes=ax, show=False)
    plt.text((lat_N1 - 0.1), (amp_N1*1e6 - 1.8), "N1:\n{:.3} ms\n{:.3} V".format(lat_N1, amp_N1), axes=ax)
    plt.show()