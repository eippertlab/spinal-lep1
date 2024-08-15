import mne.viz
import matplotlib as mpl
import prepro_functions as pp
import os

subjects = ['esgpilot01', 'esgpilot02', 'esg01', 'esg02', 'esg03', 'esg04', 'esg05']
montage_file = '/data/pt_02889/main/code/standard-10-5-cap.elp'

run_1_ReadFilter = 0
run_2_ManualRejectionPreICA = 0
run_3_quickAverageSingleSubject = 0
run_4_grandAverage = 1

sr = 500  # sampling rate for downsampling before prepro
l_freq = 1  # high-pass edge frequency for ICA
muscle_correct_type = 'corrected'  # whether muscle tool was used or not
threshold_amp = 100  # epochs containing values exceeding +- this value µV after ICA cleaning will be dropped automatically
threshold_jump = 50  # epochs containing jumps between adjacent datapoints exceeding +- this value µV after ICA cleaning will be dropped automatically
reject_mode = 'strict'  # whether to use the strict or liberal selection of epochs to reject based on TFA
h_freq = 30  # low-pass edge frequency for LEPs
n_runs = 10
evo_all_avg = []
evo_all_Fz = []
dict_avg = {}
dict_Fz = {}

for sub in subjects:
    if sub in ['esgpilot01', 'esgpilot02']:
        data_path = '/data/pt_02889/piloting/raw_data/'
        save_path = '/data/pt_02889/piloting/processed_data/'
    else:
        data_path = '/data/pt_02889/main/raw_data/'
        save_path = '/data/pt_02889/main/processed_data/'

    if not os.path.isdir(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg')):
        os.makedirs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg'))

    if run_1_ReadFilter:
        pp.readFilter(subject=sub, data_path=data_path, l_freq=l_freq, sr_new=sr, save_path=save_path, n_runs=n_runs)

    if run_2_ManualRejectionPreICA:
        pp.manRejectPre(subject=sub, data_path=save_path)

    if run_3_quickAverageSingleSubject:
        pp.quickAverage(subject=sub, data_path=save_path)

    if run_4_grandAverage:
        epochs = mne.read_epochs(os.path.join(save_path, 'sub-{}'.format(sub), 'eeg',
                                              'sub-{}_woAborts_hpfilter_manReject_epo.fif'.format(
                                                  sub)),
                                 preload=True)

        if sub in ['esgpilot01', 'esgpilot02']:
            epochs.drop_channels(['P6', 'F6', 'CPz', 'C5', 'Fpz', 'AF8', 'C2', 'PO4', 'C6', 'PO3', 'P1', 'AF4', 'CP4', 'AF3', 'FC3', 'CP3', 'FT8', 'P2', 'PO7', 'AFz', 'F2', 'F1', 'FT7', 'C1', 'F5', 'Oz', 'TP7', 'FC4', 'P5', 'PO8', 'AF7', 'TP8'])

        epochs.filter(l_freq=None, h_freq=30, method='iir')
        evo = epochs.average()
        evo_avg = evo.copy().set_eeg_reference('average')
        evo_Fz = evo.copy().set_eeg_reference(['Fz'])
        evo_all_avg.append(evo_avg)
        evo_all_Fz.append(evo_Fz)
        dict_avg[sub] = evo_avg
        dict_Fz[sub] = evo_Fz

if run_4_grandAverage:
    mne.viz.plot_compare_evokeds({'pain':evo_all_avg}, picks=['Cz'], title='Grand average, Cz-avg', ylim=dict(eeg=[-5.5, 4.5]), ci=0.95)
    mne.viz.plot_compare_evokeds({'pain':evo_all_Fz}, picks=['T8'], title='Grand average, T8-Fz', ylim=dict(eeg=[-5.5, 4.5]), ci=0.95)

    ga_avg = mne.grand_average(list(dict_avg.values()))
    ga_Fz = mne.grand_average(list(dict_Fz.values()))

    _, lat_N1 = ga_Fz.copy().pick(['T8']).get_peak(tmin=0.1, tmax=0.3, mode='neg')
    _, lat_N2 = ga_avg.copy().pick(['Cz']).get_peak(tmin=0.2, tmax=0.4, mode='neg')
    _, lat_P2 = ga_avg.copy().pick(['Cz']).get_peak(tmin=0.3, tmax=0.5, mode='pos')

    montage = mne.channels.read_custom_montage(montage_file)
    ga_avg.set_montage(montage)
    ga_Fz.set_montage(montage)

    new_rc_params = {"font.family": 'Arial', "font.size": 12, "font.serif": []}
    mpl.rcParams.update(new_rc_params)

    fig1 = ga_avg.plot_joint(times=[lat_N2, lat_P2])
    fig1.savefig('/data/pt_02889/main/results/nociceptive_cortical_N2P2.eps', format='eps')
    fig2 = ga_Fz.plot_joint(times=[lat_N1])
    fig2.savefig('/data/pt_02889/main/results/nociceptive_cortical_N1.eps', format='eps')