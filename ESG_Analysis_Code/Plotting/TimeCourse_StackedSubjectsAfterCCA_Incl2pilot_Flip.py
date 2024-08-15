import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from Functions.invert import invert
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    hpf = 30  # Can be 30 or 1
    folders = ['main', 'piloting']
    peak_lat = 0.052

    # Testing laser stuff
    sampling_rate = 1000
    conditions = ['pain']

    flip_list = []
    for condition in conditions:
        erp_list = []
        # Create dataframe of each subjects evoked response
        df = pd.DataFrame()
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
                          'esg02': [1, True],
                          'esg03': [1, False],
                          'esg04': [1, False],
                          'esg05': [1, False]}

            for subject_id in subject_ids:
                flip = chosen[subject_id][1]
                if flip is True:
                    flip_list.append(subject_id)
                if hpf == 30:
                    input_path = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30/sub-{subject_id}/'
                    fname = f'{condition}.fif'
                    figure_path = f'/data/pt_02889/main&pilot/Time_Combined_hpf30_testflip/'
                    os.makedirs(figure_path, exist_ok=True)

                else:
                    input_path = f'/data/pt_02889/{folder}/esg_analysis/cca/sub-{subject_id}/'
                    fname = f'{condition}.fif'
                    figure_path = f'/data/pt_02889/main&pilot/Time_Combined_testflip/'
                    os.makedirs(figure_path, exist_ok=True)

                epochs = mne.read_epochs(input_path + fname, preload=True)

                iv_baseline = [-0.1, -0.01]
                iv_epoch = [-0.1, 0.3]

                if flip:
                    erp = epochs.apply_function(invert).pick('Cor1').average()
                else:
                    erp = epochs.pick('Cor1').average()
                erp_list.append(erp.data.reshape(-1))

                if folder == 'main' and subject_id == 'esg01':
                    df['Time'] = erp.times
                df[f'{subject_id}'] = erp.data.reshape(-1)

        # Write dataframe to excel
        if not os.path.isfile(f'/data/pt_02889/main&pilot/SX_Data.xlsx'):
            with pd.ExcelWriter(f'/data/pt_02889/main&pilot/SX_Data.xlsx', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='StackedTimeCourses')
        else:
            with pd.ExcelWriter(f'/data/pt_02889/main&pilot/SX_Data.xlsx', mode='a', engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='StackedTimeCourses')

        fig, axes = plt.subplots(len(erp_list), 1, figsize=(6, 12))
        for erp, axis in zip(erp_list, axes):
            axis.plot(epochs.times, erp)
            axis.set_xlim([-0.01, 0.15])
            axis.axvline(peak_lat, color='red', linewidth=1)
            axis.set_ylabel('Amplitude')
            axis.set_xlabel('Time (s)')
            ylabels = ['{:,.2f}'.format(x) for x in axis.get_yticks()]
            axis.set_yticklabels(ylabels)
        plt.suptitle(f'n={len(erp_list)}, Comp 1, {condition}\n')
        plt.tight_layout()
        plt.savefig(figure_path + f'Combined_{condition}_flipped:{flip_list}.png')
        plt.savefig(figure_path + f'Combined_{condition}_flipped:{flip_list}.pdf',
                    bbox_inches='tight', format="pdf")
        # plt.show()
