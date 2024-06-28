###############################################################################################
# Emma Bailey, 14/12/2023
# Wrapper script for project to investigate somatosensory pilot ESG data
###############################################################################################

import numpy as np
import os
from Functions.import_data import import_data
from Functions.SSP import apply_SSP
from Functions.bad_channel_check import bad_channel_check
from Functions.run_CCA_spinal_hpf30 import run_CCA_hpf30
from Archive.SSP_anterior import apply_SSP_anterior
from Archive.run_CCA_spinal import run_CCA
from Archive.run_CCA_spinal_hpf30_anterior import run_CCA_hpf30_anterior
from Archive.run_CCA_spinal_hpf30_sns import run_CCA_hpf30_sns


if __name__ == '__main__':
    folder = 'piloting'  # Can be main or piloting

    # Testing laser stuff
    sampling_rate = 1000  # Frequency to downsample to from original of 10kHz
    if folder == 'piloting':
        subjects = np.arange(1, 3)
        subject_ids = [f'esgpilot{str(subject).zfill(2)}' for subject in subjects]
    else:
        subjects = np.arange(1, 7)
        subject_ids = [f'esg{str(subject).zfill(2)}' for subject in subjects]

    conditions = ['pain', 'cold', 'somatosensory']

    ######## 1. Import ############
    import_d = False  # Prep work

    ######### 2. Clean the heart artefact using SSP ########
    SSP_flag = False
    no_projections = 6

    ######## 3. Bad Channel Check, Performed but no channels excluded before CCA #######
    check_channels = False

    ######## 4. Perform CCA with hpf at 30Hz to remove cardiac noise #######
    perform_cca_hpf30 = False

    ############################################
    # Import Data from BIDS directory
    # Select channels to analyse
    # Remove stimulus artefact by PCHIP interpolation
    # Downsample and concatenate blocks of the same conditions
    # Also notch filters powerline noise and hpf at 1Hz
    ############################################
    if import_d:
        bids_root = f'/data/pt_02889/{folder}/raw_data/'  # Taking data from the bids folder
        for subject_id in subject_ids:
            if folder == 'main':
                if subject_id == 'esg06':
                    conditions = ['cold', 'somatosensory']
            input_path = f'/data/pt_02889/{folder}/raw_data/sub-{subject_id}/eeg/'  # Taking data from the bids folder
            save_path = f'/data/pt_02889/{folder}/esg_analysis/imported/sub-{subject_id}/'
            os.makedirs(save_path, exist_ok=True)
            for condition in conditions:
                import_data(subject_id, condition, sampling_rate, bids_root, input_path, save_path)

    ##################################################
    # To remove heart artifact using SSP method in MNE
    ###################################################
    if SSP_flag:
        for subject_id in subject_ids:
            input_path = f'/data/pt_02889/{folder}/esg_analysis/imported/sub-{subject_id}/'
            save_path = f'/data/pt_02889/{folder}/esg_analysis/ssp_cleaned/sub-{subject_id}/'
            os.makedirs(save_path, exist_ok=True)
            if subject_id == 'esg06' and folder == 'main':
                conditions = ['cold', 'somatosensory']
            for condition in conditions:
                apply_SSP(subject_id, condition, sampling_rate, no_projections, input_path, save_path)

    ###################################################
    # Bad Channel Check
    ###################################################
    if check_channels:
        bad_channel_check(subject_id, condition, sampling_rate)

    ###################################################
    # Run CCA on stimulus trials with 30Hz high pass filter
    ###################################################
    if perform_cca_hpf30:
        for subject_id in subject_ids:
            input_path = f'/data/pt_02889/{folder}/esg_analysis/ssp_cleaned/sub-{subject_id}/'
            save_path = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30/sub-{subject_id}/'
            os.makedirs(save_path, exist_ok=True)
            if subject_id == 'esg06' and folder == 'main':
                conditions = ['somatosensory', 'cold']
            for condition in conditions:
                cond_info = get_conditioninfo(condition)
                trigger = cond_info.trigger_name

                run_CCA_hpf30(subject_id, condition, trigger, sampling_rate, input_path, save_path, folder)

    #############################################################################################################
    # Old code
    #############################################################################################################
    # ######### 2. Clean the heart artefact using SSP on anterior rereference data ########
    # SSP_anterior_flag = False
    # ##################################################
    # # To remove heart artifact using SSP method in MNE
    # ###################################################
    # if SSP_anterior_flag:
    #     for subject_id in subject_ids:
    #         input_path = f'/data/pt_02889/{folder}/esg_analysis/imported/sub-{subject_id}/'
    #         save_path = f'/data/pt_02889/{folder}/esg_analysis/ssp_cleaned_anterior/sub-{subject_id}/'
    #         os.makedirs(save_path, exist_ok=True)
    #         if subject_id == 'esg06' and folder == 'main':
    #             conditions = ['cold', 'somatosensory']
    #         conditions = ['pain']
    #         for condition in conditions:
    #             apply_SSP_anterior(subject_id, condition, sampling_rate, no_projections, input_path, save_path)
    #
    # ###################################################
    # # Run CCA on stimulus trials
    # ###################################################
    # ######## 4. Perform CCA #######
    # perform_cca = False
    # if perform_cca:
    #     for subject_id in subject_ids:
    #         input_path = f'/data/pt_02889/{folder}/esg_analysis/ssp_cleaned/sub-{subject_id}/'
    #         save_path = f'/data/pt_02889/{folder}/esg_analysis/cca/sub-{subject_id}/'
    #         os.makedirs(save_path, exist_ok=True)
    #         if subject_id == 'esg06' and folder == 'main':
    #             conditions = ['somatosensory', 'cold']
    #         for condition in conditions:
    #             cond_info = get_conditioninfo(condition)
    #             trigger = cond_info.trigger_name
    #
    #             run_CCA(subject_id, condition, trigger, sampling_rate, input_path, save_path, folder)
    #
    # ######## 4. Perform CCA with hpf at 30Hz to remove cardiac noise on anterior reref data #######
    # perform_cca_hpf30_anterior = False
    # ###################################################
    # # Run CCA on stimulus trials with 30Hz high pass filter
    # ###################################################
    # if perform_cca_hpf30_anterior:
    #     for subject_id in subject_ids:
    #         input_path = f'/data/pt_02889/{folder}/esg_analysis/ssp_cleaned_anterior/sub-{subject_id}/'
    #         save_path = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30_anterior/sub-{subject_id}/'
    #         os.makedirs(save_path, exist_ok=True)
    #         if subject_id == 'esg06' and folder == 'main':
    #             conditions = ['somatosensory', 'cold']
    #         conditions = ['pain']
    #         for condition in conditions:
    #             cond_info = get_conditioninfo(condition)
    #             trigger = cond_info.trigger_name
    #
    #             run_CCA_hpf30_anterior(subject_id, condition, trigger, sampling_rate, input_path, save_path, folder)
    #
    # ######## 4. Perform CCA with hpf at 30Hz and sns #######
    # perform_cca_hpf30_sns = False
    # ###################################################
    # # Run CCA on stimulus trials with 30Hz high pass filter and sns before CCA
    # ###################################################
    # if perform_cca_hpf30_sns:
    #     conditions = ['pain']
    #     for subject_id in subject_ids:
    #         input_path = f'/data/pt_02889/{folder}/esg_analysis/ssp_cleaned/sub-{subject_id}/'
    #         save_path = f'/data/pt_02889/{folder}/esg_analysis/cca_hpf30_sns/sub-{subject_id}/'
    #         os.makedirs(save_path, exist_ok=True)
    #         if subject_id == 'esg06' and folder == 'main':
    #             conditions = ['somatosensory', 'cold']
    #         for condition in conditions:
    #             cond_info = get_conditioninfo(condition)
    #             trigger = cond_info.trigger_name
    #
    #             run_CCA_hpf30_sns(subject_id, condition, trigger, sampling_rate, input_path, save_path, folder)




