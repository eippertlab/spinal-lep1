def get_channels(includesEcg):
    esg_chans = ['CA1', 'CA2', 'CA3',
                 'C57', 'C97',
                 'C33', 'C53', 'C73', 'C93', 'C113',
                 'C21', 'C41', 'C61', 'C81', 'C101', 'C121',
                 'CS1', 'C1z', 'C3z', 'C5z', 'C7z', 'C9z', 'C11z', 'C13z',
                 'C22', 'C42', 'C62', 'C82', 'C102',
                 'C34', 'C54', 'C74', 'C94', 'C114',
                 'C58', 'C98']

    # include ECG
    if includesEcg:
        esg_chans.append('ECG')

    return esg_chans
