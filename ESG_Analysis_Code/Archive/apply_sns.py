from meegkit import sns


def apply_sns(data):
    data_correctform = data.swapaxes(0, 2)
    y, r = sns.sns(data_correctform)
    data_return = y.swapaxes(0, 2)

    return data_return
