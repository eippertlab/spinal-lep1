from scipy.stats import zscore
def transform(data):
    data = zscore(data, axis=0)
    return data
