# Contains the relevant functions needed to generate the isopotential plots

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, Rbf
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl


# Labels is a list of the names associated with the electrodes with non nan locations
# Elec Pos is a 2xelectrodes array containing the average position of each electrode over all participants
# Grid Pos is a electrodesx2 array containing position of each electrode - some weird combo Birgit devised
# Grid Size is a 2x array containing the size of the grid to plot
# Some of these dimensions are opposite to the matlab code so be careful
def get_gridparameters(subjects):
    # Electrode positions in mm
    # import electrode positions
    x_pos = []
    z_pos = []
    # Legacy implementation - all subjects use standardised grid now
    for subj in np.arange(1, len(subjects) + 1):
        raw_path = '/data/pt_02835/Emma/'
        fname = raw_path + 'space-Other_electrodes.tsv'
        electrode_pos = pd.read_csv(fname, sep='\t')
        electrode_pos = electrode_pos[electrode_pos.name != 'TH6']  # Remove reference channel
        electrode_pos.reset_index(drop=True, inplace=True)

        # remove electrode positions with NaN value
        # Don't want to drop nan, just want to generate logical indices where they are or are not nan
        x_idx = electrode_pos['x'].notna().to_numpy()
        y_idx = electrode_pos['y'].notna().to_numpy()
        z_idx = electrode_pos['z'].notna().to_numpy()
        my_list = np.array([x_idx, y_idx, z_idx])
        elec_idx = np.argwhere(my_list.sum(axis=0) == 3).reshape(-1)

        labels = np.array(electrode_pos['name'][elec_idx])
        x_pos.append(np.array(electrode_pos['x'][elec_idx]))
        z_pos.append(np.array(electrode_pos['z'][elec_idx]))

    # take mean over all subjects
    # x_pos has shape (n_subjects, n_electrodes not nan position)

    # Get average over participants - now have 1 location for each electrode
    x = np.mean(x_pos, axis=0)
    z = np.mean(z_pos, axis=0)
    elec_pos = [z, x]

    # define grid positions
    grid_pos = np.empty(np.shape(elec_pos))
    grid_pos[0] = np.abs(elec_pos[0] - np.max(elec_pos[0])) + 10
    grid_pos[1] = elec_pos[1] + 60
    # grid_pos = grid_pos - 50

    # define grid size - ndarray of shape (2, no. of non nan electrodes)
    grid_size = np.max(grid_pos, axis=1) + 10
    grid_pos = grid_pos.T  # transpose to match matlab code

    return labels, elec_pos, grid_size, grid_pos


def plot_esg_isopotential(chanvalues, colorbar_axes, gridsize, chan_pos_grid, labels, colorbar, time, axis,
                          colorbar_label):

    # Input arguments:
    # chanvalues = data values for each channel
    # gridsize = size of grid
    # chan_pos_grid = in matrix indexes - grid_pos
    # labels = channel labels
    # time = current_time

    ## create electrode grid
    chan_grid = np.zeros((int(gridsize[0]), int(gridsize[1])))

    # fill grid with channels
    for ichan in np.arange(0, np.size(chan_pos_grid[:, 0])):
        chan_grid[chan_pos_grid[ichan].astype(int)] = 1

    # create meshgrid
    # x-positions
    x = chan_pos_grid[:, 1].transpose()
    # y-positions
    y = chan_pos_grid[:, 0].transpose()
    # regrid and interpolate
    xq, yq = np.meshgrid(np.arange(0, gridsize[1]), np.arange(0, gridsize[0]))
    vq = griddata((x, y), np.reshape(chanvalues, -1), (xq, yq), 'nearest')

    rbf = Rbf(x, y, chanvalues, epsilon=2)
    vq = rbf(xq, yq)

    vq *= 1e6
    levels = MaxNLocator(nbins=150).tick_values(colorbar_axes[0], colorbar_axes[1])
    cf = axis.contourf(xq, yq, np.clip(vq, a_min=colorbar_axes[0], a_max=colorbar_axes[1]),
                       levels=levels, cmap='turbo', origin='upper', extend='neither')
    if colorbar:
        ticks = [colorbar_axes[0], 0, colorbar_axes[1]]
        plt.colorbar(cf, ax=axis, label=colorbar_label, ticks=ticks)
    axis.scatter(x, y, c='k', s=12)
    axis.set_ylim(max(y), min(y))  # For some reason the graph is flipped - need to flip vertical axis
    return


def mrmr_esg_isopotentialplot(subjects, chanvalues, colorbar_axes, chan_labels, colorbar, time, axis, colorbar_label):

    # Get grid parameters
    labels, elec_pos, grid_size, grid_pos = get_gridparameters(subjects)

    # Sort channels
    ordered_chanvalues = np.zeros((1, np.size(labels)))
    for ichan in np.arange(0, np.size(labels)):
        chan_idx = np.argwhere(np.isin(chan_labels, labels[ichan]))
        if chan_idx.size != 0:
            ordered_chanvalues[0, ichan] = chanvalues[chan_idx]

    # create isopotential plot
    labels = []  # set to empty if you do not want to plot the labels
    plot_esg_isopotential(ordered_chanvalues, colorbar_axes, grid_size, grid_pos, labels, colorbar, time, axis,
                          colorbar_label)

    return
