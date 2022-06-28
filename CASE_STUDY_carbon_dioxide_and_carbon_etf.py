
#     ________
#            /
#      \    /
#       \  /
#        \/

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from AdvEMDpy import EMD
from emd_hilbert import hilbert_spectrum, Hilbert, omega, theta

sns.set(style='darkgrid')

# load raw data
raw_data_ecf1 = pd.read_csv('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ECF1', header=0)
raw_data_ecf1 = raw_data_ecf1.set_index('time')
raw_data_ecf2 = pd.read_csv('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ECF2', header=0)
raw_data_ecf2 = raw_data_ecf2.set_index('time')
efc1_contains = raw_data_ecf1.index.difference(raw_data_ecf2.index)
raw_data_ecf1 = raw_data_ecf1.reindex(raw_data_ecf2.index)
full_index = raw_data_ecf1.index.append(efc1_contains)
raw_data_ecf1 = raw_data_ecf1.reindex(full_index).interpolate()
raw_data_ecf2 = raw_data_ecf2.reindex(full_index).interpolate()
raw_data_ecf1_close = np.array(raw_data_ecf1['close'])
raw_data_ecf2_close = np.array(raw_data_ecf2['close'])

plt.plot(raw_data_ecf1_close)
plt.plot(raw_data_ecf2_close)
plt.show()

# emd = EMD(time=np.arange(len(raw_data_ecf1_close)), time_series=raw_data_ecf1_close)
# imfs, hts, ifs, _, _, dthts, dtifs = emd.empirical_mode_decomposition(knots=np.linspace(0, len(raw_data_ecf1_close),
#                                                                                         1001), verbose=True,
#                                                                       debug=False, max_internal_iter=30,
#                                                                       matrix=True, dtht=True)
# np.save('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf1_imfs.npy', imfs)
# np.save('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf1_hts.npy', hts)
# np.save('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf1_ifs.npy', ifs)
# np.save('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf1_dthts.npy', dthts)
# np.save('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf1_dtifs.npy', dtifs)
#
# emd = EMD(time=np.arange(len(raw_data_ecf2_close)), time_series=raw_data_ecf2_close)
# imfs, hts, ifs, _, _, dthts, dtifs = emd.empirical_mode_decomposition(knots=np.linspace(0, len(raw_data_ecf2_close),
#                                                                                         1001), verbose=True,
#                                                                       debug=False, max_internal_iter=30, matrix=True,
#                                                                       dtht=True)
# np.save('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf2_imfs.npy', imfs)
# np.save('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf2_hts.npy', hts)
# np.save('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf2_ifs.npy', ifs)
# np.save('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf2_dthts.npy', dthts)
# np.save('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf2_dtifs.npy', dtifs)

ecf1_imfs = np.load('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf1_imfs.npy')
ecf1_hts = np.load('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf1_hts.npy')
ecf1_ifs = np.load('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf1_ifs.npy')
ecf1_dthts = np.load('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf1_dthts.npy')
ecf1_dtifs = np.load('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf1_dtifs.npy')

plt.plot(ecf1_imfs[1:-1, :].T)
plt.show()

ecf2_imfs = np.load('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf2_imfs.npy')
ecf2_hts = np.load('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf2_hts.npy')
ecf2_ifs = np.load('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf2_ifs.npy')
ecf2_dthts = np.load('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf2_dthts.npy')
ecf2_dtifs = np.load('/home/cole/Desktop/Cole/Cole Documents/AdvEMDpy/AdvEMDpy/Carbon Data/ecf2_dtifs.npy')

plt.plot(ecf2_imfs[1:-1, :].T)
plt.show()

plt.plot(ecf1_imfs[1, :].T)
plt.plot(ecf2_imfs[1, :].T)
plt.show()

time = np.arange(len(raw_data_ecf2_close))

imfs_test = [1]
max_freq = 1 / (2 * np.pi)
max_amp = 0.05
freq_incr = 101

hs_ouputs = hilbert_spectrum(time, ecf1_imfs, ecf1_dthts, ecf1_dtifs / (2 * np.pi), max_frequency=max_freq, plot=False,
                             freq_increments=freq_incr, filter_sigma=5, which_imfs=imfs_test)

ax = plt.subplot(111)
figure_size = plt.gcf().get_size_inches()
factor = 0.9
plt.gcf().set_size_inches((figure_size[0], factor * figure_size[1]))
ax.set_title(textwrap.fill(r'Instantaneous Frequency of ECF1', 55))
x_hs, y, z = hs_ouputs
z_min, z_max = 0, max_amp * np.abs(z).max()
ax.pcolormesh(x_hs, y, np.abs(z), cmap='gist_rainbow', vmin=z_min, vmax=z_max)
plt.plot((1 / 14) * np.ones(len(time[:-1])), 'k--')
plt.ylabel('Frequency (day$^{-1}$)')
plt.xlabel('Time (days)')

box_0 = ax.get_position()
ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.85, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

hs_ouputs = hilbert_spectrum(time, ecf2_imfs, ecf2_dthts, ecf2_dtifs / (2 * np.pi), max_frequency=max_freq, plot=False,
                             freq_increments=freq_incr, filter_sigma=5, which_imfs=imfs_test)

ax = plt.subplot(111)
figure_size = plt.gcf().get_size_inches()
factor = 0.9
plt.gcf().set_size_inches((figure_size[0], factor * figure_size[1]))
ax.set_title(textwrap.fill(r'Instantaneous Frequency of ECF2', 55))
x_hs, y, z = hs_ouputs
z_min, z_max = 0, max_amp * np.abs(z).max()
ax.pcolormesh(x_hs, y, np.abs(z), cmap='gist_rainbow', vmin=z_min, vmax=z_max)
plt.plot((1 / 14) * np.ones(len(time[:-1])), 'k--')
plt.ylabel('Frequency (day$^{-1}$)')
plt.xlabel('Time (days)')

box_0 = ax.get_position()
ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.85, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
