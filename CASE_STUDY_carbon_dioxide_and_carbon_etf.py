
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

from sklearn import linear_model

sns.set(style='darkgrid')

# load raw data
raw_data_ecf1 = pd.read_csv('Carbon Data/ECF1', header=0)
raw_data_ecf1 = raw_data_ecf1.set_index('time')
raw_data_ecf2 = pd.read_csv('Carbon Data/ECF2', header=0)
raw_data_ecf2 = raw_data_ecf2.set_index('time')
efc1_contains = raw_data_ecf1.index.difference(raw_data_ecf2.index)
raw_data_ecf1 = raw_data_ecf1.reindex(raw_data_ecf2.index)
full_index = raw_data_ecf1.index.append(efc1_contains)
raw_data_ecf1 = raw_data_ecf1.reindex(full_index).interpolate()
raw_data_ecf2 = raw_data_ecf2.reindex(full_index).interpolate()
raw_data_ecf1_close = np.array(raw_data_ecf1['close'])
raw_data_ecf2_close = np.array(raw_data_ecf2['close'])

temp = np.array(raw_data_ecf1.index)

ax = plt.subplot(111)
figure_size = plt.gcf().get_size_inches()
factor = 0.9
plt.gcf().set_size_inches((figure_size[0], factor * figure_size[1]))
ax.set_title(textwrap.fill(r'ECF1 and ECF2 in Euros', 55))
plt.plot(raw_data_ecf1_close, label='ECF1')
plt.plot(raw_data_ecf2_close, label='ECF2')
plt.ylabel('Euros')
plt.xlabel('Time (days)')
plt.xticks([0, 176, 433, 690, 949, 1207, 1466, 1724, 1982, 2239, 2496, 2754, 3012, 3269, 3527,
            3785, 4044, 4303],
           ['22-04-2005', '03-01-2006', '03-01-2007', '02-01-2008', '02-01-2009', '04-01-2010',
            '03-01-2011', '03-01-2012', '02-01-2013', '02-01-2014', '02-01-2015', '04-01-2016',
            '03-01-2017', '02-01-2018', '02-01-2019', '02-01-2020', '04-01-2021', '03-01-2022'],
           fontsize=8, rotation=-45, ha='left')
box_0 = ax.get_position()
ax.set_position([box_0.x0, box_0.y0 + 0.10, box_0.width * 0.95, box_0.height * 0.85])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/ecf1_and_ecf2.png')
plt.show()

# emd = EMD(time=np.arange(len(raw_data_ecf1_close)), time_series=raw_data_ecf1_close)
# imfs, hts, ifs, _, _, dthts, dtifs = emd.empirical_mode_decomposition(knots=np.linspace(0, len(raw_data_ecf1_close),
#                                                                                         1001), verbose=True,
#                                                                       debug=False, max_internal_iter=30,
#                                                                       matrix=True, dtht=True)
# np.save('Carbon Data/ecf1_imfs.npy', imfs)
# np.save('Carbon Data/ecf1_hts.npy', hts)
# np.save('Carbon Data/ecf1_ifs.npy', ifs)
# np.save('Carbon Data/ecf1_dthts.npy', dthts)
# np.save('Carbon Data/ecf1_dtifs.npy', dtifs)
#
# emd = EMD(time=np.arange(len(raw_data_ecf2_close)), time_series=raw_data_ecf2_close)
# imfs, hts, ifs, _, _, dthts, dtifs = emd.empirical_mode_decomposition(knots=np.linspace(0, len(raw_data_ecf2_close),
#                                                                                         1001), verbose=True,
#                                                                       debug=False, max_internal_iter=30, matrix=True,
#                                                                       dtht=True)
# np.save('Carbon Data/ecf2_imfs.npy', imfs)
# np.save('Carbon Data/ecf2_hts.npy', hts)
# np.save('Carbon Data/ecf2_ifs.npy', ifs)
# np.save('Carbon Data/ecf2_dthts.npy', dthts)
# np.save('Carbon Data/ecf2_dtifs.npy', dtifs)

ecf1_imfs = np.load('Carbon Data/ecf1_imfs.npy')
ecf1_hts = np.load('Carbon Data/ecf1_hts.npy')
ecf1_ifs = np.load('Carbon Data/ecf1_ifs.npy')
ecf1_dthts = np.load('Carbon Data/ecf1_dthts.npy')
ecf1_dtifs = np.load('Carbon Data/ecf1_dtifs.npy')

ecf2_imfs = np.load('Carbon Data/ecf2_imfs.npy')
ecf2_hts = np.load('Carbon Data/ecf2_hts.npy')
ecf2_ifs = np.load('Carbon Data/ecf2_ifs.npy')
ecf2_dthts = np.load('Carbon Data/ecf2_dthts.npy')
ecf2_dtifs = np.load('Carbon Data/ecf2_dtifs.npy')

# x = np.arange(len(ecf2_dtifs[1, :]))
# y = 1 / (ecf2_dtifs[1, :] / (2 * np.pi))
# X = np.ones((2, len(y)))
# X[1, :] = x
#
# coef = np.linalg.lstsq(X.T, y, rcond=None)[0]
# clf = linear_model.Lasso(alpha=1000000)
# clf.fit(X.T, y)
# m, b = np.polyfit(x, y, 1)
# plt.plot(x, y)
# plt.plot(x, np.median(y) * np.ones(len(y)), label='Median')
# plt.plot(x, m * x + b, label='Least squares')
# plt.plot(x, np.median(y) - (clf.intercept_ - b) + m * x, label='Median least squares')
# plt.ylim(-5, 25)
# plt.legend(loc='best')
# plt.show()

# returns = np.ones(len(ecf2_imfs[1, :]))
# for day in range(len(ecf2_imfs[1, :])):
#     if day % 12 == 9:
#         returns[day:] *= (raw_data_ecf2_close[day] / raw_data_ecf2_close[int(day - 6)])
# plt.plot(returns)
# plt.show()

time = np.arange(len(raw_data_ecf2_close))

imfs_test = [1]
max_freq = 1 / (2 * np.pi)
max_amp = 0.1
freq_incr = 101

hs_ouputs = hilbert_spectrum(time, ecf1_imfs, ecf1_dthts, ecf1_dtifs / (2 * np.pi), max_frequency=max_freq, plot=False,
                             freq_increments=freq_incr, filter_sigma=5, which_imfs=imfs_test)

ax = plt.subplot(111)
figure_size = plt.gcf().get_size_inches()
factor = 0.9
plt.gcf().set_size_inches((figure_size[0], factor * figure_size[1]))
ax.set_title(textwrap.fill(r'Instantaneous Frequency of IMF 1 of ECF1', 55))
x_hs, y, z = hs_ouputs
z_min, z_max = 0, 0.1 * max_amp * np.abs(z).max()
ax.pcolormesh(x_hs, y, np.abs(z), cmap='gist_rainbow', vmin=z_min, vmax=z_max)
plt.ylabel('Frequency (day$^{-1}$)')
plt.xlabel('Time (days)')

x = np.arange(np.shape(x_hs)[1])
y = ecf1_dtifs[1, :] / (2 * np.pi)
X = np.ones((2, len(y)))
X[1, :] = x

coef = np.linalg.lstsq(X.T, y, rcond=None)[0]
clf = linear_model.Lasso(alpha=1000000)
clf.fit(X.T, y)
m, b = np.polyfit(x, y, 1)
plt.plot(x, np.median(y) * np.ones(len(y)), '--', label='Median', linewidth=3)
plt.plot(x, m * x + b, '--', label='Least squares', linewidth=3)
plt.plot(x, np.median(y) - (clf.intercept_ - b) + m * x, '--', label=textwrap.fill('Median least squares', 15), linewidth=3)

box_0 = ax.get_position()
ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.85, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/if_imf_1_ecf1.png')
plt.show()

hs_ouputs = hilbert_spectrum(time, ecf2_imfs, ecf2_dthts, ecf2_dtifs / (2 * np.pi), max_frequency=max_freq, plot=False,
                             freq_increments=freq_incr, filter_sigma=5, which_imfs=imfs_test)

ax = plt.subplot(111)
figure_size = plt.gcf().get_size_inches()
factor = 0.9
plt.gcf().set_size_inches((figure_size[0], factor * figure_size[1]))
ax.set_title(textwrap.fill(r'Instantaneous Frequency of IMF 1 of ECF2', 55))
x_hs, y, z = hs_ouputs
z_min, z_max = 0, max_amp * np.abs(z).max()
ax.pcolormesh(x_hs, y, np.abs(z), cmap='gist_rainbow', vmin=z_min, vmax=z_max)
plt.ylabel('Frequency (day$^{-1}$)')
plt.xlabel('Time (days)')

x = np.arange(np.shape(x_hs)[1])
y = ecf2_dtifs[1, :] / (2 * np.pi)
X = np.ones((2, len(y)))
X[1, :] = x

coef = np.linalg.lstsq(X.T, y, rcond=None)[0]
clf = linear_model.Lasso(alpha=1000000)
clf.fit(X.T, y)
m, b = np.polyfit(x, y, 1)
plt.plot(x, np.median(y) * np.ones(len(y)), '--', label='Median', linewidth=3)
plt.plot(x, m * x + b, '--', label='Least squares', linewidth=3)
plt.plot(x, np.median(y) - (clf.intercept_ - b) + m * x, '--', label=textwrap.fill('Median least squares', 15), linewidth=3)

box_0 = ax.get_position()
ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.85, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/if_imf_1_ecf2.png')
plt.show()

fig, axs = plt.subplots(2, 2)
plt.suptitle('Instantaneous Frequency of IMFs of ECF1 and ECF2')
plt.subplots_adjust(hspace=0.35)
plt.subplots_adjust(wspace=0.25)
for i in np.arange(0, 2):
    for j in np.arange(0, 2):
        if i == 0:
            x_hs, y, z = hilbert_spectrum(time, ecf1_imfs, ecf1_dthts, ecf1_dtifs / (2 * np.pi), max_frequency=max_freq / 2,
                                          plot=False, freq_increments=freq_incr, filter_sigma=5,
                                          which_imfs=[int(j + 2)])
            axs[i, j].plot(x_hs[0, :], np.median(ecf1_dtifs[int(j + 2), :] / (2 * np.pi)) * np.ones_like(x_hs[0, :]), 'k--',
                           label='Median')
        if i == 1:
            x_hs, y, z = hilbert_spectrum(time, ecf2_imfs, ecf2_dthts, ecf2_dtifs / (2 * np.pi), max_frequency=max_freq / 2,
                                          plot=False, freq_increments=freq_incr, filter_sigma=5,
                                          which_imfs=[int(j + 2)])
            axs[i, j].plot(x_hs[0, :], np.median(ecf1_dtifs[int(j + 2), :] / (2 * np.pi)) * np.ones_like(x_hs[0, :]), 'k--',
                           label='Median')
        axs[i, j].pcolormesh(x_hs, y, np.abs(z), cmap='gist_rainbow', vmin=z_min, vmax=z_max)
        axs[i, j].set_title('ECF{} IMF {}'.format(int(i+1), int(j+2)))

# axs[0, 1].plot(time, imfs[0, :], label='Smoothed')
axs[1, 1].legend(loc='upper left')

axis = 0
for ax in axs.flat:
    if axis == 0:
        ax.set(ylabel=R'Frequency (day$^{-1}$)')
    if axis == 1:
        pass
    if axis == 2:
        ax.set(ylabel=R'Frequency (day$^{-1}$)')
        ax.set(xlabel='Time (days)')
    if axis == 3:
        ax.set(xlabel='Time (days)')
    axis += 1

plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig('Real-World Figures/if_ecf1_and_ecf2.png')
plt.show()

fig, axs = plt.subplots(2, 2)
plt.suptitle('Instantaneous Frequency of IMFs of ECF1 and ECF2')
plt.subplots_adjust(hspace=0.35)
plt.subplots_adjust(wspace=0.35)
for i in np.arange(0, 2):
    for j in np.arange(0, 2):
        if i == 0:
            x_hs, y, z = hilbert_spectrum(time, ecf1_imfs, ecf1_dthts, ecf1_dtifs / (2 * np.pi), max_frequency=max_freq / 10,
                                          plot=False, freq_increments=freq_incr, filter_sigma=5,
                                          which_imfs=[int(j + 4)])
            axs[i, j].plot(x_hs[0, :], np.median(ecf1_dtifs[int(j + 4), :] / (2 * np.pi)) * np.ones_like(x_hs[0, :]), 'k--',
                           label='Median')

            if j == 1:
                axs[i, j].plot(x_hs[0, :], (17.2 / 4411) * np.ones_like(x_hs[0, :]),
                               '--', c='gold',
                               label='Annual', linewidth=2)

        if i == 1:
            x_hs, y, z = hilbert_spectrum(time, ecf2_imfs, ecf2_dthts, ecf2_dtifs / (2 * np.pi), max_frequency=max_freq / 10,
                                          plot=False, freq_increments=freq_incr, filter_sigma=5,
                                          which_imfs=[int(j + 4)])
            axs[i, j].plot(x_hs[0, :], np.median(ecf1_dtifs[int(j + 4), :] / (2 * np.pi)) * np.ones_like(x_hs[0, :]), 'k--',
                           label='Median')
            if j == 1:
                axs[i, j].plot(x_hs[0, :], (17.2 / 4411) * np.ones_like(x_hs[0, :]),
                            '--', c='gold',
                            label='Annual', linewidth=2)
        axs[i, j].pcolormesh(x_hs, y, np.abs(z), cmap='gist_rainbow', vmin=z_min, vmax=z_max)
        axs[i, j].set_title('ECF{} IMF {}'.format(int(i+1), int(j+4)))

# axs[0, 1].plot(time, imfs[0, :], label='Smoothed')
axs[1, 1].legend(loc='upper left')

axis = 0
for ax in axs.flat:
    if axis == 0:
        ax.set(ylabel=R'Frequency (day$^{-1}$)')
    if axis == 1:
        pass
    if axis == 2:
        ax.set(ylabel=R'Frequency (day$^{-1}$)')
        ax.set(xlabel='Time (days)')
    if axis == 3:
        ax.set(xlabel='Time (days)')
    axis += 1

# plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
# box_0 = ax.get_position()
# ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.10, box_0.width * 0.85, box_0.height * 0.9])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/if_ecf1_and_ecf2_45.png')
plt.show()
