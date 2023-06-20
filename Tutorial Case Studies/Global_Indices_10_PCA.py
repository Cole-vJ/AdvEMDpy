
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import datetime
import textwrap
from sklearn.decomposition import PCA

sns.set(style='darkgrid')
np.random.seed(0)

from emd_hilbert import Hilbert, hilbert_spectrum, omega, theta  # stft_custom, morlet_wavelet_custom
from emd_utils import Utility
from AdvEMDpy import EMD

from sklearn.decomposition import FastICA

DJI_data = pd.read_csv('10 Index Data/^DJI.csv', header=0, index_col='Date')
FCHI_data = pd.read_csv('10 Index Data/^FCHI.csv', header=0, index_col='Date')
GDAXI_data = pd.read_csv('10 Index Data/^GDAXI.csv', header=0, index_col='Date')
HSI_data = pd.read_csv('10 Index Data/^HSI.csv', header=0, index_col='Date')
IBX50_data = pd.read_csv('10 Index Data/^IBX50.csv', header=0, index_col='Date')
N225_data = pd.read_csv('10 Index Data/^N225.csv', header=0, index_col='Date')
NSEI_data = pd.read_csv('10 Index Data/^NSEI.csv', header=0, index_col='Date')
ASHT40_data = pd.read_csv('10 Index Data/ASHT40.JO.csv', header=0, index_col='Date')
ISFL_data = pd.read_csv('10 Index Data/ISF.L.csv', header=0, index_col='Date')
STWAX_data = pd.read_csv('10 Index Data/STW.AX.csv', header=0, index_col='Date')

DJI_open = DJI_data['Open']
FCHI_open = FCHI_data['Open']
GDAXI_open = GDAXI_data['Open']
HSI_open = HSI_data['Open']
IBX50_open = IBX50_data['Open']
N225_open = N225_data['Open']
NSEI_open = NSEI_data['Open']
ASHT40_open = ASHT40_data['Open']
ISFL_open = ISFL_data['Open']
STWAX_open = STWAX_data['Open']

all_open_data = pd.concat([DJI_open, FCHI_open, GDAXI_open, HSI_open, IBX50_open,
                           N225_open, NSEI_open, ASHT40_open, ISFL_open, STWAX_open],
                           join='outer', axis=1, sort=True)

pca = PCA(n_components=5)
dates = np.asarray(all_open_data.index)
dates_int = np.zeros(len(dates))
for i in range(len(dates)):
    dates_int[i] = datetime.datetime.strptime(dates[i], '%Y-%m-%d').toordinal()
dates_int -= dates_int[0]
dates_int = dates_int[:-1]
column_names = ['DJI', 'FCHI', 'GDAXI', 'HSI', 'IBX50', 'N225', 'NSEI', 'ASHT40', 'ISFL', 'STWAX']
all_open_data_array = np.asarray(all_open_data)
for stock in range(10):
    data = all_open_data_array[:, stock]
    data = data[:-1]
    nan_bool = np.isnan(data)
    if sum(nan_bool) > 0:
        data[nan_bool] = sp.interpolate.interp1d(dates_int[~nan_bool], data[~nan_bool])(dates_int[nan_bool])
    all_open_data_array[:-1, stock] = data
pca.fit(all_open_data_array[:-1, :].T)
pca_components = pca.components_
pca_singular_values = pca.singular_values_
pca_weights = pca_components * (pca_singular_values / pca_singular_values.sum()).reshape(-1, 1)

date_check_data = all_open_data.copy()
date_check_data = np.hstack((np.asarray(range(1308)).reshape(-1, 1), np.asarray(all_open_data.index).reshape(-1, 1)))
all_open_data.columns = ['DJI', 'FCHI', 'GDAXI', 'HSI', 'IBX50', 'N225', 'NSEI', 'ASHT40', 'ISFL', 'STWAX']
column_names = ['DJI', 'FCHI', 'GDAXI', 'HSI', 'IBX50', 'N225', 'NSEI', 'ASHT40', 'ISFL', 'STWAX']

dates = np.asarray(all_open_data.index)
dates_int = np.zeros(len(dates))
for i in range(len(dates)):
    dates_int[i] = datetime.datetime.strptime(dates[i], '%Y-%m-%d').toordinal()
dates_int -= dates_int[0]

dates_int = dates_int[:-1]

no_knots = 200

emd_storage = np.zeros((3, 10)).astype(np.object)
for stock in range(10):
    data = np.asarray(all_open_data[column_names[stock]])
    data = data[:-1]
    nan_bool = np.isnan(data)
    if sum(nan_bool) > 0:
        data[nan_bool] = sp.interpolate.interp1d(dates_int[~nan_bool], data[~nan_bool])(dates_int[nan_bool])
    emd = EMD(time=dates_int, time_series=data)
    emd = emd.empirical_mode_decomposition(knot_envelope=np.linspace(0, dates_int[-1], no_knots), smooth=True,
                                           smoothing_penalty=1, edge_effect='symmetric_anchor', sym_alpha=0.1,
                                           stop_crit='S_stoppage', stop_crit_threshold=20, mean_threshold=0.1,
                                           debug=False, verbose=True, spline_method='b_spline', dtht=False,
                                           dtht_method='kak', max_internal_iter=30, max_imfs=5, matrix=True,
                                           initial_smoothing=True, dft='envelopes')[:3]
    emd_storage[:, stock] = emd

# DJI FCHI GDAXI HSI IBX50 N225 NSEI ASHT40 ISFL STWAX

np.save('global_indices_emd', emd_storage)

emd_storage = np.load('global_indices_emd.npy', allow_pickle=True)

imf_1_corr = np.zeros((10, len(dates_int)))
imf_2_corr = np.zeros((10, len(dates_int)))
imf_3_corr = np.zeros((10, len(dates_int)))
imf_4_corr = np.zeros((10, len(dates_int)))
imf_5_corr = np.zeros((10, len(dates_int)))
for stock in range(10):
    print(np.shape(emd_storage[0, stock]))
    imf_1_corr[stock, :] = emd_storage[0, stock][1, :]
    imf_2_corr[stock, :] = emd_storage[0, stock][2, :]
    imf_3_corr[stock, :] = emd_storage[0, stock][3, :]
    imf_4_corr[stock, :] = emd_storage[0, stock][4, :]
    imf_5_corr[stock, :] = emd_storage[0, stock][5, :]
plt.show()

# IMF1
ica = FastICA(n_components=1)
variables_reconstruct_1 = ica.fit_transform(imf_1_corr.T)
A_estimate = ica.mixing_
# plt.plot(dates_int, variables_reconstruct_1)
# plt.show()

# IMF2
ica = FastICA(n_components=1)
variables_reconstruct_2 = ica.fit_transform(imf_2_corr.T)
A_estimate = ica.mixing_
# plt.plot(dates_int, variables_reconstruct_2)
# plt.show()

# IMF3
ica = FastICA(n_components=1)
variables_reconstruct_3 = ica.fit_transform(imf_3_corr.T)
A_estimate = ica.mixing_
# plt.plot(dates_int, variables_reconstruct_3)
# plt.show()

# IMF4
ica = FastICA(n_components=1)
variables_reconstruct_4 = ica.fit_transform(imf_4_corr.T)
A_estimate = ica.mixing_
# plt.plot(dates_int, variables_reconstruct_4)
# plt.show()

# IMF5
ica = FastICA(n_components=1)
variables_reconstruct_5 = ica.fit_transform(imf_5_corr.T)
A_estimate = ica.mixing_
# plt.plot(dates_int, variables_reconstruct_5)
# plt.show()

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = plt.subplot(111)
ax.set_title('Plot of Normalised PCA Component 1 and EMD-ICA Component 5', fontsize=16, pad=15.0)
plt.plot((pca_weights[0, :] - pca_weights[0, 0]) / (np.max(pca_weights[0, :] - pca_weights[0, 0]) -
                                                    np.min(pca_weights[0, :] - pca_weights[0, 0])),
         label=r'PCA 1$^{st}$')
plt.plot(np.abs((variables_reconstruct_5 - variables_reconstruct_5[0]) / (np.max(variables_reconstruct_5 -
                                                                          variables_reconstruct_5[0]) -
                                                                   np.min(variables_reconstruct_5 -
                                                                          variables_reconstruct_5[0]))),
         label=r'EMD-ICA 5$^{th}$')
plt.plot((all_open_data_array[:-1, 0] - all_open_data_array[0, 0]) /
         (np.max(all_open_data_array[:-1, 0] - all_open_data_array[0, 0]) -
          np.min(all_open_data_array[:-1, 0] - all_open_data_array[0, 0])), label=r'DJI')
plt.plot((imf_5_corr[0, :-1] - imf_5_corr[0, 0]) /
         (np.max(imf_5_corr[0, :-1] - imf_5_corr[0, 0]) -
          np.min(imf_5_corr[0, :-1] - imf_5_corr[0, 0])), label=r'DJI IMF 5')
ax.set_xlabel('Trading days', fontsize=16, labelpad=5.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
ax.set_xticks([0, 262, 523, 784, 1045, 1306])
ax.set_xticklabels(['22-02-2016', '22-02-2017', '22-02-2018', '22-02-2019', '21-02-2020', '19-02-2021'], fontsize=12)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0 + 0.0, box_0.width * 1.10, box_0.height * 1.05])
ax.legend(loc='upper left')
plt.savefig('Figures/pca_1_emd_ica_5.pdf')
plt.show()

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = plt.subplot(111)
ax.set_title('Plot of Normalised PCA Component 2 and EMD-ICA Component 4', fontsize=16, pad=15.0)
minus_pca_2 = -(pca_weights[1, :] - pca_weights[1, 0]) / (np.max(pca_weights[1, :] - pca_weights[1, 0]) -
                                                     np.min(pca_weights[1, :] - pca_weights[1, 0]))
plt.plot(minus_pca_2 + (1 - np.max(minus_pca_2)), label=r'- PCA 2$^{nd}$')
minus_emd_ica_4 = -(variables_reconstruct_4 - variables_reconstruct_4[0]) / (np.max(variables_reconstruct_4 -
                                                                           variables_reconstruct_4[0]) -
                                                                    np.min(variables_reconstruct_4 -
                                                                           variables_reconstruct_4[0]))
plt.plot(minus_emd_ica_4 + (1 - np.max(minus_emd_ica_4)), label=r'- EMD-ICA 4$^{th}$')
n225 = (all_open_data_array[:-1, 5] - all_open_data_array[0, 5]) / (np.max(all_open_data_array[:-1, 5] - all_open_data_array[0, 5]) -
                                                        np.min(all_open_data_array[:-1, 5] - all_open_data_array[0, 5]))
plt.plot(n225 + (1 - np.max(n225)), label=r'N225')
n225_emd_4 = (imf_4_corr[5, :-1] - imf_4_corr[5, 0]) / \
             (np.max(imf_4_corr[5, :-1] - imf_4_corr[5, 0]) -
              np.min(imf_4_corr[5, :-1] - imf_4_corr[5, 0]))
plt.plot(n225_emd_4 + (1 - np.max(n225_emd_4)), label=r'N225 IMF 4')
ax.set_xlabel('Trading days', fontsize=16, labelpad=5.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
ax.set_xticks([0, 262, 523, 784, 1045, 1306])
ax.set_xticklabels(['22-02-2016', '22-02-2017', '22-02-2018', '22-02-2019', '21-02-2020', '19-02-2021'], fontsize=12)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0 + 0.0, box_0.width * 1.10, box_0.height * 1.05])
ax.legend(loc='upper left')
plt.savefig('Figures/pca_2_emd_ica_4.pdf')
plt.show()

# DJI, FCHI, GDAXI, HSI, IBX50, N225, NSEI, ASHT40, ISFL, STWAX

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = plt.subplot(111)
ax.set_title('Plot of Normalised PCA Component 3 and EMD-ICA Component 3', fontsize=16, pad=15.0)
pca_3 = -(pca_weights[2, :] - pca_weights[2, 0]) / (np.max(pca_weights[2, :] - pca_weights[2, 0]) -
                                                   np.min(pca_weights[2, :] - pca_weights[2, 0]))
plt.plot(pca_3 + (1 - np.max(pca_3)), label=r'- PCA 3$^{rd}$')
emd_ica_3 = -(variables_reconstruct_3 - variables_reconstruct_3[0]) / (np.max(variables_reconstruct_3 -
                                                                             variables_reconstruct_3[0]) -
                                                                     np.min(variables_reconstruct_3 -
                                                                            variables_reconstruct_3[0]))
plt.plot(emd_ica_3 + (1 - np.max(emd_ica_3)), label=r'- EMD-ICA 3$^{rd}$')
plt.plot((all_open_data_array[:-1, 1] - all_open_data_array[0, 1]) /
         (np.max(all_open_data_array[:-1, 1] - all_open_data_array[0, 1]) -
          np.min(all_open_data_array[:-1, 1] - all_open_data_array[0, 1])), label=r'FCHI')
plt.plot((all_open_data_array[:-1, 2] - all_open_data_array[0, 2]) /
         (np.max(all_open_data_array[:-1, 2] - all_open_data_array[0, 2]) -
          np.min(all_open_data_array[:-1, 2] - all_open_data_array[0, 2])), label=r'GDAXI')
plt.plot((all_open_data_array[:-1, 8] - all_open_data_array[0, 8]) /
         (np.max(all_open_data_array[:-1, 8] - all_open_data_array[0, 8]) -
          np.min(all_open_data_array[:-1, 8] - all_open_data_array[0, 8])), label=r'ISFL')
ax.set_xlabel('Trading days', fontsize=16, labelpad=5.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
ax.set_xticks([0, 262, 523, 784, 1045, 1306])
ax.set_xticklabels(['22-02-2016', '22-02-2017', '22-02-2018', '22-02-2019', '21-02-2020', '19-02-2021'], fontsize=12)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0 + 0.0, box_0.width * 1.10, box_0.height * 1.05])
ax.legend(loc='upper left')
plt.savefig('Figures/pca_3_emd_ica_3.pdf')
plt.show()

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = plt.subplot(111)
ax.set_title('Plot of Normalised PCA Component 4 and EMD-ICA Component 2', fontsize=16, pad=15.0)
pca_4 = (pca_weights[3, :] - pca_weights[3, 0]) / (np.max(pca_weights[3, :] - pca_weights[3, 0]) -
                                                   np.min(pca_weights[3, :] - pca_weights[3, 0]))
plt.plot(pca_4 + (1 - np.max(pca_4)), label=r'PCA 4$^{th}$')
emd_ica_2 = -(variables_reconstruct_2 - variables_reconstruct_2[0]) / (np.max(variables_reconstruct_2 -
                                                                             variables_reconstruct_2[0]) -
                                                                     np.min(variables_reconstruct_2 -
                                                                            variables_reconstruct_2[0]))
plt.plot(emd_ica_2 + (1 - np.max(emd_ica_2)), label=r'- EMD-ICA 2$^{nd}$')
ax.set_xlabel('Trading days', fontsize=16, labelpad=5.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
ax.set_xticks([0, 262, 523, 784, 1045, 1306])
ax.set_xticklabels(['22-02-2016', '22-02-2017', '22-02-2018', '22-02-2019', '21-02-2020', '19-02-2021'], fontsize=12)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0 + 0.0, box_0.width * 1.10, box_0.height * 1.05])
ax.legend(loc='upper left')
plt.savefig('Figures/pca_4_emd_ica_2.pdf')
plt.show()

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = plt.subplot(111)
ax.set_title('Plot of Normalised PCA Component 5 and EMD-ICA Component 1', fontsize=16, pad=15.0)
pca_5 = (pca_weights[4, :] - pca_weights[4, 0]) / (np.max(pca_weights[4, :] - pca_weights[4, 0]) -
                                                   np.min(pca_weights[4, :] - pca_weights[4, 0]))
plt.plot(pca_5 + (1 - np.max(pca_5)), label=r'PCA 5$^{th}$')
emd_ica_1 = (variables_reconstruct_1 - variables_reconstruct_1[0]) / (np.max(variables_reconstruct_1 -
                                                                             variables_reconstruct_1[0]) -
                                                                     np.min(variables_reconstruct_1 -
                                                                            variables_reconstruct_1[0]))
plt.plot(emd_ica_1 + (1 - np.max(emd_ica_1)), label=r'EMD-ICA 1$^{st}$')
ax.set_xlabel('Trading days', fontsize=16, labelpad=5.0)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12)
ax.set_xticks([0, 262, 523, 784, 1045, 1306])
ax.set_xticklabels(['22-02-2016', '22-02-2017', '22-02-2018', '22-02-2019', '21-02-2020', '19-02-2021'], fontsize=12)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0 + 0.0, box_0.width * 1.10, box_0.height * 1.05])
ax.legend(loc='upper left')
plt.savefig('Figures/pca_5_emd_ica_1.pdf')
plt.show()

# 5 x 2 plot
variables_reconstruct = np.vstack((variables_reconstruct_1.T, variables_reconstruct_2.T, variables_reconstruct_3.T,
                                   variables_reconstruct_4.T, variables_reconstruct_5.T)).T
variables_reconstruct_ht = np.zeros_like(variables_reconstruct)
variables_reconstruct_if = np.zeros_like(variables_reconstruct)[:-1, :]
variables_reconstruct_orig = variables_reconstruct.copy()

# estimate trading strategy

fig, axs = plt.subplots(5, 2)
fig.suptitle(textwrap.fill(r'Market Index Forecast', 50))

return_hold = np.ones(10)
return_optimal = np.ones(10)
return_trading = np.ones(10)

for index in range(10):

    dji_smoothed = np.asarray(emd_storage[0, index][0, :])
    dji_original = np.asarray(all_open_data[:-1, [column_names[index] == column_names[i] for i in range(10)]])

    imfs = 5

    dji_smoothed_trun = dji_smoothed[:-200]
    trend = variables_reconstruct[:, -1]
    if sum(trend - trend[0]) < 1:
        trend += variables_reconstruct[:, -2]
    variables_reconstruct = variables_reconstruct[:, 1:int(imfs + 1)]
    variables_reconstruct = np.hstack((np.asarray(np.ones(np.shape(variables_reconstruct)[0])).reshape(-1, 1),
                                       variables_reconstruct))
    variables_reconstruct_trun = variables_reconstruct[:-200, 1:int(imfs + 1)]
    variables_reconstruct_trun = np.hstack((np.asarray(np.ones(np.shape(variables_reconstruct_trun)[0])).reshape(-1, 1),
                                            variables_reconstruct_trun))

    gamma = np.linalg.lstsq(variables_reconstruct_trun, np.log(dji_smoothed_trun / dji_smoothed_trun[0]), rcond=None)
    gamma = gamma[0]

    trend = np.asarray(emd_storage[0, index][-1, :]) + np.asarray(emd_storage[0, index][-2, :])
    trend = trend * 0

    axs[index % 5, index // 5].plot(dates_int, np.log(dji_smoothed / dji_smoothed[0]))
    utils = Utility(time_series=np.log(dji_smoothed / dji_smoothed[0]), time=np.log(dji_smoothed / dji_smoothed[0]))
    axs[index % 5, index // 5].scatter(dates_int[utils.min_bool_func_1st_order_fd()],
                                       np.log(dji_smoothed / dji_smoothed[0])[utils.min_bool_func_1st_order_fd()])
    axs[index % 5, index // 5].scatter(dates_int[utils.max_bool_func_1st_order_fd()],
                                       np.log(dji_smoothed / dji_smoothed[0])[utils.max_bool_func_1st_order_fd()])
    axs[index % 5, index // 5].plot(dates_int, trend + np.matmul(gamma, variables_reconstruct.transpose()))
    utils_trend = Utility(time_series=trend + np.matmul(gamma, variables_reconstruct.transpose()), time=trend + np.matmul(gamma, variables_reconstruct.transpose()))
    axs[index % 5, index // 5].scatter(dates_int[utils_trend.min_bool_func_1st_order_fd()],
                                       (trend + np.matmul(gamma, variables_reconstruct.transpose()))[utils_trend.min_bool_func_1st_order_fd()])
    axs[index % 5, index // 5].scatter(dates_int[utils_trend.max_bool_func_1st_order_fd()],
                                       (trend + np.matmul(gamma, variables_reconstruct.transpose()))[utils_trend.max_bool_func_1st_order_fd()])
    axs[index % 5, index // 5].plot(dates_int[-200] * np.ones(100), np.linspace(0, 1, 100), 'r--')

    min_max_alt = 0
    for i in np.sort(np.append(dates_int[utils.min_bool_func_1st_order_fd()],
                               dates_int[utils.max_bool_func_1st_order_fd()])):

        if i > 1547:

            first_min = dates_int[utils.min_bool_func_1st_order_fd()][dates_int[utils.min_bool_func_1st_order_fd()] > 1547][0]
            first_max = dates_int[utils.max_bool_func_1st_order_fd()][dates_int[utils.max_bool_func_1st_order_fd()] > 1547][0]

            if first_min < first_max and i != previous_time:
                if min_max_alt % 2 == 0:
                    pass
                else:
                    return_optimal[index] = return_optimal[index] * np.exp((
                                            ((np.log(dji_smoothed / dji_smoothed[0]))[dates_int == i] -
                                             (np.log(dji_smoothed / dji_smoothed[0]))[dates_int == previous_time])))
            elif first_max < first_min:
                if min_max_alt == 0:
                    return_optimal[index] = return_optimal[index] * np.exp(
                                            (np.log(dji_smoothed / dji_smoothed[0])[dates_int == i] -
                                             np.log(dji_smoothed / dji_smoothed[0])[-200]))
                elif min_max_alt % 2 == 0:
                    return_optimal[index] = return_optimal[index] * np.exp((
                                            ((np.log(dji_smoothed / dji_smoothed[0]))[dates_int == i] -
                                             (np.log(dji_smoothed / dji_smoothed[0]))[dates_int == previous_time])))
                else:
                    pass

            min_max_alt += 1
            previous_time = i.copy()

    min_max_alt = 0
    for i in np.sort(np.append(dates_int[utils_trend.min_bool_func_1st_order_fd()],
                               dates_int[utils_trend.max_bool_func_1st_order_fd()])):

        if i > 1547:

            first_min = dates_int[utils_trend.min_bool_func_1st_order_fd()][
                dates_int[utils_trend.min_bool_func_1st_order_fd()] > 1547][0]
            first_max = dates_int[utils_trend.max_bool_func_1st_order_fd()][
                dates_int[utils_trend.max_bool_func_1st_order_fd()] > 1547][0]

            if first_min < first_max and i != previous_time:
                if min_max_alt % 2 == 0:
                    pass
                else:
                    return_trading[index] = return_trading[index] * np.exp(
                                            ((np.log(dji_smoothed / dji_smoothed[0]))[dates_int == i] - (np.log(dji_smoothed / dji_smoothed[0]))[dates_int == previous_time]))
            elif first_max < first_min:
                if min_max_alt == 0:
                    return_trading[index] = return_trading[index] * np.exp(
                                            (np.log(dji_smoothed / dji_smoothed[0])[dates_int == i] - np.log(dji_smoothed / dji_smoothed[0])[-200]))
                elif min_max_alt % 2 == 0:
                    return_trading[index] = return_trading[index] * np.exp(
                                            ((np.log(dji_smoothed / dji_smoothed[0]))[dates_int == i] -
                                             (np.log(dji_smoothed / dji_smoothed[0]))[dates_int == previous_time]))
                else:
                    pass

            min_max_alt += 1
            previous_time = i.copy()

    return_hold[index] = return_hold[index] * np.exp((np.log(dji_smoothed / dji_smoothed[0]))[-1] -
                                               (np.log(dji_smoothed / dji_smoothed[0]))[-200])

plt.show()

print(sum(return_hold))
print(sum(return_trading))
print(sum(return_optimal))

plt.plot(utils.derivative_forward_diff(np.log(dji_smoothed / dji_smoothed[0]), dates_int))
plt.plot(utils_trend.derivative_forward_diff(np.matmul(gamma, variables_reconstruct.transpose()), dates_int))
plt.show()

# estimate trading strategy

# estimate portfolio weights



# estimate portfolio weights

variables_reconstruct = variables_reconstruct_orig

for i in range(np.shape(variables_reconstruct)[1]):
    hilbert = Hilbert(time=np.arange(variables_reconstruct[:, i]), time_series=variables_reconstruct[:, i])
    variables_reconstruct_ht[:, i] = hilbert.dtht_kak()
    variables_reconstruct_if[:, i] = omega(dates_int, theta(variables_reconstruct[:, i], variables_reconstruct_ht[:, i]))

fig, axs = plt.subplots(5, 2)
fig.suptitle(textwrap.fill(r'Market Factors and Instantaneous Frequencies', 50))
fig.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.15)
for i in range(0, 5):
    if int(i + 1) != 5:
        axs[i, 0].plot(dates_int, variables_reconstruct[:, i] - np.mean(variables_reconstruct[:, i]), Linewidth=1)
    else:
        axs[i, 0].plot(dates_int, variables_reconstruct[:, i], Linewidth=1)
    if int(i + 1) == 1 or int(i + 1) == 2:
        axs[i, 0].set_yticks([-0.1, 0.0, 0.1])
        axs[i, 0].set_yticklabels(['-0.10', '0.00', '0.10'])
    elif int(i + 1) == 3 or int(i + 1) == 4 or int(i + 1) == 5:
        axs[i, 0].set_yticks([-0.05, 0.0, 0.05])
        axs[i, 0].set_yticklabels(['-0.05', '0.00', '0.05'])
    if int(i + 1) == 1:
        axs[i, 1].set_yticks([0.0, 0.5])
        axs[i, 1].set_yticklabels(['0.00', '0.50'])
    if int(i + 1) == 2:
        axs[i, 1].set_yticks([0.0, 0.25])
        axs[i, 1].set_yticklabels(['0.00', '0.25'])
    if int(i + 1) == 3:
        axs[i, 1].set_yticks([0.0, 0.1])
        axs[i, 1].set_yticklabels(['0.00', '0.10'])
    if int(i + 1) == 4:
        axs[i, 1].set_yticks([0.0, 0.05])
        axs[i, 1].set_yticklabels(['0.00', '0.05'])
    if int(i + 1) == 5:
        axs[i, 1].set_yticks([0.0, 0.01])
        axs[i, 1].set_yticklabels(['0.00', '0.01'])

axis = 0
for i in range(0, 5):
    axs[i, 0].set_ylabel('Factor {}'.format(axis + 1), fontsize=10)
    plt.setp(axs[i, 0].get_xticklabels(), visible=True, fontsize=8, rotation=20)
    plt.setp(axs[i, 0].get_yticklabels(), visible=True, fontsize=8)
    if axis != 4:
        plt.setp(axs[i, 0].get_xticklabels(), visible=False)
    if axis == 4:
        axs[i, 0].set_xlabel('Trading days', fontsize=10)
        axs[i, 0].set_xticks([0, 366, 731, 1096, 1460, 1824])
        axs[i, 0].set_xticklabels(['22-02-2016', '22-02-2017', '22-02-2018', '22-02-2019', '21-02-2020', '19-02-2021'])
    axis += 1

# max_frequency = 0.5
for i in range(0, 5):
    if i == 0:
        max_frequency = 0.75
    elif i == 1:
        max_frequency = 0.3
    elif i == 2:
        max_frequency = 0.1
    elif i == 3:
        max_frequency = 0.05
    elif i == 4:
        max_frequency = 0.01
    hs = hilbert_spectrum(dates_int, variables_reconstruct[:, i].reshape(1, -1), variables_reconstruct_ht[:, i].reshape(1, -1),
                          variables_reconstruct_if[:, i].reshape(1, -1), max_frequency, plot=False, which_imfs=[0])
    axs[i, 1].pcolormesh(hs[0], hs[1], hs[2], cmap='gist_rainbow', vmin=0,
                         vmax=hs[2].max())

axis = 0
for i in range(0, 5):
    axs[i, 1].set_ylabel('IF {}'.format(axis + 1), fontsize=10)
    plt.setp(axs[i, 1].get_xticklabels(), visible=True, fontsize=8, rotation=20)
    plt.setp(axs[i, 1].get_yticklabels(), visible=True, fontsize=8)
    if axis != 4:
        plt.setp(axs[i, 1].get_xticklabels(), visible=False)
    if axis == 4:
        axs[i, 1].set_xlabel('Trading days', fontsize=10)
        axs[i, 1].set_xticks([0, 366, 731, 1096, 1460, 1824])
        axs[i, 1].set_xticklabels(['22-02-2016', '22-02-2017', '22-02-2018', '22-02-2019', '21-02-2020', '19-02-2021'])
    axis += 1
    box_0 = axs[i, 0].get_position()
    axs[i, 0].set_position([box_0.x0 - 0.02, box_0.y0, box_0.width * 1.10, box_0.height])
    box_0 = axs[i, 1].get_position()
    axs[i, 1].set_position([box_0.x0 + 0.01, box_0.y0, box_0.width * 1.10, box_0.height])

plt.savefig('Figures/global_indices_factors.png')

plt.show()

### New Plot

variables_reconstruct = variables_reconstruct_orig

for i in range(np.shape(variables_reconstruct)[1]):
    hilbert = Hilbert(time=np.arange(variables_reconstruct[:, i]), time_series=variables_reconstruct[:, i])
    variables_reconstruct_ht[:, i] = hilbert.dtht_kak()
    variables_reconstruct_if[:, i] = omega(dates_int, theta(variables_reconstruct[:, i], variables_reconstruct_ht[:, i]))

window = 30

fig, axs = plt.subplots(5, 1)
fig.set_size_inches(8, 6)
fig.suptitle(textwrap.fill(r'Market Factors and {}-Day Moving Window Correlations with Corresponding IMFs'.format(window), 45))
fig.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.15)

# max_frequency = 0.5
for i in range(0, 5):
    for j in np.arange(0, 10):
        corr_vector = np.zeros_like(variables_reconstruct[:-window, i])
        for k in np.arange(len(corr_vector)):
            # exec('corr_vector[k] = np.corrcoef(np.diff(variables_reconstruct[k:int(window + k), i]), np.diff(imf_{}_corr[i, k:int(window + k)]))[0, 1]'.format(int(i + 1)))
            exec('corr_vector[k] = np.corrcoef(np.diff(variables_reconstruct[k:int(window + k), i]), np.diff(imf_{}_corr[j, k:int(window + k)]))[0, 1]'.format(int(i + 1)))
        axs[i].plot(dates_int[window:], corr_vector, label=column_names[j])


axis = 0
for i in range(0, 5):
    axs[i].set_ylabel(r'$\rho$'.format(axis + 1), fontsize=10)
    axs[i].set_title('', fontsize=12)
    plt.setp(axs[i].get_xticklabels(), visible=True, fontsize=8, rotation=20)
    plt.setp(axs[i].get_yticklabels(), visible=True, fontsize=8)
    if axis != 4:
        plt.setp(axs[i].get_xticklabels(), visible=False)
    if axis == 4:
        axs[i].set_xlabel('Trading days', fontsize=14)
        axs[i].set_xticks([0, 366, 731, 1096, 1460, 1824])
        axs[i].set_xticklabels(['22-02-2016', '22-02-2017', '22-02-2018', '22-02-2019', '21-02-2020', '19-02-2021'])
    else:
        axs[i].set_xticks([0, 366, 731, 1096, 1460, 1824])
        axs[i].set_xticklabels(['', '', '', '', '', ''])
    axis += 1
    box_0 = axs[i].get_position()
    axs[i].set_position([box_0.x0 - 0.06, box_0.y0 - 0.02, box_0.width * 0.99, box_0.height * 1.05])
axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig('Figures/global_indices_factors_correlation_plot.pdf')

plt.show()

### New Plot

imf_1_corr = np.corrcoef(imf_1_corr)
imf_2_corr = np.corrcoef(imf_2_corr)
imf_3_corr = np.corrcoef(imf_3_corr)
imf_4_corr = np.corrcoef(imf_4_corr)
imf_5_corr = np.corrcoef(imf_5_corr)

fig, axs = plt.subplots(2, 3)
fig.suptitle(textwrap.fill(r'Standardised Indices and Correlation Between IMFs', 50))
# plt.subplots_adjust(hspace=0.5)
for i in range(10):
    axs[0, 0].plot(dates_int, (emd_storage[0, i][0, :] - min(emd_storage[0, i][0, :])) /
                   (max(emd_storage[0, i][0, :]) - min(emd_storage[0, i][0, :])))
axs[0, 1].pcolormesh(np.arange(10), np.arange(10), imf_1_corr, vmin=-1, vmax=1)
colour_bar_1 = axs[0, 2].pcolormesh(np.arange(10), np.arange(10), imf_2_corr, vmin=-1, vmax=1)
fig.colorbar(colour_bar_1, ax=axs[0, 2])
axs[1, 0].pcolormesh(np.arange(10), np.arange(10), imf_3_corr, vmin=-1, vmax=1)
axs[1, 1].pcolormesh(np.arange(10), np.arange(10), imf_4_corr, vmin=-1, vmax=1)
colour_bar_2 = axs[1, 2].pcolormesh(np.arange(10), np.arange(10), imf_5_corr, vmin=-1, vmax=1)
fig.colorbar(colour_bar_2, ax=axs[1, 2])

x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x_names = ['DJI', 'FCHI', 'GDAXI', 'HSI', 'IBX50', 'N225', 'NSEI', 'ASHT40', 'ISFL', 'STWAX']
x_ticks_blank = [0, 1]
x_names_blank = ['', '']

axis = 0
for ax in axs.flat:
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45, ha='left')
    if axis == 0:
        ax.set_xticks(x_ticks_blank)
        ax.set_xticklabels(x_names_blank)
        ax.set_yticks(x_ticks_blank)
        ax.set_yticklabels(x_names_blank)
    if axis // 3 > 0:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_names, rotation=-45, size=8)
    else:
        ax.set_xticks(x_ticks_blank)
        ax.set_xticklabels(x_names_blank)
    if axis % 3 == 0 and axis // 3 > 0:
        ax.set_yticks(x_ticks)
        ax.set_yticklabels(x_names, size=8)
    else:
        ax.set_yticks(x_ticks_blank)
        ax.set_yticklabels(x_names_blank)
    axis += 1

axs[0, 0].set_title('Standardised Indices')
axs[0, 1].set_title('Correlation IMF 1')
axs[0, 2].set_title('Correlation IMF 2')
axs[1, 0].set_title('Correlation IMF 3')
axs[1, 1].set_title('Correlation IMF 4')
axs[1, 2].set_title('Correlation IMF 5')

plt.gcf().subplots_adjust(top=0.85, bottom=0.15)
# plt.savefig('Figures/global_indices.png')
plt.show()

max_frequency = 0.5

# DJI FCHI GDAXI HSI IBX50 N225 NSEI ASHT40 ISFL STWAX

hs_storage = np.zeros((3, 10)).astype(np.object)

for stock in range(10):
    hs_storage[:, stock] = hilbert_spectrum(dates_int, emd_storage[0, stock], emd_storage[1, stock],
                                            emd_storage[2, stock], max_frequency)

temp = 0
