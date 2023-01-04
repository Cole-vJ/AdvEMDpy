
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
from scipy.interpolate import interp1d

from AdvEMDpy import EMD
from emd_hilbert import hilbert_spectrum, Hilbert, omega, theta

# https://www.mortality.org/cgi-bin/hmd/country.php?cntr=GBR&level=2

# https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/livebirths/bulletins/birthsummarytablesenglandandwales/2015-07-15

sns.set(style='darkgrid')

# load raw data
raw_data = pd.read_csv('Human Mortality Database Data/monthly_birth_rate', header=0)
uk_monthly_births = np.array(raw_data['Births'])
monthly_births_refined = []

for i in range(len(uk_monthly_births)):
    if int(i + 1) % 13 == 0:
        pass
    else:
        monthly_births_refined.append(uk_monthly_births[i])
monthly_births_refined = np.asarray(monthly_births_refined)

fig = plt.figure(figsize=(9, 4))
ax = plt.subplot(111)
plt.plot(np.arange(len(monthly_births_refined)), monthly_births_refined)
plt.plot((7 + 7 * 12) * np.ones(100), np.linspace(50000, 90000, 100), 'k--', label=textwrap.fill('End of World War 2', 13))
plt.plot((11 + 23 * 12) * np.ones(100), np.linspace(60000, 100000, 100), 'b--',
         label=textwrap.fill('NHS provides pill to married women', 16))
plt.plot((11 + 29 * 12) * np.ones(100), np.linspace(60000, 100000, 100), 'g--',
         label=textwrap.fill('NHS provides pill to unmarried women & Abortion Act', 14))
plt.plot((11 + 52 * 12) * np.ones(100), np.linspace(40000, 80000, 100), '--', c='magenta',
         label=textwrap.fill('Women delay childbearing to older ages', 18))
plt.plot((11 + 62 * 12) * np.ones(100), np.linspace(40000, 80000, 100), '--', c='magenta')
plt.plot((11 + 73 * 12) * np.ones(100), np.linspace(40000, 80000, 100), '--', c='orange',
         label=textwrap.fill('Stricter wellfare reforms', 18))
plt.title('Births in the United Kingdom of Great Britain and Northern Ireland')
plt.ylabel('Births')
plt.xlabel('Years')
plt.xticks(np.arange(11, 972, int(5 * 12)), np.arange(1938, 2019, 5), rotation=-45, fontsize=8)
plt.yticks(np.arange(40000, 100001, 10000), fontsize=8)

box_0 = ax.get_position()
ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.85, box_0.height * 0.95])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/births.png')
plt.show()

plt.show()

time = np.arange(len(monthly_births_refined))

emd = EMD(time=time, time_series=monthly_births_refined)
imfs, _, ifs = emd.empirical_mode_decomposition(knots=np.arange(-1, 972, 3), verbose=True, debug=False,
                                                max_internal_iter=10)[:3]

hs_ouputs = hilbert_spectrum(time, imfs[:3], _[:3], 2 * np.pi * ifs[:3], max_frequency=20, plot=False)

ax = plt.subplot(111)
figure_size = plt.gcf().get_size_inches()
factor = 0.9
plt.gcf().set_size_inches((figure_size[0], factor * figure_size[1]))
ax.set_title(textwrap.fill(r'Instantaneous Frequency of First IMF from UK Birth Rates', 44))
x_hs, y, z = hs_ouputs
z_min, z_max = 0, np.abs(z).max()
ax.pcolormesh(x_hs, y, np.abs(z), cmap='gist_rainbow', vmin=z_min, vmax=z_max)
plt.plot(time[:-1], np.ones_like(time[:-1]), 'k--', label=textwrap.fill('Annual cycle', 10), linewidth=3)
plt.plot(time[:-1], (1 / 0.75) * np.ones_like(time[:-1]), '--', c='gold', label=textwrap.fill('Every nine months', 10),
         linewidth=3)
plt.plot((11 + 23 * 12) * np.ones(100), np.linspace(0, 2, 100), 'b--',
         label=textwrap.fill('NHS provides pill to married women', 10), linewidth=3)
plt.plot((11 + 29 * 12) * np.ones(100), np.linspace(0, 2, 100), 'g--',
         label=textwrap.fill('NHS provides pill to unmarried women', 10), linewidth=3)
plt.xticks(np.arange(11, 972, int(5 * 12)), np.arange(1938, 2019, 5), rotation=-45, fontsize=8)
plt.ylim(0, 2)
plt.ylabel('Frequency (year$^{-1}$)')
plt.xlabel('Time (years)')

box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.025, box_0.y0 + 0.05, box_0.width * 0.85, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/births_if_emd.png')
plt.show()


def henderson_kernel(order=13, start=np.nan, end=np.nan):  # has easily calculable asymmetric weighting

    if np.isnan(start) == True:
        start = -int((order - 1) / 2)
    if np.isnan(end) == True:
        end = int((order - 1) / 2)
    t = np.asarray(range(start, end + 1)) / (int((order - 1) / 2) + 1)
    # exact Henderson Kernel - differs slightly from classical Henderson smoother
    y = (15 / 79376) * (5184 - 12289 * t ** 2 + 9506 * t ** 4 - 2401 * t ** 6) * ((2175 / 1274) - (1372 / 265) * t ** 2)

    y = y / sum(y)  # renormalise when incomplete - does nothing when complete as weights sum to zero

    return y


def henderson_weights(order=13, start=np.nan, end=np.nan):  # does not have easily calculable asymmetric weighting

    if np.isnan(start) == True:
        start = -int((order - 1) / 2)
    if np.isnan(end) == True:
        end = int((order - 1) / 2)
    # classical Henderson smooth - differs slightly from exact Henderson Kernel
    p = int((order - 1) / 2)
    n = p + 2
    vector = np.asarray(range(start, (end + 1)))
    y = (315 * ((n - 1) ** 2 - vector ** 2) * (n ** 2 - vector ** 2) * ((n + 1) ** 2 - vector ** 2) *
         (3 * n ** 2 - 16 - 11 * vector ** 2)) / \
        (8 * n * (n ** 2 - 1) * (4 * n ** 2 - 1) * (4 * n ** 2 - 9) * (4 * n ** 2 - 25))

    # renormalising does not result in classical asymmetric Henderson weighting
    y = y / sum(y)  # renormalise when incomplete - does nothing when complete as weights sum to zero

    return y


def henderson_ma(time_series, order=13, method='kernel'):

    henderson_filtered_time_series = np.zeros_like(time_series)

    if method == 'renormalise':
        weights = henderson_weights(order=order)
    elif method == 'kernel':
        weights = henderson_kernel(order=order)

    # need asymmetric weights that sum to approximately one on the edges - multiple options:
    # (1) use asymmetric filter (truncate and renormalise)
    # (2) extrapolate and use symmetric filter (X11ARIMA)
    # (3) Reproducing Kernel Hilbert Space Method
    # (4) Classical asymmetric results (unknown calculation)

    for k in range(len(time_series)):

        if k < ((order - 1) / 2):
            if method == 'renormalise':
                asymmetric_weights = henderson_weights(order=order, start=(0 - k))
            elif method == 'kernel':
                asymmetric_weights = henderson_kernel(order=order, start=(0 - k))
            henderson_filtered_time_series[k] = \
                np.sum(asymmetric_weights * time_series[:int(k + ((order - 1) / 2) + 1)])
        elif k > len(time_series) - ((order - 1) / 2) - 1:
            if method == 'renormalise':
                asymmetric_weights = henderson_weights(order=order, end=(len(time_series) - k - 1))
            elif method == 'kernel':
                asymmetric_weights = henderson_kernel(order=order, end=(len(time_series) - k - 1))
            henderson_filtered_time_series[k] = \
                np.sum(asymmetric_weights * time_series[int(k - ((order - 1) / 2)):])
        else:
            henderson_filtered_time_series[k] = \
                np.sum(weights * time_series[int(k - ((order - 1) / 2)):int(k + ((order - 1) / 2) + 1)])

    return henderson_filtered_time_series


def seasonal_ma(time_series, factors='3x3', seasonality='monthly'):

    seasonal_filtered_time_series = np.zeros_like(time_series)

    if factors == '3x3':
        weighting = np.asarray((1 / 9, 2 / 9, 1 / 3, 2 / 9, 1 / 9))
        season_window_width = 5
    elif factors == '3x5':
        weighting = np.asarray((1 / 15, 2 / 15, 1 / 5, 1 / 5, 1 / 5, 2 / 15, 1 / 15))
        season_window_width = 7
    elif factors == '3x7':
        weighting = np.asarray((1 / 21, 2 / 21, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 2 / 21, 1 / 21))
        season_window_width = 9
    elif factors == '3x9':
        weighting = np.asarray((1 / 27, 2 / 27, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 2 / 27, 1 / 27))
        season_window_width = 11

    if seasonality == 'monthly':
        for month in range(12):

            month_bool = (np.asarray(range(len(time_series))) % 12 - month == 0)
            index_values = np.asarray(range(len(time_series)))[month_bool]

            for point in np.asarray(range(len(index_values)))[
                         int((season_window_width - 1) / 2):int(len(index_values) - ((season_window_width - 1) / 2))]:
                relative_points = \
                    time_series[index_values[int(point - ((season_window_width - 1) / 2)):int(
                        point + ((season_window_width + 1) / 2))]]

                seasonal_filtered_time_series[index_values[point]] = \
                    sum(weighting * relative_points)

        # repeat closest estimates at edges
        seasonal_filtered_time_series[:int(((season_window_width - 1) / 2) * 12)] = np.append(
            seasonal_filtered_time_series[int(((season_window_width - 1) / 2) * 12):int(((season_window_width + 1) / 2) * 12)],
            seasonal_filtered_time_series[int(((season_window_width - 1) / 2) * 12):int(((season_window_width + 1) / 2) * 12)])

        seasonal_filtered_time_series[int(len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 12):] = np.append(
            seasonal_filtered_time_series[int(len(seasonal_filtered_time_series) - ((season_window_width + 1) / 2) * 12):int(
                len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 12)], seasonal_filtered_time_series[int(
                len(seasonal_filtered_time_series) - ((season_window_width + 1) / 2) * 12):int(
                len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 12)])

    # need to test
    elif seasonality == 'quarterly':
        for quarter in range(4):

            month_bool = (np.asarray(range(len(time_series))) % 4 - quarter == 0)
            index_values = np.asarray(range(len(time_series)))[month_bool]

            for point in np.asarray(range(len(index_values)))[
                         int((season_window_width - 1) / 2):int(len(index_values) - ((season_window_width - 1) / 2))]:
                relative_points = \
                    time_series[index_values[int(point - ((season_window_width - 1) / 2)):int(
                        point + ((season_window_width + 1) / 2))]]

                seasonal_filtered_time_series[index_values[point]] = \
                    sum(weighting * relative_points)

        # repeat closest estimates at edges
        seasonal_filtered_time_series[:int(((season_window_width - 1) / 2) * 4)] = np.append(
            seasonal_filtered_time_series[
            int(((season_window_width - 1) / 2) * 4):int(((season_window_width + 1) / 2) * 4)],
            seasonal_filtered_time_series[
            int(((season_window_width - 1) / 2) * 4):int(((season_window_width + 1) / 2) * 4)])

        seasonal_filtered_time_series[
        int(len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 4):] = np.append(
            seasonal_filtered_time_series[
            int(len(seasonal_filtered_time_series) - ((season_window_width + 1) / 2) * 4):int(
                len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 4)],
            seasonal_filtered_time_series[int(
                len(seasonal_filtered_time_series) - ((season_window_width + 1) / 2) * 4):int(
                len(seasonal_filtered_time_series) - ((season_window_width - 1) / 2) * 4)])

    return seasonal_filtered_time_series


def x11(time, time_series, model_type='additive', seasonality='monthly', seasonal_factor='3x3', trend_window_width_1=13,
        trend_window_width_2=13, trend_window_width_3=13):

    if model_type == 'additive':

        # step 1
        # initial estimate of trend-cycle
        first_estimate_trend = np.zeros_like(time_series)

        for point in range(len(time_series))[int((trend_window_width_1 - 1) / 2): int(len(time_series) - ((trend_window_width_1 - 1) / 2))]:

            first_estimate_trend[point] = \
                np.mean(time_series[int(point - ((trend_window_width_1 - 1) / 2)):int(point + ((trend_window_width_1 - 1) / 2))])

        # interpolate edges
        # relevant to remove some trend at edges
        interpolation = \
            interp1d(time[int((trend_window_width_1 - 1) / 2): int(len(time_series) - ((trend_window_width_1 - 1) / 2))],
                     first_estimate_trend[int((trend_window_width_1 - 1) / 2): int(len(time_series) - ((trend_window_width_1 - 1) / 2))],
                     fill_value='extrapolate')

        left_extrapolate = interpolation(time[:int((trend_window_width_1 - 1) / 2)])
        right_extrapolate = interpolation(time[int(len(time_series) - ((trend_window_width_1 - 1) / 2)):])

        first_estimate_trend[:int((trend_window_width_1 - 1) / 2)] = left_extrapolate
        first_estimate_trend[int(len(time_series) - ((trend_window_width_1 - 1) / 2)):] = right_extrapolate

        # time series without trend-cycle component
        no_trend_cycle = time_series - first_estimate_trend

        # step 2
        # initial estimate of seasonality
        # assume monthly for now
        first_estimate_season = seasonal_ma(no_trend_cycle, factors=seasonal_factor, seasonality=seasonality)

        # step 3
        # estimate seasonality adjusted data
        # time series without seasonality component (with trend-cycle component)
        no_seasonality = time_series - first_estimate_season

        # step 4
        # better estimate of trend
        next_estimate_trend = henderson_ma(no_seasonality, trend_window_width_2)
        no_trend_cycle_2 = time_series - next_estimate_trend

        # step 5
        final_estimate_season = seasonal_ma(no_trend_cycle_2, factors=seasonal_factor, seasonality=seasonality)

        # step 6
        no_seasonality_2 = time_series - final_estimate_season

        # step 7
        final_estimate_trend = henderson_ma(no_seasonality_2, trend_window_width_3)

        # step 8
        final_estimate_irregular = no_seasonality_2 - final_estimate_trend

    return final_estimate_trend, final_estimate_season, final_estimate_irregular


three_factors_CO2 = x11(time=time, time_series=imfs[1, :])

dtht = Hilbert(time=time / 12, time_series=three_factors_CO2[1])
dtht_season = dtht.dtht_kak()
if_season = omega(time / 12, theta(three_factors_CO2[1], dtht_season))

hs_ouputs_x11_emd_x11 = hilbert_spectrum(time / 12, three_factors_CO2[1].reshape(1, 972), dtht_season.reshape(1, 972),
                                         if_season.reshape(1, 971), max_frequency=2 * 2 * np.pi, which_imfs=[0])

x_hs_emd_x11, y_emd_x11, z_emd_x11 = hs_ouputs_x11_emd_x11
y_emd_x11 /= 2 * np.pi

z_min_emd_x11, z_max_emd_x11 = 0, np.abs(z_emd_x11).max()
fig, ax = plt.subplots()
ax.pcolormesh(x_hs_emd_x11, y_emd_x11, np.abs(z_emd_x11), cmap='gist_rainbow', vmin=z_min_emd_x11, vmax=z_max_emd_x11)
ax.set_title(textwrap.fill(r'Instantaneous Frequency of EMD augmented with X11 Seasonal Component of First IMF from UK Birth Rates', 55))
plt.plot(time, np.ones_like(time), 'k--', label=textwrap.fill('Annual cycle', 10))
plt.plot(time, (1 / 0.75) * np.ones_like(time), '--', c='gold', label=textwrap.fill('Every nine months', 10))
plt.plot(((11 + 23 * 12) / 12) * np.ones(100), np.linspace(0, 2, 100), 'b--',
         label=textwrap.fill('NHS provides pill to married women', 10))
plt.plot(((11 + 29 * 12) / 12) * np.ones(100), np.linspace(0, 2, 100), 'g--',
         label=textwrap.fill('NHS provides pill to unmarried women', 10))
ax.axis([x_hs_emd_x11.min(), x_hs_emd_x11.max(), y_emd_x11.min(), y_emd_x11.max()])
plt.xticks(np.arange(11, 972, int(5 * 12)) / 12, np.arange(1938, 2019, 5), rotation=-45, fontsize=8)
plt.ylim(0, 2)
plt.ylabel('Frequency (year$^{-1}$)')
plt.xlabel('Time (years)')

box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.025, box_0.y0 + 0.05, box_0.width * 0.85, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/births_if_emd_x11.png')
plt.show()

temp = 0
