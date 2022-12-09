
#     ________
#            /
#      \    /
#       \  /
#        \/

import textwrap
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
import colorednoise as cn
import seaborn as sns
from AdvEMDpy import EMD
from emd_preprocess import Preprocess
from emd_basis import Basis
import scipy.fft as spfft
from matplotlib import mlab
from emd_hilbert import hilbert_spectrum

sns.set(style='darkgrid')
np.random.seed(0)

# Full Spectrum Ensemble Empirical Mode Decomposition (FSEEMD)

n = 1001
time = np.linspace(0, 5 * np.pi, n)
time_series = np.cos(time) + np.cos(5 * time) + np.random.normal(0, 1, len(time))
preprocess = Preprocess(time=time, time_series=time_series)
median = preprocess.median_filter()[1]
sd_time_series = np.std(time_series-median)

x_points = [0, (np.pi / 2), np.pi, (3 * np.pi / 2), 2 * np.pi, (5 * np.pi / 2),
            3 * np.pi, (7 * np.pi / 2), 4 * np.pi, (9 * np.pi / 2), 5 * np.pi]
x_names = (r'$ 0 $', r'$ \dfrac{\pi}{2} $', r'$ \pi $', r'$ \dfrac{3\pi}{2} $', r'$ 2\pi $', r'$ \dfrac{5\pi}{2} $',
           r'$ 3\pi $', r'$ \dfrac{7\pi}{2} $', r'$ 4\pi $', r'$ \dfrac{9\pi}{2} $', r'$ 5\pi $')


number_of_samples = np.shape(time)[0]

colour_dict = {}
colour_dict[-2] = 'Violet'
colour_dict[-1] = 'Blue'
colour_dict[0] = 'White'
colour_dict[1] = 'Pink'
colour_dict[2] = 'Brown'

for beta in colour_dict.keys():

    noise = cn.powerlaw_psd_gaussian(beta, number_of_samples)
    dt = 0.04
    s, f = mlab.psd(noise, Fs=1 / dt)
    plt.gcf().subplots_adjust(bottom=0.15)
    if beta == 2:
        plt.plot(np.log10(f), np.log10((1 / f) ** 2) - np.mean(np.log10((1 / f)[1:] ** 2) - np.log10(s)[1:]),
                 color='black', label=r'$P \propto \frac{1}{f^2}$ ', linewidth=2, zorder=2)
    elif beta == 1:
        plt.plot(np.log10(f), np.log10(1 / f) - np.mean(np.log10(1 / f)[1:] - np.log10(s)[1:]),
                 color='black', label=r'$P \propto \frac{1}{f}$ ', linewidth=2, zorder=2)
    elif beta == 0:
        plt.plot(np.log10(f), np.ones_like(f) - np.mean(np.ones_like(f)[1:] - np.log10(s)[1:]),
                 color='black', label=r'$P \propto k$ ', linewidth=2, zorder=2)
    elif beta == -1:
        plt.plot(np.log10(f), np.log10(f) - np.mean(np.log10(f)[1:] - np.log10(s)[1:]),
                 color='black', label=r'$P \propto f$ ', linewidth=2, zorder=2)
    elif beta == -2:
        plt.plot(np.log10(f), np.log10(f ** 2) - np.mean(np.log10(f ** 2)[1:] - np.log10(s)[1:]),
                 color='black', label=r'$P \propto f^2$ ', linewidth=2, zorder=2)
    if beta != 0:
        colour = colour_dict[beta]
    else:
        colour = 'Grey'
    plt.plot(np.log10(f), np.log10(s), color=f'{colour}', label=f'{colour_dict[beta]} noise', linewidth=2, zorder=1)
    plt.legend(loc='lower left')
    plt.title(f'{colour_dict[beta]} Noise Power Spectral Density')
    plt.xlabel('log10(Frequency)')
    plt.ylabel('log10(Power)')
    plt.show()

    noisy_time_series = time_series + noise
    plt.title(f'{colour_dict[beta]} Noise Example')
    if beta != 0:
        colour = colour_dict[beta]
    else:
        colour = 'Grey'
    plt.plot(time, time_series, '-', c='black', label='Original time series', zorder=2)
    plt.plot(time, noisy_time_series, c=f'{colour}', label=f'Time series with added {colour_dict[beta].lower()} noise',
             zorder=1)
    plt.xticks(x_points, x_names)
    plt.legend(loc='lower left')
    plt.show()

# test FSEEMD
fseemd_sd = 0.1
imf_1 = np.zeros_like(time_series)
imf_2 = np.zeros_like(time_series)
iterations = 10

for i in range(iterations):

    beta = np.random.randint(-2, 2 + 1)
    new_time_series = time_series + cn.powerlaw_psd_gaussian(beta, len(time_series)) * sd_time_series * fseemd_sd
    emd = EMD(time=time, time_series=new_time_series)
    imfs = emd.empirical_mode_decomposition(smooth=True, stop_crit_threshold=1, max_imfs=3, verbose=False)[0]

    imf_1 += imfs[2, :]
    imf_2 += imfs[3, :]

plt.title('Full Spectrum Ensemble Empirical Mode Decomposition Example')
plt.plot(time, imf_1 / iterations, label='IMF 1')
plt.plot(time, np.cos(5 * time), '--', label='cos(5t)')
plt.plot(time, imf_2 / iterations, label='IMF 2')
plt.plot(time, np.cos(time), '--', label='cos(t)')
plt.legend(loc='lower left')
plt.show()
print(f'IMF 1 Error {np.sum(np.abs(imf_1 / iterations - np.cos(5 * time)))}')
print(f'IMF 2 Error {np.sum(np.abs(imf_2 / iterations - np.cos(time)))}')

# Compressive Sampling Empirical Mode Decomposition (CSEMD)
# Constant frequency and amplitude example


def frequency(time_signal, frequency_period, min_frequency, max_frequency):

    end_time = time_signal[-1]  # last time point value
    # time over which frequency changes from minimum frequency to maximum frequency or vice versa
    freq_half_mod = frequency_period / 2  # half period
    time_points = len(time_signal)  # total number of time points
    time_diff = np.diff(time_signal)  # difference between time points
    increments = int(end_time / freq_half_mod)
    increment_length = int(max((time_points - 1) / increments, 1))  # minimum set to 1 as trivial
    new_time_signal = np.zeros_like(time_signal)
    new_time_signal[0] = time_signal[0]

    for i in range(1, time_points):
        if np.mod(i / increment_length, 1) == 0:
            temp_mod = 1
        else:
            temp_mod = np.mod(i / increment_length, 1)
        if ((i - 1) // increment_length) % 2 == 0:
            new_time_signal[i] = new_time_signal[i - 1] + (min_frequency + temp_mod * (max_frequency - min_frequency)) \
                                 * time_diff[i - 1]
        else:
            new_time_signal[i] = new_time_signal[i - 1] + (max_frequency - temp_mod * (max_frequency - min_frequency)) \
                                 * time_diff[i - 1]
    return new_time_signal


def ibst(inst_freq, knot_time, knots):

    output = np.zeros_like(knot_time)
    output[0] = 1

    diff_time = np.diff(knot_time)

    for i in range(1, len(knot_time)):

        output[i] = np.cos(np.sum(inst_freq[:i] * diff_time[:i]))

    basis = Basis(time=knot_time, time_series=output)
    output = basis.basis_function_approximation(knots, knot_time)[0]

    return output


def bst_processing(signal, compression_bool):

    end = signal[-1]
    begin = signal[0]

    A = spfft.idct(np.identity(len(signal)), norm='ortho', axis=0)
    sampling_matrix = np.zeros((np.sum(compression_bool), n))
    row = 0

    ri = np.linspace(0, len(signal)-1, len(signal))
    ri = ri[compression_bool]

    for r in ri:
        sampling_matrix[row, r] = 1
        row += 1

    A = np.matmul(sampling_matrix, A)
    signal = np.matmul(sampling_matrix, signal)

    # do L1 optimization
    vx = cvx.Variable(n)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [A * vx == signal]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve(verbose=True, solver=cvx.ECOS)

    x = np.array(vx.value)
    sig = spfft.idct(x, norm='ortho', axis=0)

    x_sorted = x.copy()
    x_sorted.sort()
    x_sorted = np.flip(x_sorted)

    indice_storage = np.linspace(0, len(x_sorted) - 1, len(x_sorted))

    indices = []

    level = x_sorted[0]

    for k in x_sorted:

        if k > 0.1 * level:
            indices = np.hstack((indices, indice_storage[x == k]))

    frequency_content = (np.pi / (end - begin)) * indices

    return frequency_content


def imf_construction(signal, compression_bool, knot_time, knots):

    frequencies = bst_processing(signal, compression_bool)

    b_spline_approx = np.zeros_like(signal)

    for f in frequencies:

        b_spline_approx += ibst(f, knot_time, knots)

    return b_spline_approx


def imf_compressive_sampling(signal, signal_time, knots, knots_time, indices, sig_level=0.1, debug=False):

    # establish time domain of frequency
    end_time = signal_time[-1]
    begin_time = signal_time[0]

    # construct basis matrices
    A = spfft.idct(np.identity(len(signal)), norm='ortho', axis=0)
    sampling_matrix = np.zeros((len(indices), len(signal)))
    row = 0
    for r in indices:
        sampling_matrix[row, r] = 1
        row += 1

    # sample matrices
    A = np.matmul(sampling_matrix, A)
    signal = np.matmul(sampling_matrix, signal)

    # do L1 optimization
    vx = cvx.Variable(n)
    objective = cvx.Minimize(cvx.norm(vx, 1))
    constraints = [A * vx == signal]
    prob = cvx.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cvx.ECOS)

    # extract and find largest frequency components
    x = np.array(vx.value)

    if debug:
        plt.plot(range(len(signal_time)) / ((end_time - begin_time) / np.pi), x)
        plt.title('Underlying Frequency Structure')
        # plt.xlim(-3, 8)  # temporary for demonstrative plots
        plt.show()

    x_sorted = x.copy()
    x_sorted.sort()
    x_sorted = np.flip(x_sorted)
    indice_storage = np.linspace(0, len(x_sorted) - 1, len(x_sorted))

    indices = []
    level = x_sorted[0]

    for k in x_sorted:

        if k > sig_level * level:
            indices = np.hstack((indices, indice_storage[x == k]))

    frequency_content = (np.pi / (end_time - begin_time)) * indices

    # construct B-spline approximation
    b_spline_approx = np.zeros_like(knots_time)
    diff_time = np.diff(knots_time)

    magnitude = 0

    for j in frequency_content:

        component = np.zeros_like(b_spline_approx)
        component[0] = 1

        inst_freq = j * np.ones_like(component)[:-1]

        for i in range(1, len(knots_time)):

            component[i] = np.cos(np.sum(inst_freq[:i] * diff_time[:i]))

        basis = Basis(time=knots_time, time_series=component)
        b_spline_approx += \
            (x_sorted[magnitude] / np.sqrt(len(signal_time) / 2)) * basis.basis_function_approximation(knots,
                                                                                                       knots_time)[0]
        magnitude += 1

    return b_spline_approx


time = np.linspace(0, 5 * np.pi, 1001)
knots = np.linspace(0, 5 * np.pi, 401)

time_series = np.cos(50 * time)
time_series += np.cos(30 * time)
time_series += np.cos(10 * time)
time_series += np.cos(5 * time)

m = 51  # 5% sample
ri = np.random.choice(n, m, replace=False)  # random sample of indices
ri.sort()  # sorting not strictly necessary, but convenient for plotting

reconstructed_time_series = imf_compressive_sampling(time_series, time, knots, time, ri, debug=True)

plt.title('Compressive Sampling EMD Example')
plt.plot(time, time_series, zorder=1, label='Time series')
plt.plot(time, reconstructed_time_series, '--', zorder=2, label='Reconstruction')
plt.scatter(time[ri], time_series[ri], c='r', zorder=3, label='Samples')
plt.legend(loc='lower left')
plt.xticks(x_points, x_names)
plt.show()

emd = EMD(time=time, time_series=reconstructed_time_series)
imfs, hts, ifs, _, _, dtht, dtif = emd.empirical_mode_decomposition(knots=knots, knot_time=time, smooth=True,
                                                                    stop_crit_threshold=1, dtht=True,
                                                                    dtht_method='fft')

hilbert = hilbert_spectrum(time=time, imf_storage=imfs, ht_storage=dtht, if_storage=dtif, max_frequency=60, plot=False)

z_min, z_max = 0, np.abs(hilbert[2]).max()
fig, ax = plt.subplots()
ax.pcolormesh(hilbert[0], hilbert[1], hilbert[2], cmap='gist_rainbow', vmin=z_min, vmax=z_max)
plt.plot(hilbert[0][0, :], 50 * np.ones_like(hilbert[0][0, :]), '--', label='cos(50t)')
plt.plot(hilbert[0][0, :], 30 * np.ones_like(hilbert[0][0, :]), '--', label='cos(30t)')
plt.plot(hilbert[0][0, :], 10 * np.ones_like(hilbert[0][0, :]), '--', label='cos(10t)')
plt.plot(hilbert[0][0, :], 5 * np.ones_like(hilbert[0][0, :]), '--', label='cos(5t)')
ax.set_title(textwrap.fill('Gaussian Filtered Compressive Sampling EMD Hilbert Spectrum', 40))
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
ax.axis([hilbert[0].min(), hilbert[0].max(), hilbert[1].min(), hilbert[1].max()])

box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.03, box_0.y0 + 0.05, box_0.width * 0.9, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# Compressive Sampling Empirical Mode Decomposition (CSEMD)
# Non-constant frequency and non-constant amplitude example

frequency_modulation_period = np.pi
frequency_max = 4
frequency_min = 2
knots = np.linspace(0, 5 * np.pi, 51)

modulated_freq = frequency(time, frequency_modulation_period, frequency_min, frequency_max)

cum_signal_CAMF = np.cos(modulated_freq)

m = 51
ri = np.random.choice(n, m, replace=False)
ri.sort()

temp_freq = imf_compressive_sampling(cum_signal_CAMF, time, knots, time, ri, sig_level=0, debug=True)

plt.plot(time, cum_signal_CAMF, zorder=1)
plt.plot(time, temp_freq, '--', zorder=2)
plt.scatter(time[ri], cum_signal_CAMF[ri], c='r', zorder=3)
plt.show()

emd = EMD(time=time, time_series=temp_freq)
imfs, _, _, _, _, hts, ifs = emd.empirical_mode_decomposition(knots=knots, knot_time=time, smooth=True,
                                                              stop_crit_threshold=1, dtht=True)

hilbert = hilbert_spectrum(time=time, imf_storage=imfs, ht_storage=hts, if_storage=ifs, max_frequency=5, plot=False)

z_min, z_max = 0, np.abs(hilbert[2]).max()
fig, ax = plt.subplots()
ax.pcolormesh(hilbert[0], hilbert[1], hilbert[2], cmap='gist_rainbow', vmin=z_min, vmax=z_max)
plt.plot(hilbert[0][0, :], (modulated_freq[:-1] - modulated_freq[1:]) / (time[:-1] - time[1:]),
         '--', label=textwrap.fill('Modulated frequency', 10))
ax.set_title(textwrap.fill('Gaussian Filtered Compressive Sampling EMD Hilbert Spectrum', 40))
ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
ax.axis([hilbert[0].min(), hilbert[0].max(), hilbert[1].min(), hilbert[1].max()])

box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.04, box_0.y0 + 0.05, box_0.width * 0.9, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
