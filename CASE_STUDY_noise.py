
import textwrap
import numpy as np
import seaborn as sns
import colorednoise as cn
import matplotlib.pyplot as plt

import AdvEMDpy
from emd_hilbert import hilbert_spectrum, Hilbert, omega, theta

begin = 0
end = 1
points = int(7.5 * 512)
x = np.linspace(begin, end, points)

for dB in ['brown', 'brown', 'red', 'blue', 'violet']:

    signal_1 = np.sin(250 * np.pi * x ** 2)
    signal_2 = np.sin(80 * np.pi * x ** 2)

    signal = signal_1 + signal_2

    # noise = np.random.normal(0, (np.sum(signal ** 2) / len(signal)) / (10 ** dB), points)
    # plt.plot(signal + noise)
    # plt.show()
    if dB == 'brown':
        noise = cn.powerlaw_psd_gaussian(2, points)
    if dB == 'red':
        noise = cn.powerlaw_psd_gaussian(1, points)
    if dB == 'blue':
        noise = cn.powerlaw_psd_gaussian(-1, points)
    if dB == 'purple':
        noise = cn.powerlaw_psd_gaussian(-2, points)

    signal = signal + noise

    x_points = np.arange(0, 1.1, 0.1)
    x_names = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

    plt.figure(2)
    ax = plt.subplot(111)
    plt.title('Frequency Modulation Example')
    ax.plot(x, signal)
    plt.xticks(x_points, x_names)
    plt.ylabel('g(t)')
    plt.xlabel('t')

    box_0 = ax.get_position()
    # print(box_0)
    ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.05, box_0.width * 0.9, box_0.height * 0.9])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('FM_signals/frequency_modulation_{}.png'.format(dB))

    # plt.show()

    # stft

    hilbert = Hilbert(x, signal)
    t_stft, f_stft, z_stft = hilbert.stft_custom(window_width=512, angular_frequency=False)

    plt.figure(2)
    ax = plt.subplot(111)
    plt.gcf().subplots_adjust(bottom=0.10)
    ax.pcolormesh((t_stft - (1 / (3840 / 128))), f_stft, np.abs(z_stft), vmin=0, vmax=np.max(np.max(np.abs(z_stft))))
    ax.plot(t_stft[:-1], 250 * t_stft[:-1], 'b--', label='f(t) = 250t', linewidth=2)
    ax.plot(t_stft[:-1], 80 * t_stft[:-1], 'g--', label='f(t) = 80t', linewidth=2)
    plt.title(r'STFT - $ |\mathcal{STF}(g(t))(t,f)|^2 $')
    plt.ylabel('f')
    plt.xlabel('t')
    plt.xticks(x_points, x_names)
    plt.ylim(0, 260)

    box_0 = ax.get_position()
    ax.set_position([box_0.x0 + 0.0125, box_0.y0 + 0.075, box_0.width * 0.8, box_0.height * 0.9])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('FM_signals/frequency_modulation_stft_{}.png'.format(dB))

    # plt.show()

    # morlet wavelet

    hilbert = Hilbert(x, signal)
    t_morlet, f_morlet, z_morlet = hilbert.morlet_wavelet_custom(window_width=512, angular_frequency=False)
    # z_morlet[0, :] = z_morlet[1, :]

    plt.figure(2)
    ax = plt.subplot(111)
    plt.gcf().subplots_adjust(bottom=0.10)
    ax.pcolormesh((t_morlet - (1 / (3840 / 128))), f_morlet, np.abs(z_morlet), vmin=0, vmax=np.max(np.max(np.abs(z_morlet))))
    ax.plot(t_morlet[:-1], 250 * t_morlet[:-1], 'b--', label='f(t) = 250t', linewidth=2)
    ax.plot(t_morlet[:-1], 80 * t_morlet[:-1], 'g--', label='f(t) = 80t', linewidth=2)
    plt.title(r'Morlet - $ |(g(t))(t,f)|^2 $')
    plt.ylabel('f')
    plt.xlabel('t')
    plt.xticks(x_points, x_names)
    plt.ylim(0, 260)


    box_0 = ax.get_position()
    ax.set_position([box_0.x0 + 0.0125, box_0.y0 + 0.075, box_0.width * 0.8, box_0.height * 0.9])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('FM_signals/frequency_modulation_morlet_{}.png'.format(dB))

    # plt.show()

    # joint plot - top

    fig, axs = plt.subplots(1, 2)
    plt.subplots_adjust(hspace=0.5)
    axs[0].pcolormesh((t_stft - (1 / (3840 / 128))), f_stft, np.abs(z_stft), vmin=0, vmax=np.max(np.max(np.abs(z_stft))))
    axs[0].plot(t_morlet[:-1], 250 * t_morlet[:-1], 'b--', label='f(t) = 250t', linewidth=2)
    axs[0].plot(t_morlet[:-1], 80 * t_morlet[:-1], 'g--', label='f(t) = 80t', linewidth=2)
    axs[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[0].set_xticklabels(['$0$', '$0.2$', '$0.4$', '$0.6$', '$0.8$', '$1.0$'])
    axs[1].pcolormesh((t_morlet - (1 / (3840 / 128))), f_morlet, np.abs(z_morlet), vmin=0, vmax=np.max(np.max(np.abs(z_morlet))))
    axs[1].plot(t_morlet[:-1], 250 * t_morlet[:-1], 'b--', label='f(t) = 250t', linewidth=2)
    axs[1].plot(t_morlet[:-1], 80 * t_morlet[:-1], 'g--', label='f(t) = 80t', linewidth=2)
    axs[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axs[1].set_xticklabels(['$0$', '$0.2$', '$0.4$', '$0.6$', '$0.8$', '$1.0$'])

    axis = 0
    for ax in axs.flat:
        ax.set_ylim(0, 260)
        ax.set(xlabel='t')
        if axis == 0:
            ax.set(ylabel='Frequency')
            ax.legend(loc='upper left')
        if axis == 1:
            ax.set_yticks([0])
            ax.set_yticklabels([''])
        axis += 1

    plt.gcf().subplots_adjust(bottom=0.15)

    axs[0].set_title(r'$ |\mathcal{STF}(g(t))(t,f)|^2 $')
    axs[1].set_title(r'$ |\mathcal{MW}(g(t))(t,f)|^2 $')

    plt.gcf().subplots_adjust(bottom=0.15)

    plt.savefig('FM_signals/frequency_modulation_STFT_Morlet_{}.png'.format(dB))

    # plt.show()

    # joint plot - bottom

    # emd

    knots = np.linspace(begin, end, int(points / 10))

    advemd = AdvEMDpy.EMD(time=x, time_series=signal)
    imfs, _, _, _, _, hts, ifs = advemd.empirical_mode_decomposition(knots=knots, knot_time=x, matrix=True,
                                                                     max_internal_iter=20, mean_threshold=10,
                                                                     stopping_criterion_threshold=10, dtht=True)

    # for i in range(np.shape(imfs)[0]):
    #     plt.plot(imfs[i, :])
    #     plt.show()

    hs_ouputs = hilbert_spectrum(x, imfs, hts, ifs, max_frequency=260 * 2 * np.pi, plot=False)

    x_hs_stft, y_stft, z_stft = hs_ouputs
    y_stft = y_stft / (2 * np.pi)

    z_min, z_max = 0, np.abs(z_stft).max()
    fig, ax = plt.subplots()
    ax.pcolormesh(x_hs_stft, y_stft, z_stft, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
    ax.plot(t_stft, 250 * t_stft, 'b--', label='f(t) = 250t', linewidth=2)
    ax.plot(t_stft, 80 * t_stft, 'g--', label='f(t) = 80t', linewidth=2)
    ax.set_title(f'Gaussian Filtered Hilbert Spectrum - {int(points / 10)} Knots')
    ax.set_xlabel('t')
    ax.set_ylabel('f')
    ax.axis([x_hs_stft.min(), x_hs_stft.max(), y_stft.min(), y_stft.max()])

    box_0 = ax.get_position()
    ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.75, box_0.height * 0.9])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('FM_signals/frequency_modulation_emd_384_{}.png'.format(dB))

    # plt.show()

    knots = np.linspace(begin, end, int(points / 5))

    advemd = AdvEMDpy.EMD(time=x, time_series=signal)
    imfs, _, _, _, _, hts, ifs = advemd.empirical_mode_decomposition(knots=knots, knot_time=x, matrix=True, dtht=True,
                                                                     max_internal_iter=20, mean_threshold=10,
                                                                     stopping_criterion_threshold=10)

    # for i in range(np.shape(imfs)[0]):
    #     plt.plot(imfs[i, :])
    #     plt.show()

    hs_ouputs = hilbert_spectrum(x, imfs, hts, ifs, max_frequency=260 * 2 * np.pi)

    x_hs_morlet, y_morlet, z_morlet = hs_ouputs
    y_morlet = y_morlet / (2 * np.pi)

    z_min, z_max = 0, np.abs(z_morlet).max()
    fig, ax = plt.subplots()
    ax.pcolormesh(x_hs_morlet, y_morlet, z_morlet, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
    ax.plot(t_stft, 250 * t_stft, 'b--', label='f(t) = 250t', linewidth=2)
    ax.plot(t_stft, 80 * t_stft, 'g--', label='f(t) = 80t', linewidth=2)
    ax.set_title(f'Gaussian Filtered Hilbert Spectrum - {int(points / 5)} Knots')
    ax.set_xlabel('t')
    ax.set_ylabel('f')
    ax.axis([x_hs_morlet.min(), x_hs_morlet.max(), y_morlet.min(), y_morlet.max()])

    box_0 = ax.get_position()
    ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.75, box_0.height * 0.9])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig('FM_signals/frequency_modulation_emd_768_{}.png'.format(dB))

    # plt.show()

    # joint plot - top

    fig, axs = plt.subplots(1, 2)
    plt.subplots_adjust(hspace=0.5, top=0.8)
    axs[0].pcolormesh(x_hs_stft, y_stft, z_stft, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
    axs[0].plot(t_stft, 250 * t_stft, 'b--', label='f(t) = 250t', linewidth=2)
    axs[0].plot(t_stft, 80 * t_stft, 'g--', label='f(t) = 80t', linewidth=2)
    axs[0].set_title(textwrap.fill(f'Gaussian Filtered Hilbert Spectrum - {int(points / 10)} Knots', 20))
    axs[0].axis([x_hs_stft.min(), x_hs_stft.max(), y_stft.min(), y_stft.max()])
    axs[1].pcolormesh(x_hs_morlet, y_morlet, z_morlet, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
    axs[1].plot(t_stft, 250 * t_stft, 'b--', label='f(t) = 250t', linewidth=2)
    axs[1].plot(t_stft, 80 * t_stft, 'g--', label='f(t) = 80t', linewidth=2)
    axs[1].set_title(textwrap.fill(f'Gaussian Filtered Hilbert Spectrum - {int(points / 5)} Knots', 20))
    axs[1].axis([x_hs_morlet.min(), x_hs_morlet.max(), y_morlet.min(), y_morlet.max()])

    axis = 0
    for ax in axs.flat:
        ax.set_ylim(0, 260)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_xticklabels(['$0$', '$0.2$', '$0.4$', '$0.6$', '$0.8$'])
        ax.set(xlabel='t')
        if axis == 0:
            ax.set(ylabel='Frequency')
            ax.legend(loc='upper left')
        if axis == 1:
            ax.set(ylabel='')
            ax.set_yticks([0])
            ax.set_yticklabels([''])
        axis += 1

    plt.gcf().subplots_adjust(bottom=0.15)

    axs[0].set_title(textwrap.fill(f'Gaussian Filtered Hilbert Spectrum - {int(points / 10)} Knots', 20))
    axs[1].set_title(textwrap.fill(f'Gaussian Filtered Hilbert Spectrum - {int(points / 5)} Knots', 20))

    plt.savefig('FM_signals/frequency_modulation_emd_384_768_{}.png'.format(dB))

    # plt.show()

    # joint plot - bottom

    # # knot point optimisation
    #
    # knots = np.linspace(begin, end, int(points / 5))
    #
    # advemd = AdvEMDpy.EMD(time=x, time_series=signal)
    # imfs, _, _, _, knots, hts, ifs = advemd.empirical_mode_decomposition(knots=knots, knot_time=x, matrix=True,
    #                                                                      max_internal_iter=20,
    #                                                                      mean_threshold=10, stopping_criterion_threshold=10,
    #                                                                      optimise_knots=True, output_knots=True, dtht=True)
    #
    # for i in range(np.shape(imfs)[0]):
    #     plt.plot(imfs[i, :])
    #     plt.show()
    #
    # plt.figure(2)
    # ax = plt.subplot(111)
    # plt.title('Frequency Modulation Optimised Knot Allocation')
    # plt.plot(x, signal)
    # for knot in knots[:-1]:
    #     plt.plot(knot * np.ones(100), np.linspace(-1, 1, 100), 'r--')
    # plt.plot(knots[-1] * np.ones(100), np.linspace(-1, 1, 100), 'r--', label='optimised knots')
    # plt.xticks(x_points, x_names)
    # plt.legend(loc='lower left')
    # plt.ylabel('g(t)')
    # plt.xlabel('t')
    #
    # box_0 = ax.get_position()
    # ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.05, box_0.width * 0.9, box_0.height * 0.9])
    # # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #
    # plt.savefig('FM_signals/frequency_modulation_knots_{}.png'.format(dB))
    #
    # plt.show()
    #
    # hs_ouputs = hilbert_spectrum(x, imfs, hts, ifs, max_frequency=260 * 2 * np.pi)
    #
    # x_hs_morlet, y_morlet, z_morlet = hs_ouputs
    # y_morlet = y_morlet / (2 * np.pi)
    #
    # z_min, z_max = 0, np.abs(z_morlet).max()
    # fig, ax = plt.subplots()
    # ax.pcolormesh(x_hs_morlet, y_morlet, z_morlet, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
    # plt.plot(t_stft, 250 * t_stft, 'b--', label='f(t) = 250t', linewidth=2)
    # plt.plot(t_stft, 80 * t_stft, 'g--', label='f(t) = 80t', linewidth=2)
    # ax.set_title(textwrap.fill('Gaussian Filtered Hilbert Spectrum - Serial Bisection Knot Point Optimisation', 40))
    # ax.set_xlabel('t')
    # ax.set_ylabel('f')
    # ax.axis([x_hs_morlet.min(), x_hs_morlet.max(), y_morlet.min(), y_morlet.max()])
    #
    # box_0 = ax.get_position()
    # ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.75, box_0.height * 0.9])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #
    # plt.savefig('FM_signals/frequency_modulation_knot_optimisation_{}.png'.format(dB))
    #
    # plt.show()
    #
    # # knot distance decrease and heat map plot
    #
    # fig, axs = plt.subplots(1, 2)
    # fig.suptitle(textwrap.fill('Gaussian Filtered Hilbert Spectrum - Serial Bisection Knot Point Optimisation', 40))
    # plt.subplots_adjust(hspace=0.5, wspace=0.3)
    # axs[0].scatter(knots[:-1], np.diff(knots))
    # axs[0].set_title(textwrap.fill('Knot Location versus Successive Knot Distance', 25))
    # axs[1].pcolormesh(x_hs_morlet, y_morlet, z_morlet, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
    # axs[1].plot(x_hs_morlet[0, :], 250 * x_hs_morlet[0, :], 'b--', label='f(t) = 250t', linewidth=2)
    # axs[1].plot(x_hs_morlet[0, :], 80 * x_hs_morlet[0, :], 'g--', label='f(t) = 80t', linewidth=2)
    # axs[1].set_title(textwrap.fill('Hilbert Transform with Optimised Knots', 25))
    # # axs[1].legend(loc='upper centre')
    # axis = 0
    # for ax in axs.flat:
    #     if axis == 0:
    #         ax.set_ylabel(r'Distance between successive knots $ (\Delta{\tau}) $', size=8)
    #         ax.set_xlabel(r'Knots $ (\tau) $', size=8)
    #         ax.set_yticks([0, 0.005, 0.01, 0.015])
    #         ax.set_yticklabels([0, 0.005, 0.01, 0.015], size=8)
    #         ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    #         ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], size=8)
    #     if axis == 1:
    #         ax.set_xlabel('t', size=8)
    #         ax.set_ylabel('f', size=8)
    #         ax.set_yticks([0, 50, 100, 150, 200, 250])
    #         ax.set_yticklabels([0, 50, 100, 150, 200, 250], size=8)
    #         ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    #         ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8], size=8)
    #         ax.legend(loc='upper left')
    #     axis += 1
    # plt.gcf().subplots_adjust(top=0.775)
    # plt.savefig('FM_signals/frequency_modulation_knot_and_plot_{}.png'.format(dB))
    # plt.show()

temp = 0