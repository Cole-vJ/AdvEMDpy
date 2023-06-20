
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from mpl_toolkits.mplot3d import Axes3D
import textwrap

from emd_hilbert import Hilbert, hilbert_spectrum  # stft_custom, morlet_wavelet_custom
from AdvEMDpy import EMD


def generate_garch_noise(n, alpha, beta, omega=0):

    noise = np.zeros(n)
    variance = np.zeros(n)
    random_normal_noise = np.random.normal(0, 1, n)

    for i in range(1, n):

        variance[i] = omega + alpha * noise[i - 1] ** 2 + beta * variance[i - 1]
        noise[i] = np.sqrt(variance[i]) * random_normal_noise[i]

    return noise


np.random.seed(0)

begin = 0
end = 1
points = int(7.5 * 512)

x = np.linspace(begin, end, points)

signal_1 = np.sin(250 * np.pi * x ** 2)
signal_2 = np.sin(80 * np.pi * x ** 2)

signal = signal_1 + signal_2

alpha = 0.25
beta = 0.25
y = generate_garch_noise(n=points, alpha=alpha, beta=beta, omega=0.1)
plt.plot(y)
plt.show()

v1 = np.arange(1.0, -0.01, -0.025)
v2 = np.arange(1.0, -0.01, -0.025)

v1_mesh, v2_mesh = np.meshgrid(v1, v2)
kurtosis_surface = np.zeros_like(v1_mesh)
theory_kurtosis_surface =  np.zeros_like(v1_mesh)

x_points = np.arange(0, 1.1, 0.1)
x_names = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

for num_i, i in enumerate(v1):
    # print(num_i)
    for num_j, j in enumerate(v2):
        garch_noise = generate_garch_noise(n=points, alpha=i, beta=j, omega=0.1)
        kurt = kurtosis(garch_noise)
        kurtosis_surface[num_i, num_j] = kurt
        # print(kurt)
        theory_kurtosis_surface[num_i, num_j] = 3 / (1 - 2 * ((i ** 2) / (1 - (i + j) ** 2))) - 3
        # print(theory_kurtosis_surface[num_i, num_j])

lim = 10

kurtosis_surface[kurtosis_surface > lim ] = lim
kurtosis_surface[kurtosis_surface < 0] = 0
kurtosis_surface[np.isnan(kurtosis_surface)] = lim
theory_kurtosis_surface[theory_kurtosis_surface > lim ] = lim
theory_kurtosis_surface[theory_kurtosis_surface < 0] = lim

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = plt.axes(projection='3d')
ax.view_init(30, -30)
ax.set_title(r'Truncated Excess Kurtosis Surface of GARCH($1$,$1$)', fontsize=10)
cov_plot = ax.plot_surface(v1_mesh, v2_mesh, kurtosis_surface, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none')
ax.set_xticks(ticks=[1, 0])
ax.set_xticklabels(labels=['1', '0'], fontsize=8, ha="left", rotation_mode="anchor")
ax.set_xlabel(r'$\alpha$', fontsize=10)
ax.set_yticks(ticks=[0, 1])
ax.set_yticklabels(labels=[0, 1], rotation=0, fontsize=8)
ax.set_ylabel(r'$\beta$', fontsize=10)
ax.set_zticks(ticks=[0, lim])
ax.set_zticklabels([0, lim], fontsize=8)
ax.set_zlabel('Kurtosis', fontsize=8, rotation=180)
ax.set_zlim(-1.0, lim + 1.0)
plt.colorbar(cov_plot)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0, box_0.width, box_0.height])
plt.savefig('Figures/truncated_excess_kurtosis_surface_GARCH_11.pdf')
plt.show()

fig = plt.figure()
fig.set_size_inches(8, 6)
ax = plt.axes(projection='3d')
ax.view_init(30, -30)
ax.set_title(r'Truncated Theoretical Excess Kurtosis Surface of GARCH($1$,$1$)', fontsize=10)
cov_plot = ax.plot_surface(v1_mesh, v2_mesh, theory_kurtosis_surface, rstride=1, cstride=1, cmap='gist_rainbow', edgecolor='none')
ax.set_xticks(ticks=[1, 0])
ax.set_xticklabels(labels=['1', '0'], fontsize=8, ha="left", rotation_mode="anchor")
ax.set_xlabel(r'$\alpha$', fontsize=10)
ax.set_yticks(ticks=[0, 1])
ax.set_yticklabels(labels=[0, 1], rotation=0, fontsize=8)
ax.set_ylabel(r'$\beta$', fontsize=10)
ax.set_zticks(ticks=[0, lim])
ax.set_zticklabels([0, lim], fontsize=8)
ax.set_zlabel('Kurtosis', fontsize=8, rotation=180)
ax.set_zlim(-1.0, lim + 1.0)
plt.colorbar(cov_plot)
box_0 = ax.get_position()
ax.set_position([box_0.x0 - 0.05, box_0.y0, box_0.width, box_0.height])
plt.savefig('Figures/truncated_theory_kurtosis_surface_GARCH_11.pdf')
plt.show()

normal = np.random.normal(0, 1, points * 1000)
print('Excess Kurtosis: {}'.format(kurtosis(y)))

signal += y

x_points = np.arange(0, 1.1, 0.1)
x_names = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

plt.figure(2)
ax = plt.subplot(111)
plt.title(r'Frequency Modulation Example with GARCH($1$,$1$) Noise')
plt.plot(x, signal)
plt.xticks(x_points, x_names)
plt.ylabel('g(t)')
plt.xlabel('t')

box_0 = ax.get_position()
# print(box_0)
ax.set_position([box_0.x0 + 0.05, box_0.y0 + 0.05, box_0.width * 0.9, box_0.height * 0.9])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Figures/frequency_modulation_garch.pdf')
plt.show()

# stft

hilbert_stft = Hilbert(time=x, time_series=signal)
t_stft, f_stft, z_stft = hilbert_stft.stft_custom(window_width=512, angular_frequency=True)
f_stft = f_stft / (2 * np.pi)

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.10)
plt.pcolormesh((t_stft - (1 / (3840 / 128))), f_stft, np.abs(z_stft), vmin=0, vmax=np.max(np.max(np.abs(z_stft))))
plt.plot(t_stft[:-1], 250 * t_stft[:-1], 'b--', label='f(t) = 250t', LineWidth=2)
plt.plot(t_stft[:-1], 80 * t_stft[:-1], 'g--', label='f(t) = 80t', LineWidth=2)
plt.title(r'STFT - $ |\mathcal{STF}(g(t))(t,f)|^2 $')
plt.ylabel('f')
plt.xlabel('t')
plt.xticks(x_points, x_names)
plt.ylim(0, 260)

box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.0125, box_0.y0 + 0.075, box_0.width * 0.8, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Figures/frequency_modulation_stft_garch.pdf')
plt.show()

# morlet wavelet

hilbert_mwt = Hilbert(time=x, time_series=signal)
t_morlet, f_morlet, z_morlet = hilbert_mwt.morlet_wavelet_custom(window_width=512, angular_frequency=True)
f_morlet = f_morlet / (2 * np.pi)
# z_morlet[0, :] = z_morlet[1, :]

ax = plt.subplot(111)
plt.gcf().subplots_adjust(bottom=0.10)
plt.pcolormesh((t_morlet - (1 / (3840 / 128))), f_morlet, np.abs(z_morlet), vmin=0, vmax=np.max(np.max(np.abs(z_morlet))))
plt.plot(t_morlet[:-1], 250 * t_morlet[:-1], 'b--', label='f(t) = 250t', LineWidth=2)
plt.plot(t_morlet[:-1], 80 * t_morlet[:-1], 'g--', label='f(t) = 80t', LineWidth=2)
plt.title(r'Morlet - $ |(g(t))(t,f)|^2 $')
plt.ylabel('f')
plt.xlabel('t')
plt.xticks(x_points, x_names)
plt.ylim(0, 260)

box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.0125, box_0.y0 + 0.075, box_0.width * 0.8, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Figures/frequency_modulation_morlet_garch.pdf')
plt.show()

# joint plot - top

fig, axs = plt.subplots(1, 2)
plt.subplots_adjust(hspace=0.5)
axs[0].pcolormesh((t_stft - (1 / (3840 / 128))), f_stft, np.abs(z_stft), vmin=0, vmax=np.max(np.max(np.abs(z_stft))))
axs[0].plot(t_morlet[:-1], 250 * t_morlet[:-1], 'b--', label='f(t) = 250t', LineWidth=2)
axs[0].plot(t_morlet[:-1], 80 * t_morlet[:-1], 'g--', label='f(t) = 80t', LineWidth=2)
axs[0].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axs[0].set_xticklabels(['$0$', '$0.2$', '$0.4$', '$0.6$', '$0.8$', '$1.0$'])
axs[1].pcolormesh((t_morlet - (1 / (3840 / 128))), f_morlet, np.abs(z_morlet), vmin=0, vmax=np.max(np.max(np.abs(z_morlet))))
axs[1].plot(t_morlet[:-1], 250 * t_morlet[:-1], 'b--', label='f(t) = 250t', LineWidth=2)
axs[1].plot(t_morlet[:-1], 80 * t_morlet[:-1], 'g--', label='f(t) = 80t', LineWidth=2)
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
plt.savefig('Figures/frequency_modulation_STFT_Morlet_garch.pdf')
plt.show()

# joint plot - bottom

# emd

knots = np.linspace(begin, end, int(points / 10))

emd_384 = EMD(time=x, time_series=signal)
imfs, _, _, _, _, hts, ifs = emd_384.empirical_mode_decomposition(knots=knots, matrix=True, max_internal_iter=20,
                                                                  mean_threshold=10, stopping_criterion_threshold=10,
                                                                  dtht=True)

for i in range(np.shape(imfs)[0]):
    plt.plot(imfs[i, :])
    plt.show()

hs_ouputs = hilbert_spectrum(x, imfs, hts, ifs, max_frequency=260 * 2 * np.pi)

x_hs_emd_384, y_emd_384, z_emd_384 = hs_ouputs
y_emd_384 = y_emd_384 / (2 * np.pi)

z_min, z_max = 0, np.abs(z_emd_384).max()
fig, ax = plt.subplots()
ax.pcolormesh(x_hs_emd_384, y_emd_384, z_emd_384, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
plt.plot(t_stft, 250 * t_stft, 'b--', label='f(t) = 250t', LineWidth=2)
plt.plot(t_stft, 80 * t_stft, 'g--', label='f(t) = 80t', LineWidth=2)
ax.set_title(f'Gaussian Filtered Hilbert Spectrum - {int(points / 10)} Knots')
ax.set_xlabel('t')
ax.set_ylabel('f')
ax.axis([x_hs_emd_384.min(), x_hs_emd_384.max(), y_emd_384.min(), y_emd_384.max()])

box_0 = ax.get_position()
ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.75, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Figures/frequency_modulation_emd_384_garch.png')
plt.show()

knots = np.linspace(begin, end, int(points / 5))

emd_768 = EMD(time=x, time_series=signal)
imfs, _, _, _, _, hts, ifs = emd_768.empirical_mode_decomposition(knots=knots, matrix=True, max_internal_iter=20,
                                                                  mean_threshold=10, stopping_criterion_threshold=10,
                                                                  dtht=True)

for i in range(np.shape(imfs)[0]):
    plt.plot(imfs[i, :])
    plt.show()

hs_ouputs = hilbert_spectrum(x, imfs, hts, ifs, max_frequency=260 * 2 * np.pi)

x_hs_emd_768, y_emd_768, z_emd_768 = hs_ouputs
y_emd_768 = y_emd_768 / (2 * np.pi)

z_min, z_max = 0, np.abs(z_emd_768).max()
fig, ax = plt.subplots()
ax.pcolormesh(x_hs_emd_768, y_emd_768, z_emd_768, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
plt.plot(t_stft, 250 * t_stft, 'b--', label='f(t) = 250t', LineWidth=2)
plt.plot(t_stft, 80 * t_stft, 'g--', label='f(t) = 80t', LineWidth=2)
ax.set_title(f'Gaussian Filtered Hilbert Spectrum - {int(points / 5)} Knots')
ax.set_xlabel('t')
ax.set_ylabel('f')
ax.axis([x_hs_emd_768.min(), x_hs_emd_768.max(), y_emd_768.min(), y_emd_768.max()])

box_0 = ax.get_position()
ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.75, box_0.height * 0.9])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Figures/frequency_modulation_emd_768_garch.png')
plt.show()

# joint plot - top

fig, axs = plt.subplots(1, 2)
plt.subplots_adjust(hspace=0.5, top=0.8)
axs[0].pcolormesh(x_hs_emd_384, y_emd_384, z_emd_384, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
axs[0].plot(t_stft, 250 * t_stft, 'b--', label='f(t) = 250t', LineWidth=2)
axs[0].plot(t_stft, 80 * t_stft, 'g--', label='f(t) = 80t', LineWidth=2)
axs[0].set_title(textwrap.fill(f'Gaussian Filtered Hilbert Spectrum - {int(points / 10)} Knots', 20))
axs[0].axis([x_hs_emd_384.min(), x_hs_emd_384.max(), y_emd_384.min(), y_emd_384.max()])
axs[1].pcolormesh(x_hs_emd_768, y_emd_768, z_emd_768, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
axs[1].plot(t_stft, 250 * t_stft, 'b--', label='f(t) = 250t', LineWidth=2)
axs[1].plot(t_stft, 80 * t_stft, 'g--', label='f(t) = 80t', LineWidth=2)
axs[1].set_title(textwrap.fill(f'Gaussian Filtered Hilbert Spectrum - {int(points / 5)} Knots', 20))
axs[1].axis([x_hs_emd_768.min(), x_hs_emd_768.max(), y_emd_768.min(), y_emd_768.max()])

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
plt.savefig('Figures/frequency_modulation_emd_384_768_garch.png')
plt.show()
