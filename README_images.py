
#     ________
#            /
#      \    /
#       \  /
#        \/

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from AdvEMDpy import EMD

sns.set(style='darkgrid')

sample_time = np.linspace(0, 5 * np.pi, 1001)
sample_time_series = np.cos(sample_time) + np.cos(5 * sample_time)

sample_knots = np.linspace(0, 5 * np.pi, 51)
sample_knot_time = np.linspace(0, 5 * np.pi, 1001)

emd = EMD(time=sample_time, time_series=sample_time_series)
imfs, hts, ifs, _, _, _, _ = emd.empirical_mode_decomposition(knots=sample_knots, knot_time=sample_knot_time,
                                                              edge_effect='characteristic_wave_Huang')

plt.figure(1)
plt.title('Sample EMD of First Component')
plt.plot(sample_time, np.cos(5 * sample_time))
plt.plot(sample_time, imfs[1, :], '--')
plt.savefig('README_Images/Figure_1.png')
plt.show()

plt.figure(2)
plt.title('Sample EMD of Second Component')
plt.plot(sample_time, np.cos(sample_time))
plt.plot(sample_time, imfs[2, :], '--')
plt.savefig('README_Images/Figure_2.png')
plt.show()
