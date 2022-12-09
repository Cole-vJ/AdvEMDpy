
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

sns.set(style='darkgrid')

# load raw data
raw_data = \
    pd.read_csv('Human Mortality Database Data/death_rate',
                header=0, delim_whitespace=True)

for i in np.arange(0, 110, 5):

    uk_male_0 = np.array(raw_data['Male'].loc[raw_data['Age'] == str(i)])
    uk_male_1 = np.array(raw_data['Male'].loc[raw_data['Age'] == str(i + 1)])
    uk_male_2 = np.array(raw_data['Male'].loc[raw_data['Age'] == str(i + 2)])
    uk_male_3 = np.array(raw_data['Male'].loc[raw_data['Age'] == str(i + 3)])
    uk_male_4 = np.array(raw_data['Male'].loc[raw_data['Age'] == str(i + 4)])
    exec("uk_male_" + str(i) + "_" + str(i + 4) + "= uk_male_0 + uk_male_1 + uk_male_2 + uk_male_3 + uk_male_4")

    uk_female_0 = np.array(raw_data['Female'].loc[raw_data['Age'] == str(i)])
    uk_female_1 = np.array(raw_data['Female'].loc[raw_data['Age'] == str(i + 1)])
    uk_female_2 = np.array(raw_data['Female'].loc[raw_data['Age'] == str(i + 2)])
    uk_female_3 = np.array(raw_data['Female'].loc[raw_data['Age'] == str(i + 3)])
    uk_female_4 = np.array(raw_data['Female'].loc[raw_data['Age'] == str(i + 4)])
    exec("uk_female_" + str(i) + "_" + str(i + 4) + "= uk_male_0 + uk_female_1 + uk_female_2 + uk_female_3 + uk_female_4")

del i, uk_male_0, uk_male_1, uk_male_2, uk_male_3, uk_male_4, \
    uk_female_0, uk_female_1, uk_female_2, uk_female_3, uk_female_4

upper_limit = 100

ax = plt.subplot(111)
plt.title(textwrap.fill("United Kingdom Male Deaths in 5 Year Stratifications from 1922 to 2020", 40))
for i in np.arange(0, upper_limit, 5):
    if i < 50:
        exec("plt.plot(np.arange(1922, 2021, 1), uk_male_" + str(i) + "_" + str(i + 4) + ", label='Ages " + str(
            i) + "-" + str(i + 4) + "')")
    else:
        exec("plt.plot(np.arange(1922, 2021, 1), uk_male_" + str(i) + "_" + str(i + 4) + ", '--', label='Ages " + str(
            i) + "-" + str(i + 4) + "')")
plt.xlabel('Years')
plt.ylabel('Deaths')
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.01, box_0.y0 + 0.02, box_0.width * 0.85, box_0.height * 1.0])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/male_deaths.png')
plt.show()

ax = plt.subplot(111)
plt.title(textwrap.fill("United Kingdom Female Deaths in 5 Year Stratifications from 1922 to 2020", 40))
for i in np.arange(0, upper_limit, 5):
    if i < 50:
        exec("plt.plot(np.arange(1922, 2021, 1), uk_female_" + str(i) + "_" + str(i + 4) + ", label='Ages " + str(
            i) + "-" + str(i + 4) + "')")
    else:
        exec("plt.plot(np.arange(1922, 2021, 1), uk_female_" + str(i) + "_" + str(i + 4) + ", '--', label='Ages " + str(
            i) + "-" + str(i + 4) + "')")
plt.xlabel('Years')
plt.ylabel('Deaths')
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.01, box_0.y0 + 0.02, box_0.width * 0.85, box_0.height * 1.0])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/female_deaths.png')
plt.show()

time = np.arange(1922, 2021, 1)
ax = plt.subplot(111)
plt.title('IMF 1 for Each 5 Year Stratified Age Group for Males')
for i in np.arange(0, upper_limit, 5):
    exec("emd = EMD(time=time, time_series=uk_male_" + str(i) + "_" + str(i + 4) + ")")
    exec("imfs, _, ifs = emd.empirical_mode_decomposition(knots=np.arange(1922, 2020, 4), verbose=True, debug=False, "
         "max_internal_iter=10)[:3]")
    # if i < 50:
    #     exec("plt.plot(np.arange(1922, 2021, 1), imfs[1, :], label='Ages " + str(
    #         i) + "-" + str(i + 4) + "')")
    # else:
    #     exec("plt.plot(np.arange(1922, 2021, 1), imfs[1, :], '--', label='Ages " + str(
    #         i) + "-" + str(i + 4) + "')")
    if i < 50:
        pass
    else:
        exec("plt.plot(np.arange(1922, 2021, 1), imfs[1, :], label='Ages " + str(
            i) + "-" + str(i + 4) + "')")
plt.plot(1990 * np.ones(100), np.linspace(3000, 5000, 100), 'k')
plt.plot(np.linspace(1989.5, 1990, 100), np.linspace(3500, 3000, 100), 'k')
plt.plot(np.linspace(1990.5, 1990, 100), np.linspace(3500, 3000, 100), 'k', label=textwrap.fill('World War 1 baby boom', 12))
plt.plot(1995 * np.ones(100), np.linspace(5000, 7000, 100), 'k')
plt.plot(np.linspace(1994.5, 1995, 100), np.linspace(5500, 5000, 100), 'k')
plt.plot(np.linspace(1995.5, 1995, 100), np.linspace(5500, 5000, 100), 'k')
plt.plot(1986 * np.ones(100), np.linspace(2500, 4500, 100), 'k')
plt.plot(np.linspace(1985.5, 1986, 100), np.linspace(3000, 2500, 100), 'k')
plt.plot(np.linspace(1986.5, 1986, 100), np.linspace(3000, 2500, 100), 'k')
plt.plot(1999 * np.ones(100), np.linspace(5000, 7000, 100), 'k')
plt.plot(np.linspace(1998.5, 1999, 100), np.linspace(5500, 5000, 100), 'k')
plt.plot(np.linspace(1999.5, 1999, 100), np.linspace(5500, 5000, 100), 'k')
plt.plot(2005 * np.ones(100), np.linspace(3500, 5500, 100), 'k')
plt.plot(np.linspace(2004.5, 2005, 100), np.linspace(4000, 3500, 100), 'k')
plt.plot(np.linspace(2005.5, 2005, 100), np.linspace(4000, 3500, 100), 'k')
plt.plot(2011 * np.ones(100), np.linspace(3500, 5500, 100), 'k')
plt.plot(np.linspace(2010.5, 2011, 100), np.linspace(4000, 3500, 100), 'k')
plt.plot(np.linspace(2011.5, 2011, 100), np.linspace(4000, 3500, 100), 'k')
plt.plot(2014 * np.ones(100), np.linspace(2000, 4000, 100), 'k')
plt.plot(np.linspace(2013.5, 2014, 100), np.linspace(2500, 2000, 100), 'k')
plt.plot(np.linspace(2014.5, 2014, 100), np.linspace(2500, 2000, 100), 'k')
plt.plot(1979 * np.ones(100), np.linspace(3000, 5000, 100), 'k')
plt.plot(np.linspace(1978.5, 1979, 100), np.linspace(3500, 3000, 100), 'k')
plt.plot(np.linspace(1979.5, 1979, 100), np.linspace(3500, 3000, 100), 'k')
plt.xlabel('Years')
plt.ylabel('Deaths')
plt.xlim(1955, 2025)
plt.ylim(-6000, 16000)
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.01, box_0.y0 + 0.02, box_0.width * 0.85, box_0.height * 1.0])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/male_deaths_ww1_boom.png')
plt.show()

time = np.arange(1922, 2021, 1)
ax = plt.subplot(111)
plt.title('IMF 1 for Each 5 Year Stratified Age Group for Males')
for i in np.arange(0, upper_limit, 5):
    exec("emd = EMD(time=time, time_series=uk_male_" + str(i) + "_" + str(i + 4) + ")")
    exec("imfs, _, ifs = emd.empirical_mode_decomposition(knots=np.arange(1922, 2020, 4), verbose=True, debug=False, "
         "max_internal_iter=10)[:3]")
    if i < 50:
        pass
    else:
        exec("plt.plot(np.arange(1922, 2021, 1) - i + 50, imfs[1, :], label='Ages " + str(
            i) + "-" + str(i + 4) + "')")
plt.xlabel('Born')
plt.ylabel('Deaths')
# plt.xlim(1955, 2025)
plt.xticks([1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020], [1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970], rotation='-45')
plt.ylim(-6000, 16000)
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.01, box_0.y0 + 0.05, box_0.width * 0.85, box_0.height * 0.98])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/male_deaths_ww1_boom_other.png')
plt.show()

ax = plt.subplot(111)
plt.title('IMF 1 for Each 5 Year Stratified Age Group for Females')
for i in np.arange(0, upper_limit, 5):
    exec("emd = EMD(time=time, time_series=uk_female_" + str(i) + "_" + str(i + 4) + ")")
    exec("imfs, _, ifs = emd.empirical_mode_decomposition(knots=np.arange(1922, 2020, 4), verbose=True, debug=False, "
         "max_internal_iter=10)[:3]")
    # if i < 50:
    #     exec("plt.plot(np.arange(1922, 2021, 1), imfs[1, :], label='Ages " + str(
    #         i) + "-" + str(i + 4) + "')")
    # else:
    #     exec("plt.plot(np.arange(1922, 2021, 1), imfs[1, :], '--', label='Ages " + str(
    #         i) + "-" + str(i + 4) + "')")
    if i < 50:
        pass
    else:
        exec("plt.plot(np.arange(1922, 2021, 1), imfs[1, :], label='Ages " + str(
            i) + "-" + str(i + 4) + "')")
plt.plot(1990 * np.ones(100), np.linspace(3000, 5000, 100), 'k')
plt.plot(np.linspace(1989.5, 1990, 100), np.linspace(3500, 3000, 100), 'k')
plt.plot(np.linspace(1990.5, 1990, 100), np.linspace(3500, 3000, 100), 'k', label=textwrap.fill('World War 1 baby boom', 12))
plt.plot(1995 * np.ones(100), np.linspace(4500, 6500, 100), 'k')
plt.plot(np.linspace(1994.5, 1995, 100), np.linspace(5000, 4500, 100), 'k')
plt.plot(np.linspace(1995.5, 1995, 100), np.linspace(5000, 4500, 100), 'k')
plt.plot(1985 * np.ones(100), np.linspace(2000, 4000, 100), 'k')
plt.plot(np.linspace(1984.5, 1985, 100), np.linspace(2500, 2000, 100), 'k')
plt.plot(np.linspace(1985.5, 1985, 100), np.linspace(2500, 2000, 100), 'k')
plt.plot(1999 * np.ones(100), np.linspace(4500, 6500, 100), 'k')
plt.plot(np.linspace(1998.5, 1999, 100), np.linspace(5000, 4500, 100), 'k')
plt.plot(np.linspace(1999.5, 1999, 100), np.linspace(5000, 4500, 100), 'k')
plt.plot(2005 * np.ones(100), np.linspace(4000, 6000, 100), 'k')
plt.plot(np.linspace(2004.5, 2005, 100), np.linspace(4500, 4000, 100), 'k')
plt.plot(np.linspace(2005.5, 2005, 100), np.linspace(4500, 4000, 100), 'k')
plt.plot(2010 * np.ones(100), np.linspace(2500, 4500, 100), 'k')
plt.plot(np.linspace(2009.5, 2010, 100), np.linspace(3000, 2500, 100), 'k')
plt.plot(np.linspace(2010.5, 2010, 100), np.linspace(3000, 2500, 100), 'k')
plt.plot(2014 * np.ones(100), np.linspace(4500, 6500, 100), 'k')
plt.plot(np.linspace(2013.5, 2014, 100), np.linspace(5000, 4500, 100), 'k')
plt.plot(np.linspace(2014.5, 2014, 100), np.linspace(5000, 4500, 100), 'k')
plt.plot(1979 * np.ones(100), np.linspace(2000, 4000, 100), 'k')
plt.plot(np.linspace(1978.5, 1979, 100), np.linspace(2500, 2000, 100), 'k')
plt.plot(np.linspace(1979.5, 1979, 100), np.linspace(2500, 2000, 100), 'k')
plt.xlabel('Years')
plt.ylabel('Deaths')
plt.xlim(1955, 2025)
plt.ylim(-6000, 16000)
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.01, box_0.y0 + 0.02, box_0.width * 0.85, box_0.height * 1.0])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/female_deaths_ww1_boom.png')
plt.show()

time = np.arange(1922, 2021, 1)
ax = plt.subplot(111)
plt.title('IMF 1 for Each 5 Year Stratified Age Group for Females')
for i in np.arange(0, upper_limit, 5):
    exec("emd = EMD(time=time, time_series=uk_female_" + str(i) + "_" + str(i + 4) + ")")
    exec("imfs, _, ifs = emd.empirical_mode_decomposition(knots=np.arange(1922, 2020, 4), verbose=True, debug=False, "
         "max_internal_iter=10)[:3]")
    if i < 50:
        pass
    else:
        exec("plt.plot(np.arange(1922, 2021, 1) - i + 50, imfs[1, :], label='Ages " + str(
            i) + "-" + str(i + 4) + "')")
plt.xlabel('Born')
plt.ylabel('Deaths')
# plt.xlim(1955, 2025)
plt.xticks([1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020], [1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970], rotation='-45')
plt.ylim(-6000, 16000)
box_0 = ax.get_position()
ax.set_position([box_0.x0 + 0.01, box_0.y0 + 0.05, box_0.width * 0.85, box_0.height * 0.98])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
plt.savefig('Real-World Figures/female_deaths_ww1_boom_other.png')
plt.show()

temp = 0
