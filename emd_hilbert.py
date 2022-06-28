
#     ________
#            /
#      \    /
#       \  /
#        \/

# emd class to handle all Hilbert transform related methods

import numpy as np
import scipy as sp
import seaborn as sns
from scipy import signal as sig
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

sns.set(style='darkgrid')


def theta(real: np.ndarray, imaginary: np.ndarray) -> np.ndarray:
    """
    Function for extracting phase of analytical signal.

    Parameters
    ----------
    real : real ndarray
        Real component of signal.

    imaginary : real ndarray
        Imaginary component of signal.

    Returns
    -------
    phase : real ndarray
        Phase of analytical signal.

    Notes
    -----

    """
    phase = my_arctan(imaginary / real)

    return phase


def omega(x: np.ndarray, theta_input: np.ndarray) -> np.ndarray:
    """
    Function for calculating instantaneous frequency.

    Parameters
    ----------
    x : real ndarray
        x co-ordinate.

    theta_input : real ndarray
        Phase of signal.

    Returns
    -------
    inst_freq : real ndarray
        Instantaneous frequency.

    Notes
    -----

    """
    inst_freq = (np.array(theta_input[1:]) - np.array(theta_input[:-1])) / (np.array(x[1:]) - np.array(x[:-1]))

    return inst_freq


def my_arctan(y: np.ndarray) -> np.ndarray:
    """
    Custom arctan function to prevent discontinuities.

    Parameters
    ----------
    y : real ndarray
        Imaginary/real components of signal.

    Returns
    -------
    custom_arctan : real ndarray
        Continuous arctan of continuous signal.

    Notes
    -----

    """
    cycle_count = (y[1:] - y[0:-1])
    cycle_count = (cycle_count < -1)
    count = np.cumsum(np.append(0, cycle_count))

    custom_arctan = np.arctan(y) + count * np.pi

    return custom_arctan


def hilbert_spectrum(time: np.ndarray, imf_storage: np.ndarray, ht_storage: np.ndarray, if_storage: np.ndarray,
                     max_frequency: float, freq_increments: int = 1001, which_imfs: str = 'all', plot: bool = True,
                     filter_spectrum: bool = True, filter_sigma: int = 5) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Produce Hilbert spectrum using outputs of empirical_mode_decomposition().

    Parameters
    ----------
    time : real ndarray
        Time corresponding to IMFs.

    imf_storage : real ndarray
        Stores IMFs - smoothed (or unsmoothed) signal in first row, IMFs in consecutive rows, and trend in last row.

    ht_storage : real ndarray
        Stores HT of IMFs - HT of smoothed (or unsmoothed) signal in first row (for consistency),
        HT of IMFs in consecutive rows, and HT of trend in last row (for consistency).

    if_storage : real ndarray
        Stores IF of IMFs - IF of smoothed (or unsmoothed) signal in first row (for consistency),
        IF of IMFs in consecutive rows, and IF of trend in last row (for consistency).

    max_frequency : float
        Maximum frequency resolution to be displayed.

    freq_increments : integer
        Number of frequency increments to use in frequency resolution.

    which_imfs : string_like or List[int]
        Which IMFs to use in creating Hilbert spectrum:
            'all' : uses all the IMFs
            list of indices : uses only specified IMFs - example : [1, 2, 3].

    plot : bool
        Whether to plot during execution of function for debugging purposes.

    filter_spectrum : bool
        Whether to filter the Hilbert spectrum using a Gaussian filter.

    filter_sigma : float
        If Gaussian filtering - determines the standard deviation of Gaussian filter.

    Returns
    -------
    x : real ndarray
        Meshgrid of time values for Hilbert spectrum plotting.

    y : real ndarray
        Meshgrid of frequency values for Hilbert spectrum plotting.

        z : real ndarray
            Unfiltered Hilbert spectrum.

        or

        filtered_z : real ndarray
            Filtered Hilbert spectrum.

    Notes
    -----

    """
    # generate mesh grids for the x & y bounds
    x, y = np.meshgrid(time[:-1], np.linspace(0, max_frequency, freq_increments))

    z = np.zeros(np.shape(x))
    if which_imfs == 'all':
        range_to_return = range(1, np.shape(imf_storage)[0] - 1)
    else:
        range_to_return = which_imfs
    for k in range_to_return:  # not 1st or last - 1st is smoothed signal - last is trend
        for i in range(len(time) - 1):  # left Riemann approximation in a sense
            for j in range(freq_increments - 1):  # also left Riemann approximation in a sense
                if y[j, 0] <= if_storage[k, :][i] < y[j + 1, 0]:  # if instantaneous frequency in range
                    z[j, i] += np.sqrt((imf_storage[k, :][i]) ** 2 + (ht_storage[k, :][i]) ** 2)
                    # assign corresponding instantaneous amplitude to correct z value

    z = z[:-1, :-1]

    if plot:
        z_min, z_max = 0, np.abs(z).max()
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, z, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
        ax.set_title('Raw Hilbert Spectrum')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.axis([x.min(), x.max(), y.min(), y.max()])

        box_0 = ax.get_position()
        ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.9, box_0.height * 0.9])

        plt.show()

    filtered_z = {}  # avoids unnecessary error

    if filter_spectrum:
        filtered_z = gaussian_filter(z, sigma=filter_sigma)

    if plot and filter_spectrum:
        z_min, z_max = 0, np.abs(filtered_z).max()
        fig, ax = plt.subplots()
        ax.pcolormesh(x, y, filtered_z, cmap='gist_rainbow', vmin=z_min, vmax=z_max)
        ax.set_title('Gaussian Filtered Hilbert Spectrum')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        ax.axis([x.min(), x.max(), y.min(), y.max()])

        box_0 = ax.get_position()
        ax.set_position([box_0.x0, box_0.y0 + 0.05, box_0.width * 0.9, box_0.height * 0.9])

        plt.show()

    if filter_spectrum:
        return x, y, filtered_z
    else:
        return x, y, z


def morlet_window(width: int, sigma: float) -> np.ndarray:
    """
    Unadjusted Morlet window function.

    Parameters
    ----------
    width : integer (positive power of 2)
        Window width to use - power of two as window of two corresponds to Nyquist rate.

    sigma : float
        Corresponds to the frequency of the frequency of the wavelet.

    Returns
    -------
    output : real ndarray
        Normalised Morlet wavelet vector.

    Notes
    -----
    https://en.wikipedia.org/wiki/Morlet_wavelet

    """
    # fixed width wavelet translates to a fixed width Fourier transformed wavelet in frequency spectrum

    # Definition - https://en.wikipedia.org/wiki/Morlet_wavelet
    c_pi = (1 + np.exp(- sigma ** 2) - 2 * np.exp(- 0.75 * sigma ** 2)) ** (-1 / 2)
    t = (np.arange(width + 1) - (width / 2)) * (10 / width)
    wavelet = c_pi * (np.pi ** (-1 / 4)) * (np.exp(1j * sigma * t) - np.exp(- (1 / 2) * sigma ** 2))
    output = np.exp(- (1 / 2) * t ** 2) * wavelet.real

    return output


def morlet_window_adjust(width: int, sigma: float, cycles: float, significance: float) -> np.ndarray:
    """
    Adjusted Morlet window function.

    Parameters
    ----------
    width : integer (positive power of 2)
        Window width to use - power of two as window window of two corresponds to Nyquist rate.

    sigma : float
        Corresponds to the frequency of the frequency of the wavelet.

    cycles : float
        Number of full cycles to be within significance value - higher frequency wavelet will taper off more quickly.

    significance : float
        Significance corresponding to confidence interval of Normal distribution.

    Returns
    -------
    output : real ndarray
        Normalised adjusted Morlet wavelet vector.

    Notes
    -----
    https://en.wikipedia.org/wiki/Morlet_wavelet

    There will be 50 cycles of whatever frequency signal within 99.7% confidence bound of Normal Districbution.
    This statement is accurate within a normalisation constant.

    """

    # adjustable wavelet where 'cycles' and 'significance' adjusts Gaussian window relative to cycles in wavelet
    # this assists by improving frequency resolution at low frequencies (at the expense of low time resolution)
    # improves time resolution at high frequencies (at the expense of low frequency resolution)

    # Definition - https://en.wikipedia.org/wiki/Morlet_wavelet
    c_pi = (1 + np.exp(- sigma ** 2) - 2 * np.exp(- 0.75 * sigma ** 2)) ** (-1 / 2)
    t = (np.arange(width + 1) - (width / 2)) * (10 / width)  # domain of [-5, 5]
    wavelet = c_pi * (np.pi ** (-1 / 4)) * (np.exp(1j * sigma * t) - np.exp(- (1 / 2) * sigma ** 2))
    output = np.exp(
        - (1 / 2) * (((significance / 5) / (cycles / (sigma * (10 / (2 * np.pi))))) * t) ** 2) * wavelet.real

    return output


class Hilbert:

    def __init__(self, time: np.ndarray, time_series: np.ndarray):

        self.time = time
        self.time_series = time_series

    def dtht_kak(self) -> np.array:
        """
        Basic discrete-time Hilbert transform.

        Returns
        -------
        dtht : real ndarray
            Basic discrete-time Hilbert transform approximation.

        Notes
        -----
        S Kak. The discrete hilbert transform. Proceedings of the IEEE, 58(4):585–586, 1970.

        This out performs the inbuilt Python FFT DTHT method.

        """
        time_series = self.time_series

        all_odds = np.arange(1, len(time_series), 2)
        all_evens = np.arange(0, len(time_series), 2)

        dtht = np.zeros_like(time_series)

        for i in range(len(dtht)):

            if i % 2 == 0:
                f = time_series[all_odds]
                g = (2 / np.pi) / (i - all_odds)
            else:
                f = time_series[all_evens]
                g = (2 / np.pi) / (i - all_evens)

            dtht[i] = np.sum(f * g)

        return dtht

    def dtht_fft(self) -> np.array:
        """
        Fast-Fourier transform discrete-time Hilbert transform (FFT-DTHT).

        Returns
        -------
        dtht : real ndarray
            FFT discrete-time Hilbert transform approximation.

        Notes
        -----
        Simply calls inbuilt Python FFT DTHT method.
        Inbuilt code is easy to follow and differs only slightly between even and odd length vectors.

        """
        time_series = self.time_series

        dtht = sig.hilbert(time_series).imag

        return dtht

    def stft_custom(self, window: str = 'hann', window_width: int = 256, plot: bool = False,
                    angular_frequency: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Short-time Fourier transform signal.

        Parameters
        ----------
        window : string_like
            Window to use when tapering section of signal before transform.

        window_width : integer (positive power of 2)
            Window width to use when tapering signal - power of two as window window of two corresponds to Nyquist rate.

        plot : boolean
            Plot output as part pf debugging.

        angular_frequency: boolean
            Display angular frequency or standard frequency.

        Returns
        -------
        t : real ndarray
            Meshgrid of time corresponding to time of transformed signal.

        f : real ndarray
            Meshgrid of frequency corresponding to frequency of transformed signal.

        z : real ndarray
            Short-time Fourier transform of signal.

        Notes
        -----
        Output corresponds to angular frequency.

        """
        time = self.time
        signal = self.time_series

        # normalise and left Riemann sum approximate window
        window_func = sig.get_window(window=window, Nx=(window_width + 1), fftbins=False)
        window_func = window_func[:-1] / np.sum(window_func[:-1])

        # how much must be added to create an integer multiple of window length
        addition = int(int(window_width / 2) - int((len(signal) + window_width) % int(window_width / 2)))

        # integer multiple of window length
        signal_new = np.hstack((np.zeros(int(window_width / 2)), signal, np.zeros(int(window_width / 2 + addition))))

        # storage of windowed Fourier transform
        z = np.zeros((int(len(signal_new) / (window_width / 2) - 1), int(window_width)))

        # calculate sampling rate for frequency
        sampling_rate = len(time) / (time[-1] - time[0])
        # sampling_rate = len(time) / np.pi

        for row in range(np.shape(z)[0]):
            # multiply window_func onto interval 'row'
            z[row, :] = window_func * signal_new[int(row * (window_width / 2)):int((row + 2) * (window_width / 2))]
        # real Fourier transform the matrix
        z = np.transpose(sp.fft.rfft(z, n=window_width))
        # calculate frequency vector
        if angular_frequency:
            constant = 2 * np.pi
        else:
            constant = 1
        f = constant * np.linspace(0, (sampling_rate / 2), int(window_width / 2 + 1))
        # calculate time vector
        t = np.linspace(time[0], time[-1] + (time[-1] - time[0]) * (addition / len(signal)),
                        int(len(signal_new) / (window_width / 2) - 1))

        if plot:
            plt.pcolormesh(t, f, np.abs(z), vmin=0, vmax=np.max(np.max(np.abs(z))))
            plt.ylabel('f')
            plt.xlabel('t')
            plt.show()

        return t, f, z

    def morlet_wavelet_custom(self, window_width: int = 256, adjust: bool = True, cycles: float = 50,
                              significance: float = 3,
                              angular_frequency: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Morlet wavelet transform signal.

        Parameters
        ----------
        window_width : integer (positive power of 2)
            Window width - power of two as window window of two corresponds to Nyquist rate.

        adjust : bool
            Whether to use morlet_window() or morlet_window_adjust().

        cycles : float
            Number of full cycles to be within significance value -
            higher frequency wavelet will taper off more quickly.

        significance : float
            Significance corresponding to confidence interval of Normal distribution.

        angular_frequency: boolean
            Display angular frequency or standard frequency.

        Returns
        -------
        t : real ndarray
            Meshgrid of time corresponding to time of transformed signal.

        f : real ndarray
            Meshgrid of frequency corresponding to frequency of transformed signal.

        z : real ndarray
            Morlet wavelet transform of signal.

        Notes
        -----
        Output corresponds to angular frequency.

        """
        time = self.time
        signal = self.time_series

        # how much must be added to create an integer multiple of window length
        addition = int(int(window_width / 2) - int((len(signal) + window_width) % int(window_width / 2)))

        # integer multiple of window length
        signal_new = np.hstack((np.zeros(int(window_width / 2)), signal, np.zeros(int(window_width / 2 + addition))))

        # storage of windowed Morlet transform
        z = np.zeros((int(len(signal_new) / (window_width / 2) - 1), int((window_width / 2) + 1)))

        # calculate sampling rate for frequency
        sampling_rate = len(time) / (time[-1] - time[0])
        # sampling_rate = len(time) / np.pi

        for row in range(np.shape(z)[0]):
            for col in range(np.shape(z)[1]):

                # This takes advantage of the relationship between Fourier transform and convolution
                # Rather than convoluting the Morlet wavelet and the signal
                # Fourier transform Morlet wavelet and section of signal
                # Then multiply them together and inverse Fourier transform the product

                # Fourier transform window of signal
                temp_z = np.fft.fft(signal_new[int(row * (window_width / 2)):int((row + 2) * (window_width / 2))],
                                    n=window_width)
                # calculate Morlet window function
                if adjust:
                    if col == 0:
                        window_morlet = np.zeros(window_width)
                    else:
                        window_morlet = morlet_window_adjust(window_width, (col * (2 * np.pi) / 10), cycles,
                                                             significance)[:-1]
                else:
                    window_morlet = morlet_window(window_width, (col * (2 * np.pi) / 10))[:-1]
                # Fourier transform Morlet window function
                fft_window_morlet = np.abs(np.fft.fft(window_morlet))
                # multiply onto Fourier transformed signal
                temp_z *= fft_window_morlet
                # normalise Morlet wavelets in frequency spectrum - not as important with fixed window
                temp_z *= (1 / np.max(fft_window_morlet))
                # inverse Fourier transform the product
                temp_z = sp.fft.ifft(temp_z, n=window_width)
                # fill appropriate z[row, col] value
                z[row, col] = sum(np.abs(temp_z))

        z = np.transpose(z)
        # calculate frequency vector
        if angular_frequency:
            constant = 2 * np.pi
        else:
            constant = 1
        f = constant * np.linspace(0, (sampling_rate / 2), int(window_width / 2 + 1))
        # calculate time vector
        t = np.linspace(time[0], time[-1] + (time[-1] - time[0]) * (addition / len(signal)),
                        int(len(signal_new) / (window_width / 2) - 1))
        z[0, :] = z[1, :]

        return t, f, z
