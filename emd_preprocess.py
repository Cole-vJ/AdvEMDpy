
#     ________
#            /
#      \    /
#       \  /
#        \/

# emd class to handle all preprocessing

import numpy as np
import cvxpy as cvx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal as sig

sns.set(style='darkgrid')


def henderson_kernel(order: int = 13, start: int = np.nan, end: int = np.nan) -> np.ndarray:
    """
    RKHS re-derivation of classical Henderson smoothing weights with easily translatable asymmetric weighting.

    Parameters
    ----------
    order : integer
        Width of weighting vector.

    start : integer
        Starting point relative to central point of order width.

    end : integer
        End point relative to central point of order width.

    Returns
    -------
    y : real ndarray
        Henderson kernel weighting.

    Notes
    -----
    To be used in hw method.

    """
    if np.isnan(start):
        start = -int((order - 1) / 2)
    if np.isnan(end):
        end = int((order - 1) / 2)
    t = np.asarray(range(start, end + 1)) / (int((order - 1) / 2) + 1)
    # exact Henderson Kernel - differs slightly from classical Henderson smoother
    y = (15 / 79376) * (5184 - 12289 * t ** 2 + 9506 * t ** 4 - 2401 * t ** 6) * (
                (2175 / 1274) - (1372 / 265) * t ** 2)

    y = y / sum(y)  # renormalise when incomplete - does nothing when complete as weights sum to zero

    return y


def henderson_weights(order: int = 13, start: int = np.nan, end: int = np.nan) -> np.ndarray:
    """
    Classical Henderson smoothing weights.

    Parameters
    ----------
    order : integer
        Width of weighting vector.

    start : integer
        Starting point relative to central point of order width.

    end : integer
        End point relative to central point of order width.

    Returns
    -------
    y : real ndarray
        Henderson weighting.

    Notes
    -----
    To be used in hw method.

    """
    if np.isnan(start):
        start = -int((order - 1) / 2)
    if np.isnan(end):
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


class Preprocess:
    """
    Preprocess time series.

    Parameters
    ----------
    time : array_like
        Time corresponding to time series to be preprocessed.

    time_series : array_like
        Time series to be preprocessed.

    Notes
    -----

    """
    def __init__(self, time: np.ndarray, time_series: np.ndarray):

        self.time = time
        self.time_series = time_series

    def downsample(self, window_function: str = 'hamming', decimation_level: int = 20,
                   window_factor: int = 20, decimate: bool = True) -> (np.ndarray, np.ndarray):
        """
        Downsample time series.

        Parameters
        ----------
        window_function : string_like
            Window function to use when downsampling.

        decimation_level : integer
            Level of decimation - modular of index values to keep.

        window_factor : integer
            Used in calculation of window width to be used in downsampling.

        decimate : bool
            Whether or not to decimate.

        Returns
        -------
        decimated_time : real ndarray
            Decimated time.

        downsampled_and_decimated_time_series : real ndarray
            Downsampled and decimated time series.

        Notes
        -----
        Two stages: (1) Downsample
                    (2) Decimate

        """
        window_length = int(window_factor * decimation_level)
        window_half_length = int(window_length / 2)
        window_function = sig.get_window(window=window_function, Nx=window_length + 1, fftbins=False)
        unnormalised_window = window_function * np.sinc((np.linspace(0, window_length, window_length + 1) -
                                                         window_half_length) / decimation_level)
        window_function_calculation = unnormalised_window / sum(unnormalised_window)

        downsampled_time_series = np.zeros_like(self.time_series)

        # downsample

        for t in range(len(self.time_series)):

            if window_half_length <= t <= (len(self.time_series) - window_half_length - 1):
                downsampled_time_series[t] = sum(
                    self.time_series[int(t - window_half_length):int(t + window_half_length + 1)] *
                    window_function_calculation)

            elif t < window_half_length:
                downsampled_time_series[t] = sum(self.time_series[:int(t + window_half_length + 1)] *
                                                 window_function_calculation[int(window_half_length - t):])

            elif t > (len(self.time_series) - window_half_length - 1):
                downsampled_time_series[t] = sum(self.time_series[int(t - window_half_length):] *
                                                 window_function_calculation[:int(window_half_length +
                                                                                  len(self.time_series) - t)])

        # decimate

        if decimate:

            downsampled_and_decimated_time_series = np.zeros(len(downsampled_time_series) // decimation_level + 1)
            decimated_time = np.zeros_like(downsampled_and_decimated_time_series)

            for t in range(len(downsampled_time_series)):

                if t % decimation_level == 0:
                    downsampled_and_decimated_time_series[int(t // decimation_level)] = downsampled_time_series[t]
                    decimated_time[int(t // decimation_level)] = self.time[t]

        else:
            decimated_time = self.time.copy()
            downsampled_and_decimated_time_series = downsampled_time_series

        return decimated_time, downsampled_and_decimated_time_series

    def median_filter(self, window_width: int = 51) -> (np.ndarray, np.ndarray):
        """
        Median filters the provided time series.

        Parameters
        ----------
        window_width : integer (odd positive)
            Window width to be used when filtering time series.

        Returns
        -------
        median_filtered_time : real ndarray
            Time associated with median filtered time series.

        median_filtered_time_series : real ndarray
            Median filtered time series.

        Notes
        -----
        Only works with odd window lengths as original point needs to be centred.

        """
        median_filtered_time_series = np.zeros_like(self.time_series)
        median_filtered_time = self.time

        boundary = (window_width - 1) / 2

        for i in range(len(self.time_series)):
            if i < boundary:
                median_filtered_time_series[i] = np.median(self.time_series[:(i + i + 1)])
            elif i >= len(self.time_series) - boundary:
                median_filtered_time_series[i] = \
                    np.median(self.time_series[-int(2 * (len(self.time_series) - i) - 1):])
            else:
                median_filtered_time_series[i] = \
                    np.median(self.time_series[int(i - boundary):int(i + boundary + 1)])

        return median_filtered_time, median_filtered_time_series

    def mean_filter(self, window_width: int = 51) -> (np.ndarray, np.ndarray):
        """
        Mean filters the provided time series.

        Parameters
        ----------
        window_width : integer (odd positive)
            Window width to be used when filtering time series.

        Returns
        -------
        mean_filtered_time : real ndarray
            Time associated with unsmoothed mean filtered time series.

        mean_filtered_time_series : real ndarray
            Unsmoothed mean filtered time series.

        Notes
        -----
        Only works with odd window lengths as original point needs to be centred.

        """
        mean_filtered_time_series = np.zeros_like(self.time_series)
        mean_filtered_time = self.time

        boundary = (window_width - 1) / 2

        for i in range(len(self.time_series)):
            if i < boundary:
                mean_filtered_time_series[i] = np.mean(self.time_series[:(i + i + 1)])
            elif i >= len(self.time_series) - boundary:
                mean_filtered_time_series[i] = np.mean(self.time_series[-int(2 * (len(self.time_series) - i) - 1):])
            else:
                mean_filtered_time_series[i] = np.mean(self.time_series[int(i - boundary):int(i + boundary + 1)])

        return mean_filtered_time, mean_filtered_time_series

    def quantile_filter(self, window_width: int = 51, q: float = 0.95) -> (np.ndarray, np.ndarray):
        """
        Quantile filters the provided time series.

        Parameters
        ----------
        window_width : integer (odd postive)
            Window length to be used when filtering time series.

        q : float
            Confidence bound (one-sided).

        Returns
        -------
        quantile_filtered_time : real ndarray
            Time associated with unsmoothed quantile filtered time series.

        quantile_filtered_time_series : real ndarray
            Unsmoothed quantile filtered time series.

        Notes
        -----
        Only works with odd window lengths as original point needs to be centred.
        Not be used directly - creates bounds for Winsorization.

        """
        quantile_filtered_time_series = np.zeros_like(self.time_series)
        quantile_filtered_time = self.time

        boundary = (window_width - 1) / 2

        for i in range(len(self.time_series)):
            if i < boundary:
                quantile_filtered_time_series[i] = np.quantile(self.time_series[:(i + i + 1)], q)
            elif i >= len(self.time_series) - boundary:
                quantile_filtered_time_series[i] = \
                    np.quantile(self.time_series[-int(2 * (len(self.time_series) - i) - 1):], q)
            else:
                quantile_filtered_time_series[i] = \
                    np.quantile(self.time_series[int(i - boundary):int(i + boundary + 1)], q)

        return quantile_filtered_time, quantile_filtered_time_series

    def winsorize(self, window_width: int = 51, a: float = 0.9) -> (np.ndarray, np.ndarray):
        """
        Winsorize the provided time series.

        Parameters
        ----------
        window_width : integer (odd postive)
            Window length to be used when filtering time series.

        a : float
            Confidence bound (two-sided).

        Returns
        -------
        winsorized_time : real ndarray
            Time associated with unsmoothed Winsorized time series.

        winsorized_time_series : real ndarray
            Unsmoothed Winsorized time series.

        Notes
        -----
        Only works with odd window lengths as original point needs to be centred.

        """
        winsorized_time = self.time
        winsorized_time_series = self.time_series.copy()

        upper_quantile = Preprocess.quantile_filter(self, window_width=window_width, q=(1 - (1 - a) / 2))[1]

        lower_quantile = Preprocess.quantile_filter(self, window_width=window_width, q=((1 - a) / 2))[1]

        winsorized_time_series[self.time_series > upper_quantile] = upper_quantile[self.time_series > upper_quantile]
        winsorized_time_series[self.time_series < lower_quantile] = lower_quantile[self.time_series < lower_quantile]

        return winsorized_time, winsorized_time_series

    def winsorize_interpolate(self, window_width: int = 51, a: float = 0.9) -> (np.ndarray, np.ndarray):
        """
        Winsorize and interpolate the provided time series.

        Parameters
        ----------
        window_width : integer (odd postive)
            Window length to be used when filtering time series.

        a : float
            Confidence bound (two-sided).

        Returns
        -------
        winsorized_and_interpolated_time : real ndarray
            Time associated with unsmoothed Winsorized and interpolated time series.

        winsorized_and_interpolated_time_series : real ndarray
            Unsmoothed Winsorized and interpolated time series.

        Notes
        -----
        Only works with odd window lengths as original point needs to be centred.

        """
        winsorized_and_interpolated_time = self.time
        winsorized_and_interpolated_time_series = self.time_series.copy()

        upper_quantile = Preprocess.quantile_filter(self, window_width=window_width, q=(1 - (1 - a) / 2))[1]

        lower_quantile = Preprocess.quantile_filter(self, window_width=window_width, q=((1 - a) / 2))[1]

        winsorized_and_interpolated_time_series[self.time_series > upper_quantile] = np.nan
        winsorized_and_interpolated_time_series[self.time_series < lower_quantile] = np.nan
        nan_bool = np.isnan(winsorized_and_interpolated_time_series)  # find nan values and create bool
        winsorized_and_interpolated_time_series[nan_bool] = \
            np.interp(winsorized_and_interpolated_time[nan_bool], winsorized_and_interpolated_time[~nan_bool],
                      winsorized_and_interpolated_time_series[~nan_bool], left=0, right=0)  # interpolate nan values

        return winsorized_and_interpolated_time, winsorized_and_interpolated_time_series

    def hp(self, smoothing_penalty: float = 1, order: int = 2, norm_1: int = 2, norm_2: int = 1) -> (np.ndarray,
                                                                                                     np.ndarray):
        """
        Extension of Hodrick-Prescott Filter to other orders of penalisation and additional norms.
        Default inputs is the Hodrick-Prescott Filter with lasso norm on curvature penalisation term.

        Parameters
        ----------
        smoothing_penalty : float
            Smoothing penalty to penalise finite-difference order term.

        order : integer
            Order of penalisation.

        norm_1 : integer
            Norm type to use to fit to time series.

        norm_2 : integer
            Norm type to use when penalising order of filtered time series.

        Returns
        -------
        hp_time : real ndarray
            Time associated with generalised Hodrick-Prescott filtered time series.

        hp_time_series : real ndarray
            Generalised Hodrick-Prescott filtered time series.

        Notes
        -----
        Part of section on discrete splines.

        """
        hp_time = self.time

        # create generalised variable structure
        vx = cvx.Variable(len(self.time_series))
        objective = {}  # removes unnecessary error owing to conditional creation below

        # create objective function - depends on order of penalisation
        if order == 1:
            objective = cvx.Minimize(cvx.norm(self.time_series - vx, norm_1) +
                                     smoothing_penalty * cvx.norm(vx[:-1] - vx[1:], norm_2))
        elif order == 2:
            objective = cvx.Minimize(cvx.norm(self.time_series - vx, norm_1) +
                                     smoothing_penalty * cvx.norm(vx[:-2] - 2 * vx[1:-1] + vx[2:], norm_2))
        elif order == 3:
            objective = cvx.Minimize(cvx.norm(self.time_series - vx, norm_1) +
                                     smoothing_penalty * cvx.norm(vx[:-3] - 3 * vx[1:-2] + 3 * vx[2:-1] - vx[3:],
                                                                  norm_2))
        elif order == 4:
            objective = cvx.Minimize(cvx.norm(self.time_series - vx, norm_1) +
                                     smoothing_penalty * cvx.norm(
                vx[:-4] - 4 * vx[1:-3] + 6 * vx[2:-2] - 4 * vx[3:-1] + vx[4:], norm_2))

        # create problem to solve from objective function
        prob = cvx.Problem(objective)
        # solve problem
        prob.solve(solver=cvx.ECOS)

        # reconstruct time series
        hp_time_series = np.array(vx.value)

        return hp_time, hp_time_series

    def hw(self, order: int = 13, method: str = 'kernel') -> (np.ndarray, np.ndarray):
        """
        Henderson smoothing filter.

        Parameters
        ----------
        order : integer
            Width of weighting vector.

        method : string_like
            Method to use in calculating weights.

        Returns
        -------
        hw_time : real ndarray
            Time associated with Henderson weighted and filtered time series.

        hw_time_series : real ndarray
            Henderson weighting filtered time series.

        Notes
        -----
        RKHS weights preferred to be used in algorithm as easily tranlatable to asymmetric weights.

        """
        hw_time = self.time.copy()
        hw_time_series = np.zeros_like(self.time_series)
        weights = {}  # removes unnecessary error owing to conditional creation below
        asymmetric_weights = {}  # removes unnecessary error owing to conditional creation below

        if method == 'renormalise':
            weights = henderson_weights(order=order)
        elif method == 'kernel':
            weights = henderson_kernel(order=order)

        # need asymmetric weights that sum to approximately one on the edges - multiple options:
        # (1) use asymmetric filter (truncate and renormalise)
        # (2) Reproducing Kernel Hilbert Space Method

        for k in range(len(self.time_series)):

            if k < ((order - 1) / 2):
                if method == 'renormalise':
                    asymmetric_weights = henderson_weights(order=order, start=(0 - k))
                elif method == 'kernel':
                    asymmetric_weights = henderson_kernel(order=order, start=(0 - k))
                hw_time_series[k] = \
                    np.sum(asymmetric_weights * self.time_series[:int(k + ((order - 1) / 2) + 1)])
            elif k > len(self.time_series) - ((order - 1) / 2) - 1:
                if method == 'renormalise':
                    asymmetric_weights = henderson_weights(order=order,
                                                           end=(len(self.time_series) - k - 1))
                elif method == 'kernel':
                    asymmetric_weights = henderson_kernel(order=order,
                                                          end=(len(self.time_series) - k - 1))
                hw_time_series[k] = \
                    np.sum(asymmetric_weights * self.time_series[int(k - ((order - 1) / 2)):])
            else:
                hw_time_series[k] = \
                    np.sum(weights * self.time_series[int(k - ((order - 1) / 2)):int(k + ((order - 1) / 2) + 1)])

        return hw_time, hw_time_series
