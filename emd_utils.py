
#     ________
#            /
#      \    /
#       \  /
#        \/

# emd class to house all utility methods

import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='darkgrid')


def time_extension(vector: np.ndarray) -> np.ndarray:
    """
    Time or knot extension for edge fitting.

    Returns
    -------
    extension : real ndarray
        Extended vector as detailed in notes.

    Notes
    -----
    Reflects all points about both end points:

    old_vector = [t_0, t_1, t_2]
    new_vector = [t_0 - (t_2 - t_0), t_0 - (t_1 - t_0), t_0, t_1, t_2, t_2 + (t_2 - t_1), t_2 + (t_2 - t_0)]

    """
    start = vector[0]
    end = vector[-1]

    diff_1 = (start * np.ones_like(vector[1:]) - vector[1:])
    diff_2 = (end * np.ones_like(vector[:-1]) - vector[:-1])
    extension = np.hstack((vector[0] * np.ones_like(diff_1) + np.flip(diff_1), vector,
                           vector[-1] * np.ones_like(diff_2) + np.flip(diff_2)))

    return extension


def my_factorial(vector) -> np.ndarray:
    """
    Loops over math.factorial() function to create vector of factorials.

    Parameters
    ----------
    vector : array_like
        Vector of integers to be raised to factorials.

    Returns
    -------
    factorial_vector : real ndarray
        Vector of factorials.

    Notes
    -----
    Example: my_factorial((1, 2, 3, 4)) = (1, 2, 6, 24)

    """
    factorial_vector = np.zeros_like(vector)

    for y, z in enumerate(vector):
        factorial_vector[y] = math.factorial(z)

    return factorial_vector


class Utility:
    """
    Useful time series manipulation.

    Parameters
    ----------
    time : array_like
        Time corresponding to time series to be preprocessed.

    Notes
    -----

    """
    def __init__(self, time: np.ndarray, time_series: np.ndarray):

        self.time = time
        self.time_series = time_series

    def zero_crossing(self) -> np.ndarray:
        """
        Returns an array of booleans marking zero-crossings of input time series.

        Returns
        -------
        out : bool
            Boolean for finite difference zero-crossing of time series.

        Notes
        -----
        Only works when product of adjacent points is negative, not zero - using zeros could lead to proliferation.

        """
        zero_crossing_time_series = self.time_series.copy()

        return np.r_[zero_crossing_time_series[1:] * zero_crossing_time_series[:-1] <= 0, False]

    def max_bool_func_1st_order_fd(self) -> np.ndarray:
        """
        Maximum boolean method:
        Returns an array of booleans marking local maxima of input time series.

        Returns
        -------
        max_bool_order_1 : bool
            Boolean for finite difference maxima of time series.

        Notes
        -----
        Boundary process relies on False appended to end points.
        Uses first-order forward difference.

        """
        max_time_series = self.time_series
        max_bool_order_1 = np.r_[False,
                                 max_time_series[1:] >=
                                 max_time_series[:-1]] & np.r_[max_time_series[:-1] >
                                                               max_time_series[1:],
                                                               False]

        return max_bool_order_1

    # calculates second-order maxima using finite difference method

    def max_bool_func_2nd_order_fd(self) -> np.ndarray:
        """
        Maximum boolean method:
        Returns an array of booleans marking local second-order maxima of input time series.

        Returns
        -------
        max_bool_order_2 : bool
            Boolean for second order finite difference maxima of time series.

        Notes
        -----
        Uses second-order central difference and then second-order forward difference.

        """
        max_time_series = self.time_series
        max_bool_order_2 = np.r_[False,
                                 (max_time_series[:-2] - 2 * max_time_series[1:-1] + max_time_series[2:]) <= 0,
                                 False] & np.r_[(max_time_series[:-2] - 2 *
                                                 max_time_series[1:-1] + max_time_series[2:]) >= 0,
                                                False, False]

        return max_bool_order_2

    # calculates first-order minima using finite difference method

    def min_bool_func_1st_order_fd(self) -> np.ndarray:
        """
        Minimum boolean method:
        Returns an array of booleans marking local minima of input time series.

        Returns
        -------
        min_bool_order_1 : bool
            Boolean for finite difference minima of time series.

        Notes
        -----
        Boundary process relies on False appended to end points
        Uses first-order forward difference.

        """
        min_time_series = self.time_series
        min_bool_order_1 = np.r_[False,
                                 min_time_series[1:] <=
                                 min_time_series[:-1]] & np.r_[min_time_series[:-1] <
                                                               min_time_series[1:],
                                                               False]

        return min_bool_order_1

    # calculates second-order minima using finite difference method

    def min_bool_func_2nd_order_fd(self) -> np.ndarray:
        """
        Minimum boolean method:
        Returns an array of booleans marking local second-order minima of input time series.

        Returns
        -------
        min_bool_order_2 : bool
            Boolean for second order finite difference minima of time series.

        Notes
        -----
        Uses second-order central difference and then second-order forward difference.

        """
        min_time_series = self.time_series
        min_bool_order_2 = np.r_[False,
                                 (min_time_series[:-2] - 2 * min_time_series[1:-1] + min_time_series[2:]) >= 0,
                                 False] & np.r_[(min_time_series[:-2] - 2 *
                                                 min_time_series[1:-1] + min_time_series[2:]) <= 0,
                                                False, False]

        return min_bool_order_2

    # Inflection point boolean function

    def inflection_point(self) -> np.ndarray:
        """
        Inflection point boolean method.

        Returns
        -------
        inflection_bool : bool
            Boolean for second order finite difference maxima and minima of time series.

        Notes
        -----
        Uses second-order central difference.

        """
        inflection_bool = np.logical_or(Utility.max_bool_func_2nd_order_fd(self),
                                        Utility.min_bool_func_2nd_order_fd(self))

        return inflection_bool

    def derivative_forward_diff(self) -> np.ndarray:
        """
        First forward difference derivative calculator.

        Returns
        -------
        derivative : real ndarray
            Derivative of time series.

        Notes
        -----
        Uses first forward difference.

        """
        time = self.time.copy()
        time_series = self.time_series.copy()

        derivative = (time_series[:-1] - time_series[1:]) / (time[:-1] - time[1:])

        return derivative

    def energy(self) -> float:
        """
        Energy calculation for energy difference tracking.

        Returns
        -------
        energy_calc : float
            Energy approximation.

        Notes
        -----
        Left Riemann sum approximation.

        """
        energy_calc = sum((self.time_series[:-1] ** 2) * np.diff(self.time))

        return energy_calc

    def binomial_average(self, order) -> np.ndarray:  # choose odd order for central weighting
        """
        Creates a vector with a weighted binomial average of surrounding points.

        Parameters
        ----------
        order : integer (odd positive)
            Number of surrounding points to use when taking binomial average.

        Returns
        -------
        bin_av : real ndarray
            Vector of binomial averages.

        Notes
        -----
        Special considerations are made for edges of time series.
        Example: binomial_average((1, 2, 3), 3) = (1, 2, 3)

        """
        time_series = self.time_series

        bin_av = np.array(np.zeros_like(time_series), dtype=float)

        for j in range(len(time_series)):

            if j in np.hstack((range(int((order - 1) / 2)),
                               range(int((len(time_series) - (order - 1) / 2)), int(len(time_series))))):
                # if index is within range of boundary - need to truncate binomial averaging

                temp_nparray = np.intersect1d(np.array(range(int(j - (order - 1) / 2), int(j + (order - 1) / 2 + 1))),
                                              np.array(range(len(time_series))))

                temp_values = time_series[int(temp_nparray[0]):int(temp_nparray[-1] + 1)]

                temp_indices = range(len(temp_values))

                if j in range(int((order - 1) / 2)):
                    temp_indices = np.flip(temp_indices)

                vectorize = np.ones_like(temp_values)

                bin_av[j] = sum(np.multiply(np.multiply((1 / (sum((math.factorial(order - 1)) /
                                                                  np.multiply(my_factorial(
                                                                      (order - 1) * vectorize - temp_indices),
                                                                              my_factorial(
                                                                                  temp_indices))))) * vectorize,
                                                        ((math.factorial(order - 1)) /
                                                         np.multiply(
                                                             my_factorial((order - 1) * vectorize - temp_indices),
                                                             my_factorial(temp_indices)))), temp_values))

            else:
                # otherwise normal binomial average function

                temp_values = time_series[int(j - (order - 1) / 2):int(j + (order - 1) / 2 + 1)]
                temp_indices = np.asarray(range(len(temp_values)))
                vectorize = np.ones_like(temp_values)

                bin_av[j] = sum(np.multiply(np.multiply(((1 / 2) ** (order - 1)) * vectorize,
                                                        ((math.factorial(order - 1)) /
                                                         np.multiply(
                                                             my_factorial((order - 1) * vectorize - temp_indices),
                                                             my_factorial(temp_indices)))), temp_values))

        return bin_av
