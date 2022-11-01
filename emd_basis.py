
#     ________
#            /
#      \    /
#       \  /
#        \/

# emd class to house all basis construction methods

import numpy as np
from numpy.linalg import solve
import seaborn as sns
import matplotlib.pyplot as plt
from emd_utils import time_extension

sns.set(style='darkgrid')

np.seterr(divide='ignore', invalid='ignore')
# divide='ignore' deals with division by zero in basis constructions
# invalid='ignore' deals with NaN values in basis constructions


def chsi(t: np.ndarray, p0: float, m0: float, p1: float, m1: float) -> np.ndarray:
    """
    Cubic Hermite Spline Interpolation basis function construction.

    Parameters
    ----------
    t : real ndarray
        Time series over which basis is constructed.

    p0 : float
        Basis function for intercept equal to 1 at t[0].

    m0 : float
        Basis function for tangent equal to 1 at t[0].

    p1 : float
        Basis function for intercept equal to 1 at t[-1].

    m1 : float
        Basis function for tangent equal to 1 at t[-1]

    Returns
    -------
    chsi_basis : real ndarray
        Returns a single interval cubic Hermite spline interpolation.

    Notes
    -----
    Scaling factor for tangent bases is introduced later (essential for non-unit intervals).

    """
    t = (t - t[0]) / (t[-1] - t[0])

    chsi_basis = (2 * t ** 3 - 3 * t ** 2 + 1) * p0 + (t ** 3 - 2 * t ** 2 + t) * m0 + \
                 (-2 * t ** 3 + 3 * t ** 2) * p1 + (t ** 3 - t ** 2) * m1

    return chsi_basis


def m(x0: float, x1: float, y0: float, y1: float) -> float:
    """
    Simple gradient function.

    Parameters
    ----------
    x0 : float
        x co-ordinate at time 0.

    x1 : float
        x co-ordinate at time 1.

    y0 : float
        y co-ordinate at time 0.

    y1 : float
        y co-ordinate at time 1.

    Returns
    -------
    grad : float
        Gradient value.

    Notes
    -----

    """
    grad = (y1 - y0) / (x1 - x0)

    return grad


def asi_si(x_2: float, x_1: float, x0: float, x1: float, x2: float, y_2: float, y_1: float, y0: float, y1: float,
           y2: float) -> float:
    """
    Akima spline tangent value calculation.

    Parameters
    ----------
    x_2 : float
        x co-ordinate at time -2.

    x_1 : float
        x co-ordinate at time -1.

    x0 : float
        x co-ordinate at time 0.

    x1 : float
        x co-ordinate at time 1.

    x2 : float
        x co-ordinate at time 2.

    y_2 : float
        y co-ordinate at time -2.

    y_1 : float
        y co-ordinate at time -1.

    y0 : float
        y co-ordinate at time 0.

    y1 : float
        y co-ordinate at time 1.

    y2 : float
        y co-ordinate at time 2.

    Returns
    -------
    tangent : float
        Akima spline tangent value.

    Notes
    -----

    """
    tangent = (np.abs(m(x1, x2, y1, y2) - m(x0, x1, y0, y1)) * m(x_1, x0, y_1, y0) +
               np.abs(m(x_1, x0, y_1, y0) - m(x_2, x_1, y_2, y_1)) * m(x0, x1, y0, y1)) / \
              (np.abs(m(x1, x2, y1, y2) - m(x0, x1, y0, y1)) + np.abs(m(x_1, x0, y_1, y0) - m(x_2, x_1, y_2, y_1)))

    return tangent


class Basis:

    def __init__(self, time: np.ndarray, time_series: np.ndarray):

        self.time = time
        self.time_series = time_series

    def b(self, knots: np.ndarray, degree: int) -> np.ndarray:
        """
        Recursive method for building basis functions - concise and effective.

        Parameters
        ----------
        knots : real ndarray
            Entire knot vector or subset of knot vector depending on level of recursion.
            Number of knots provided depends on degree of basis function i.e. degree = 3 -> len(knots) = 5

        degree : int
            Degree of basis spline to be constructed.

        Returns
        -------
        b_basis : real ndarray
            Single basis spline of degree: "degree".

        Notes
        -----
        Continually subsets knot vector by one increment until base case is reached.

        """
        basis_time = self.time.copy()

        if degree == 0:

            return np.asarray(1.0 * ((knots[0] <= basis_time) & (basis_time < knots[1])))

        else:

            c1 = (basis_time - knots[0] * np.ones_like(basis_time)) / \
                 (knots[-2] * np.ones_like(basis_time) - knots[0] * np.ones_like(basis_time)) * \
                 Basis.b(self, knots[:-1], int(degree - 1))

            c2 = (knots[-1] * np.ones_like(basis_time) - basis_time) / \
                 (knots[-1] * np.ones_like(basis_time) - knots[1] * np.ones_like(basis_time)) * Basis.b(self, knots[1:],
                                                                                                        int(degree - 1))

        b_basis = c1 + c2

        return b_basis

    def hilbert_b(self, knots: np.ndarray, degree: int) -> np.ndarray:
        """
        Recursive method for building hilbert basis functions - concise and effective.

        Parameters
        ----------
        knots : array_like
            Entire knot vector or subset of knot vector depending on level of recursion.
            Number of knots provided depends on degree of basis function i.e. degree = 3 -> len(knots) = 5

        degree : int
            Degree of hilbert basis spline to be constructed.

        Returns
        -------
        hilbert_b_basis : real ndarray
            Single hilbert transformed basis spline of degree: "degree".

        Notes
        -----
        Continually subsets knot vector by one increment until base case is reached.

        """
        hilbert_basis_time = self.time.copy()

        if degree == 0:

            return np.asarray((1 / np.pi) * np.log(np.abs((hilbert_basis_time - knots[0]) /
                                                          (hilbert_basis_time - knots[1]))))

        else:

            c1 = (hilbert_basis_time - knots[0] * np.ones_like(hilbert_basis_time)) / \
                 (knots[-2] * np.ones_like(hilbert_basis_time) - knots[0] * np.ones_like(hilbert_basis_time)) * \
                 Basis.hilbert_b(self, knots[:-1], int(degree - 1))

            c2 = (knots[-1] * np.ones_like(hilbert_basis_time) - hilbert_basis_time) / \
                 (knots[-1] * np.ones_like(hilbert_basis_time) - knots[1] *
                  np.ones_like(hilbert_basis_time)) * Basis.hilbert_b(self, knots[1:], int(degree - 1))

        hilbert_b_basis = c1 + c2

        return hilbert_b_basis

    def quadratic_b_spline(self, knots: np.ndarray) -> np.ndarray:
        """
        Returns a (len(knots) - 3) x (len(time)) array. Each row is an individual quadratic basis.
        Matrix is sparse. Each column contains at most 3 non-zero values (only three bases overlap at any point).

        Parameters
        ----------
        knots : array_like
            Knot points to be used (not necessarily evenly spaced).

        Returns
        -------
        matrix_c : real ndarray
            Each row of matrix contains an individual quadratic basis spline.

        Notes
        -----
        Follows work from :

        Q. Chen, N. Huang, S. Riemenschneider, and Y. Xu. A B-spline Approach for Em-
        pirical Mode Decompositions. Advances in Computational Mathematics, 24(1-4):
        171–195, 2006.

        Derivative of cubic B-spline (or combination thereof) is simply a combination of quadratic B-splines.

        """
        num_c = len(knots) - 3  # quadratic basis-spline -> 3 fewer coefficients than knots
        matrix_c = np.zeros((num_c, len(self.time)))  # each row is a single basis function

        for tau in range(num_c):  # watch inequalities

            subset_knots = knots[tau:(tau + 4)]  # select 4 knots applicable to current cubic spline
            matrix_c[tau, :] = Basis.b(self, subset_knots, 2)

        return matrix_c

    def derivative_cubic_b_spline(self, knots: np.ndarray) -> np.ndarray:
        """
        Returns a (len(knots) - 4) x (len(time)) array. Each row is an individual cubic basis derivative.
        Matrix is sparse. Each column contains at most 4 non-zero values (only 4 bases overlap at any point).

        Parameters
        ----------
        knots : array_like
            Knot points to be used (not necessarily evenly spaced).

        Returns
        -------
        matrix_c : real ndarray
            Each row of matrix contains an individual cubic basis derivative spline.

        Notes
        -----
        Follows work from :

        Q. Chen, N. Huang, S. Riemenschneider, and Y. Xu. A B-spline Approach for Em-
        pirical Mode Decompositions. Advances in Computational Mathematics, 24(1-4):
        171–195, 2006.

        Derivative of cubic B-spline (or combination thereof) is simply a combination of quadratic B-splines.

        """
        knots_plus_8 = time_extension(knots)
        knots_plus_8 = knots_plus_8[int(len(knots) - 5):int(2 * len(knots) + 3)]
        quad_mat = Basis.quadratic_b_spline(self, knots_plus_8)
        factor_mat = np.zeros((len(knots_plus_8) - 4, len(knots_plus_8) - 3))

        for a in range(len(knots_plus_8) - 4):
            factor_mat[a, a:(a + 2)] = 3 / (knots_plus_8[a + 3] - knots_plus_8[a]), -3 / (
                        knots_plus_8[a + 4] - knots_plus_8[a + 1])

        matrix_c = np.matmul(factor_mat, quad_mat)

        return matrix_c

    def cubic_b_spline(self, knots: np.ndarray) -> np.ndarray:
        """
        Returns a (len(knots) - 4)x(len(time)) array. Each row is an individual cubic basis.
        Matrix is sparse. Each column contains at most 4 non-zero values (only four bases overlap at any point).

        Parameters
        ----------
        knots : real ndarray
            Knot points to be used (not necessarily evenly spaced).

        Returns
        -------
        matrix_c : real ndarray
            Each row of matrix contains an individual cubic basis spline.

        Notes
        -----
        When used in basis_function_approximation and envelope_basis_function_approximation a vector 'c' is calculated
        such that with output of this function being array 'B' and original signal being 's' the objective function
        ||(B^T)c - s||^2 is minimized.

        """
        # optimise speed
        if all(np.isclose(np.diff(knots), (knots[1] - knots[0]) * np.ones_like(np.diff(knots)))) and \
                all(np.isclose(np.diff(self.time),
                               (self.time[1] - self.time[0]) * np.ones_like(np.diff(self.time)))) and \
                np.isclose(knots[0], self.time[0]) and np.isclose(knots[-1], self.time[-1]):
            optimise_speed = True
        else:
            optimise_speed = False

        num_c = len(knots) - 4  # cubic basis-spline -> 4 fewer coefficients than knots
        matrix_c = np.zeros((num_c, len(self.time)))  # each row is a single basis function

        if not optimise_speed:
            for tau in range(num_c):  # watch inequalities
                subset_knots = knots[tau:(tau + 5)]  # select 5 knots applicable to current cubic spline
                matrix_c[tau, :] = Basis.b(self, subset_knots, 3)
        elif optimise_speed:
            try:
                # can improve speed
                matrix_c[0, :] = Basis.b(self, knots[0:5], 3)
                for tau in range(1, num_c):
                    matrix_c[tau, :] = \
                        np.hstack((np.zeros(int((len(self.time) - 1) / (len(knots) - 1))),
                                   matrix_c[int(tau - 1), :][:-int((len(self.time) - 1) / (len(knots) - 1))]))
            except:
                for tau in range(num_c):  # watch inequalities
                    subset_knots = knots[tau:(tau + 5)]  # select 5 knots applicable to current cubic spline
                    matrix_c[tau, :] = Basis.b(self, subset_knots, 3)

        return matrix_c

    # hilbert cubic basis splines as function of knots points

    def hilbert_cubic_b_spline(self, knots: np.ndarray) -> np.ndarray:
        """
        Returns a (len(knots)- 4)x(len(time)) array. Each row is an individual hilbert cubic basis.
        Matrix is sparse. Each column contains at most 4 non-zero values (only four bases overlap at any point).

        Parameters
        ----------
        knots : array_like
            Knot points to be used (not necessarily evenly spaced).

        Returns
        -------
        matrix_c : real ndarray
            Each row of matrix contains an individual hilbert transformed cubic basis spline.

        Notes
        -----
        The coefficients are optimised elsewhere.

        """
        # optimise speed
        if all(np.isclose(np.diff(knots), (knots[1] - knots[0]) * np.ones_like(np.diff(knots)))) and \
                all(np.isclose(np.diff(self.time), (self.time[1] - self.time[0]) * np.ones_like(np.diff(self.time)))):
            optimise_speed = True
        else:
            optimise_speed = False

        basis_time = self.time
        num_c = len(knots) - 4  # hilbert cubic basis-spline -> 4 fewer coefficients than knots
        matrix_c = np.zeros((num_c, len(basis_time)))  # each row is a single basis function

        if not optimise_speed:
            for tau in range(num_c):  # watch inequalities
                subset_knots = knots[tau:(tau + 5)]  # select 5 knots applicable to current hilbert cubic spline
                un_interpolated = Basis.hilbert_b(self, subset_knots, 3)  # some nan values as a result of base case
                nan_bool = np.isnan(un_interpolated)  # find nan values and create bool
                un_interpolated[nan_bool] =\
                    np.interp(basis_time[nan_bool], basis_time[~nan_bool],
                              un_interpolated[~nan_bool], left=0, right=0)  # interpolate nan values
                matrix_c[tau, :] = un_interpolated
        elif optimise_speed:
            time_extended = time_extension(self.time)
            time_extended = time_extension(time_extended)
            basis_opt = Basis(time=time_extended, time_series=time_extended)
            total_knots = len(knots)
            mid_knots = total_knots
            un_interpolated = basis_opt.hilbert_b(knots=knots[0:5], degree=3)
            nan_bool = np.isnan(un_interpolated)  # find nan values and create bool
            un_interpolated[nan_bool] = np.interp(time_extended[nan_bool], time_extended[~nan_bool],
                                                  un_interpolated[~nan_bool], left=0, right=0)  # interpolate nan values
            interpolated_subset = un_interpolated[int(4 * len(self.time) - 1):int(5 * len(self.time) - 1)]
            matrix_c[0, :] = interpolated_subset
            # plt.plot(matrix_c[0, :])
            for tau in range(1, num_c):
                if tau == 11:
                    test = 0
                matrix_c[tau, :] = \
                    un_interpolated[int(4 * len(self.time) - 1 - int(tau * ((len(self.time) - 1) / (len(knots) - 7)))):
                                    int(5 * len(self.time) - 1 - int(tau * ((len(self.time) - 1) / (len(knots) - 7))))]
                # plt.plot(matrix_c[tau, :])

        return matrix_c

    def optimize_knot_points(self, error: float = 10, lamda: float = 1,
                             epsilon: float = 0.05, method: str = 'ser_bisect') -> np.ndarray:
        """

        Parameters
        ----------
        error : float
            Error term that causes bisecting to stop (and serial bisecting to begin).

        lamda : float
            Level of second-order smoothing.

        epsilon : float
            Minimum distance allowed between successive potential knot points.

        method : string_like
            bisection : vanilla bisection method that bisects until error bound is satisfied.
            serial_bisection : Serial Bisection method from citation below.

        Returns
        -------
        optimized_knots : real ndarray
            Vector of optimised knot locations.

        Notes
        -----
        Number of knots is not fixed.

        Implements Serial Bisection knot point optimisation method from:

            V. Dung and T. Tjahjowidodo. A direct method to solve optimal knots of B-spline
            curves: An application for non-uniform B-spline curves fitting. PLoS ONE, 12(3):
            e0173857, 2017. doi: https://doi.org/10.1371/journal.pone.0173857.

        Basis object redefined iteratively to accommodate continuous redefining of time, time series, and knot sequence.

        """
        optimize_time = self.time.copy()
        optimize_time_series = self.time_series.copy()

        b = optimize_time[-1]
        a = optimize_time[0]
        optimized_knots = [a]
        old_t = {}  # fixes assignment error
        old_time_series = {}  # fixes assignment error
        b_diff = {}  # fixes assignment error

        if method[-9:] == 'bisection' or method[-6:] == 'bisect':

            while a < optimize_time[-4]:

                new_t = optimize_time.copy()
                new_time_series = optimize_time_series.copy()
                optimized_knots += [b]
                new_error = error + 1  # to initiate second while loop

                # works - now need to loop over all time points
                while new_error > error:

                    # store original vectors
                    old_t = new_t.copy()
                    old_time_series = new_time_series.copy()

                    # truncate original vector to new potential knot
                    new_t = old_t[old_t <= b]
                    new_time_series = old_time_series[old_t <= b]

                    # calculate new knots
                    new_knots = np.linspace(a, b, 2)
                    optimized_knots = optimized_knots[:-1] + [new_knots[1]]
                    all_knots_new = time_extension(optimized_knots)
                    all_knots_new = time_extension(all_knots_new)
                    all_knots_new = time_extension(all_knots_new)
                    all_knots_start_bool = all_knots_new == optimize_time[0]
                    all_knots_start_bool = np.append(all_knots_start_bool[3:], np.asarray((0, 0, 0)))
                    all_knots_b_bool = all_knots_new == b
                    all_knots_b_bool = np.append(np.asarray((0, 0, 0, 0)), all_knots_b_bool[:-4])
                    all_knots_bool = (all_knots_start_bool + all_knots_b_bool) == 1
                    all_knots_bool = np.cumsum(all_knots_bool) == 1
                    all_knots_extension = all_knots_new[all_knots_bool]

                    optimize_basis = Basis(time=new_t, time_series=new_time_series)

                    b_spline_matrix = optimize_basis.cubic_b_spline(all_knots_extension)  # using redefined Basis object

                    second_order_matrix = np.zeros((np.shape(b_spline_matrix)[0],
                                                    np.shape(b_spline_matrix)[0] - 2))  # note transpose
                    for c in range(np.shape(b_spline_matrix)[0] - 2):
                        second_order_matrix[c:(c + 3), c] = [1, -2,
                                                             1]  # filling values for second-order difference matrix
                    b_spline_matrix_2 = np.append(b_spline_matrix, lamda *
                                                  second_order_matrix, axis=1)  # l2 norm trick

                    new_time_series_2 = np.append(new_time_series, np.zeros(np.shape(b_spline_matrix_2)[0] - 2), axis=0)
                    # l2 norm trick

                    coef = np.linalg.lstsq(b_spline_matrix_2.transpose(), new_time_series_2, rcond=None)
                    coef = coef[0]

                    new_approximation = np.matmul(coef, b_spline_matrix)

                    new_error = np.sum(np.abs(new_time_series[np.r_[a <= new_t] & np.r_[new_t <= b]] -
                                              new_approximation[np.r_[a <= new_t] & np.r_[new_t <= b]]))

                    # new end point
                    b = (a + b) / 2

                b_left = a + 2 * (b - a)  # return to actual position
                b_right = b_left + (1 / 2) * (b_left - a)  # move knot back half the distance
                temp = 1

                if method == 'ser_bisect':

                    while np.abs(b_left - b_right) > epsilon and b_left < optimize_time[-1]:

                        new_t = old_t[old_t <= b_right]
                        new_time_series = old_time_series[old_t <= b_right]

                        # calculate new knots
                        new_knots = np.linspace(a, b_right, 2)
                        optimized_knots = optimized_knots[:-1] + [new_knots[1]]
                        all_knots_new = time_extension(optimized_knots)
                        all_knots_new = time_extension(all_knots_new)
                        all_knots_new = time_extension(all_knots_new)
                        all_knots_start_bool = all_knots_new == optimize_time[0]
                        all_knots_start_bool = np.append(all_knots_start_bool[3:], np.asarray((0, 0, 0)))
                        all_knots_b_bool = all_knots_new == b_right
                        all_knots_b_bool = np.append(np.asarray((0, 0, 0, 0)), all_knots_b_bool[:-4])
                        all_knots_bool = (all_knots_start_bool + all_knots_b_bool) == 1
                        all_knots_bool = np.cumsum(all_knots_bool) == 1
                        all_knots_extension = all_knots_new[all_knots_bool]  # assume correct for now

                        optimize_basis = Basis(time=new_t, time_series=new_time_series)  #

                        b_spline_matrix = optimize_basis.cubic_b_spline(all_knots_extension)
                        # using redefined Basis object

                        second_order_matrix = np.zeros((np.shape(b_spline_matrix)[0],
                                                        np.shape(b_spline_matrix)[0] - 2))  # note transpose
                        for d in range(np.shape(b_spline_matrix)[0] - 2):
                            second_order_matrix[d:(d + 3), d] = [1, -2,
                                                                 1]  # filling values for second-order difference matrix
                        b_spline_matrix_2 = np.append(b_spline_matrix, lamda *
                                                      second_order_matrix, axis=1)  # l2 norm trick

                        # l2 norm trick
                        new_time_series_2 = np.append(new_time_series, np.zeros(np.shape(b_spline_matrix_2)[0] - 2),
                                                      axis=0)

                        coef = np.linalg.lstsq(b_spline_matrix_2.transpose(), new_time_series_2, rcond=None)
                        coef = coef[0]

                        new_approximation = np.matmul(coef, b_spline_matrix)

                        new_error = np.sum(np.abs(new_time_series[np.r_[a <= new_t] & np.r_[new_t <= b_right]] -
                                                  new_approximation[np.r_[a <= new_t] & np.r_[new_t <= b_right]]))

                        b_diff = (b_right - b_left)

                        if new_error > error:

                            b_right = b_left + (1 / 2) * b_diff
                            temp = 0

                        else:

                            b_left = b_right
                            b_right = b_left + (1 / 2) * b_diff
                            temp = 1

                if temp == 0:
                    b = b_right + (1 / 2) * b_diff
                elif temp == 1:
                    b = b_left
                a = b
                b = optimize_time[-1]

        return optimized_knots

    def chsi_basis(self, knots: np.ndarray, full_or_not: str = 'full') -> np.ndarray:
        """
        Cubic Hermite Spline Interpolation basis matrix construction.

        Parameters
        ----------
        knots : array_like
            Knots for piecewise cubic spline.

        full_or_not : string_like
            'full' - input or estimate all parameters.
            'derivative' - input or estimate only tangent values.
            'intersect' - input or estimate only intersect values.

        Returns
        -------
        basis_matrix : real ndarray
            Matrix constructed in a way as to accommodate different optimization as detailed in notes.

        Notes
        -----
        Basis is already constructed in such as a way as to ensure required parameters are equal to ensure continuity.
        Scaling factor for tangent bases is done here (essential for non-unit intervals).
        'knots' should be a proper subset of 'time' (unlike cubic b-spline).

        """
        chsi_time = self.time
        k = {}  # removes unnecessary error
        bool_vector = {}  # removes unnecessary error
        subset_time = {}  # removes unnecessary error

        if full_or_not == 'full':

            basis_matrix = np.zeros((2 * (len(knots)), len(chsi_time)))

            for k in range(len(knots) - 1):

                bool_vector = np.r_[np.round(chsi_time, 5) >= np.round(knots[k], 5)] & np.r_[
                    np.round(chsi_time, 5) <= np.round(knots[k + 1], 5)]
                subset_time = chsi_time[bool_vector]

                if k == 0:

                    basis_matrix[int(2 * k), bool_vector] = chsi(subset_time, 1, 0, 0, 0)
                    basis_matrix[int(2 * k + 1), bool_vector] = (subset_time[-1] -
                                                                 subset_time[0]) * chsi(subset_time, 0, 1, 0, 0)

                else:

                    bool_vector_prev = np.r_[np.round(chsi_time, 5) >= np.round(knots[k - 1], 5)] & np.r_[
                        np.round(chsi_time, 5) < np.round(knots[k], 5)]
                    temp_time_prev = chsi_time[bool_vector_prev]

                    basis_matrix[int(2 * k), bool_vector] = chsi(subset_time, 1, 0, 0, 0)
                    basis_matrix[int(2 * k), bool_vector_prev] += chsi(temp_time_prev, 0, 0, 1, 0)

                    basis_matrix[int(2 * k + 1), bool_vector] = (subset_time[-1] - subset_time[0]) * chsi(subset_time,
                                                                                                          0, 1, 0, 0)
                    basis_matrix[int(2 * k + 1), bool_vector_prev] += (temp_time_prev[-1] - temp_time_prev[0]) * chsi(
                        temp_time_prev, 0, 0, 0, 1)

            basis_matrix[2 * (k + 1), bool_vector] = chsi(subset_time, 0, 0, 1, 0)
            basis_matrix[2 * (k + 1) + 1, bool_vector] = (subset_time[-1] - subset_time[0]) * chsi(subset_time,
                                                                                                   0, 0, 0, 1)

        else:

            basis_matrix = np.zeros((len(knots), len(chsi_time)))

            if full_or_not == 'derivative':

                for k in range(len(knots) - 1):

                    bool_vector = np.r_[np.round(chsi_time, 5) >= np.round(knots[k], 5)] & np.r_[
                        np.round(chsi_time, 5) <= np.round(knots[k + 1], 5)]
                    subset_time = chsi_time[bool_vector]

                    if k == 0:

                        basis_matrix[k, bool_vector] = (subset_time[-1] - subset_time[0]) * chsi(subset_time,
                                                                                                 0, 1, 0, 0)

                    else:

                        bool_vector_prev = np.r_[np.round(chsi_time, 5) >= np.round(knots[k - 1], 5)] & np.r_[
                            np.round(chsi_time, 5) <= np.round(knots[k], 5)]
                        temp_time_prev = chsi_time[bool_vector_prev]

                        basis_matrix[k, bool_vector] = (subset_time[-1] - subset_time[0]) * chsi(subset_time,
                                                                                                 0, 1, 0, 0)
                        basis_matrix[k, bool_vector_prev] = (temp_time_prev[-1] - temp_time_prev[0]) * chsi(
                            temp_time_prev,
                            0, 0, 0, 1)

                basis_matrix[k + 1, bool_vector] = (subset_time[-1] - subset_time[0]) * chsi(subset_time, 0, 0, 0, 1)

            elif full_or_not == 'intersect':

                for k in range(len(knots) - 1):

                    bool_vector = np.r_[np.round(chsi_time, 5) >= np.round(knots[k], 5)] & np.r_[
                        np.round(chsi_time, 5) <= np.round(knots[k + 1], 5)]
                    subset_time = chsi_time[bool_vector]

                    if k == 0:

                        basis_matrix[k, bool_vector] = chsi(subset_time, 1, 0, 0, 0)

                    else:

                        bool_vector_prev = np.r_[np.round(chsi_time, 5) >= np.round(knots[k - 1], 5)] & np.r_[
                            np.round(chsi_time, 5) <= np.round(knots[k], 5)]
                        temp_time_prev = chsi_time[bool_vector_prev]

                        basis_matrix[k, bool_vector] = chsi(subset_time, 1, 0, 0, 0)
                        basis_matrix[k, bool_vector_prev] = chsi(temp_time_prev, 0, 0, 1, 0)

                basis_matrix[k + 1, bool_vector] = chsi(subset_time, 0, 0, 1, 0)

        return basis_matrix

    def chsi_fit(self, knots_time: np.ndarray, knots: np.ndarray, type_chsi_fit: str = 'signal') -> np.ndarray:
        """
        Cubic Hermite Spline Interpolation fitting function.

        Parameters
        ----------
        knots_time : array_like
            Time series over which basis functions and signal approximation will be defined.

        knots : array_like
            Knots for basis function construction.

        type_chsi_fit : string_like
            'signal' - approximate entire signal
            'envelope' - approximate extrema envelopes

        Returns
        -------
        chsi_approx : real ndarray
            Cubic Hermite spline interpolation.

        Notes
        -----
        "knots" and "knots_time" are extrema when constructing extrema envelope.

        """
        chsi_approx = {}  # removes unnecessary error

        chsi_time = self.time
        chsi_time_series = self.time_series

        if type_chsi_fit == 'signal':

            ps = np.zeros_like(knots)

            for t_1 in range(len(knots)):

                subset_time = chsi_time - knots[t_1]
                min_bool_1 = np.abs(chsi_time - knots[t_1]) == np.min(
                    np.abs(chsi_time - knots[t_1]))  # finds nearest

                if sum(min_bool_1) > 1:  # two time points equal in distance from knot
                    min_bool_1 = (np.cumsum(min_bool_1) == 1)

                min_time = subset_time[min_bool_1]

                # points match exactly - this should always be the case otherwise there will be discontinuities
                if min_time == 0:

                    ps[t_1] = chsi_time_series[min_bool_1]

                elif min_time < 0:  # knot point to the right of nearest point
                    # need to use point to right for interpolation
                    # time_point_min, knot, next_time_point

                    min_bool_2 = (np.append(0, min_bool_1[:-1]) == 1)
                    time_left = chsi_time[min_bool_1]
                    time_right = chsi_time[min_bool_2]

                    ps[t_1] = (chsi_time_series[min_bool_1] * (time_right - knots[t_1]) +
                               chsi_time_series[min_bool_2] * (knots[t_1] - time_left)) / (time_right - time_left)

                elif min_time > 0:  # knot point to the left of nearest point
                    # need to use point to left for interpolation
                    # next_time_point, knot, min_time_point

                    min_bool_2 = (np.append(min_bool_1[1:], 0) == 1)
                    time_right = chsi_time[min_bool_1]
                    time_left = chsi_time[min_bool_2]

                    ps[t_1] = (chsi_time_series[min_bool_2] * (time_right - knots[t_1]) +
                               chsi_time_series[min_bool_1] * (knots[t_1] - time_left)) / (time_right - time_left)

            new_signal = chsi_time_series - np.matmul(ps, Basis.chsi_basis(self, knots, full_or_not='intersect'))
            # removes intersect bases - does not affect estimation of parameters

            new_basis_matrix = Basis.chsi_basis(self, knots, full_or_not='derivative')
            # constructs basis for tangent values

            coef = np.linalg.lstsq(new_basis_matrix.transpose(), new_signal, rcond=None)[0]
            # estimates coefficients for tangent basis matrix

            new_basis_matrix_extended = Basis.chsi_basis(self, knots, full_or_not='derivative')
            # constructs extended matrix

            chsi_approx = np.matmul(coef,
                                    new_basis_matrix_extended) + np.matmul(ps,
                                                                           Basis.chsi_basis(self, knots,
                                                                                            full_or_not='intersect'))
            # recombines intersects and tangents into approximation

        elif type_chsi_fit == 'envelope':

            ps = np.hstack((chsi_time_series[0], chsi_time_series, chsi_time_series[-1]))
            knots = np.hstack((knots_time[0], chsi_time, knots_time[-1]))

            ms = np.zeros_like(ps)
            ms[1:-1] = (1 / 2) * ((ps[2:] - ps[1:-1]) / (knots[2:] - knots[1:-1]) +
                                  (ps[1:-1] - ps[:-2]) / (knots[1:-1] - knots[:-2]))  # finite difference method

            basis_chsi = Basis(time=knots_time, time_series=knots_time)

            chsi_approx = \
                np.matmul(ps, basis_chsi.chsi_basis(knots, full_or_not='intersect'))\
                + np.matmul(ms, basis_chsi.chsi_basis(knots, full_or_not='derivative'))

        return chsi_approx

    def asi_fit(self, knots_time: np.ndarray, knots: np.ndarray, type_asi_fit: str = 'signal') -> np.ndarray:
        """
        Akima Spline Interpolation fitting method.

        Parameters
        ----------
        knots_time : array_like
            Time series over which basis functions and signal approximation will be defined.

        knots : array_like
            Knots for basis function construction.

        type_asi_fit : string_like
            'signal' - approximate entire signal
            'envelope' - approximate extrema envelopes

        Returns
        -------
        akima_spline : real ndarray
            Akima spline interpolation.

        Notes
        -----
        "knots" and "knots_time" are extrema when constructing extrema envelope.

        """
        ys = {}  # removes unnecessary error
        ps = {}  # removes unnecessary error
        ms = {}  # removes unnecessary error
        xs = {}  # removes unnecessary error

        asi_time = self.time
        asi_time_series = self.time_series

        if type_asi_fit == 'signal':

            ys = np.zeros_like(knots)

            for t_1 in range(len(knots)):

                subset_time = asi_time - knots[t_1]
                min_bool_1 = np.abs(asi_time - knots[t_1]) == np.min(
                    np.abs(asi_time - knots[t_1]))  # finds nearest

                if sum(min_bool_1) > 1:
                    min_bool_1 = (np.cumsum(min_bool_1) == 1)

                min_time = subset_time[min_bool_1]

                # points match exactly - this should always be the case otherwise there will be discontinuities
                if min_time == 0:

                    ys[t_1] = asi_time_series[min_bool_1]

                elif min_time < 0:  # knot point to the right of nearest point
                    # need to use point to right for interpolation
                    # time_point_min, knot, next_time_point

                    min_bool_2 = (np.append(0, min_bool_1[:-1]) == 1)
                    time_left = asi_time[min_bool_1]
                    time_right = asi_time[min_bool_2]

                    ys[t_1] = (asi_time_series[min_bool_1] * (time_right - knots[t_1]) + asi_time_series[min_bool_2] * (
                            knots[t_1] - time_left)) / (time_right - time_left)

                elif min_time > 0:  # knot point to the left of nearest point
                    # need to use point to left for interpolation
                    # next_time_point, knot, min_time_point

                    min_bool_2 = (np.append(min_bool_1[1:], 0) == 1)
                    time_right = asi_time[min_bool_1]
                    time_left = asi_time[min_bool_2]

                    ys[t_1] = (asi_time_series[min_bool_2] * (time_right - knots[t_1]) + asi_time_series[min_bool_1] * (
                            knots[t_1] - time_left)) / (time_right - time_left)

            # ys calculated above
            xs = knots

            ms = np.zeros_like(ys)
            ps = np.zeros_like(ys)

        elif type_asi_fit == 'envelope':

            ys = np.hstack((asi_time_series[0], asi_time_series, asi_time_series[-1]))
            xs = np.hstack((knots_time[0], asi_time, knots_time[-1]))

            ms = np.zeros_like(ys)
            ps = np.zeros_like(ys)

            knots = xs.copy()

        akima_spline = np.zeros_like(knots_time)

        c = -1

        for d in range(len(ys)):

            ps[d] = ys[d]

            if d in [0, 1, int(len(ys) - 2), int(len(ys) - 1)]:

                if d == 0:
                    ms[d] = m(xs[d], xs[d + 1], ys[d], ys[d + 1])  # use first forward difference

                elif d == int(len(ys) - 1):
                    ms[d] = m(xs[d - 1], xs[d], ys[d - 1], ys[d])  # use first backward difference

                else:
                    ms[d] = (m(xs[d], xs[d + 1], ys[d], ys[d + 1]) + m(xs[d - 1], xs[d], ys[d - 1],
                                                                       ys[d])) / 2  # average

            else:

                tau = range(d - 2, d + 3)

                ms[d] = asi_si(xs[tau[0]], xs[tau[1]], xs[tau[2]], xs[tau[3]], xs[tau[4]],
                               ys[tau[0]], ys[tau[1]], ys[tau[2]], ys[tau[3]], ys[tau[4]])  # standard akima

            if c >= 0:  # this is where spline is iteratively built

                bool_vector = np.r_[np.round(knots_time, 5) >= np.round(knots[c], 5)] & \
                              np.r_[np.round(knots_time, 5) <= np.round(knots[c + 1], 5)]
                # rounding error at final decimal point

                subset_time = knots_time[bool_vector]

                # need to scale tangent values

                scale = subset_time[-1] - subset_time[0]

                akima_spline[bool_vector] = chsi(subset_time, ps[c], ms[c] * scale, ps[c + 1], ms[c + 1] * scale)

            c += 1

        return akima_spline

    def basis_function_approximation(self, knots: np.ndarray, knot_time: np.ndarray,
                                     spline_method: str = 'b_spline') -> (np.ndarray, np.ndarray):
        """
        Fits a smoothing spline to the original signal.

        Least square fits parameter vector 'coef' using basis functions such that:
        ||(cubic_basis_function_matrix^T)(coef) - raw_signal||^2 is minimized.

        Approximation is then calculated by:
        (cubic_basis_function_matrix_extended^T)(coef).

        This increases (or decreases) the number of time points over which approximation calculated.
        New smoothed signal will have same number of points as knot_time vector.

        Parameters
        ----------
        knots : real ndarray
            Knot points to be used in cubic basis spline construction.
            Possible extension to be optimized over.

        knot_time : real ndarray
            Time points used in construction of basis functions.
            This is done for two reasons:
                - assists in smoothing the signal at the edges, and
                - is later used in mirror technique for dealing with edges.

        spline_method : string_like
            Spline method to be used.

        Returns
        -------
        approximation : real ndarray
            Basis spline approximation.

        coef : real ndarray
            Corresponding coefficients.

        Notes
        -----
        Essentially a least square fitting function.

        """
        approximation = {}  # avoids unnecessary error
        coef = {}  # avoids unnecessary error

        time = self.time
        time_series = self.time_series

        if spline_method == 'b_spline':

            # knots_fix
            # require four knots either side for non-natural spline
            new_knots = time_extension(knots)
            new_knots_bool = ((new_knots >= knots[0]) & (new_knots <= knots[-1]))
            column_number = len(knots) - 1
            new_knots_bool[int(column_number - 3):int(column_number)] = [True, True, True]
            new_knots_bool[(- int(column_number)):(- int(column_number - 3))] = [True, True, True]
            knots = new_knots[new_knots_bool]

            new_signal_time = time_extension(time)
            new_signal_time = new_signal_time[(new_signal_time >= knots[0]) & (new_signal_time <= knots[-1])]

            basis_function_extended_time = Basis(time=new_signal_time, time_series=time_series)
            # basis_function_extended_time = Basis(time=time, time_series=time_series)

            cubic_basis_function_matrix = basis_function_extended_time.cubic_b_spline(knots)
            cubic_basis_function_matrix = cubic_basis_function_matrix[:, ((new_signal_time >= time[0]) &
                                                                          (new_signal_time <= time[-1]))]

            # optimise speed - pseudo inverse
            try:
                coef = solve(cubic_basis_function_matrix.dot(cubic_basis_function_matrix.T),
                             cubic_basis_function_matrix.dot(time_series))
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(cubic_basis_function_matrix.T, time_series, rcond=None)
                coef = coef[0]

            new_knots_time = time_extension(knot_time)
            new_knots_time = new_knots_time[(new_knots_time >= knots[0]) & (new_knots_time <= knots[-1])]

            basis_function_extended_knot_time = Basis(time=new_knots_time, time_series=time_series)

            cubic_basis_function_matrix_extended = basis_function_extended_knot_time.cubic_b_spline(knots)
            cubic_basis_function_matrix_extended = cubic_basis_function_matrix_extended[:, ((new_knots_time >=
                                                                                             time[0]) &
                                                                                            (new_knots_time <=
                                                                                             time[-1]))]

            approximation = np.matmul(coef, cubic_basis_function_matrix_extended)

        elif spline_method == 'chsi':

            approximation = Basis.chsi_fit(self, knots_time=knot_time, knots=knots)

            coef = np.zeros_like(approximation)

        elif spline_method == 'asi':

            approximation = Basis.asi_fit(self, knots_time=knot_time, knots=knots)

            coef = np.zeros_like(approximation)

        return approximation, coef

    def basis_function_approximation_matrix(self, b_spline_matrix_signal: np.ndarray,
                                            b_spline_matrix_smooth: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        When using B-splines this increases the speed of the algorithm.
        Relevant matrices are calculated once at the outset.
        'b-spline_matrix_signal' is used to estimate the coefficients.
        'b_spline_matrix_smooth' is used to calculate smoothed signal.
        'b_spline_matrix_signal' and 'b_spline_matrix_smooth' have same number of rows, but different number of columns.

        Fits a smoothing spline to the original signal using b_spline_matrix_signal.

        Least square fits parameter vector 'coef' using basis functions such that:
        ||(b_spline_matrix_signal^T)(coef) - raw_signal||^2 is minimized.

        Approximation is then calculated by:
        (b_spline_matrix_smooth^T)(coef).

        This increases (or decreases) the number of time points over which approximation calculated.

        Parameters
        ----------
        b_spline_matrix_signal : real ndarray
            Each row is a basis function with number of columns dependant length of signal.

        b_spline_matrix_smooth : real ndarray
            Each row is a basis function with number of columns variable.

        Returns
        -------
        approximation : real ndarray
            Basis spline approximation.

        coef : real ndarray
            Corresponding coefficients.

        Notes
        -----
        Essentially a least square fitting function.

        """
        time_series = self.time_series

        # optimise speed - pseudo inverse
        try:
            coef = solve(b_spline_matrix_signal.dot(b_spline_matrix_signal.T),
                         b_spline_matrix_signal.dot(time_series))
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(b_spline_matrix_signal.transpose(), time_series, rcond=None)
            coef = coef[0]

        approximation = np.matmul(coef, b_spline_matrix_smooth)

        return approximation, coef
