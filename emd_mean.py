
#     ________
#            /
#      \    /
#       \  /
#        \/

# emd class to handle all detrended fluctuation analysis

import numpy as np
from numpy.linalg import solve
import seaborn as sns
import matplotlib.pyplot as plt

from emd_utils import Utility, time_extension
from emd_basis import Basis

sns.set(style='darkgrid')


def envelope_basis_function(time: np.ndarray, time_series: np.ndarray, extrema_bool_max: np.ndarray,
                            extrema_bool_min: np.ndarray, cubic_basis_function_matrix_extended: np.ndarray,
                            new_knots_time: np.ndarray, subset_time_vector_bool: np.ndarray,
                            knots_for_envelope: np.ndarray, extrema_type: str, smooth: bool, smoothing_penalty: float,
                            edge_effect: str, alpha: float, nn_m: int, nn_k: int, nn_method: str,
                            nn_learning_rate: float, nn_iter: int) -> (np.ndarray, np.ndarray):
    """
    Function called by methods separate, but related methods below to prevent unnecessary repetition.

    Parameters
    ----------
    time : real ndarray
        Time associated with time series.

    time_series : real ndarray
        Time series used in construction of local mean.

    extrema_bool_max : real ndarray (boolean)
        Array containing booleans of maxima positions.

    extrema_bool_min : real ndarray (boolean)
        Array containing booleans of minima positions.

    knots_for_envelope : array_like
        Knot points to be used in cubic basis spline envelope construction.
        Possible extension to be optimized over.

    cubic_basis_function_matrix_extended : real ndarray
        Extended basis matrix for edge fitting and smoothing.

    new_knots_time : real ndarray
        Extended knot array for edge fitting.

    subset_time_vector_bool : real ndarray (boolean)
        Boolean array to extract original time vector from extended time vector.

    extrema_type : string_like
        'maxima' or 'minima'.

    smooth: boolean
        Whether or not to smooth signal.

    smoothing_penalty: float
        Smoothing penalty for second-order smoothing of parameter estimates to be multiplied on to
        basis function matrix.

    edge_effect: string_like
        What edge technique to use. Default set to symmetric that mirrors nearest extrema.

    alpha : float
        Used in symmetric edge-effect to decide whether to anchor spline.
        When alpha = 1 technique reduces to symmetric 'Symmetric Anchor' in:

        K. Zeng and M. He. A Simple Boundary Process Technique for Empirical Mode
        Decomposition. In IEEE International Geoscience and Remote Sensing Symposium,
        volume 6, pages 4258–4261. IEEE, 2004.

        or

        J. Zhao and D. Huang. Mirror Extending and Circular Spline Function for Empirical
        Mode Decomposition Method. Journal of Zhejiang University - Science A, 2(3):
        247–252, 2001.

        Other extreme is alpha = -infinity. Time series is reflected and next extremum is taken.

    nn_m : integer
        Number of points (outputs) on which to train in neural network edge effect.

    nn_k : integer
        Number of points (inputs) to use when estimating weights for neuron.

    nn_method : string_like
        Gradient descent method used to estimate weights.

    nn_learning_rate : float
        Learning rate to use when adjusting weights.

    nn_iter : integer
        Number of iterations to perform when estimating weights.

    Returns
    -------
    approximation : real ndarray
        Cubic basis spline envelope fitted through extrema.

    coef : real ndarray
        Corresponding coefficients.

    Notes
    -----
    Least square fitting function of extrema.
    Envelopes are second-order smoothed.

    (1) Add CHSI interpolation option. SOLVED.
    (2) Add ASI interpolation option. SOLVED.
    (3) Add 'Symmetric Discard' edge effect. SOLVED.
    (4) Add 'Symmetric Anchor' edge effect. SOLVED.
    (5) Add 'Symmetric' edge effect. SOLVED.
    (6) Add 'Conditional Symmetric Anchor' edge effect. SOLVED.
        - either 'Symmetric Anchor' or 'Symmetric' depending on condition.
        - analogous to 'Improved Slope-Based Method'.
    (7) Add 'Anti-Symmetric' edge effect. SOLVED.
    (8) Add 'Slope-Based' method. SOLVED.
    (9) Add 'Improved Slope-Based' method. SOLVED.
    (10) Add 'Huang Characteristic Wave' edge effect. SOLVED.
    (11) Add 'Coughlin Characteristic Wave' edge effect. SOLVED.
    (12) Add 'Average Characteristic Wave' edge effect. SOLVED.
    (13) Add 'Neural Network' edge effect. SOLVED.

    """
    extrema_basis_function_matrix = {}  # avoids unnecessary error
    extrema = {}  # avoids unnecessary error
    cubic_basis_function_matrix = {}  # avoids unnecessary error
    points_until_first_extreme_from_left_extract = {}  # avoids unnecessary error
    points_until_first_extreme_from_right_extract = {}  # avoids unnecessary error
    points_until_first_minima_from_left_extract = {}  # avoids unnecessary error
    points_until_first_minima_from_right_extract = {}  # avoids unnecessary error
    points_until_first_maxima_from_left_extract = {}  # avoids unnecessary error
    points_until_first_maxima_from_right_extract = {}  # avoids unnecessary error
    left_length = {}  # avoids unnecessary error
    right_length = {}  # avoids unnecessary error

    utility_extrema = Utility(time, time_series)

    try:
        # Left values
        left_maxima = time_series[extrema_bool_max][0]
        left_maxima_x = np.array(time)[extrema_bool_max][0]
        left_minima = time_series[extrema_bool_min][0]
        left_minima_x = np.array(time)[extrema_bool_min][0]
        diff_left_1 = left_maxima - left_minima
        left_signal = time_series[0]

        # Right values
        right_maxima = time_series[extrema_bool_max][-1]
        right_maxima_x = np.array(time)[extrema_bool_max][-1]
        right_minima = time_series[extrema_bool_min][-1]
        right_minima_x = np.array(time)[extrema_bool_min][-1]
        diff_right_1 = right_maxima - right_minima
        right_signal = time_series[-1]
    except IndexError:  # deals with trend error
        approximation = {}  # avoids unnecessary error
        coef = {}  # avoids unnecessary error
        if extrema_type == 'maxima':
            approximation = max(time_series) * np.ones_like(time_series)
            coef = max(time_series) * np.ones(len(knots_for_envelope) + int(2))
        elif extrema_type == 'minima':
            approximation = min(time_series) * np.ones_like(time_series)
            coef = min(time_series) * np.ones(len(knots_for_envelope) + int(2))
        return approximation, coef

    # add neural network edge effect - top

    if edge_effect == 'neural_network':

        time_extended = time_extension(time)
        time_series_extended = np.zeros_like(time_extended) / 0
        time_series_extended[int(len(time_series) - 1):int(2 * (len(time_series) - 1) + 1)] = time_series

        cubic_basis_function_matrix = cubic_basis_function_matrix_extended[:, subset_time_vector_bool]

        # forward ->

        p_matrix = np.zeros((nn_k, nn_m))
        for col in range(nn_m):
            p_matrix[:, col] = time_series[(-(nn_m + nn_k - col)):(-(nn_m - col))]
        t = time_series[-nn_m:]

        # calculate right weights - top

        seed_weights = np.ones(nn_k) / nn_k
        weights_right = seed_weights.copy()  # seed weights
        train_input = p_matrix  # training input matrix
        adjustment = {}  # avoids unnecessary error

        for iteration in range(nn_iter):

            output = np.matmul(weights_right, train_input)  # calculate output vector with given weights

            error = (t - output)  # calculate error vector for given output

            gradients = error * (- train_input)  # calculate matrix containing all gradient values

            # calculate average of gradient values over ever output
            average_gradient_vector = np.mean(gradients, axis=1)

            # adjustment uses all gradients
            if nn_method == 'grad_descent':

                adjustment = - nn_learning_rate * average_gradient_vector

            # adjustment uses only largest absolute gradient
            elif nn_method == 'steep_descent':

                # calculate maximum of gradient over average gradient values
                max_gradient_vector = \
                    average_gradient_vector * \
                    (np.abs(average_gradient_vector) == max(np.abs(average_gradient_vector)))

                adjustment = - nn_learning_rate * max_gradient_vector

            weights_right += adjustment

        # calculate right weights - bottom

        max_count_right = 0
        min_count_right = 0
        i_right = 0
        extended_time_right = 0

        while ((max_count_right < 1) or (min_count_right < 1)) and (i_right < len(time_series) - 1) and \
                (np.nanmax(np.abs(extended_time_right)) < 10 * np.max(np.abs(time_series))):

            time_series_extended[int(2 * (len(time_series) - 1) + 1 + i_right)] = \
                sum(weights_right * time_series_extended[int(
                    2 * (len(time_series) - 1) + 1 - nn_k + i_right):int(
                    2 * (len(time_series) - 1) + 1 + i_right)])

            i_right += 1

            if i_right > 1:

                extended_time_right = \
                    time_extended[int(2 * (len(time_series) - 1) + 1):
                                  int(2 * (len(time_series) - 1) + 1 + i_right + 1)]
                extended_time_series_right = \
                    time_series_extended[int(2 * (len(time_series) - 1) + 1):
                                         int(2 * (len(time_series) - 1) + 1 + i_right + 1)]
                extended_utility_right = Utility(time=extended_time_right,
                                                 time_series=extended_time_series_right)

                if sum(extended_utility_right.max_bool_func_1st_order_fd()) > 0:
                    max_count_right += 1

                if sum(extended_utility_right.min_bool_func_1st_order_fd()) > 0:
                    min_count_right += 1

        # manage exponential edge explosion
        if np.nanmax(np.abs(extended_time_right)) >= 10 * np.max(np.abs(time_series)):
            time_series_extended[int(2 * (len(time_series) - 1) + 1):] = time_series[::-1][1:]

        # backward <-

        p_matrix = np.zeros((nn_k, nn_m))
        for col in range(nn_m):
            p_matrix[:, col] = time_series[int(col + 1):int(col + nn_k + 1)]
        t = time_series[:nn_m]

        # calculate left weights - top

        seed_weights = np.ones(nn_k) / nn_k
        weights_left = seed_weights.copy()  # seed weights
        train_input = p_matrix  # training input matrix

        for iteration in range(nn_iter):

            output = np.matmul(weights_left, train_input)  # calculate output vector with given weights

            error = (t - output)  # calculate error vector for given output

            gradients = error * (- train_input)  # calculate matrix containing all gradient values

            # calculate average of gradient values over ever output
            average_gradient_vector = np.mean(gradients, axis=1)

            # adjustment uses all gradients
            if nn_method == 'grad_descent':

                adjustment = - nn_learning_rate * average_gradient_vector

            # adjustment uses only largest absolute gradient
            elif nn_method == 'steep_descent':

                # calculate maximum of gradient over average gradient values
                max_gradient_vector = average_gradient_vector * (
                        np.abs(average_gradient_vector) == max(np.abs(average_gradient_vector)))

                adjustment = - nn_learning_rate * max_gradient_vector

            weights_left += adjustment

        # calculate left weights - bottom

        max_count_left = 0
        min_count_left = 0
        i_left = 0
        extended_time_left = 0

        while ((max_count_left < 1) or (min_count_left < 1)) and (i_left < len(time_series) - 1) and \
                (np.nanmax(np.abs(extended_time_left)) < 10 * np.max(np.abs(time_series))):

            time_series_extended[int(len(time_series) - 2 - i_left)] = \
                sum(weights_left * time_series_extended[int(len(time_series) - 1 - i_left):int(
                    len(time_series) - 1 - i_left + nn_k)])

            i_left += 1

            if i_left > 1:

                extended_time_left = time_extended[int(len(time_series) - 1 - i_left):
                                                   int(len(time_series))]
                extended_time_series_left = time_series_extended[int(len(time_series) - 1 - i_left):
                                                                 int(len(time_series))]
                extended_utility_left = Utility(time=extended_time_left, time_series=extended_time_series_left)

                if sum(extended_utility_left.max_bool_func_1st_order_fd()) > 0:
                    max_count_left += 1

                if sum(extended_utility_left.min_bool_func_1st_order_fd()) > 0:
                    min_count_left += 1

        # manage exponential edge explosion
        if np.nanmax(np.abs(extended_time_left)) >= 10 * np.max(np.abs(time_series)):
            time_series_extended[:int(len(time_series)-1)] = time_series[::-1][:-1]

        extrema_bool = {}  # avoids unnecessary error
        extended_utility = Utility(time=time_extended, time_series=time_series_extended)
        if extrema_type == 'maxima':
            extrema_bool = extended_utility.max_bool_func_1st_order_fd()
        elif extrema_type == 'minima':
            extrema_bool = extended_utility.min_bool_func_1st_order_fd()

        extrema_basis_function_matrix = cubic_basis_function_matrix_extended[:, extrema_bool]

        extrema = time_series_extended[extrema_bool]

    # add neural network edge effect - bottom

    elif edge_effect[:9] == 'symmetric':

        extrema_bool = {}  # avoids unnecessary error
        # symmetric edge-effect only requires one set of extrema
        if extrema_type == 'maxima':
            extrema_bool = extrema_bool_max
        elif extrema_type == 'minima':
            extrema_bool = extrema_bool_min

        points_until_first_extreme_from_left = (
                np.cumsum(np.cumsum(extrema_bool)) <= 1)  # boolean for points until first extreme from left

        first_extreme_from_left = (np.cumsum(np.cumsum(extrema_bool)) == 1)
        # boolean for first extreme from left

        points_until_first_extreme_from_right = np.flip(
            np.cumsum(np.cumsum(np.flip(extrema_bool))) <= 1)
        # boolean for points until first extreme from right

        first_extreme_from_right = np.flip(
            np.cumsum(np.cumsum(np.flip(extrema_bool))) == 1)  # boolean for first extreme from right

        points_until_first_extreme_from_left_extract = points_until_first_extreme_from_left[
            points_until_first_extreme_from_left]  # extracts the correct size boolean vector on left

        points_until_first_extreme_from_right_extract = points_until_first_extreme_from_right[
            points_until_first_extreme_from_right]  # extracts the correct size boolean vector on right

        # from left
        # better way to do this

        temp_left = np.cumsum(subset_time_vector_bool)  # continued below

        temp_left = (temp_left == 1)  # finds column for first time point from left in signal

        left_column_number = np.array(range(len(temp_left)))
        # creates vector of column numbers - continued below

        left_column_number = left_column_number[temp_left]  # extracts column number

        subset_time_vector_bool[int(left_column_number - len(points_until_first_extreme_from_left_extract)):int(
            left_column_number)] = points_until_first_extreme_from_left_extract
        # inserts boolean (vector of ones) using column number

        # from right
        # better way to do this

        temp_right = np.cumsum(np.flip(subset_time_vector_bool))  # continued below

        temp_right = (np.flip(temp_right) == 1)  # finds column for first time point from right in signal

        right_column_number = np.array(range(len(temp_right)))
        # creates vector of column numbers - continued below

        right_column_number = right_column_number[temp_right]  # extracts column number

        subset_time_vector_bool[(int(right_column_number) + 1): int(right_column_number + len(
            points_until_first_extreme_from_right_extract) + 1)] = points_until_first_extreme_from_right_extract
        # inserts boolean (vector of ones) using column number #

        # combines all of the above

        cubic_basis_function_matrix = cubic_basis_function_matrix_extended[:, subset_time_vector_bool]
        # extract points only relevant to "adjusted" signal

        first_extreme_from_left_extract = first_extreme_from_left[
            points_until_first_extreme_from_left]  # extracts vector to add to left of extrema vector

        first_extreme_from_right_extract = first_extreme_from_right[
            points_until_first_extreme_from_right]  # extracts vector to add to right of extrema vector

        if edge_effect[(-6):] == 'anchor':
            if extrema_type == 'maxima':
                # If left minima is left-most extremum
                if left_minima_x < left_maxima_x:
                    diff_left_2 = left_signal - left_minima
                    if diff_left_2 > (1 - alpha) * diff_left_1:
                        first_extreme_from_left_extract = np.zeros_like(first_extreme_from_left_extract)
                        first_extreme_from_left = np.zeros_like(first_extreme_from_left)
                        extrema_bool[0] = 1
                # If right minima is right-most extremum
                if right_minima_x > right_maxima_x:
                    diff_right_2 = np.abs(right_signal - right_minima)
                    if diff_right_2 > (1 - alpha) * diff_right_1:
                        first_extreme_from_right_extract = np.zeros_like(first_extreme_from_right_extract)
                        first_extreme_from_right = np.zeros_like(first_extreme_from_right)
                        extrema_bool[-1] = 1
            elif extrema_type == 'minima':
                # If left maxima is left-most extremum
                if left_maxima_x < left_minima_x:
                    diff_left_2 = np.abs(left_signal - left_maxima)
                    if diff_left_2 > (1 - alpha) * diff_left_1:
                        first_extreme_from_left_extract = np.zeros_like(first_extreme_from_left_extract)
                        first_extreme_from_left = np.zeros_like(first_extreme_from_left)
                        extrema_bool[0] = 1
                # If right maxima is right-most extremum
                if right_maxima_x > right_minima_x:
                    diff_right_2 = np.abs(right_signal - right_maxima)
                    if diff_right_2 > (1 - alpha) * diff_right_1:
                        first_extreme_from_right_extract = np.zeros_like(first_extreme_from_right_extract)
                        first_extreme_from_right = np.zeros_like(first_extreme_from_right)
                        extrema_bool[-1] = 1

        extrema_basis_function_matrix = cubic_basis_function_matrix[
                                        :, np.hstack((np.flip(first_extreme_from_left_extract),
                                                      extrema_bool, np.flip(first_extreme_from_right_extract)))]

        extrema = np.hstack(
            (time_series[first_extreme_from_left], time_series[extrema_bool],
             time_series[first_extreme_from_right]))
        # appends mirrored extrema to either side of extrema vector
        # adds them to extrema vector #

        if edge_effect[(-7):] == 'discard':

            maxima = time_series[extrema_bool_max]
            minima = time_series[extrema_bool_min]
            indices = np.arange(len(time))

            if extrema_type == 'maxima':
                if len(maxima) > 1:
                    if left_maxima_x < left_minima_x:
                        left_index = int(indices[extrema_bool_max][0] - (indices[extrema_bool_max][1] -
                                                                         indices[extrema_bool_max][0]))
                        left_new_maxima = time_series[extrema_bool_max][1]
                    else:
                        left_index = int(indices[extrema_bool_min][0] - (indices[extrema_bool_max][0] -
                                                                         indices[extrema_bool_min][0]))
                        left_new_maxima = time_series[extrema_bool_max][0]
                    if right_maxima_x > right_minima_x:
                        right_index = \
                            int(indices[extrema_bool_max][-1] + (indices[extrema_bool_max][-1] -
                                                                 indices[extrema_bool_max][-2]) - (len(time)))
                        right_new_maxima = time_series[extrema_bool_max][-2]
                    else:
                        right_index = \
                            int(indices[extrema_bool_min][-1] + (indices[extrema_bool_min][-1] -
                                                                 indices[extrema_bool_max][-1]) - (len(time)))
                        right_new_maxima = time_series[extrema_bool_max][-1]

                    # this is to not extract incorrect values from signal
                    extrema_bool_max_original = extrema_bool_max.copy()

                    left_max_bool = np.zeros_like(new_knots_time[new_knots_time < time[0]])
                    if left_index < 0:
                        left_max_bool[left_index] = 1
                        left_max_bool = left_max_bool == 1
                    else:
                        extrema_bool_max[left_index] = 1
                        extrema_bool_max = extrema_bool_max == 1
                        left_max_bool = left_max_bool == 1

                    right_max_bool = np.zeros_like(new_knots_time[new_knots_time > time[-1]])
                    if right_index >= 0:
                        right_max_bool[right_index] = 1
                        right_max_bool = right_max_bool == 1
                    else:
                        extrema_bool_max[right_index] = 1
                        extrema_bool_max = extrema_bool_max == 1
                        right_max_bool = right_max_bool == 1

                else:
                    left_max_bool = np.flip(extrema_bool_max[1:])
                    right_max_bool = np.flip(extrema_bool_max[:-1])
                    left_new_maxima = time_series[extrema_bool_max]
                    right_new_maxima = time_series[extrema_bool_max]
                    extrema_bool_max_original = extrema_bool_max.copy()

                extrema_basis_function_matrix = \
                    cubic_basis_function_matrix_extended[:, np.hstack((left_max_bool,
                                                                       extrema_bool_max,
                                                                       right_max_bool))]

                extrema = np.hstack((left_new_maxima, time_series[extrema_bool_max_original], right_new_maxima))

            elif extrema_type == 'minima':
                if len(minima) > 1:
                    if left_minima_x < left_maxima_x:
                        left_index = int(indices[extrema_bool_min][0] - (indices[extrema_bool_min][1] -
                                                                         indices[extrema_bool_min][0]))
                        left_new_minima = time_series[extrema_bool_min][1]
                    else:
                        left_index = int(indices[extrema_bool_max][0] - (indices[extrema_bool_min][0] -
                                                                         indices[extrema_bool_max][0]))
                        left_new_minima = time_series[extrema_bool_min][0]
                    if right_minima_x > right_maxima_x:
                        right_index = \
                            int(indices[extrema_bool_min][-1] + (indices[extrema_bool_min][-1] -
                                                                 indices[extrema_bool_min][-2]) - (len(time)))
                        right_new_minima = time_series[extrema_bool_min][-2]
                    else:
                        right_index = \
                            int(indices[extrema_bool_max][-1] + (indices[extrema_bool_max][-1] -
                                                                 indices[extrema_bool_min][-1]) - (len(time)))
                        right_new_minima = time_series[extrema_bool_min][-1]

                    # this is to not extract incorrect values from signal
                    extrema_bool_min_original = extrema_bool_min.copy()

                    left_min_bool = np.zeros_like(new_knots_time[new_knots_time < time[0]])
                    if left_index < 0:
                        left_min_bool[left_index] = 1
                        left_min_bool = left_min_bool == 1
                    else:
                        extrema_bool_min[left_index] = 1
                        extrema_bool_min = extrema_bool_min == 1
                        left_min_bool = left_min_bool == 1

                    right_min_bool = np.zeros_like(new_knots_time[new_knots_time > time[-1]])
                    if right_index >= 0:
                        right_min_bool[right_index] = 1
                        right_min_bool = right_min_bool == 1
                    else:
                        extrema_bool_min[right_index] = 1
                        extrema_bool_min = extrema_bool_min == 1
                        right_min_bool = right_min_bool == 1

                else:
                    left_min_bool = np.flip(extrema_bool_min[1:])
                    right_min_bool = np.flip(extrema_bool_min[:-1])
                    left_new_minima = time_series[extrema_bool_min]
                    right_new_minima = time_series[extrema_bool_min]
                    extrema_bool_min_original = extrema_bool_min.copy()

                extrema_basis_function_matrix = \
                    cubic_basis_function_matrix_extended[:, np.hstack((left_min_bool,
                                                                       extrema_bool_min,
                                                                       right_min_bool))]

                extrema = np.hstack((left_new_minima, time_series[extrema_bool_min_original], right_new_minima))

    # anti-symmetric - top

    elif edge_effect == 'anti-symmetric':

        signal_left = time_series[0]
        signal_right = time_series[-1]

        # maxima - top

        points_until_first_maxima_from_left = (
                np.cumsum(np.cumsum(extrema_bool_max)) <= 1)
        # boolean for all point before (inclusive) first maxima from left
        first_maxima_from_left = (np.cumsum(np.cumsum(extrema_bool_max)) == 1)
        # boolean for first maxima from left

        points_until_first_maxima_from_right = np.flip(
            np.cumsum(np.cumsum(np.flip(extrema_bool_max))) <= 1)
        # boolean for all point before (inclusive) first maxima from right
        first_maxima_from_right = np.flip(np.cumsum(np.cumsum(np.flip(extrema_bool_max))) == 1)
        # boolean for first maxima from right

        points_until_first_maxima_from_left_extract = points_until_first_maxima_from_left[
            points_until_first_maxima_from_left]  # extracts the correct size maxima boolean vector on left
        points_until_first_maxima_from_right_extract = points_until_first_maxima_from_right[
            points_until_first_maxima_from_right]  # extracts the correct size maxima boolean vector on right

        # maxima - bottom

        # minima - top

        points_until_first_minima_from_left = (
                np.cumsum(np.cumsum(extrema_bool_min)) <= 1)
        # boolean for all point before (inclusive) first minima from left
        first_minima_from_left = (np.cumsum(np.cumsum(extrema_bool_min)) == 1)
        # boolean for first minima from left

        points_until_first_minima_from_right = np.flip(
            np.cumsum(np.cumsum(np.flip(extrema_bool_min))) <= 1)
        # boolean for all point before (inclusive) first minima from right
        first_minima_from_right = np.flip(np.cumsum(np.cumsum(np.flip(extrema_bool_min))) == 1)
        # boolean for first minima from right

        points_until_first_minima_from_left_extract = points_until_first_minima_from_left[
            points_until_first_minima_from_left]  # extracts the correct size minima boolean vector on left
        points_until_first_minima_from_right_extract = points_until_first_minima_from_right[
            points_until_first_minima_from_right]  # extracts the correct size minima boolean vector on right

        # minima - bottom

        if extrema_type == 'maxima':

            # from left

            left_column_number = int(len(time) - 1)

            subset_time_vector_bool[int(left_column_number -
                                        len(points_until_first_minima_from_left_extract)):
                                    int(left_column_number)] = points_until_first_minima_from_left_extract
            # inserts boolean (vector of ones) using column number

            # from right

            right_column_number = int(2 * (len(time) - 1))

            subset_time_vector_bool[(int(right_column_number) + 1): int(right_column_number + len(
                points_until_first_minima_from_right_extract) + 1)] = \
                points_until_first_minima_from_right_extract
            # inserts boolean (vector of ones) using column number #

            # combines left and right

            cubic_basis_function_matrix = cubic_basis_function_matrix_extended[:, subset_time_vector_bool]
            # extract points only relevant to "adjusted" signal

            first_minima_from_left_extract = first_minima_from_left[
                points_until_first_minima_from_left]  # extracts vector to add to left of extrema vector

            first_minima_from_right_extract = first_minima_from_right[
                points_until_first_minima_from_right]  # extracts vector to add to right of extrema vector

            extrema_basis_function_matrix = cubic_basis_function_matrix[
                                            :, np.hstack((np.flip(first_minima_from_left_extract),
                                                          extrema_bool_max,
                                                          np.flip(first_minima_from_right_extract)))]

            extrema = np.hstack(
                (signal_left + (signal_left - time_series[first_minima_from_left]),
                 time_series[extrema_bool_max],
                 signal_right + (signal_right - time_series[first_minima_from_right])))
            # appends mirrored extrema to either side of extrema vector
            # adds them to extrema vector #

        elif extrema_type == 'minima':

            # from left

            left_column_number = int(len(time) - 1)

            subset_time_vector_bool[int(left_column_number -
                                        len(points_until_first_maxima_from_left_extract)): int(
                left_column_number)] = points_until_first_maxima_from_left_extract
            # inserts boolean (vector of ones) using column number

            # from right

            right_column_number = int(2 * (len(time) - 1))

            subset_time_vector_bool[(int(right_column_number) + 1): int(right_column_number + len(
                points_until_first_maxima_from_right_extract) + 1)] = \
                points_until_first_maxima_from_right_extract
            # inserts boolean (vector of ones) using column number #

            # combines left and right

            cubic_basis_function_matrix = cubic_basis_function_matrix_extended[:, subset_time_vector_bool]
            # extract points only relevant to "adjusted" signal

            first_maxima_from_left_extract = first_maxima_from_left[
                points_until_first_maxima_from_left]  # extracts vector to add to left of extrema vector

            first_maxima_from_right_extract = first_maxima_from_right[
                points_until_first_maxima_from_right]  # extracts vector to add to right of extrema vector

            extrema_basis_function_matrix = cubic_basis_function_matrix[
                                            :, np.hstack((np.flip(first_maxima_from_left_extract),
                                                          extrema_bool_min,
                                                          np.flip(first_maxima_from_right_extract)))]

            extrema = np.hstack(
                (signal_left + (signal_left - time_series[first_maxima_from_left]),
                 time_series[extrema_bool_min],
                 signal_right + (signal_right - time_series[first_maxima_from_right])))
            # appends mirrored extrema to either side of extrema vector
            # adds them to extrema vector #

    # anti-symmetric - bottom

    elif edge_effect == 'characteristic_wave_Huang':

        left_bool = {}
        left_extrema = {}
        extrema_bool = {}
        right_bool = {}
        right_extrema = {}

        # Description nebulous in original paper - open to interpretation.
        # Rationalised better interpretation - need to fix!
        # Improved interpretations needs

        cubic_basis_function_matrix = \
            cubic_basis_function_matrix_extended[:, (time[0] <= new_knots_time) & (new_knots_time <= time[-1])]

        maxima = time_series[utility_extrema.max_bool_func_1st_order_fd()]
        maxima_time = time[utility_extrema.max_bool_func_1st_order_fd()]

        left_two_maxima = maxima[:2]
        right_two_maxima = maxima[-2:]

        minima = time_series[utility_extrema.min_bool_func_1st_order_fd()]
        minima_time = time[utility_extrema.min_bool_func_1st_order_fd()]

        left_two_minima = minima[:2]
        right_two_minima = minima[-2:]

        extended_time_huang = time_extension(time)
        left_time = extended_time_huang[:int(len(time) - 1)]
        right_time = extended_time_huang[int(2 * len(time) - 1):]

        time_series_util = Utility(time=time, time_series=time_series)

        if len(left_two_maxima) > 1 and len(left_two_minima) > 1:

            left_period_1 = 2 * np.abs(minima_time[0] - maxima_time[0])
            left_period_2 = 2 * np.abs(minima_time[1] - maxima_time[1])
            right_period_1 = 2 * np.abs(minima_time[-1] - maxima_time[-1])
            right_period_2 = 2 * np.abs(minima_time[-2] - maxima_time[-2])

            left_amp_1 = np.abs(left_two_maxima[0] - left_two_minima[0])
            left_amp_2 = np.abs(left_two_maxima[1] - left_two_minima[1])
            right_amp_1 = np.abs(right_two_maxima[-1] - right_two_minima[-1])
            right_amp_2 = np.abs(right_two_maxima[-2] - right_two_minima[-2])

            if extrema_type == 'maxima':

                extrema_bool = time_series_util.max_bool_func_1st_order_fd()

                if maxima_time[0] < minima_time[0]:

                    left_min_time = maxima_time[0] - (left_period_1 / left_period_2) * np.abs(minima_time[0] -
                                                                                              maxima_time[1])
                    left_min = left_two_maxima[0] - (left_amp_1 / left_amp_2) * np.abs(left_two_maxima[0] -
                                                                                       left_two_minima[1])

                    left_max_time = left_min_time - (left_period_1 / left_period_2) * np.abs(maxima_time[0] -
                                                                                             minima_time[0])
                    left_max = left_min + (left_amp_1 / left_amp_2) * np.abs(left_two_maxima[0] - left_two_minima[0])

                else:

                    left_max_time = minima_time[0] - (left_period_1 / left_period_2) * np.abs(maxima_time[0] -
                                                                                              minima_time[1])
                    left_max = left_two_maxima[0] + (left_amp_1 / left_amp_2) * np.abs(left_two_maxima[0] -
                                                                                       left_two_maxima[1])

                left_bool = min(np.abs(left_max_time - left_time)) == np.abs(left_max_time - left_time)
                left_extrema = left_max

                if minima_time[-1] < maxima_time[-1]:

                    right_min_time = maxima_time[-1] + (right_period_1 / right_period_2) * np.abs(minima_time[-1] -
                                                                                                  maxima_time[-2])
                    right_min = right_two_maxima[-1] - (right_amp_1 / right_amp_2) * np.abs(right_two_minima[-1] -
                                                                                            right_two_maxima[-2])

                    right_max_time = right_min_time + (right_period_1 / right_period_2) * np.abs(minima_time[-1] -
                                                                                                 maxima_time[-1])
                    right_max = right_min + (right_amp_1 / right_amp_2) * (right_two_maxima[-1] - right_two_minima[-1])

                else:

                    right_max_time = minima_time[-1] + (right_period_1 / right_period_2) * np.abs(minima_time[-1] -
                                                                                                  maxima_time[-2])
                    right_max = right_two_minima[-1] + (right_amp_1 / right_amp_2) * np.abs(right_two_minima[-2] -
                                                                                            right_two_maxima[-1])

                right_bool = min(np.abs(right_max_time - right_time)) == np.abs(right_max_time - right_time)
                right_extrema = right_max

            elif extrema_type == 'minima':

                extrema_bool = time_series_util.min_bool_func_1st_order_fd()

                if maxima_time[0] < minima_time[0]:

                    left_min_time = maxima_time[0] - (left_period_1 / left_period_2) * np.abs(minima_time[0] -
                                                                                              maxima_time[1])
                    left_min = left_two_maxima[0] - (left_amp_1 / left_amp_2) * np.abs(left_two_maxima[1] -
                                                                                       left_two_minima[0])

                else:

                    left_max_time = minima_time[0] - (left_period_1 / left_period_2) * np.abs(maxima_time[0] -
                                                                                              minima_time[1])
                    left_max = left_two_maxima[0] + (left_amp_1 / left_amp_2) * np.abs(left_two_maxima[1] -
                                                                                       left_two_maxima[0])

                    left_min_time = left_max_time - (left_period_1 / left_period_2) * np.abs(minima_time[0] -
                                                                                             maxima_time[0])
                    left_min = left_max - (left_amp_1 / left_amp_2) * np.abs(left_two_maxima[0] - left_two_minima[0])

                left_bool = min(np.abs(left_min_time - left_time)) == np.abs(left_min_time - left_time)
                left_extrema = left_min

                if minima_time[-1] < maxima_time[-1]:

                    right_min_time = maxima_time[-1] + (right_period_1 / right_period_2) * np.abs(maxima_time[-2] -
                                                                                                  minima_time[-1])
                    right_min = right_two_maxima[-1] - (right_amp_1 / right_amp_2) * np.abs(right_two_maxima[-2] -
                                                                                            right_two_minima[-1])

                else:

                    right_max_time = minima_time[-1] + (right_period_1 / right_period_2) * np.abs(maxima_time[-1] -
                                                                                                  minima_time[-2])
                    right_max = right_two_minima[-1] + (right_amp_1 / right_amp_2) * np.abs(right_two_maxima[-1] -
                                                                                            right_two_minima[-2])

                    right_min_time = right_max_time + (left_period_1 / left_period_2) * np.abs(minima_time[-1] -
                                                                                               maxima_time[-1])
                    right_min = right_max - (left_amp_1 / left_amp_2) * np.abs(right_two_maxima[-1] -
                                                                               right_two_minima[-1])

                right_bool = min(np.abs(right_min_time - right_time)) == np.abs(right_min_time - right_time)
                right_extrema = right_min

            extrema_basis_function_matrix = \
                cubic_basis_function_matrix_extended[:, np.hstack((left_bool, extrema_bool, right_bool))]

            extrema = np.hstack((left_extrema, time_series[extrema_bool], right_extrema))

        else:

            time_series_util = Utility(time=time, time_series=time_series)

            if extrema_type == 'maxima':

                extrema_bool = time_series_util.max_bool_func_1st_order_fd()
                extrema = np.hstack((left_maxima, time_series[extrema_bool], right_maxima))

                if sum(extrema_bool) > 1:

                    diff = maxima_time[-1] - maxima_time[-2]
                    right_max_time = maxima_time[-1] + diff
                    right_bool = min(np.abs(right_max_time - right_time)) == np.abs(right_max_time - right_time)
                    left_max_time = maxima_time[-2] - diff
                    left_bool = min(np.abs(left_max_time - left_time)) == np.abs(left_max_time - left_time)

                    extrema_basis_function_matrix = \
                        cubic_basis_function_matrix_extended[:, np.hstack((left_bool, extrema_bool, right_bool))]

                else:

                    extrema_basis_function_matrix = \
                        cubic_basis_function_matrix_extended[:, np.hstack((extrema_bool[:-1], extrema_bool,
                                                                           extrema_bool[1:]))]

            if extrema_type == 'minima':

                extrema_bool = time_series_util.min_bool_func_1st_order_fd()
                extrema = np.hstack((left_minima, time_series[extrema_bool], right_minima))

                if sum(extrema_bool) > 1:

                    diff = minima_time[-1] - minima_time[-2]
                    right_min_time = minima_time[-1] + diff
                    right_bool = min(np.abs(right_min_time - right_time)) == np.abs(right_min_time - right_time)
                    left_min_time = minima_time[-2] - diff
                    left_bool = min(np.abs(left_min_time - left_time)) == np.abs(left_min_time - left_time)

                    extrema_basis_function_matrix = \
                        cubic_basis_function_matrix_extended[:, np.hstack((left_bool, extrema_bool, right_bool))]

                else:

                    extrema_basis_function_matrix = \
                        cubic_basis_function_matrix_extended[:, np.hstack((extrema_bool[:-1],
                                                                           extrema_bool, extrema_bool[1:]))]

    elif edge_effect == 'characteristic_wave_Coughlin':

        cubic_basis_function_matrix = \
            cubic_basis_function_matrix_extended[:, (time[0] <= new_knots_time) & (new_knots_time <= time[-1])]

        maxima = time_series[utility_extrema.max_bool_func_1st_order_fd()]
        maxima_time = time[utility_extrema.max_bool_func_1st_order_fd()]

        minima = time_series[utility_extrema.min_bool_func_1st_order_fd()]
        minima_time = time[utility_extrema.min_bool_func_1st_order_fd()]

        left_maxima = maxima[0]
        left_maxima_time = maxima_time[0]
        right_maxima = maxima[-1]
        right_maxima_time = maxima_time[-1]

        left_minima = minima[0]
        left_minima_time = minima_time[0]
        right_minima = minima[-1]
        right_minima_time = minima_time[-1]

        left_period = 2 * np.abs(left_maxima_time - left_minima_time)
        right_period = 2 * np.abs(right_maxima_time - right_minima_time)

        left_min_time = np.min([left_minima_time, left_maxima_time])
        left_max_time = np.max([left_minima_time, left_maxima_time])
        right_min_time = np.min([right_minima_time, right_maxima_time])
        right_max_time = np.max([right_minima_time, right_maxima_time])

        left_local_mean = np.mean(time_series[np.r_[left_min_time <= time] & np.r_[time <= left_max_time]])
        right_local_mean = np.mean(time_series[np.r_[right_min_time <= time] & np.r_[time <= right_max_time]])

        left_time_extension = new_knots_time[new_knots_time <= left_max_time]
        right_time_extension = new_knots_time[new_knots_time >= right_min_time]

        left_amp = (1 / 2) * np.abs(left_maxima - left_minima)
        right_amp = (1 / 2) * np.abs(right_maxima - right_minima)

        if left_maxima_time < left_minima_time:
            left_amp *= -1
        if right_maxima_time > right_minima_time:
            right_amp *= -1

        left_signal = \
            left_amp * np.cos(2 * np.pi * (left_time_extension -
                                           left_max_time) / left_period) + left_local_mean

        right_signal = \
            right_amp * np.cos(2 * np.pi * (right_time_extension -
                                            right_min_time) / right_period) + right_local_mean

        utility_huang_wave_right = Utility(time=right_time_extension, time_series=right_signal)
        utility_huang_wave_left = Utility(time=left_time_extension, time_series=left_signal)

        if extrema_type == 'maxima':

            right_new_maxima_bool = utility_huang_wave_right.max_bool_func_1st_order_fd()
            left_new_maxima_bool = utility_huang_wave_left.max_bool_func_1st_order_fd()

            if left_maxima_time < left_minima_time:
                left_first_max_value = left_signal[left_new_maxima_bool][-2]
            else:
                left_first_max_value = left_signal[left_new_maxima_bool][-1]
            if right_maxima_time > right_minima_time:
                right_first_max_value = right_signal[right_new_maxima_bool][1]
            else:
                right_first_max_value = right_signal[right_new_maxima_bool][0]

            left_max_bool_truncate = left_new_maxima_bool[:int(len(time) - 1)]
            left_max_bool_truncate_one_max = np.flip(np.cumsum(np.cumsum(np.flip(left_max_bool_truncate))) == 1)

            right_max_bool_truncate = right_new_maxima_bool[-int(len(time) - 1):]
            right_max_bool_truncate_one_max = np.cumsum(np.cumsum(right_max_bool_truncate)) == 1

            extrema_basis_function_matrix = \
                cubic_basis_function_matrix_extended[:, np.hstack((left_max_bool_truncate_one_max,
                                                                   extrema_bool_max,
                                                                   right_max_bool_truncate_one_max))]
            # adds them to extrema vector #

            extrema = np.hstack((left_first_max_value, time_series[extrema_bool_max], right_first_max_value))

        elif extrema_type == 'minima':

            right_new_minima_bool = utility_huang_wave_right.min_bool_func_1st_order_fd()
            left_new_minima_bool = utility_huang_wave_left.min_bool_func_1st_order_fd()

            if left_minima_time < left_maxima_time:
                left_first_min_value = left_signal[left_new_minima_bool][-2]
            else:
                left_first_min_value = left_signal[left_new_minima_bool][-1]
            if right_maxima_time < right_minima_time:
                right_first_min_value = right_signal[right_new_minima_bool][1]
            else:
                right_first_min_value = right_signal[right_new_minima_bool][0]

            left_min_bool_truncate = left_new_minima_bool[:int(len(time) - 1)]
            left_min_bool_truncate_one_min = np.flip(np.cumsum(np.cumsum(np.flip(left_min_bool_truncate))) == 1)

            right_min_bool_truncate = right_new_minima_bool[-int(len(time) - 1):]
            right_min_bool_truncate_one_min = np.cumsum(np.cumsum(right_min_bool_truncate)) == 1

            extrema_basis_function_matrix = \
                cubic_basis_function_matrix_extended[:, np.hstack((left_min_bool_truncate_one_min,
                                                                   extrema_bool_min,
                                                                   right_min_bool_truncate_one_min))]
            # adds them to extrema vector #

            extrema = np.hstack((left_first_min_value, time_series[extrema_bool_min], right_first_min_value))

    elif edge_effect[-18:] == 'slope_based_method':

        cubic_basis_function_matrix = cubic_basis_function_matrix_extended[:, subset_time_vector_bool]

        maxima = time_series[extrema_bool_max]
        maxima_time = time[extrema_bool_max]

        minima = time_series[extrema_bool_min]
        minima_time = time[extrema_bool_min]

        # need to make provision for there only being one maxima or one minima
        # need to make provision for there only being one maxima or one minima
        # need to make provision for there only being one maxima or one minima

        if (len(maxima) > 1) and (len(minima) > 1):

            left_maxima_0 = maxima[0]
            left_maxima_time_0 = maxima_time[0]
            left_maxima_1 = maxima[1]
            left_maxima_time_1 = maxima_time[1]

            left_minima_0 = minima[0]
            left_minima_time_0 = minima_time[0]
            left_minima_1 = minima[1]
            left_minima_time_1 = minima_time[1]

            right_maxima_n = maxima[-1]
            right_maxima_time_n = maxima_time[-1]
            right_maxima_n_1 = maxima[-2]
            right_maxima_time_n_1 = maxima_time[-2]

            right_minima_n = minima[-1]
            right_minima_time_n = minima_time[-1]
            right_minima_n_1 = minima[-2]
            right_minima_time_n_1 = minima_time[-2]

            if left_maxima_time_0 < left_minima_time_0:

                s1 = (left_maxima_1 - left_minima_0) / (left_maxima_time_1 - left_minima_time_0)
                s2 = (left_minima_0 - left_maxima_0) / (left_minima_time_0 - left_maxima_time_0)

                left_min_new = left_maxima_0 - s1 * (left_maxima_time_1 - left_minima_time_0)

                if (edge_effect[:8] == 'improved') and (left_signal < left_min_new):
                    left_min_new = left_signal

                left_max_new = left_min_new - s2 * (left_minima_time_0 - left_maxima_time_0)

            else:

                s1 = (left_minima_1 - left_maxima_0) / (left_minima_time_1 - left_maxima_time_0)
                s2 = (left_maxima_0 - left_minima_0) / (left_maxima_time_0 - left_minima_time_0)

                left_max_new = left_minima_0 - s1 * (left_minima_time_1 - left_maxima_time_0)

                if (edge_effect[:8] == 'improved') and (left_signal > left_max_new):
                    left_max_new = left_signal

                left_min_new = left_max_new - s2 * (left_maxima_time_0 - left_minima_time_0)

            if right_maxima_time_n > right_minima_time_n:

                s1 = (right_maxima_n_1 - right_minima_n) / (right_maxima_time_n_1 - right_minima_time_n)
                s2 = (right_minima_n - right_maxima_n) / (right_minima_time_n - right_maxima_time_n)

                right_min_new = right_maxima_n - s1 * (right_maxima_time_n_1 - right_minima_time_n)

                if (edge_effect[:8] == 'improved') and (right_min_new > right_signal):
                    right_min_new = right_signal

                right_max_new = right_min_new - s2 * (right_minima_time_n - right_maxima_time_n)

            else:

                s1 = (right_minima_n_1 - right_maxima_n) / (right_minima_time_n_1 - right_maxima_time_n)
                s2 = (right_maxima_n - right_minima_n) / (right_maxima_time_n - right_minima_time_n)

                right_max_new = right_minima_n - s1 * (right_minima_time_n_1 - right_maxima_time_n)

                if (edge_effect[:8] == 'improved') and (right_max_new < right_signal):
                    right_max_new = right_signal

                right_min_new = right_max_new - s2 * (right_maxima_time_n - right_minima_time_n)

            if extrema_type == 'maxima':

                indices = np.arange(len(time))
                left_index = int(indices[extrema_bool_max][0] - (indices[extrema_bool_max][1] -
                                                                 indices[extrema_bool_max][0]))
                right_index = int(indices[extrema_bool_max][-1] + (indices[extrema_bool_max][-1] -
                                                                   indices[extrema_bool_max][-2]) - (len(time)))

                left_max_bool = np.zeros_like(new_knots_time[new_knots_time < time[0]])
                left_max_bool[left_index] = 1
                left_max_bool = left_max_bool == 1
                right_max_bool = np.zeros_like(new_knots_time[new_knots_time > time[-1]])
                right_max_bool[right_index] = 1
                right_max_bool = right_max_bool == 1

                extrema_basis_function_matrix = \
                    cubic_basis_function_matrix_extended[:, np.hstack((left_max_bool,
                                                                       extrema_bool_max,
                                                                       right_max_bool))]

                extrema = np.hstack((left_max_new, time_series[extrema_bool_max], right_max_new))

            elif extrema_type == 'minima':

                indices = np.arange(len(time))
                left_index = int(indices[extrema_bool_min][0] - (indices[extrema_bool_min][1] -
                                                                 indices[extrema_bool_min][0]))
                right_index = int(indices[extrema_bool_min][-1] + (indices[extrema_bool_min][-1] -
                                                                   indices[extrema_bool_min][-2]) - (len(time)))

                left_min_bool = np.zeros_like(new_knots_time[new_knots_time < time[0]])
                left_min_bool[left_index] = 1
                left_min_bool = left_min_bool == 1
                right_min_bool = np.zeros_like(new_knots_time[new_knots_time > time[-1]])
                right_min_bool[right_index] = 1
                right_min_bool = right_min_bool == 1

                extrema_basis_function_matrix = \
                    cubic_basis_function_matrix_extended[:, np.hstack((left_min_bool,
                                                                       extrema_bool_min,
                                                                       right_min_bool))]

                extrema = np.hstack((left_min_new, time_series[extrema_bool_min], right_min_new))

        else:

            # simply reflects extrema as require at least 2 maxima and 2 minima to perform slope-based method

            if extrema_type == 'maxima':

                indices = np.arange(len(time))
                left_index = int(-indices[extrema_bool_max][0])
                right_index = int(((len(time) - 2) - indices[extrema_bool_max][-1]))

                left_max_bool = np.zeros_like(new_knots_time[new_knots_time < time[0]])
                left_max_bool[left_index] = 1
                left_max_bool = left_max_bool == 1
                right_max_bool = np.zeros_like(new_knots_time[new_knots_time > time[-1]])
                right_max_bool[right_index] = 1
                right_max_bool = right_max_bool == 1

                left_max_new = maxima[0]
                right_max_new = maxima[-1]

                extrema_basis_function_matrix = \
                    cubic_basis_function_matrix_extended[:, np.hstack((left_max_bool,
                                                                       extrema_bool_max,
                                                                       right_max_bool))]

                extrema = np.hstack((left_max_new, time_series[extrema_bool_max], right_max_new))

            elif extrema_type == 'minima':

                indices = np.arange(len(time))
                left_index = int(-indices[extrema_bool_min][0])
                right_index = int(((len(time) - 2) - indices[extrema_bool_min][-1]))

                left_min_bool = np.zeros_like(new_knots_time[new_knots_time < time[0]])
                left_min_bool[left_index] = 1
                left_min_bool = left_min_bool == 1
                right_min_bool = np.zeros_like(new_knots_time[new_knots_time > time[-1]])
                right_min_bool[right_index] = 1
                right_min_bool = right_min_bool == 1

                left_min_new = minima[0]
                right_min_new = minima[-1]

                extrema_basis_function_matrix = \
                    cubic_basis_function_matrix_extended[:, np.hstack((left_min_bool,
                                                                       extrema_bool_min,
                                                                       right_min_bool))]

                extrema = np.hstack((left_min_new, time_series[extrema_bool_min], right_min_new))

    elif edge_effect == 'average':

        cubic_basis_function_matrix = \
            cubic_basis_function_matrix_extended[:, (time[0] <= new_knots_time) & (new_knots_time <= time[-1])]

        if extrema_type == 'maxima':

            maxima = time_series[extrema_bool_max]

            if len(maxima) > 1:

                left_two_maxima = maxima[:2]
                right_two_maxima = maxima[-2:]

                indices = np.arange(len(time))
                left_index = int(indices[extrema_bool_max][0] - (indices[extrema_bool_max][1] -
                                                                 indices[extrema_bool_max][0]))
                right_index = int(indices[extrema_bool_max][-1] + (indices[extrema_bool_max][-1] -
                                                                   indices[extrema_bool_max][-2]) - (len(time)))

                left_max_bool = np.zeros_like(new_knots_time[new_knots_time < time[0]])
                left_max_bool[left_index] = 1
                left_max_bool = left_max_bool == 1
                right_max_bool = np.zeros_like(new_knots_time[new_knots_time > time[-1]])
                right_max_bool[right_index] = 1
                right_max_bool = right_max_bool == 1

                left_new_maxima = np.mean(left_two_maxima)
                right_new_maxima = np.mean(right_two_maxima)

                extrema_basis_function_matrix = \
                    cubic_basis_function_matrix_extended[:, np.hstack((left_max_bool,
                                                                       extrema_bool_max,
                                                                       right_max_bool))]

                extrema = np.hstack((left_new_maxima, time_series[extrema_bool_max], right_new_maxima))

            else:

                maximum = time_series[extrema_bool_max]

                left_max_bool = np.flip(extrema_bool_max[1:])
                right_max_bool = np.flip(extrema_bool_max[:-1])

                extrema_basis_function_matrix = \
                    cubic_basis_function_matrix_extended[:, np.hstack((left_max_bool,
                                                                       extrema_bool_max,
                                                                       right_max_bool))]

                extrema = np.hstack((maximum, maximum, maximum))

        if extrema_type == 'minima':

            minima = time_series[extrema_bool_min]

            if len(minima) > 1:

                left_two_minima = minima[:2]
                right_two_minima = minima[-2:]

                indices = np.arange(len(time))
                left_index = int(indices[extrema_bool_min][0] - (indices[extrema_bool_min][1] -
                                                                 indices[extrema_bool_min][0]))
                right_index = int(indices[extrema_bool_min][-1] + (indices[extrema_bool_min][-1] -
                                                                   indices[extrema_bool_min][-2]) - (len(time)))

                left_min_bool = np.zeros_like(new_knots_time[new_knots_time < time[0]])
                left_min_bool[left_index] = 1
                left_min_bool = left_min_bool == 1
                right_min_bool = np.zeros_like(new_knots_time[new_knots_time > time[-1]])
                right_min_bool[right_index] = 1
                right_min_bool = right_min_bool == 1

                left_new_minima = np.mean(left_two_minima)
                right_new_minima = np.mean(right_two_minima)

                extrema_basis_function_matrix = \
                    cubic_basis_function_matrix_extended[:, np.hstack((left_min_bool,
                                                                       extrema_bool_min,
                                                                       right_min_bool))]

                extrema = np.hstack((left_new_minima, time_series[extrema_bool_min], right_new_minima))

            else:

                minimum = time_series[extrema_bool_min]

                left_min_bool = np.flip(extrema_bool_min[1:])
                right_min_bool = np.flip(extrema_bool_min[:-1])

                extrema_basis_function_matrix = \
                    cubic_basis_function_matrix_extended[:, np.hstack((left_min_bool,
                                                                       extrema_bool_min,
                                                                       right_min_bool))]

                extrema = np.hstack((minimum, minimum, minimum))

    elif edge_effect == 'none':

        extrema_bool = {}  # avoids unnecessary error
        # no edge-effect only requires one set of extrema
        if extrema_type == 'maxima':
            extrema_bool = extrema_bool_max
        elif extrema_type == 'minima':
            extrema_bool = extrema_bool_min

        cubic_basis_function_matrix = cubic_basis_function_matrix_extended[:, subset_time_vector_bool]
        # extract points only relevant to signal

        extrema_basis_function_matrix = cubic_basis_function_matrix[:, extrema_bool]
        # adds them to extrema vector #

        extrema = time_series[extrema_bool]

    # b-spline 2nd order smoothing
    # forms part of Statistical Empirical Mode Decomposition (SEMD)

    if smooth:

        # optimised
        second_order_matrix = np.diag(np.ones(np.shape(extrema_basis_function_matrix)[0]), 0)[:, :-2] + \
                              np.diag(-2 * np.ones(int(np.shape(extrema_basis_function_matrix)[0] - 1)), -1)[:, :-2] + \
                              np.diag(np.ones(int(np.shape(extrema_basis_function_matrix)[0] - 2)),  -2)[:, :-2]

        extrema_basis_function_matrix = np.append(extrema_basis_function_matrix, smoothing_penalty *
                                                  second_order_matrix, axis=1)  # l2 norm trick

        extrema = np.append(extrema, np.zeros(np.shape(extrema_basis_function_matrix)[0] - 2), axis=0)
        # l2 norm trick

        # optimse speed - pseudo inverse
        # pseudo inverse sometimes results in singular matrix problem
        try:
            coef = solve(extrema_basis_function_matrix.dot(extrema_basis_function_matrix.T),
                         extrema_basis_function_matrix.dot(extrema))
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(extrema_basis_function_matrix.transpose(), extrema,
                                   rcond=None)
            coef = coef[0]
        # extrema_basis_function_matrix and extrema matrix augmented for smoothness

        approximation = np.matmul(coef, cubic_basis_function_matrix)  # basis_function_matrix is used here

    else:

        # optimse speed - pseudo inverse
        # pseudo inverse sometimes results in singular matrix problem
        try:
            coef = solve(extrema_basis_function_matrix.dot(extrema_basis_function_matrix.T),
                         extrema_basis_function_matrix.dot(extrema))
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(extrema_basis_function_matrix.transpose(), extrema,
                                   rcond=None)
            coef = coef[0]
        approximation = np.matmul(coef, cubic_basis_function_matrix)  # basis_function_matrix is used here

    # knots_fix - 2 more basis functions than knots
    if edge_effect[:9] == 'symmetric':  # possible improvement to code layout needed?

        # must now extract from approximation only points applicable to original signal
        approximation = approximation[int(len(points_until_first_extreme_from_left_extract)): int(
            len(approximation) - len(points_until_first_extreme_from_right_extract))]

        # knots_fix - 2 more basis functions than knots
        # must now extract from coef only coefficients necessary for envelope
        tot_coef = len(coef)  # total number of coef in extension
        tot_internal_knots = len(knots_for_envelope)  # original knots
        tot_coef_needed = tot_internal_knots - 4 + (4 + 2)  # extended math for demonstration purposes
        tot_coef_not_needed = tot_coef - tot_coef_needed  # coefficients not needed
        coef = coef[int(tot_coef_not_needed / 2):int(tot_coef_not_needed / 2 + tot_coef_needed)]
        # extract relevant coefficient

    elif edge_effect == 'anti-symmetric':

        if extrema_type == 'maxima':
            left_length = len(points_until_first_minima_from_left_extract)
            right_length = len(points_until_first_minima_from_right_extract)
        elif extrema_type == 'minima':
            left_length = len(points_until_first_maxima_from_left_extract)
            right_length = len(points_until_first_maxima_from_right_extract)

        # must now extract from approximation only points applicable to original signal
        approximation = approximation[int(left_length): int(len(approximation) - right_length)]

        # knots_fix - 2 more basis functions than knots
        # must now extract from coef only coefficients necessary for envelope
        tot_coef = len(coef)  # total number of coef in extension
        tot_internal_knots = len(knots_for_envelope)  # original knots
        tot_coef_needed = tot_internal_knots - 4 + (4 + 2)  # extended math for demonstration purposes
        tot_coef_not_needed = tot_coef - tot_coef_needed  # coefficients not needed
        coef = coef[int(tot_coef_not_needed / 2):int(tot_coef_not_needed / 2 + tot_coef_needed)]
        # extract relevant coefficient

    elif edge_effect[:19] == 'characteristic_wave' or edge_effect[-18:] == 'slope_based_method' or \
            edge_effect == 'average' or edge_effect == 'none' or edge_effect == 'neural_network':

        # knots_fix - 2 more basis functions than knots
        # must now extract from coef only coefficients necessary for envelope
        tot_coef = len(coef)  # total number of coef in extension
        tot_internal_knots = len(knots_for_envelope)  # original knots
        tot_coef_needed = tot_internal_knots - 4 + (4 + 2)  # extended math for demonstration purposes
        tot_coef_not_needed = tot_coef - tot_coef_needed  # coefficients not needed
        coef = coef[int(tot_coef_not_needed / 2):int(tot_coef_not_needed / 2 + tot_coef_needed)]
        # extract relevant coefficient

    return approximation, coef


class Fluctuation:

    def __init__(self, time: np.ndarray, time_series: np.ndarray):

        self.time = time
        self.time_series = time_series

    def envelope_basis_function_approximation(self, knots_for_envelope: np.ndarray, extrema_type: str,
                                              smooth: bool = True, smoothing_penalty: float = 1.0,
                                              edge_effect: str = 'symmetric', spline_method: str = 'b_spline',
                                              alpha: float = 0.1, nn_m: int = 200, nn_k: int = 100,
                                              nn_method: str = 'grad_descent', nn_learning_rate: float = 0.01,
                                              nn_iter: int = 10) -> (np.ndarray, np.ndarray):
        """
        Least square fits parameter vector 'coef' using basis functions 'cubic_basis_function_matrix' such that
        ||(cubic_basis_function_matrix^T)(coef) - extrema||^2 is minimized.

        Parameters
        ----------
        knots_for_envelope : array_like
            Knot points to be used in cubic basis spline envelope construction.
            Possible extension to be optimized over.

        extrema_type : string_like
            'maxima' or 'minima'.

        smooth: boolean
            Whether or not to smooth signal.

        smoothing_penalty: float
            Smoothing penalty for second-order smoothing of parameter estimates to be multiplied on to
            basis function matrix.

        edge_effect: string_like
            What edge technique to use. Default set to symmetric that mirrors nearest extrema.

        spline_method : string_like
            Spline method to be used.

        alpha : float
            Used in symmetric edge-effect to decide whether to anchor spline.
            When alpha = 1 technique reduces to symmetric 'Symmetric Anchor' in:

            K. Zeng and M. He. A Simple Boundary Process Technique for Empirical Mode
            Decomposition. In IEEE International Geoscience and Remote Sensing Symposium,
            volume 6, pages 4258–4261. IEEE, 2004.

            or

            J. Zhao and D. Huang. Mirror Extending and Circular Spline Function for Empirical
            Mode Decomposition Method. Journal of Zhejiang University - Science A, 2(3):
            247–252, 2001.

            Other extreme is alpha = -infinity. Time series is reflected and next extremum is taken.

        nn_m : integer
            Number of points (outputs) on which to train in neural network edge effect.

        nn_k : integer
            Number of points (inputs) to use when estimating weights for neuron.

        nn_method : string_like
            Gradient descent method used to estimate weights.

        nn_learning_rate : float
            Learning rate to use when adjusting weights.

        nn_iter : integer
            Number of iterations to perform when estimating weights.

        Returns
        -------
        approximation : real ndarray
            Cubic basis spline envelope fitted through extrema.

        coef : real ndarray
            Corresponding coefficients.

        Notes
        -----
        Least square fitting function of extrema.
        Envelopes are second-order smoothed.

        (1) Add CHSI interpolation option. SOLVED.
        (2) Add ASI interpolation option. SOLVED.
        (3) Add 'Symmetric Discard' edge effect. SOLVED.
        (4) Add 'Symmetric Anchor' edge effect. SOLVED.
        (5) Add 'Symmetric' edge effect. SOLVED.
        (6) Add 'Conditional Symmetric Anchor' edge effect. SOLVED.
            - either 'Symmetric Anchor' or 'Symmetric' depending on condition.
            - analogous to 'Improved Slope-Based Method'.
        (7) Add 'Anti-Symmetric' edge effect. SOLVED.
        (8) Add 'Slope-Based' method. SOLVED.
        (9) Add 'Improved Slope-Based' method. SOLVED.
        (10) Add 'Huang Characteristic Wave' edge effect. SOLVED.
        (11) Add 'Coughlin Characteristic Wave' edge effect. SOLVED.
        (12) Add 'Average Characteristic Wave' edge effect. SOLVED.
        (13) Add 'Neural Network' edge effect. SOLVED.

        """
        time = self.time
        time_series = self.time_series

        # Need both maxima and minima for edge effects other than symmetric
        utility_extrema = Utility(time, time_series)

        extrema_bool_max = utility_extrema.max_bool_func_1st_order_fd()
        extrema_bool_min = utility_extrema.min_bool_func_1st_order_fd()

        approximation = {}  # avoids unnecessary error
        coef = {}  # avoids unnecessary error

        if spline_method == 'b_spline':

            new_knots_time = time_extension(time)

            subset_time_vector_bool = ((time[0] <= new_knots_time) & (new_knots_time <= time[-1]))
            # boolean of applicable subset of points

            extended_knots = time_extension(knots_for_envelope)

            extended_basis = Basis(time=new_knots_time, time_series=new_knots_time)

            cubic_basis_function_matrix_extended = extended_basis.cubic_b_spline(extended_knots)

            # identical from this point?

            approximation, coef = envelope_basis_function(
                time=time, time_series=time_series, extrema_bool_max=extrema_bool_max,
                extrema_bool_min=extrema_bool_min,
                cubic_basis_function_matrix_extended=cubic_basis_function_matrix_extended,
                new_knots_time=new_knots_time, subset_time_vector_bool=subset_time_vector_bool,
                knots_for_envelope=knots_for_envelope, extrema_type=extrema_type, smooth=smooth,
                smoothing_penalty=smoothing_penalty, edge_effect=edge_effect, alpha=alpha, nn_m=nn_m, nn_k=nn_k,
                nn_method=nn_method, nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

        elif spline_method == 'chsi':

            extrema_bool = {}
            # chsi only requires one set of extrema
            if extrema_type == 'maxima':
                extrema_bool = extrema_bool_max
            elif extrema_type == 'minima':
                extrema_bool = extrema_bool_min
            basis_chsi = Basis(time=time[extrema_bool], time_series=time_series[extrema_bool])

            approximation = basis_chsi.chsi_fit(knots_time=time, knots=knots_for_envelope,
                                                type_chsi_fit='envelope')

            coef = np.zeros_like(approximation)

        elif spline_method == 'asi':

            extrema_bool = {}
            # asi only requires one set of extrema
            if extrema_type == 'maxima':
                extrema_bool = extrema_bool_max
            elif extrema_type == 'minima':
                extrema_bool = extrema_bool_min
            basis_asi = Basis(time=time[extrema_bool], time_series=time_series[extrema_bool])

            approximation = basis_asi.asi_fit(knots_time=time, knots=knots_for_envelope,
                                              type_asi_fit='envelope')

            coef = np.zeros_like(approximation)

        return approximation, coef

    def envelope_basis_function_approximation_matrix(self, extended_matrix: np.ndarray, knots_for_envelope: np.ndarray,
                                                     extrema_type: str, smooth: bool = True,
                                                     smoothing_penalty: float = 1, edge_effect: str = 'symmetric',
                                                     alpha: float = 0.1, nn_m: int = 200, nn_k: int = 100,
                                                     nn_method: str = 'grad_descent',
                                                     nn_learning_rate: float = 0.01,
                                                     nn_iter: int = 10) -> (np.ndarray, np.ndarray):
        """
        Least square fits parameter vector 'coef' using basis functions 'cubic_basis_function_matrix' such that
        ||(cubic_basis_function_matrix^T)(coef) - extrema||^2 is minimized.

        Parameters
        ----------
        extended_matrix : array_like
            Matrix input to improve speed of algorithm - prevents having to construct matrix everytime.

        knots_for_envelope : array_like
            Knot points to be used in cubic basis spline envelope construction.
            Possible extension to be optimized over.

        extrema_type : string_like
            'maxima' or 'minima'.

        smooth: boolean
            Whether or not to smooth signal.

        smoothing_penalty: float
            Smoothing penalty for second-order smoothing of parameter estimates to be multiplied on to
            basis function matrix.

        edge_effect: string_like
            What edge technique to use. Default set to symmetric that mirrors nearest extrema.

        alpha : float
            Used in symmetric edge-effect to decide whether to anchor spline.
            When alpha = 1 technique reduces to symmetric in:

            K. Zeng and M. He. A Simple Boundary Process Technique for Empirical Mode
            Decomposition. In IEEE International Geoscience and Remote Sensing Symposium,
            volume 6, pages 4258–4261. IEEE, 2004.

            or

            J. Zhao and D. Huang. Mirror Extending and Circular Spline Function for Empirical
            Mode Decomposition Method. Journal of Zhejiang University - Science A, 2(3):
            247–252, 2001.

            Other extreme is alpha = -infinity. Time series is reflected and next extremum is taken.

        nn_m : integer
            Number of points (outputs) on which to train in neural network edge effect.

        nn_k : integer
            Number of points (inputs) to use when estimating weights for neuron.

        nn_method : string_like
            Gradient descent method used to estimate weights.

        nn_learning_rate : float
            Learning rate to use when adjusting weights.

        nn_iter : integer
            Number of iterations to perform when estimating weights.

        Returns
        -------
        approximation : real ndarray
            Cubic basis spline envelope fitted through extrema.

        coef : real ndarray
            Corresponding coefficients.

        Notes
        -----
        Least square fitting function of extrema.
        Envelopes are second-order smoothed.

        """
        time = self.time
        time_series = self.time_series

        # Need both maxima and minima for edge effects other than symmetric
        utility = Utility(time, time_series)

        extrema_bool_max = utility.max_bool_func_1st_order_fd()
        extrema_bool_min = utility.min_bool_func_1st_order_fd()

        new_knots_time = time_extension(time)

        subset_time_vector_bool = ((time[0] <= new_knots_time) & (new_knots_time <= time[-1]))
        # boolean of applicable subset of points

        cubic_basis_function_matrix_extended = extended_matrix

        # identical from this point?

        approximation, coef = envelope_basis_function(
            time=time, time_series=time_series, extrema_bool_max=extrema_bool_max,
            extrema_bool_min=extrema_bool_min,
            cubic_basis_function_matrix_extended=cubic_basis_function_matrix_extended,
            new_knots_time=new_knots_time, subset_time_vector_bool=subset_time_vector_bool,
            knots_for_envelope=knots_for_envelope, extrema_type=extrema_type, smooth=smooth,
            smoothing_penalty=smoothing_penalty, edge_effect=edge_effect, alpha=alpha, nn_m=nn_m, nn_k=nn_k,
            nn_method=nn_method, nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

        return approximation, coef

    def envelope_basis_function_approximation_fixed_points(self, knots_for_envelope: np.ndarray, extrema_type: str,
                                                           max_bool: np.ndarray, min_bool: np.ndarray,
                                                           smooth: bool = True, smoothing_penalty: float = 1,
                                                           edge_effect: str = 'symmetric', alpha: float = 0.1,
                                                           nn_m: int = 200, nn_k: int = 100,
                                                           nn_method: str = 'grad_descent',
                                                           nn_learning_rate: float = 0.01,
                                                           nn_iter: int = 10) -> (np.ndarray, np.ndarray):
        """
        Envelope calculation that directly takes extrema as inputs to accommodate external optimisation.

        Parameters
        ----------
        knots_for_envelope : array_like
            Knots to be used to fit cubic B-splines.

        extrema_type : string_like
            Whether maxima or minima.

        max_bool : array_like
            Vector of booleans designating optimised maxima.

        min_bool : array_like
            Vector of booleans designating optimised minima.

        smooth : boolean
            Whether to smooth envelope (SEMD combined with Enhanced EMD - not be confused with Ensemble EMD)

        smoothing_penalty : float
            Penalty to use when second-order smoothing the spline.

        edge_effect: string_like
            What edge technique to use. Default set to symmetric that mirrors nearest extrema.

        alpha : float
            Used in symmetric edge-effect to decide whether to anchor spline.
            When alpha = 1 technique reduces to symmetric 'Symmetric Anchor' in:

            K. Zeng and M. He. A Simple Boundary Process Technique for Empirical Mode
            Decomposition. In IEEE International Geoscience and Remote Sensing Symposium,
            volume 6, pages 4258–4261. IEEE, 2004.

            or

            J. Zhao and D. Huang. Mirror Extending and Circular Spline Function for Empirical
            Mode Decomposition Method. Journal of Zhejiang University - Science A, 2(3):
            247–252, 2001.

            Other extreme is alpha = -infinity. Time series is reflected and next extremum is taken.

        nn_m : integer
            Number of points (outputs) on which to train in neural network edge effect.

        nn_k : integer
            Number of points (inputs) to use when estimating weights for neuron.

        nn_method : string_like
            Gradient descent method used to estimate weights.

        nn_learning_rate : float
            Learning rate to use when adjusting weights.

        nn_iter : integer
            Number of iterations to perform when estimating weights.

        Returns
        -------
        approximation : array_like
            Optimised extrema envelope.

        coef : array_like
            Coefficients corresponding to optimised extrema envelope.

        Notes
        -----
        Only major change to original function is extrema boolean vectors are now an input to accommodate external
        optimisation.

        """
        time = self.time
        time_series = self.time_series

        new_knots_time = time_extension(time)

        subset_time_vector_bool = ((time[0] <= new_knots_time) & (new_knots_time <= time[-1]))
        # boolean of applicable subset of points

        new_knots = time_extension(knots_for_envelope)

        basis = Basis(time=new_knots_time, time_series=new_knots_time)

        extrema_bool_max = max_bool
        extrema_bool_min = min_bool

        cubic_basis_function_matrix_extended = basis.cubic_b_spline(new_knots)

        # identical from this point?

        approximation, coef = envelope_basis_function(
            time=time, time_series=time_series, extrema_bool_max=extrema_bool_max,
            extrema_bool_min=extrema_bool_min,
            cubic_basis_function_matrix_extended=cubic_basis_function_matrix_extended,
            new_knots_time=new_knots_time, subset_time_vector_bool=subset_time_vector_bool,
            knots_for_envelope=knots_for_envelope, extrema_type=extrema_type, smooth=smooth,
            smoothing_penalty=smoothing_penalty, edge_effect=edge_effect, alpha=alpha, nn_m=nn_m, nn_k=nn_k,
            nn_method=nn_method, nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

        return approximation, coef

    def direct_detrended_fluctuation_estimation(self, knots: np.ndarray, smooth: bool = True,
                                                smoothing_penalty: float = 1, technique: str = 'inflection_points',
                                                order: int = 15, increment: int = 10) -> (np.ndarray, np.ndarray):
        """
        Directly estimates local mean rather than using extrema envelopes.

        Parameters
        ----------
        knots : array_like
            Knot points to use when interpolating local mean estimate.

        smooth : boolean
            Whether or not to second-order smooth the interpolation.

        smoothing_penalty : float
            Smoothing penalty for second-order smoothing of parameter estimates to be multiplied on to
            basis function matrix.

        technique : string_like
            What technique to use for direct local mean estimation.
            Default is 'inflection_points'.

        order : integer (odd positive)
            Number of surrounding points to use when averaging when technique='binomial_average'.

        increment : integer (positive)
            Difference in indices between points used for local mean estimation.

        Returns
        -------
        approximation : real ndarray
            Local mean estimate - same dimensions as lsq_signal.

        coef : real ndarray
            Coefficents associated with cubic B-spline approximation.

        Notes
        -----
        No edge effects - can include later.

        """
        time = self.time
        time_series = self.time_series

        point_basis_function_matrix = {}  # avoids unnecessary error
        interpolation_points = {}  # avoids unnecessary error
        cubic_basis_function_matrix = {}  # avoids unnecessary error

        new_knots_time = time_extension(time)  # extend time vector

        subset_time_vector_bool = ((time[0] <= new_knots_time) & (new_knots_time <= time[-1]))
        # boolean of applicable subset of points - not necessary unless using edge effect

        new_knots = time_extension(knots)  # extend knot vector

        basis = Basis(time=new_knots_time, time_series=new_knots_time)
        utility = Utility(time=time, time_series=time_series)

        cubic_basis_function_matrix_extended = basis.cubic_b_spline(new_knots)  # extended basis matrix

        if technique == 'inflection_points':

            # calculate inflection point bool and inflection points
            inflection_points_bool = utility.inflection_point()
            inflection_points = time_series[inflection_points_bool]

            # extract points only relevant to spline estimation
            cubic_basis_function_matrix = cubic_basis_function_matrix_extended[:, subset_time_vector_bool]

            interpolation_point_basis_function_matrix = cubic_basis_function_matrix[:, inflection_points_bool]

            point_basis_function_matrix = interpolation_point_basis_function_matrix.copy()
            interpolation_points = inflection_points.copy()

        elif technique == 'binomial_average':

            binomial_range = np.arange(len(time))  # vector of indices
            binomial_bool = np.mod(binomial_range, increment) == 0  # boolean where indices are at correct locations
            binomial_average_points = utility.binomial_average(order=order)  # binomial average points
            binomial_average_points = binomial_average_points[binomial_bool]  # subset of points for interpolation

            # extract points only relevant to spline estimation
            cubic_basis_function_matrix = cubic_basis_function_matrix_extended[:, subset_time_vector_bool]

            # extract relevant points of matrix
            binomial_average_point_basis_function_matrix = cubic_basis_function_matrix[:, binomial_bool]

            point_basis_function_matrix = binomial_average_point_basis_function_matrix.copy()
            interpolation_points = binomial_average_points.copy()

        if smooth:

            second_order_matrix = np.zeros((np.shape(point_basis_function_matrix)[0],
                                            np.shape(point_basis_function_matrix)[0] - 2))  # note transpose

            for a in range(np.shape(point_basis_function_matrix)[0] - 2):
                second_order_matrix[a:(a + 3), a] = [1, -2, 1]  # filling values for second-order difference matrix

            point_basis_function_matrix = np.append(point_basis_function_matrix,
                                                    smoothing_penalty * second_order_matrix, axis=1)
            # l2 norm trick

            interpolation_points = np.append(interpolation_points,
                                             np.zeros(np.shape(point_basis_function_matrix)[0] - 2), axis=0)
            # l2 norm trick

            coef = np.linalg.lstsq(point_basis_function_matrix.transpose(), interpolation_points,
                                   rcond=None)
            # intepolation_points_basis_function_matrix and extrema matrix augmented for smoothness

            coef = coef[0]

            approximation = np.matmul(coef, cubic_basis_function_matrix)  # basis_function_matrix is used here

        else:

            coef = np.linalg.lstsq(point_basis_function_matrix.transpose(), interpolation_points, rcond=None)
            # no smoothness

            coef = coef[0]

            approximation = np.matmul(coef, cubic_basis_function_matrix)  # basis_function_matrix is used here

        # knots_fix - 2 more basis functions than knots
        # must now extract from coef only coefficients necessary for envelope
        tot_coef = len(coef)  # total number of coef in extension
        tot_internal_knots = len(knots)  # original knots
        tot_coef_needed = tot_internal_knots - 4 + (4 + 2)  # extended math for demonstration purposes
        tot_coef_not_needed = tot_coef - tot_coef_needed  # coefficients not needed
        coef = coef[int(tot_coef_not_needed / 2):int(tot_coef_not_needed / 2 + tot_coef_needed)]
        # extract relevant coefficient

        return approximation, coef
