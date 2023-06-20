
#     ________
#            /
#      \    /
#       \  /
#        \/

# emd class to handle core EMD methods

import numpy as np
import seaborn as sns
from scipy import signal as sig
import matplotlib.pyplot as plt
import warnings

from emd_utils import Utility, time_extension
from emd_basis import Basis
from emd_preprocess import Preprocess
from emd_mean import Fluctuation
from emd_hilbert import Hilbert, theta, omega, hilbert_spectrum

sns.set(style='darkgrid')
np.seterr(divide='ignore', invalid='ignore')


def stopping_criterion_fail_helper(intrinsic_mode_function_storage, intrinsic_mode_function_candidate,
                                   intrinsic_mode_function_storage_coef, intrinsic_mode_function_candidate_coef,
                                   remainder, remainder_coef, imf_count, internal_iteration_count,

                                   knot_time, debug, detrended_fluctuation_technique, matrix, b_spline_matrix_extended,
                                   knot_envelope, smooth, smoothing_penalty, edge_effect, spline_method, alpha,
                                   order, increment, optimal_maxima, optimal_minima, verbose,
                                   stopping_criterion_threshold, stopping_criterion, coef_storage, optimise_knots,
                                   knots,
                                   calculated_threshold, mean_fluctuation_theta_1, mft_alpha, nn_m, nn_k,
                                   nn_method, nn_learning_rate, nn_iter):
    """
    Helper function to prevent repetition in and decrease total length of empirical_mode_decomposition parent function.

    Parameters divided into three groups:
    ->  those taken in, augmented or changed and output again,
    ->  those outputted after being calculated within function, and
    ->  those used as parameters in parent function.

    Parameters
    ----------
    intrinsic_mode_function_storage : real ndarray
        Stores confirmed IMFs (output again).

    intrinsic_mode_function_candidate : real ndarray
        Contains IMF candidate - is either added to intrinsic_mode_function_storage or recalculated (output again).

    intrinsic_mode_function_storage_coef : real ndarray
        Stores confirmed IMF coefficients (output again).

    intrinsic_mode_function_candidate_coef : real ndarray
        Contains IMF candidate coefficients - is either added to intrinsic_mode_function_storage_coef or recalculated
        (output again).

    remainder : real ndarray
        Contains remainder of time series (output again).

    remainder_coef : real ndarray
        Contains remainder coefficients of time series (output again).

    imf_count : integer
        Current IMF count (output again).

    internal_iteration_count : integer
        Current internal iteration count (output again).

    ###

    knot_time : real ndarray
        Internal knot time points for envelope construction.

    debug : boolean
        If debugging, this displays every single incremental imf with corresponding extrema, envelopes, and mean.

    detrended_fluctuation_technique : string
        What technique is used to estimate local mean of signal.

    matrix : boolean
        If true, constructs cubic-basis spline matrix once at outset - greatly increases speed.

    b_spline_matrix_extended : real ndarray
        Preconstructed matrix containing B-spline bases.

    knot_envelope : real ndarray
        Internal knot points for envelope construction.

    smooth : boolean
        Whether or not envelope smoothing takes place - Statistical Empirical Mode Decomposition (SEMD).

    smoothing_penalty : float
        Penalty to be used when smoothing - Statistical Empirical Mode Decomposition (SEMD).

    edge_effect : string
        What technique is used for the edges of the envelopes to not propagate errors.

    spline_method : string
        Spline method to use for smoothing and envelope sifting.

    alpha : float
        Value 'alpha' applied to conditional symmetric edge effect.

    order : integer (odd positive)
        The number of points to use in binomial averaging.

    increment : integer (positive)
        The incrementation of binomial averaging.

    optimal_maxima: real ndarray
        Array of booleans containing location of optimal maxima for Enhanced Empirical Mode Decomposition.

    optimal_minima: real ndarray
        Array of booleans containing location of optimal minima for Enhanced Empirical Mode Decomposition.

    verbose : boolean
        Whether or not to print success or failure of various criteria - 2 IMF conditions, stopping criteria, etc.

    stopping_criterion_threshold : float
        What threshold is used for whichever stopping criterion we use in stopping_criterion.

    stopping_criterion : string
        String representing stopping criterion used for output message.

    coef_storage : real ndarray
        Stores coefficients.

    optimise_knots : int
        What knot optimisation method has been selected.

    knots : real ndarray
        Knots used in algorithm.

    calculated_threshold : float
        Calculated threshold to compare against stopping_criterion_threshold above.

    mean_fluctuation_theta_1 : float
        Mean Fluctuation Threshold theta 1 value.

    mft_alpha : float
        Mean Fluctuation Threshold theta alpha threshold value.

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
    intrinsic_mode_function_storage : real ndarray
        Stores confirmed IMFs.

    intrinsic_mode_function_candidate : real ndarray
        Contains IMF candidate - is either added to intrinsic_mode_function_storage or recalculated.

    intrinsic_mode_function_storage_coef : real ndarray
        Stores confirmed IMF coefficients.

    intrinsic_mode_function_candidate_coef : real ndarray
        Contains IMF candidate coefficients - is either added to intrinsic_mode_function_storage_coef or recalculated.

    remainder : real ndarray
        Contains remainder of time series.

    remainder_coef : real ndarray
        Contains remainder coefficients of time series.

    imf_count : integer
        Current IMF count.

    internal_iteration_count : integer
        Current internal iteration count.

    imf_envelope_mean : real ndarray
        Calculated local mean.

    imf_envelope_mean_coef : real ndarray
        Calculated local mean coefficients.

    imf_max_time_points : real ndarray
        Time points corresponding to maxima.

    imf_min_time_points : real ndarray
        Time points corresponding to minima.

    intrinsic_mode_function_max_storage : real ndarray
        Maxima storage.

    intrinsic_mode_function_min_storage : real ndarray
        Minima storage.

    imf_envelope_max : real ndarray
        Maximum envelope fit to imf_max_time_points, intrinsic_mode_function_max_storage.

    imf_envelope_max_coef : real ndarray
        Maximum envelope coefficients fit to imf_max_time_points, intrinsic_mode_function_max_storage.

    imf_envelope_min : real ndarray
        Minimum envelope fit to imf_min_time_points, intrinsic_mode_function_min_storage.

    imf_envelope_min_coef : real ndarray
        Minimum envelope coefficients fit to imf_min_time_points, intrinsic_mode_function_min_storage.

    coef : real ndarray or Dict

    Notes
    -----
    Function is executed when stopping criterion threshold is satisfied and sifting can stop.

    """
    imf_envelope_mean = {}  # avoids unnecessary error
    imf_envelope_mean_coef = {}  # avoids unnecessary error
    imf_max_time_points = {}  # avoids unnecessary error
    imf_min_time_points = {}  # avoids unnecessary error
    imf_envelope_max = {}  # avoids unnecessary error
    imf_envelope_max_coef = {}  # avoids unnecessary error
    imf_envelope_min = {}  # avoids unnecessary error
    imf_envelope_min_coef = {}  # avoids unnecessary error

    # stores imf in storage matrix
    intrinsic_mode_function_storage = np.vstack((intrinsic_mode_function_storage,
                                                 intrinsic_mode_function_candidate))

    if optimise_knots != 2:
        # stores imf coefficients in storage matrix
        intrinsic_mode_function_storage_coef = np.vstack((intrinsic_mode_function_storage_coef,
                                                          intrinsic_mode_function_candidate_coef))
    else:
        coef_storage[imf_count] = intrinsic_mode_function_candidate_coef

    # fixes coefficient length discrepancy when optimise_knots=2
    remainder = remainder - intrinsic_mode_function_candidate
    # removes imf from what remains of smoothed signal
    if len(remainder_coef) != len(intrinsic_mode_function_candidate_coef):
        basis_coef = Basis(time=knot_time, time_series=remainder)
        remainder, remainder_coef = basis_coef.basis_function_approximation(knots=knots,
                                                                            knot_time=knot_time)
    else:
        remainder_coef = remainder_coef - intrinsic_mode_function_candidate_coef

    # new imf candidate for next loop iteration
    intrinsic_mode_function_candidate = remainder.copy()
    intrinsic_mode_function_candidate_coef = remainder_coef.copy()

    utility_imf = Utility(time=knot_time, time_series=intrinsic_mode_function_candidate)

    # finds imf candidate maxima boolean #
    intrinsic_mode_function_max_bool = utility_imf.max_bool_func_1st_order_fd()
    # extracts imf maximums
    intrinsic_mode_function_max_storage = intrinsic_mode_function_candidate[intrinsic_mode_function_max_bool]

    # finds imf candidate minima boolean #
    intrinsic_mode_function_min_bool = utility_imf.min_bool_func_1st_order_fd()
    # extracts imf minimums
    intrinsic_mode_function_min_storage = intrinsic_mode_function_candidate[intrinsic_mode_function_min_bool]

    if debug:
        imf_max_time_points = knot_time[intrinsic_mode_function_max_bool]
        imf_min_time_points = knot_time[intrinsic_mode_function_min_bool]

    # calculate envelopes
    # if there are still maxima and minima
    if any(intrinsic_mode_function_max_storage) and any(intrinsic_mode_function_min_storage):

        fluctuation_imf = Fluctuation(time=knot_time, time_series=intrinsic_mode_function_candidate)

        if detrended_fluctuation_technique == 'envelopes':

            if matrix:

                imf_envelope_max, imf_envelope_max_coef = \
                    fluctuation_imf.envelope_basis_function_approximation_matrix(
                        b_spline_matrix_extended, knot_envelope, 'maxima',
                        smooth, smoothing_penalty, edge_effect, alpha, nn_m=nn_m,
                        nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                        nn_iter=nn_iter)

                imf_envelope_min, imf_envelope_min_coef = \
                    fluctuation_imf.envelope_basis_function_approximation_matrix(
                        b_spline_matrix_extended, knot_envelope, 'minima',
                        smooth, smoothing_penalty, edge_effect, alpha, nn_m=nn_m,
                        nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                        nn_iter=nn_iter)

            else:

                imf_envelope_max, imf_envelope_max_coef = \
                    fluctuation_imf.envelope_basis_function_approximation(
                        knot_envelope, 'maxima', smooth, smoothing_penalty, edge_effect, spline_method, alpha,
                        nn_m=nn_m, nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                        nn_iter=nn_iter)

                imf_envelope_min, imf_envelope_min_coef = \
                    fluctuation_imf.envelope_basis_function_approximation(
                        knot_envelope, 'minima', smooth, smoothing_penalty, edge_effect, spline_method, alpha,
                        nn_m=nn_m, nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                        nn_iter=nn_iter)

            # calculate mean and coefficients
            imf_envelope_mean = (imf_envelope_max + imf_envelope_min) / 2
            imf_envelope_mean_coef = (imf_envelope_max_coef + imf_envelope_min_coef) / 2

        elif detrended_fluctuation_technique == 'inflection_points':

            imf_envelope_mean, imf_envelope_mean_coef = \
                fluctuation_imf.direct_detrended_fluctuation_estimation(
                    knot_envelope, smooth=smooth, smoothing_penalty=smoothing_penalty, technique='inflection_points')

        elif detrended_fluctuation_technique == 'binomial_average':

            imf_envelope_mean, imf_envelope_mean_coef = \
                fluctuation_imf.direct_detrended_fluctuation_estimation(
                    knot_envelope, smooth=smooth, smoothing_penalty=smoothing_penalty, technique='binomial_average',
                    order=order, increment=increment)

        elif detrended_fluctuation_technique == 'enhanced':

            # calculates maximum envelope & coefficients
            imf_envelope_max, imf_envelope_max_coef = \
                fluctuation_imf.envelope_basis_function_approximation_fixed_points(
                    knot_envelope, 'maxima', optimal_maxima, optimal_minima, smooth, smoothing_penalty,
                    edge_effect='symmetric', alpha=alpha, nn_m=nn_m,
                    nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                    nn_iter=nn_iter)

            # calculates minimum envelope & coefficients
            imf_envelope_min, imf_envelope_min_coef = \
                fluctuation_imf.envelope_basis_function_approximation_fixed_points(
                    knot_envelope, 'minima', optimal_maxima, optimal_minima, smooth, smoothing_penalty,
                    edge_effect='symmetric', alpha=alpha, nn_m=nn_m,
                    nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                    nn_iter=nn_iter)

            # calculate mean and coefficients
            imf_envelope_mean = (imf_envelope_min + imf_envelope_max) / 2
            imf_envelope_mean_coef = (imf_envelope_min_coef + imf_envelope_max_coef) / 2

    # if there are not maxima and minima - mean set to zero
    # investigate effect of this - suspect it leads to breaking of while loop
    # designed specifically to end sifting in while loop
    else:
        imf_envelope_mean = np.zeros_like(intrinsic_mode_function_candidate)

    # print report
    # need to customise section based on stopping criterion choice
    if verbose:

        if stopping_criterion == 'sd':

            print(f'IMF_{imf_count}{internal_iteration_count} Standard deviation STOPPING CRITERION MET with sd = '
                  f'{str(np.round(calculated_threshold, 2))} < sd threshold = {stopping_criterion_threshold}')

        elif stopping_criterion == 'sd_11a':

            print(f'IMF_{imf_count}{internal_iteration_count} Standard deviation (11a) STOPPING CRITERION MET with '
                  f'sd = {str(calculated_threshold)} < sd threshold = {stopping_criterion_threshold}')

        elif stopping_criterion == 'sd_11b':

            print(f'IMF_{imf_count}{internal_iteration_count} Standard deviation (11b) STOPPING CRITERION MET with '
                  f'sd = {str(calculated_threshold)} < sd threshold = {stopping_criterion_threshold}')

        elif stopping_criterion == 'mft':

            print(f'IMF_{imf_count}{internal_iteration_count} Mean Fluctuation Threshold STOPPING CRITERION MET '
                  f'with theta_1 threshold = {mean_fluctuation_theta_1} > {1 - mft_alpha} and theta_2 threshold = 1')

        elif stopping_criterion == 'edt':

            print(f'IMF_{imf_count}{internal_iteration_count} Energy Difference Tracking STOPPING CRITERION MET '
                  f'with energy difference = {calculated_threshold} < {stopping_criterion_threshold}')

        elif stopping_criterion == 'S_stoppage':

            print(f'IMF_{imf_count}{internal_iteration_count} S Stoppage STOPPING CRITERION MET '
                  f'with count = {stopping_criterion_threshold}')

    # imf count updated and internal iteration count reset
    imf_count += 1
    internal_iteration_count = 0

    return intrinsic_mode_function_storage, intrinsic_mode_function_candidate, \
        intrinsic_mode_function_storage_coef, intrinsic_mode_function_candidate_coef, \
        remainder, remainder_coef, imf_count, internal_iteration_count, \
        imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, imf_min_time_points, \
        intrinsic_mode_function_max_storage, intrinsic_mode_function_min_storage, \
        imf_envelope_max, imf_envelope_max_coef, imf_envelope_min, imf_envelope_min_coef, coef_storage


def stopping_criterion_pass_helper(intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef,
                                   imf_envelope_mean, imf_envelope_mean_coef,

                                   imf_count, internal_iteration_count,
                                   knot_time, debug, detrended_fluctuation_technique, matrix, b_spline_matrix_extended,
                                   knot_envelope, smooth, smoothing_penalty, edge_effect, spline_method, alpha,
                                   order, increment, optimal_maxima, optimal_minima, verbose,
                                   stopping_criterion_threshold, stopping_criterion, calculated_threshold,
                                   mean_fluctuation_theta_1, mft_alpha, mean_fluctuation_theta_2, s_stoppage_count,
                                   nn_m, nn_k, nn_method, nn_learning_rate, nn_iter):
    """
    Helper function to prevent repetition in and decrease total length of empirical_mode_decomposition parent function.

    Parameters divided into three groups:
    ->  those taken in, augmented or changed and output again,
    ->  those outputted after being calculated within function, and
    ->  those used as parameters in parent function.

    Parameters
    ----------
    intrinsic_mode_function_candidate : real ndarray
        Contains IMF candidate - is either added to intrinsic_mode_function_storage or recalculated (output again).

    intrinsic_mode_function_candidate_coef : real ndarray
        Contains IMF candidate coefficients - is either added to intrinsic_mode_function_storage_coef or recalculated
        (output again).

    imf_envelope_mean : real ndarray
        Calculated local mean (output again).

    imf_envelope_mean_coef : real ndarray
        Calculated local mean coefficients (output again).

    ###

    imf_count : integer
        Current IMF count.

    internal_iteration_count : integer
        Current internal iteration count.

    knot_time : real ndarray
        Internal knot time points for envelope construction.

    debug : boolean
        If debugging, this displays every single incremental imf with corresponding extrema, envelopes, and mean.

    detrended_fluctuation_technique : string
        What technique is used to estimate local mean of signal.

    matrix : boolean
        If true, constructs cubic-basis spline matrix once at outset - greatly increases speed.

    b_spline_matrix_extended : real ndarray
        Preconstructed matrix containing B-spline bases.

    knot_envelope : real ndarray
        Internal knot points for envelope construction.

    smooth : boolean
        Whether or not envelope smoothing takes place - Statistical Empirical Mode Decomposition (SEMD).

    smoothing_penalty : float
        Penalty to be used when smoothing - Statistical Empirical Mode Decomposition (SEMD).

    edge_effect : string
        What technique is used for the edges of the envelopes to not propagate errors.

    spline_method : string
        Spline method to use for smoothing and envelope sifting.

    alpha : float
        Value 'alpha' applied to conditional symmetric edge effect.

    order : integer (odd positive)
        The number of points to use in binomial averaging.

    increment : integer (positive)
        The incrementation of binomial averaging.

    optimal_maxima: real ndarray
        Array of booleans containing location of optimal maxima for Enhanced Empirical Mode Decomposition.

    optimal_minima: real ndarray
        Array of booleans containing location of optimal minima for Enhanced Empirical Mode Decomposition.

    verbose : boolean
        Whether or not to print success or failure of various criteria - 2 IMF conditions, stopping criteria, etc.

    stopping_criterion_threshold : float
        What threshold is used for whichever stopping criterion we use in stopping_criterion.

    stopping_criterion : string
        String representing stopping criterion used for output message.

    calculated_threshold : float
        Calculated threshold to compare against stopping_criterion_threshold above.

    mean_fluctuation_theta_1 : float
        Mean Fluctuation Threshold theta 1 value.

    mft_alpha : float
        Mean Fluctuation Threshold theta alpha threshold value.

    mean_fluctuation_theta_2 : float
        Mean Fluctuation Threshold theta 2 value.

    s_stoppage_count : integer
        S-stoppage stopping criterion value.

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
    intrinsic_mode_function_candidate : real ndarray
        Contains IMF candidate - is either added to intrinsic_mode_function_storage or recalculated.

    intrinsic_mode_function_candidate_coef : real ndarray
        Contains IMF candidate coefficients - is either added to intrinsic_mode_function_storage_coef or recalculated.

    imf_envelope_mean : real ndarray
        Calculated local mean.

    imf_envelope_mean_coef : real ndarray
        Calculated local mean coefficients.

    imf_max_time_points : real ndarray
        Time points corresponding to maxima.

    imf_min_time_points : real ndarray
        Time points corresponding to minima.

    intrinsic_mode_function_max_storage : real ndarray
        Maxima storage.

    intrinsic_mode_function_min_storage : real ndarray
        Minima storage.

    imf_envelope_max : real ndarray
        Maximum envelope fit to imf_max_time_points, intrinsic_mode_function_max_storage.

    imf_envelope_max_coef : real ndarray
        Maximum envelope coefficients fit to imf_max_time_points, intrinsic_mode_function_max_storage.

    imf_envelope_min : real ndarray
        Minimum envelope fit to imf_min_time_points, intrinsic_mode_function_min_storage.

    imf_envelope_min_coef : real ndarray
        Minimum envelope coefficients fit to imf_min_time_points, intrinsic_mode_function_min_storage.

    Notes
    -----
    Function is executed when stopping criterion threshold is not satisfied and sifting must continue.

    """
    imf_max_time_points = {}  # avoids unnecessary error
    imf_min_time_points = {}  # avoids unnecessary error
    imf_envelope_max = {}  # avoids unnecessary error
    imf_envelope_max_coef = {}  # avoids unnecessary error
    imf_envelope_min = {}  # avoids unnecessary error
    imf_envelope_min_coef = {}  # avoids unnecessary error

    # removes mean from imf candidate to create new candidate
    intrinsic_mode_function_candidate = intrinsic_mode_function_candidate - imf_envelope_mean
    # removes mean from imf candidate to create new candidate coefficients
    intrinsic_mode_function_candidate_coef = intrinsic_mode_function_candidate_coef - imf_envelope_mean_coef

    utility_imf = Utility(time=knot_time, time_series=intrinsic_mode_function_candidate)

    # finds intrinsic_mode_function_candidate maximums boolean
    intrinsic_mode_function_max_bool = utility_imf.max_bool_func_1st_order_fd()
    # stores new maximums
    intrinsic_mode_function_max_storage = intrinsic_mode_function_candidate[intrinsic_mode_function_max_bool]

    # finds intrinsic_mode_function_candidate minimums boolean
    intrinsic_mode_function_min_bool = utility_imf.min_bool_func_1st_order_fd()
    # stores new minimums
    intrinsic_mode_function_min_storage = intrinsic_mode_function_candidate[intrinsic_mode_function_min_bool]

    if debug:
        imf_max_time_points = knot_time[intrinsic_mode_function_max_bool]
        imf_min_time_points = knot_time[intrinsic_mode_function_min_bool]

    fluctuation_imf = Fluctuation(time=knot_time, time_series=intrinsic_mode_function_candidate)

    if detrended_fluctuation_technique == 'envelopes':

        # calculate envelopes
        if matrix:

            imf_envelope_max, imf_envelope_max_coef = fluctuation_imf.envelope_basis_function_approximation_matrix(
                b_spline_matrix_extended, knot_envelope, 'maxima', smooth, smoothing_penalty, edge_effect, alpha=alpha,
                nn_m=nn_m, nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                nn_iter=nn_iter)

            imf_envelope_min, imf_envelope_min_coef = fluctuation_imf.envelope_basis_function_approximation_matrix(
                b_spline_matrix_extended, knot_envelope, 'minima', smooth, smoothing_penalty, edge_effect, alpha=alpha,
                nn_m=nn_m, nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                nn_iter=nn_iter)

        else:

            imf_envelope_max, imf_envelope_max_coef = fluctuation_imf.envelope_basis_function_approximation(
                knot_envelope, 'maxima', smooth, smoothing_penalty,
                edge_effect, spline_method=spline_method, alpha=alpha, nn_m=nn_m,
                nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                nn_iter=nn_iter)

            imf_envelope_min, imf_envelope_min_coef = fluctuation_imf.envelope_basis_function_approximation(
                knot_envelope, 'minima', smooth, smoothing_penalty,
                edge_effect, spline_method=spline_method, alpha=alpha, nn_m=nn_m,
                nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                nn_iter=nn_iter)

        # calculate mean and coefficients
        imf_envelope_mean = (imf_envelope_max + imf_envelope_min) / 2
        imf_envelope_mean_coef = (imf_envelope_max_coef + imf_envelope_min_coef) / 2

    elif detrended_fluctuation_technique == 'inflection_points':

        imf_envelope_mean, imf_envelope_mean_coef = fluctuation_imf.direct_detrended_fluctuation_estimation(
            knot_envelope, smooth=smooth, smoothing_penalty=smoothing_penalty, technique='inflection_points')

    elif detrended_fluctuation_technique == 'binomial_average':

        imf_envelope_mean, imf_envelope_mean_coef = fluctuation_imf.direct_detrended_fluctuation_estimation(
            knot_envelope, smooth=smooth, smoothing_penalty=smoothing_penalty,
            technique='binomial_average', order=order, increment=increment)

    elif detrended_fluctuation_technique == 'enhanced':

        # calculates maximum envelope & coefficients
        imf_envelope_max, imf_envelope_max_coef = fluctuation_imf.envelope_basis_function_approximation_fixed_points(
            knot_envelope, 'maxima', optimal_maxima, optimal_minima, smooth,
            smoothing_penalty, edge_effect='symmetric', alpha=alpha, nn_m=nn_m,
            nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
            nn_iter=nn_iter)

        # calculates minimum envelope & coefficients
        imf_envelope_min, imf_envelope_min_coef = fluctuation_imf.envelope_basis_function_approximation_fixed_points(
            knot_envelope, 'minima', optimal_maxima, optimal_minima, smooth,
            smoothing_penalty, edge_effect='symmetric', alpha=alpha, nn_m=nn_m,
            nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
            nn_iter=nn_iter)

        # calculate mean and coefficients
        imf_envelope_mean = (imf_envelope_min + imf_envelope_max) / 2
        imf_envelope_mean_coef = (imf_envelope_min_coef + imf_envelope_max_coef) / 2

    if verbose:

        if stopping_criterion == 'sd':

            print(f'IMF_{imf_count}{internal_iteration_count} Standard deviation STOPPING CRITERION NOT MET '
                  f'with sd = ' + str(np.round(calculated_threshold, 2)))

        elif stopping_criterion == 'sd_11a':

            print(f'IMF_{imf_count}{internal_iteration_count} Standard deviation (11a) STOPPING CRITERION NOT MET '
                  f'with sd = ' + str(calculated_threshold))

        elif stopping_criterion == 'sd_11b':

            print(f'IMF_{imf_count}{internal_iteration_count} Standard deviation (11b) STOPPING CRITERION NOT MET '
                  f'with sd = ' + str(calculated_threshold))

        elif stopping_criterion == 'mft':

            print(f'IMF_{imf_count}{internal_iteration_count} Mean Fluctuation Threshold STOPPING CRITERION NOT MET '
                  f'with theta_1 threshold = {mean_fluctuation_theta_1} < {1 - mft_alpha} or '
                  f'theta_2 threshold = {mean_fluctuation_theta_2} < 1')

        elif stopping_criterion == 'edt':

            print(f'IMF_{imf_count}{internal_iteration_count} Energy Difference Tracking STOPPING CRITERION NOT MET '
                  f'with energy difference = {calculated_threshold} > {stopping_criterion_threshold}')

        elif stopping_criterion == 'S_stoppage':

            print(f'IMF_{imf_count}{internal_iteration_count} S Stoppage STOPPING CRITERION NOT MET '
                  f'with count = {s_stoppage_count} < {stopping_criterion_threshold}')

    return intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef, \
        imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, imf_min_time_points, \
        intrinsic_mode_function_max_storage, intrinsic_mode_function_min_storage, \
        imf_envelope_max, imf_envelope_max_coef, imf_envelope_min, imf_envelope_min_coef


class EMD:

    def __init__(self, time_series: np.ndarray, **kwargs):

        self.time_series = time_series
        try:
            self.time = kwargs.get('time') * 1
        except TypeError:
            self.time = np.arange(len(self.time_series))

        if len(self.time) != len(self.time_series):
            raise ValueError('Input time series and input time are incompatible lengths.')

    def empirical_mode_decomposition(self, smooth=True,
                                     smoothing_penalty=1, edge_effect='symmetric', sym_alpha=0.1,
                                     stop_crit='sd',
                                     stop_crit_threshold=10, mft_theta_1=0.05, mft_theta_2=0.5,
                                     mft_alpha=0.05,
                                     mean_threshold=10, debug=False, verbose=True, spline_method='b_spline',
                                     dtht=False, dtht_method='kak', max_internal_iter=10, max_imfs=10, matrix=False,
                                     initial_smoothing=True,
                                     dft='envelopes', order=15, increment=10,
                                     preprocess='none', preprocess_window_length=51, preprocess_quantile=0.9,
                                     preprocess_penalty=1, preprocess_order=13, preprocess_norm_1=2,
                                     preprocess_norm_2=1,
                                     ensemble=False, ensemble_sd=0.1, ensemble_iter=10, enhanced_iter=10,
                                     output_coefficients=False, optimise_knots=0,
                                     knot_method='ser_bisect', output_knots=False,
                                     knot_error=10, knot_lamda=1, knot_epsilon=0.5,
                                     downsample_window='hamming', downsample_decimation_factor=20,
                                     downsample_window_factor=20, nn_m=200, nn_k=100, nn_method='grad_descent',
                                     nn_learning_rate=0.01, nn_iter=100, **kwargs):

        """
        Main function that serves as parent function to all above functions to do the actual sifting.

        Parameters
        ----------
        smooth : boolean
            Whether or not envelope smoothing takes place - Statistical Empirical Mode Decomposition -
            from:

                D. Kim, K. Kim, and H. Oh. Extending the scope of empirical mode decomposition by
                smoothing. EURASIP Journal on Advances in Signal Processing, 2012(168):1–17,
                2012.

            Highly recommended smoothing takes place - envelope fitting still takes place with arbitrary knot sequence,
            but if Schoenberg-Whitney conditions fail nonsensical envelopes may result.

        smoothing_penalty : float
            Penalty to be used when smoothing - Statistical Empirical Mode Decomposition.

        edge_effect : string
            What technique is used for the edges of the envelopes to not propagate errors
            - used in envelope_basis_function_approximation().

            symmetric : reflect extrema with no forced extrema at reflection point.

            symmetric_anchor : reflect extrema with forced extrema at reflection point depending on alpha value -
                               alpha value of 1 is equivalent to:

                K. Zeng and M. He. A Simple Boundary Process Technique for Empirical Mode
                Decomposition. In IEEE International Geoscience and Remote Sensing Symposium,
                volume 6, pages 4258–4261. IEEE, 2004.

                or

                J. Zhao and D. Huang. Mirror Extending and Circular Spline Function for Empirical
                Mode Decomposition Method. Journal of Zhejiang University - Science A, 2(3):
                247–252, 2001.

            symmetric_discard : reflect extrema about last extremum - discards ends of signal - from:

                G. Rilling, P. Flandrin, and P. Goncalves. On Empirical Mode Decomposition and its
                Algorithms. In IEEE-EURASIP Workshop on Nonlinear Signal and Image Process-
                ing, volume 3, pages 8–11. NSIP-03, Grado (I), 2003.

                or

                F. Wu and L Qu. An improved method for restraining the end effect in empirical mode
                decomposition and its applications to the fault diagnosis of large rotating machinery.
                Journal of Sound and Vibration, 314(3-5):586–602, 2008. doi: 10.1016/j.jsv.2008.
                01.020.

            anti-symmetric : reflects about end point on both axes -
                             reflected about x = x_end_point and y = y_end_point - modified version of:

                K. Zeng and M. He. A Simple Boundary Process Technique for Empirical Mode
                Decomposition. In IEEE International Geoscience and Remote Sensing Symposium,
                volume 6, pages 4258–4261. IEEE, 2004.

            characteristic_wave_Huang : calculate characteristic wave (sinusoid) using first/last two maxima or minima -
                                        from:

                N. Huang, Z. Shen, S. Long, M. Wu, H. Shih, Q. Zheng, N. Yen, C. Tung, and H. Liu.
                The Empirical Mode Decomposition and the Hilbert Spectrum for Nonlinear and
                Non-Stationary Time Series Analysis. Proceedings of the Royal Society of London
                A, 454:903–995, 1998.

            charcteristic_wave_Coughlin : calculate characterictic wave (sinusoid)
                                          using first/last maximum and minimum - from:

                K. Coughlin and K. Tung. 11-Year solar cycle in the stratosphere extracted by the
                empirical mode decomposition method. Advances in Space Research, 34(2):323–329,
                2004. doi: 10.1016/j.asr.2003.02.045.

            slope_based_method : calculate extrema using slopes between extrema and time difference - from:

                M. Dätig and T. Schlurmann. Performance and limitations of the hilbert-huang trans-
                formation (hht) with an application to irregular water waves. Ocean Engineering,
                31(14-15):1783–1834, 2004.

            improved_slope_based_method : calculate extrema using slopes betweeen extrema and time difference -
                                          takes into account end points and possibly anchors them - from:

                F. Wu and L Qu. An improved method for restraining the end effect in empirical mode
                decomposition and its applications to the fault diagnosis of large rotating machinery.
                Journal of Sound and Vibration, 314(3-5):586–602, 2008. doi: 10.1016/j.jsv.2008.
                01.020.

            average : averages the last two extrema and uses time difference to repeat pattern - from:

                F. Chiew, M. Peel, G. Amirthanathan, and G. Pegram. Identification of oscillations in
                historical global streamflow data using empirical mode decomposition. In Regional
                Hydrological Impacts of Climatic Change - Hydroclimatic Variabiltiy, volume 296,
                pages 53–62. International Association of Hydrological Sciences, 2005.

            neural_network : uses a single neuron neural network to explicitly extrapolate the
                             whole time series to approximate next extrema from:

                Deng Y, Wang W, Qian C, Wang Z, Dai D (2001). “Boundary-Processing-Technique in
                EMD Method and Hilbert Transform.”Chinese Science Bulletin,46(1), 954–960.

                ########################################################################################################
                #                                                                                                      #
                #   NOTE: Use with cause as explicit methods can result in nonsensical IMFs and basis proliferation.   #
                #         Do not blindly apply. Recommende to use debug=True for initial algorithm operation.          #
                #         Carefully constructed limits need to be imposed.                                             #
                #                                                                                                      #
                ########################################################################################################

            none : no edge-effect considered.

        sym_alpha : float
            'alpha' value applied to conditional symmetric edge effect.

        stop_crit : string
            What stopping criterion to use on the internal sifting loop.

            sd : standard deviation stopping criterion from:

                N. Huang, Z. Shen, S. Long, M. Wu, H. Shih, Q. Zheng, N. Yen, C. Tung, and H. Liu.
                The Empirical Mode Decomposition and the Hilbert Spectrum for Nonlinear and
                Non-Stationary Time Series Analysis. Proceedings of the Royal Society of London
                A, 454:903–995, 1998.

            sd_11a : standard deviation stopping criterion from:

                N. Huang and Z. Wu. A review on Hilbert-Huang transform: Method and its appli-
                cations to geophysical studies. Reviews of Geophysics, 46(RG2006):1–23, 2008. doi:
                10.1029/2007RG000228.

            sd_11b : standard deviation stopping criterion from:

                N. Huang and Z. Wu. A review on Hilbert-Huang transform: Method and its appli-
                cations to geophysical studies. Reviews of Geophysics, 46(RG2006):1–23, 2008. doi:
                10.1029/2007RG000228.

            mft : mean fluctuation threshold stopping criterion from:

                G. Rilling, P. Flandrin, and P. Goncalves. On Empirical Mode Decomposition and its
                Algorithms. In IEEE-EURASIP Workshop on Nonlinear Signal and Image Process-
                ing, volume 3, pages 8–11. NSIP-03, Grado (I), 2003.

                or

                A. Tabrizi, L. Garibaldi, A. Fasana, and S. Marchesiello. Influence of Stopping Crite-
                rion for Sifting Process of Empirical Mode Decomposition (EMD) on Roller Bear-
                ing Fault Diagnosis. In Advances in Condition Monitoring of Machinery in Non-
                Stationary Operations, pages 389–398. Springer-Verlag, Berlin Heidelberg, 2014.

            edt : energy difference tracking stopping criterion:

                C. Junsheng, Y. Dejie, and Y. Yu. Research on the Intrinsic Mode Function (IMF)
                Criterion in EMD Method. Mechanical Systems and Signal Processing, 20(4):817–
                824, 2006.

            S_stoppage : stoppage criterion based on number of successive IMF candidates with the same number of extrema
                         and zero-crossings:

                N. Huang and Z. Wu. A review on Hilbert-Huang transform: Method and its appli-
                cations to geophysical studies. Reviews of Geophysics, 46(RG2006):1–23, 2008. doi:
                10.1029/2007RG000228.

        stop_crit_threshold : float
            What threshold is used for whatever stopping criterion we use in stopping_criterion.

        mft_theta_1 : float
            theta_1 value used in mean fluctuation threshold stopping criterion.

        mft_theta_2 : float
            theta_2 value used in mean fluctuation threshold stopping criterion.

        mft_alpha : float
            alpha value used in mean fluctuation threshold stopping criterion.

        mean_threshold : float
            What threshold is used for the difference for the mean_envelope from zero.

        debug : boolean
            If we are debugging, this displays every single incremental imf
            with corresponding extrema, envelopes, and mean.

        verbose : bool
            Whether or not to print success or failure of various criteria - 2 IMF conditions, stopping criteria, etc.

        spline_method : string
            Spline method to use for smoothing and envelope sifting.

            b_spline: cubic B-spline envelope fitting from:

                Q. Chen, N. Huang, S. Riemenschneider, and Y. Xu. A B-spline Approach for Em-
                pirical Mode Decompositions. Advances in Computational Mathematics, 24(1-4):
                171–195, 2006.

            chsi: cubic Hermite spline interpolation from:

                A. Egambaram, N. Badruddin, V. Asirvadam, and T. Begum. Comparison of Envelope
                Interpolation Techniques in Empirical Mode Decomposition (EMD) for Eyeblink Ar-
                tifact Removal From EEG. In IEEE EMBS Conference on Biomedical Engineering
                and Sciences (IECBES), pages 590–595. IEEE, 2016.

            asi: Akima spline interpolation from:

                A. Egambaram, N. Badruddin, V. Asirvadam, and T. Begum. Comparison of Envelope
                Interpolation Techniques in Empirical Mode Decomposition (EMD) for Eyeblink Ar-
                tifact Removal From EEG. In IEEE EMBS Conference on Biomedical Engineering
                and Sciences (IECBES), pages 590–595. IEEE, 2016.

        dtht : boolean
            If True, performs discrete-time Hilbert transform of IMFs.

        dtht_method : string_like
            Whether to use Basic DTHT ('kak') or FFT DTHT ('fft').

        max_internal_iter : integer (positive)
            Additional stopping criterion - hard limit on number of internal siftings.

        max_imfs : integer (positive)
            Hard limit on number of external siftings.

        matrix : boolean
            If true, constructs cubic-basis spline matrix once at outset - greatly increases speed.
            IMPORTANT: Overrides 'spline_method' choice.

        initial_smoothing : boolean
            If true, performs initial smoothing and associated co-efficient fitting.
            If False, no initial smoothing is done - dtht is used to get Hilbert transform of IMF 1.

        dft : string
            What technique is used to estimate local mean of signal.

            envelopes : fits cubic spline to both maxima and minima and calculates mean by averaging envelopes - from:

                N. Huang, Z. Shen, S. Long, M. Wu, H. Shih, Q. Zheng, N. Yen, C. Tung, and H. Liu.
                The Empirical Mode Decomposition and the Hilbert Spectrum for Nonlinear and
                Non-Stationary Time Series Analysis. Proceedings of the Royal Society of London
                A, 454:903–995, 1998.

            inflection_points : estimates local mean by fitting cubic spline through inflection points - from:

                Y. Kopsinis and S. McLaughlin. Investigation of the empirical mode decomposition
                based on genetic algorithm optimization schemes. In Proceedings of the 32nd IEEE
                International Conference on Acoustics, Speech and Signal Processing (ICASSP’07),
                volume 3, pages 1397–1400, Honolulu, Hawaii, United States of America, 2007.
                IEEE.

            binomial_average : estimates local mean by taking binmoial average of surrounding points and interpolating -
                               from:

                Q. Chen, N. Huang, S. Riemenschneider, and Y. Xu. A B-spline Approach for Em-
                pirical Mode Decompositions. Advances in Computational Mathematics, 24(1-4):
                171–195, 2006.

                IMPORTANT: difficult to extract mean accurately on lower frequency structures -
                           replicates time series too closely.

            enhanced : performs Enhanced EMD on derivative of signal (or imf candidate) to approximate extrema locations
                       of highest frequency component which are better interpolation points for extrema envelopes -
                       from:

                Y. Kopsinis and S. McLaughlin. Enhanced Empirical Mode Decomposition using a
                Novel Sifting-Based Interpolation Points Detection. In Proceedings of the IEEE/SP
                14th Workshop on Statistical Signal Processing (SSP’07), pages 725–729, Madison,
                Wisconsin, United States of America, 2007b. IEEE Computer Society.

        order : integer (odd positive)
            The number of points to use in binomial averaging.

            Example: if order=5, then weighting vector will be (1/16, 4/16, 6/16, 4/16, 1/16) centred on selected point.

        increment : integer (positive)
            The incrementation of binomial averaging.

            Example: if increment=10, then point used will have indices: 0, 10, 20, etc.

        preprocess : string
            What preprocessing technique to use (if at all) - effective when dealing with heavy-tailed and mixed noise.

            median_filter : impose a median filter on signal - very robust when dealing with outliers.

            mean_filter : impose a mean filter on signal - more susceptible to outliers.

            winsorize : use quantile filters for upper and lower confidence intervals and set time series values
                        equal to upper or lower quantile values when time series exceeds these limits.

            winsorize_interpolate : use quantile filters for upper and lower confidence intervals and interpolate
                                    time series values that are discarded when time series exceeds these limits.

            HP : use generalised Hodrick-Prescott filtering.

            HW : use Henderson-Whittaker smoothing.

            none : perform no preprocessing.

        preprocess_window_length : integer (odd positive)
            Window length to use when preprocessing signal - should be odd as requires original point at centre.

        preprocess_quantile : float
            Confidence level to use when using 'winsorize' or 'winsorize_interpolate' preprocessing techniques.

        preprocess_penalty : float
            Penalty to use in generalised Hodrick-Prescott filter. Original HP filter - preprocess_penalty = 1.

        preprocess_order : integer
            if preprocess='HP':
                Order of smoothing to be used in generalised Hodrick-Prescott filter. Original HP filter -
                preprocess_order = 2.
            if preprocess='HW':
                Width of Henderson-Whittaker window used to calculate weights for smoothing.

        preprocess_norm_1 : integer
            Norm to be used on curve fit for generalised Hodrick-Prescott filter. Original HP filter -
            preprocess_norm_1 = 2.

        preprocess_norm_2 : integer
            Norm to be used on smoothing order for generalised Hodrick-Prescott filter. Original HP filter -
            preprocess_norm_2 = 2.

        ensemble : boolean
            Whether or not to use Ensemble Empirical Mode Decomposition routine - from:

                Z. Wu and N. Huang. Ensemble Empirical Mode Decomposition: a noise-assisted data
                analysis method. Advances in Adaptive Carbon Data Analysis, 1(1):1–41, 2009.

        ensemble_sd : float
            Fraction of standard deviation of detreneded signal to used when generated noise assisting noise.

        ensemble_iter : integer (positve)
            Number of iterations to use when performing Ensemble Empirical Mode Decomposition.

        enhanced_iter : integer (positve)
            Number of internal iterations to use when performing Enhanced Empirical Mode Decomposition.

        output_coefficients : bool
            Optionally output coefficients corresponding to B-spline IMFs.
            Increases outputs to 4.

        optimise_knots : int
            Optionally optimise knots.

        knot_method : string_like
            Knot point optimisation method to use:
            'bisection' - bisects until error condition is met,
            'ser_bisect' - bisects until error condition met - extends, reevaluates.

            from:

                V. Dung and T. Tjahjowidodo. A direct method to solve optimal knots of B-spline
                curves: An application for non-uniform B-spline curves fitting. PLoS ONE, 12(3):
                e0173857, 2017. doi: https://doi.org/10.1371/journal.pone.0173857.

        output_knots : bool
            Optionally output knots - only relevant when optionally optimised.

        knot_error : float
            Maximum error allowed on increment of knot distance optimisation from:

                V. Dung and T. Tjahjowidodo. A direct method to solve optimal knots of B-spline
                curves: An application for non-uniform B-spline curves fitting. PLoS ONE, 12(3):
                e0173857, 2017. doi: https://doi.org/10.1371/journal.pone.0173857.

        knot_lamda : float
            Smoothing parameter used when fitting cubic B-spline to optimise knot locations.

        knot_epsilon : float
            Minimum distance allowed between potential knot points from:

                V. Dung and T. Tjahjowidodo. A direct method to solve optimal knots of B-spline
                curves: An application for non-uniform B-spline curves fitting. PLoS ONE, 12(3):
                e0173857, 2017. doi: https://doi.org/10.1371/journal.pone.0173857.

        downsample_window : string_like
            Window to use when downsampling.

        downsample_decimation_factor : integer (positive)
            Decimation level when downsampling.
            Product of downsample_decimation_factor and downsample_window_factor must be even.

        downsample_window_factor : integer (positive)
            Downsampling level when downsampling.
            Product of downsample_decimation_factor and downsample_window_factor must be even.

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
        intrinsic_mode_function_storage : real ndarray
            Matrix containing smoothed original (optional) signal in intrinsic_mode_function_storage[0, :]
            followed by IMFs and trend in successive rows.

        intrinsic_mode_function_storage_ht : real ndarray
            Matrix containing HT of smoothed original signal in intrinsic_mode_function_storage_ht[0, :] (not used, but
            included for consistency) followed by HTs of IMFs and trend in successive rows.

        intrinsic_mode_function_storage_if : real ndarray
            Matrix containing IF of smoothed original signal in intrinsic_mode_function_storage_if[0, :] (not used, but
            included for consistency) followed by IFs of IMFs and trend in successive rows.

        intrinsic_mode_function_storage_coef : real ndarray
            Matrix containing B-spline coefficients corresponding to spline curves in intrinsic_mode_function_storage.

        knot_envelope : real ndarray
            Vector containing (possibly optimised) knots.

        intrinsic_mode_function_storage_dt_ht : real ndarray
            Discrete-time Hilbert transform.

        intrinsic_mode_function_storage_dt_if : real ndarray
            Instantaneous frequency using discrete-time Hilbert transform.

        Notes
        -----
        Speed increased (20 times) by:

        -> removing recursive B-spline construction - only applicable to uniform knot sequence.
            -> problem if knot sequence non-uniform
        -> removing recursive Hilbert B-spline construction - only applicable to uniform knot sequence.
            -> problem if knot sequence non-uniform
        -> replace Kak DTHT with FFT DTHT in dealing with boundary Hilbert B-splines.
        -> using pseudo-inverse and numpy.linalg.solve() instead of inverse and numpy.linalg.lstsq().

        To include:

        -> modified SD stopping criterion.
        -> Huang Characteristic Wave in addition to modified Huang characteristic wave.

        """
        ######################
        # Error Checking Top #
        ######################

        try:
            knots = kwargs.get('knots') * 1
        except TypeError:
            knots = np.linspace(0, self.time[-1], int(len(self.time) / 10 + 1))
        try:
            knot_time = kwargs.get('knot_time') * 1
        except TypeError:
            knot_time = self.time

        # could use math.isclose()
        if len(np.intersect1d(np.round(knots, 5), np.round(knot_time, 5))) != len(np.round(knots, 5)):
            warnings.warn('Recommended knots are subset of knot time points for discontinuity issues.')
        elif not (np.intersect1d(np.round(knots, 5), np.round(knot_time, 5)) == np.round(knots, 5)).all:
            warnings.warn('Recommended knots are subset of knot time points for discontinuity issues.')
        if not isinstance(smooth, bool):
            raise TypeError('\'smooth\' must be boolean.')
        if not isinstance(smoothing_penalty, (float, int)):
            raise TypeError('\'smoothing_penalty\' must be float or integer.')
        if edge_effect not in {'neural_network', 'symmetric', 'symmetric_anchor', 'symmetric_discard', 'anti-symmetric',
                               'characteristic_wave_Huang', 'characteristic_wave_Coughlin', 'slope_based_method',
                               'improved_slope_based_method', 'average', 'none'}:
            raise ValueError('\'edge_effect\' not an acceptable value.')
        if not isinstance(sym_alpha, (float, int)):
            raise TypeError('\'sym_alpha\' value must be float or integer.')
        if sym_alpha > 1:
            raise ValueError('\'sym_alpha\' value must not be greater than 1')
        if stop_crit not in {'sd', 'sd_11a', 'sd_11b', 'mft', 'edt', 'S_stoppage'}:
            raise ValueError('\'stop_crit\' not an acceptable value.')
        if not isinstance(stop_crit_threshold, (float, int)):
            raise TypeError('\'stop_crit_threshold\' must be float or integer.')
        if not isinstance(mft_theta_1, (float, int)) or not isinstance(mft_theta_2, (float, int)):
            raise TypeError('\'mft_theta_1\' and \'mft_theta_2\' must be floats of integers.')
        if not mft_theta_1 > 0 or not mft_theta_2 > 0 or mft_theta_1 > mft_theta_2:
            raise ValueError('\'mft_theta_1\' and \'mft_theta_2\' not acceptable values.')
        if not isinstance(mft_alpha, (float, int)):
            raise TypeError('\'mft_alpha\' must be float or integer.')
        if mft_alpha <= 0 or mft_alpha >= 1:
            raise ValueError('\'mft_alpha\' must be a percentage.')
        if not isinstance(mean_threshold, (float, int)):
            raise TypeError('\'mean_threshold\' must be a float or integer.')
        if not mean_threshold > 0:
            raise ValueError('\'mean_threshold\' must be greater than zero.')
        if not isinstance(debug, bool):
            raise TypeError('\'debug\' must be boolean.')
        if not isinstance(verbose, bool):
            raise TypeError('\'verbose\' must be boolean.')
        if spline_method not in {'b_spline', 'chsi', 'asi'}:
            raise ValueError('\'spline_method\' is not an acceptable value.')
        if not isinstance(dtht, bool):
            raise TypeError('\'dtht\' must be boolean.')
        if dtht_method not in {'kak', 'fft'}:
            raise ValueError('\'dtht_method\' is not an acceptable method.')
        if not isinstance(max_internal_iter, int) or max_internal_iter < 1:
            raise ValueError('\'max_internal_iter\' must be a non-negative integer.')
        if not isinstance(matrix, bool):
            raise TypeError('\'matrix\' must be boolean.')
        if not isinstance(initial_smoothing, bool):
            raise TypeError('\'initial_smoothing\' must be boolean.')
        if dft not in {'envelopes', 'inflection_points', 'binomial_average', 'enhanced'}:
            raise ValueError('\'dft\' not an acceptable value.')
        if not isinstance(order, int) or not order % 2 == 1 or order < 1:
            raise ValueError('\'order\' must be a positive, odd, integer.')
        if not isinstance(increment, int) or increment < 1:
            raise ValueError('\'increment\' must be a positive integer.')
        if preprocess not in {'median_filter', 'mean_filter', 'winsorize', 'winsorize_interpolate', 'HP', 'HW',
                              'downsample_decimate', 'downsample', 'none'}:
            raise ValueError('\'preprocess\' technique not an acceptable value.')
        if not isinstance(preprocess_window_length, int) or not preprocess_window_length % 2 == 1 or \
                preprocess_window_length < 1:
            raise ValueError('\'preprocess_window_length\' must be a positive, odd, integer.')
        if not isinstance(preprocess_quantile, float) or preprocess_quantile <= 0 or preprocess_quantile >= 1:
            raise ValueError('\'preprocess_quantile\' value must be a percentage.')
        if not isinstance(preprocess_penalty, (float, int)) or preprocess_penalty < 0:
            raise ValueError('\'preprocess_penalty\' must be a non-negative float or integer.')
        if not isinstance(preprocess_order, int) or preprocess_order < 1:
            raise ValueError('\'preprocess_order\' must be a positive integer.')
        if not isinstance(preprocess_norm_1, int) or preprocess_norm_1 < 1:
            raise ValueError('\'preprocess_norm_1\' must be a positive integer.')
        if not isinstance(preprocess_norm_2, int) or preprocess_norm_2 < 1:
            raise ValueError('\'preprocess_norm_2\' must be a positive integer.')
        if not isinstance(ensemble, bool):
            raise TypeError('\'ensemble\' must be boolean.')
        if not isinstance(ensemble_sd, (float, int)) or not ensemble_sd > 0:
            raise ValueError('\'ensemble_sd\' must be positive float or integer.')
        if not isinstance(ensemble_iter, int) or ensemble_iter < 1:
            raise ValueError('\'ensemble_iter\' must be a positive integer.')
        if not isinstance(output_coefficients, bool):
            raise ValueError('\'output_coefficients\' must be boolean.')
        if optimise_knots not in {0, 1, 2}:
            raise ValueError('\'optimise_knots\' is not an appropriate value.')
        if knot_method not in {'ser_bisect', 'bisection'}:
            raise ValueError('\'knot_method\' technique not an acceptable value.')
        if not isinstance(output_knots, bool):
            raise TypeError('\'output_knots\' must be boolean.')
        try:
            sig.get_window(downsample_window, Nx=256)
        except ValueError:
            raise ValueError('\'downsample_window\' unknown value.')
        if not isinstance(downsample_decimation_factor, int) or downsample_decimation_factor < 1:
            raise ValueError('\'downsample_decimation_factor\' must be a positive integer.')
        if not isinstance(downsample_window_factor, int) or downsample_window_factor < 1:
            raise ValueError('\'downsample_window_factor\' must be a positive integer.')
        if (downsample_decimation_factor * downsample_window_factor) % 2 == 1:
            raise ValueError('Product of \'downsample_decimation_factor\' and'
                             ' \'downsample_window_factor\' must be even.')
        if not isinstance(nn_m, int) or nn_m < 1:
            raise ValueError('\'nn_m\' training outputs must be a positive integer.')
        if not isinstance(nn_k, int) or nn_k < 1:
            raise ValueError('\'nn_k\' training inputs must be a positive integer.')
        if nn_method not in {'grad_descent', 'steep_descent'}:
            raise ValueError('\'nn_method\' technique not an acceptable value.')
        if not isinstance(nn_learning_rate, (float, int)) or not nn_learning_rate > 0:
            raise ValueError('\'nn_learning_rate\' must be a positive float or integer.')
        if not isinstance(nn_iter, int) or nn_iter < 1:
            raise ValueError('\'nn_iter\' must be a positive integer.')

        # joint error messages

        if preprocess == 'HP' and preprocess_order not in {1, 2, 3, 4}:
            raise ValueError('Hodrick-Prescott order must be 1, 2, 3, or 4.')
        if preprocess == 'HW' and preprocess_order % 2 == 0:
            raise ValueError('Henderson-Whittaker order must be odd.')

        #########################
        # Error Checking Bottom #
        #########################

        # initialise
        time = self.time
        time_series = self.time_series

        # extend time and knot time to create relevant Booleans for subset of matrix extraction
        extended_input_signal_time = time_extension(time)
        extended_knot_time = time_extension(knot_time)

        # if optimise_knots == 2 then store each set of knots in dictionary and store set of coefficients in dictionary
        knot_storage = {}
        coef_storage = {}

        # optimised knots
        basis = Basis(time=time, time_series=time_series)
        if optimise_knots == 1 or optimise_knots == 2:
            knots = basis.optimize_knot_points(error=knot_error, lamda=knot_lamda, epsilon=knot_epsilon,
                                               method=knot_method)
            knot_storage[0] = knots
            knot_storage[1] = knots

        # extend knots to create relevant Booleans for subset of matrix extraction
        extended_knot_envelope = time_extension(knots)

        signal_time_bool = np.r_[extended_input_signal_time >= time[0]] & np.r_[
            extended_input_signal_time <= time[-1]]
        knot_time_bool = np.r_[extended_knot_time >= knot_time[0]] & np.r_[
            extended_knot_time <= knot_time[-1]]
        knot_bool = np.r_[extended_knot_envelope >= knots[0]] & np.r_[
            extended_knot_envelope <= knots[-1]]

        # extend knot bool for implicit smoothing - non-natural spline

        # knots_fix - 2 more basis functions than knots
        column_number = len(knots) - 1
        knot_bool[int(column_number - 1)] = True
        knot_bool[(- int(column_number))] = True
        knot_bool = knot_bool[2:-2]

        # create extended B-spline basis matrix for coefficients
        basis_extend_input_time = Basis(time=extended_input_signal_time, time_series=extended_input_signal_time)
        b_spline_matrix_signal = basis_extend_input_time.cubic_b_spline(extended_knot_envelope)
        b_spline_matrix_signal = b_spline_matrix_signal[:, signal_time_bool]
        b_spline_matrix_signal = b_spline_matrix_signal[knot_bool, :]
        basis_extend_knot_time = Basis(time=extended_knot_time, time_series=extended_knot_time)
        b_spline_matrix_extended = basis_extend_knot_time.cubic_b_spline(extended_knot_envelope)
        b_spline_matrix_smooth = b_spline_matrix_extended[:, knot_time_bool]
        b_spline_matrix_smooth = b_spline_matrix_smooth[knot_bool, :]

        number_maxima_previous = {}  # avoids unnecessary error
        number_minima_previous = {}  # avoids unnecessary error
        number_zero_crossings_previous = {}  # avoids unnecessary error
        s_stoppage_count = {}  # avoids unnecessary error

        if not ensemble:

            #####################
            # Preprocessing Top #
            #####################

            preprocess_class = Preprocess(time=time, time_series=time_series)

            if preprocess == 'median_filter':
                time_series = preprocess_class.median_filter(window_width=preprocess_window_length)[1]
            elif preprocess == 'mean_filter':
                time_series = preprocess_class.mean_filter(window_width=preprocess_window_length)[1]
            elif preprocess == 'winsorize':
                time_series = \
                    preprocess_class.winsorize(window_width=preprocess_window_length, a=preprocess_quantile)[1]
            elif preprocess == 'winsorize_interpolate':
                time_series = \
                    preprocess_class.winsorize_interpolate(window_width=preprocess_window_length,
                                                           a=preprocess_quantile)[1]
            elif preprocess == 'HP':
                time_series = preprocess_class.hp(smoothing_penalty=preprocess_penalty,
                                                  order=preprocess_order, norm_1=preprocess_norm_1,
                                                  norm_2=preprocess_norm_2)[1]
            elif preprocess == 'HW':
                time_series = preprocess_class.hw(order=preprocess_order, method='kernel')[1]
            elif preprocess == 'downsample':
                time, time_series = preprocess_class.downsample(window_function=downsample_window,
                                                                decimation_level=downsample_decimation_factor,
                                                                window_factor=downsample_window_factor,
                                                                decimate=False)
                knot_time = time.copy()
            elif preprocess == 'downsample_decimate':
                time, time_series = preprocess_class.downsample(window_function=downsample_window,
                                                                decimation_level=downsample_decimation_factor,
                                                                window_factor=downsample_window_factor)
                knot_time = time.copy()

                # should improve structure of below code - deals with decimated time and time series
                extended_input_signal_time = time_extension(time)
                extended_knot_time = time_extension(knot_time)
                knots = np.linspace(0, self.time[-1], int(len(self.time) / 10 + 1))
                extended_knot_envelope = time_extension(knots)
                signal_time_bool = np.r_[extended_input_signal_time >= time[0]] & np.r_[
                    extended_input_signal_time <= time[-1]]
                knot_time_bool = np.r_[extended_knot_time >= knot_time[0]] & np.r_[
                    extended_knot_time <= knot_time[-1]]
                knot_bool = np.r_[extended_knot_envelope >= knots[0]] & np.r_[
                    extended_knot_envelope <= knots[-1]]
                column_number = len(knots) - 1
                knot_bool[int(column_number - 1)] = True
                knot_bool[(- int(column_number))] = True
                knot_bool = knot_bool[2:-2]
                basis_extend_input_time = Basis(time=extended_input_signal_time, time_series=extended_input_signal_time)
                b_spline_matrix_signal = basis_extend_input_time.cubic_b_spline(extended_knot_envelope)
                b_spline_matrix_signal = b_spline_matrix_signal[:, signal_time_bool]
                b_spline_matrix_signal = b_spline_matrix_signal[knot_bool, :]
                basis_extend_knot_time = Basis(time=extended_knot_time, time_series=extended_knot_time)
                b_spline_matrix_extended = basis_extend_knot_time.cubic_b_spline(extended_knot_envelope)
                b_spline_matrix_smooth = b_spline_matrix_extended[:, knot_time_bool]
                b_spline_matrix_smooth = b_spline_matrix_smooth[knot_bool, :]

            ########################
            # Preprocessing Bottom #
            ########################

            # need to make smoothing optional to replicate Huang paper results
            if initial_smoothing and preprocess =='none':  # smooth initial signal
                time = knot_time
                if matrix:  # use matrix constructed above to improve speed of algorithm - only B-splines
                    least_square_fit, least_square_fit_coef = \
                        basis.basis_function_approximation_matrix(b_spline_matrix_signal, b_spline_matrix_smooth)
                else:  # construct matrix everytime
                    least_square_fit, least_square_fit_coef = \
                        basis.basis_function_approximation(knots, knot_time, spline_method=spline_method)

            # else least_square_fit is simply original signal with coefficients set to zero
            # Hilbert transform of IMF 1 must be calculated using Discrete Time Hilbert Transform
            else:
                least_square_fit, least_square_fit_coef = time_series, np.zeros(len(knot_bool[knot_bool]))

            utility = Utility(time=time, time_series=time_series)
            max_bool = utility.max_bool_func_1st_order_fd()  # finds maximums boolean

            max_y = {}  # avoids unnecessary error
            max_x = {}  # avoids unnecessary error
            if debug:
                max_y = least_square_fit[max_bool]
                max_x = knot_time[max_bool]

            min_bool = utility.min_bool_func_1st_order_fd()  # finds minimums boolean

            min_y = {}  # avoids unnecessary error
            min_x = {}  # avoids unnecessary error
            if debug:
                min_y = least_square_fit[min_bool]
                min_x = knot_time[min_bool]

            ####################################
            # detrended fluctuation techniques #
            ####################################

            #############
            # envelopes #
            #############

            fluctuation = Fluctuation(time=time, time_series=least_square_fit)
            basis = Basis(time=time, time_series=least_square_fit)
            utility = Utility(time=time, time_series=least_square_fit)

            max_envelope_approximation = {}  # avoids unnecessary error
            min_envelope_approximation = {}  # avoids unnecessary error
            optimal_maxima = {}  # avoids unnecessary error
            optimal_minima = {}  # avoids unnecessary error
            mean_envelope_approximation = {}  # avoids unnecessary error
            mean_envelope_approximation_coef = {}  # avoids unnecessary error

            # returns time series without breaking algorithm if unsuitable time series provided
            if sum(utility.max_bool_func_1st_order_fd()) + sum(utility.min_bool_func_1st_order_fd()) < 3:
                return least_square_fit, None, None, None, None, None, None

            if dft == 'envelopes':
                if matrix:
                    # calculates maximum envelope & coefficients
                    max_envelope_approximation, max_envelope_approximation_coef = \
                        fluctuation.envelope_basis_function_approximation_matrix(b_spline_matrix_extended,
                                                                                 knots, 'maxima', smooth,
                                                                                 smoothing_penalty, edge_effect,
                                                                                 alpha=sym_alpha, nn_m=nn_m, nn_k=nn_k,
                                                                                 nn_method=nn_method,
                                                                                 nn_learning_rate=nn_learning_rate,
                                                                                 nn_iter=nn_iter)
                    # calculates minimum envelope & coefficients
                    min_envelope_approximation, min_envelope_approximation_coef = \
                        fluctuation.envelope_basis_function_approximation_matrix(b_spline_matrix_extended,
                                                                                 knots, 'minima', smooth,
                                                                                 smoothing_penalty, edge_effect,
                                                                                 alpha=sym_alpha, nn_m=nn_m, nn_k=nn_k,
                                                                                 nn_method=nn_method,
                                                                                 nn_learning_rate=nn_learning_rate,
                                                                                 nn_iter=nn_iter)
                else:
                    # calculates maximum envelope & coefficients
                    max_envelope_approximation, max_envelope_approximation_coef = \
                        fluctuation.envelope_basis_function_approximation(knots, 'maxima', smooth,
                                                                          smoothing_penalty, edge_effect,
                                                                          spline_method=spline_method,
                                                                          alpha=sym_alpha, nn_m=nn_m, nn_k=nn_k,
                                                                          nn_method=nn_method,
                                                                          nn_learning_rate=nn_learning_rate,
                                                                          nn_iter=nn_iter)

                    # calculates minimum envelope & coefficients
                    min_envelope_approximation, min_envelope_approximation_coef = \
                        fluctuation.envelope_basis_function_approximation(knots, 'minima', smooth,
                                                                          smoothing_penalty, edge_effect,
                                                                          spline_method=spline_method,
                                                                          alpha=sym_alpha, nn_m=nn_m, nn_k=nn_k,
                                                                          nn_method=nn_method,
                                                                          nn_learning_rate=nn_learning_rate,
                                                                          nn_iter=nn_iter)

                mean_envelope_approximation = \
                    (min_envelope_approximation + max_envelope_approximation) / 2  # mean envelope
                mean_envelope_approximation_coef = \
                    (max_envelope_approximation_coef + min_envelope_approximation_coef) / 2
                # mean envelope coefficients

            #####################
            # inflection points #
            #####################

            elif dft == 'inflection_points':

                # construct matrix everytime
                mean_envelope_approximation, mean_envelope_approximation_coef = \
                    fluctuation.direct_detrended_fluctuation_estimation(knots, smooth=smooth,
                                                                        smoothing_penalty=smoothing_penalty,
                                                                        technique='inflection_points')

            ####################
            # binomial average #
            ####################

            elif dft == 'binomial_average':

                # construct matrix everytime
                mean_envelope_approximation, mean_envelope_approximation_coef = \
                    fluctuation.direct_detrended_fluctuation_estimation(knots, smooth=smooth,
                                                                        smoothing_penalty=smoothing_penalty,
                                                                        technique='binomial_average',
                                                                        order=order, increment=increment)

            ############
            # enhanced #
            ############

            elif dft == 'enhanced':

                if np.any(least_square_fit_coef):
                    # i.e. if initial smoothing was done then coefficients would be non-zero

                    derivative_of_lsq = np.matmul(least_square_fit_coef, basis.derivative_cubic_b_spline(knots))
                    derivative_time = knot_time
                    derivative_knots = knots

                else:  # else calculate derivative using finite difference methods

                    derivative_of_lsq = utility.derivative_forward_diff()
                    derivative_time = knot_time[:-1]
                    derivative_knots = np.linspace(knots[0], knots[-1], len(knots))

                emd_derivative = EMD(time=derivative_time, time_series=derivative_of_lsq)

                # change (1) detrended_fluctuation_technique and (2) max_internal_iter and (3) debug
                # (confusing with external debugging)
                imf_1_of_derivative = \
                    emd_derivative.empirical_mode_decomposition(
                        derivative_knots, derivative_time, smooth=smooth, smoothing_penalty=smoothing_penalty,
                        edge_effect=edge_effect, stop_crit=stop_crit,
                        stop_crit_threshold=stop_crit_threshold, mft_theta_1=mft_theta_1,
                        mft_theta_2=mft_theta_2, mft_alpha=mft_alpha, mean_threshold=mean_threshold, debug=False,
                        verbose=False, spline_method=spline_method,
                        dtht=dtht, max_internal_iter=enhanced_iter, max_imfs=max_imfs, matrix=matrix,
                        initial_smoothing=initial_smoothing, dft='envelopes',
                        order=order, increment=increment, preprocess=preprocess,
                        preprocess_window_length=preprocess_window_length,
                        preprocess_quantile=preprocess_quantile)[0][1, :]

                if np.any(least_square_fit_coef):

                    utility_derivative = Utility(time=knot_time, time_series=imf_1_of_derivative)
                    derivatives_bool_max = np.r_[utility_derivative.derivative_forward_diff() < 0, False]
                    derivatives_bool_min = np.r_[utility_derivative.derivative_forward_diff() > 0, False]

                else:

                    utility_derivative = Utility(time=knot_time[:-1], time_series=imf_1_of_derivative)
                    derivatives_bool_max = np.r_[
                        False, utility_derivative.derivative_forward_diff() < 0, False]
                    derivatives_bool_min = np.r_[
                        False, utility_derivative.derivative_forward_diff() > 0, False]

                optimal_maxima = derivatives_bool_max & np.r_[utility_derivative.zero_crossing() == 1]

                optimal_minima = derivatives_bool_min & np.r_[utility_derivative.zero_crossing() == 1]

                # calculates maximum envelope & coefficients
                max_envelope_approximation, max_envelope_approximation_coef = \
                    fluctuation.envelope_basis_function_approximation_fixed_points(
                        knots, 'maxima', optimal_maxima, optimal_minima, smooth, smoothing_penalty,
                        edge_effect='symmetric', alpha=sym_alpha, nn_m=nn_m, nn_k=nn_k, nn_method=nn_method,
                        nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)
                # calculates minimum envelope & coefficients
                min_envelope_approximation, min_envelope_approximation_coef = \
                    fluctuation.envelope_basis_function_approximation_fixed_points(
                        knots, 'minima', optimal_maxima, optimal_minima, smooth, smoothing_penalty,
                        edge_effect='symmetric', alpha=sym_alpha, nn_m=nn_m, nn_k=nn_k, nn_method=nn_method,
                        nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

                mean_envelope_approximation = (min_envelope_approximation + max_envelope_approximation) / 2
                # mean envelope
                mean_envelope_approximation_coef = \
                    (max_envelope_approximation_coef + min_envelope_approximation_coef) / 2
                # mean envelope coefficients

            # plot first iteration of the algorithm
            if debug:
                plt.plot(knot_time, least_square_fit)
                plt.scatter(min_x, min_y)
                plt.scatter(max_x, max_y)
                if dft == 'envelopes':
                    plt.plot(knot_time, max_envelope_approximation)
                    plt.plot(knot_time, min_envelope_approximation)
                elif dft == 'inflection_points':
                    plt.scatter(time[utility.inflection_point()],
                                time_series[utility.inflection_point()])
                elif dft == 'binomial_average':
                    binomial_range = np.arange(len(least_square_fit))
                    binomial_bool = np.mod(binomial_range,
                                           increment) == 0  # boolean where indices are at correct locations
                    utility_bin_av = Utility(time=time[binomial_bool], time_series=least_square_fit[binomial_bool])
                    binomial_average_points = utility_bin_av.binomial_average(order=order)
                    plt.scatter(knot_time[binomial_bool], least_square_fit[binomial_bool])
                    plt.scatter(knot_time[binomial_bool], binomial_average_points)
                elif dft == 'enhanced':
                    plt.scatter(knot_time[optimal_maxima], least_square_fit[optimal_maxima], marker="x", c='r')
                    plt.scatter(knot_time[optimal_minima], least_square_fit[optimal_minima], marker="x", c='b')
                    plt.plot(knot_time, max_envelope_approximation)
                    plt.plot(knot_time, min_envelope_approximation)
                plt.plot(knot_time, mean_envelope_approximation)
                plt.xlabel('t')
                plt.ylabel('g(t)')
                plt.title('Smoothed Signal and 1st Iteration of Sifting')
                plt.show()

            # intrinsic_mode_function_candidate construction
            # coef are not correct if initial smoothing not done - this is dealt with later
            remainder = least_square_fit.copy()  # initialized
            remainder_coef = least_square_fit_coef.copy()  # initialized
            intrinsic_mode_function_candidate = remainder - mean_envelope_approximation  # initialized
            intrinsic_mode_function_candidate_coef = remainder_coef - mean_envelope_approximation_coef  # initialized

            utility_imf = Utility(time=time, time_series=intrinsic_mode_function_candidate)

            intrinsic_mode_function_max_bool = utility_imf.max_bool_func_1st_order_fd()
            # finds IMF maximums boolean
            intrinsic_mode_function_max_storage = intrinsic_mode_function_candidate[intrinsic_mode_function_max_bool]
            # stores maximums

            intrinsic_mode_function_min_bool = utility_imf.min_bool_func_1st_order_fd()
            # finds IMF minimums boolean
            intrinsic_mode_function_min_storage = intrinsic_mode_function_candidate[intrinsic_mode_function_min_bool]
            # stores minimums

            imf_max_time_points = {}  # avoids unnecessary error
            imf_min_time_points = {}  # avoids unnecessary error
            imf_envelope_max = {}  # avoids unnecessary error
            imf_envelope_min = {}  # avoids unnecessary error

            if debug:
                imf_max_time_points = knot_time[intrinsic_mode_function_max_bool]
                imf_min_time_points = knot_time[intrinsic_mode_function_min_bool]

            fluctuation_imf = Fluctuation(time=time, time_series=intrinsic_mode_function_candidate)

            imf_envelope_mean = {}  # avoids unnecessary error
            imf_envelope_mean_coef = {}  # avoids unnecessary error

            if dft == 'envelopes':

                # calculate envelope maxima and minima
                if matrix:
                    imf_envelope_max, imf_envelope_max_coef = \
                        fluctuation_imf.envelope_basis_function_approximation_matrix(b_spline_matrix_extended,
                                                                                     knots, 'maxima', smooth,
                                                                                     smoothing_penalty, edge_effect,
                                                                                     alpha=sym_alpha, nn_m=nn_m,
                                                                                     nn_k=nn_k, nn_method=nn_method,
                                                                                     nn_learning_rate=nn_learning_rate,
                                                                                     nn_iter=nn_iter)

                    imf_envelope_min, imf_envelope_min_coef = \
                        fluctuation_imf.envelope_basis_function_approximation_matrix(b_spline_matrix_extended,
                                                                                     knots, 'minima', smooth,
                                                                                     smoothing_penalty, edge_effect,
                                                                                     alpha=sym_alpha, nn_m=nn_m,
                                                                                     nn_k=nn_k, nn_method=nn_method,
                                                                                     nn_learning_rate=nn_learning_rate,
                                                                                     nn_iter=nn_iter)
                else:
                    imf_envelope_max, imf_envelope_max_coef = \
                        fluctuation_imf.envelope_basis_function_approximation(knots, 'maxima',
                                                                              smooth, smoothing_penalty, edge_effect,
                                                                              spline_method=spline_method,
                                                                              alpha=sym_alpha, nn_m=nn_m,
                                                                              nn_k=nn_k, nn_method=nn_method,
                                                                              nn_learning_rate=nn_learning_rate,
                                                                              nn_iter=nn_iter)

                    imf_envelope_min, imf_envelope_min_coef = \
                        fluctuation_imf.envelope_basis_function_approximation(knots, 'minima',
                                                                              smooth, smoothing_penalty, edge_effect,
                                                                              spline_method=spline_method,
                                                                              alpha=sym_alpha, nn_m=nn_m,
                                                                              nn_k=nn_k, nn_method=nn_method,
                                                                              nn_learning_rate=nn_learning_rate,
                                                                              nn_iter=nn_iter)

                # intialise first mean after intialising first candidate IMF
                imf_envelope_mean = (imf_envelope_max + imf_envelope_min) / 2
                imf_envelope_mean_coef = (imf_envelope_max_coef + imf_envelope_min_coef) / 2

            #####################
            # inflection points #
            #####################

            elif dft == 'inflection_points':

                imf_envelope_mean, imf_envelope_mean_coef = \
                    fluctuation_imf.direct_detrended_fluctuation_estimation(knots, smooth=smooth,
                                                                            smoothing_penalty=smoothing_penalty,
                                                                            technique='inflection_points')

            ####################
            # binomial average #
            ####################

            elif dft == 'binomial_average':

                imf_envelope_mean, imf_envelope_mean_coef = \
                    fluctuation_imf.direct_detrended_fluctuation_estimation(knots, smooth=smooth,
                                                                            smoothing_penalty=smoothing_penalty,
                                                                            technique='binomial_average', order=order,
                                                                            increment=increment)

            elif dft == 'enhanced':

                # calculates maximum envelope & coefficients
                imf_envelope_max, imf_envelope_max_coef = \
                    fluctuation_imf.envelope_basis_function_approximation_fixed_points(knots, 'maxima',
                                                                                       optimal_maxima, optimal_minima,
                                                                                       smooth, smoothing_penalty,
                                                                                       edge_effect='symmetric',
                                                                                       alpha=sym_alpha, nn_m=nn_m,
                                                                                       nn_k=nn_k, nn_method=nn_method,
                                                                                       nn_learning_rate=nn_learning_rate,
                                                                                       nn_iter=nn_iter)
                # calculates minimum envelope & coefficients
                imf_envelope_min, imf_envelope_min_coef = \
                    fluctuation_imf.envelope_basis_function_approximation_fixed_points(knots, 'minima',
                                                                                       optimal_maxima, optimal_minima,
                                                                                       smooth, smoothing_penalty,
                                                                                       edge_effect='symmetric',
                                                                                       alpha=sym_alpha, nn_m=nn_m,
                                                                                       nn_k=nn_k, nn_method=nn_method,
                                                                                       nn_learning_rate=nn_learning_rate,
                                                                                       nn_iter=nn_iter)

                imf_envelope_mean = (imf_envelope_min + imf_envelope_max) / 2
                # mean envelope
                imf_envelope_mean_coef = (imf_envelope_min_coef + imf_envelope_max_coef) / 2
                # mean envelope coefficients

            intrinsic_mode_function_storage = remainder
            # first row of storage matrix will be smoothed original signal
            intrinsic_mode_function_storage_coef = remainder_coef
            if optimise_knots == 2:
                coef_storage[0] = remainder_coef
            # first row of storage matrix will be smoothed coefficients

            imf_count = 0  # initialise IMF count
            internal_iteration_count = 0  # initialise internal IMF sifting
            initialize_stoppage = {}  # avoids unnecessary error
            if stop_crit == 'S_stoppage':
                s_stoppage_count = 0
                initialize_stoppage = 0

            # FAIL-SAFE ADDENDUM - remove imf caused by numerical errors
            num_err = 1e-06

            # while there are still maxima AND minima continue - need both to continue
            # also continues while imf count less than maximum number of imfs allowed
            # also continues while imf candidate amplitude larger than numerical error level
            while any(intrinsic_mode_function_max_storage) and any(intrinsic_mode_function_min_storage) and \
                    imf_count < max_imfs and \
                    np.any((intrinsic_mode_function_candidate -
                            np.mean(intrinsic_mode_function_candidate)) > num_err):

                imf_count += 1

                # optimised knots - needs improvement
                basis = Basis(time=knot_time, time_series=intrinsic_mode_function_candidate)
                if imf_count > 1 and optimise_knots == 2:
                    knots = basis.optimize_knot_points(error=knot_error, lamda=knot_lamda, epsilon=knot_epsilon,
                                                       method=knot_method)
                    if len(knots) < 5:
                        knots = np.linspace(knot_time[0], knot_time[-1], 5)
                    knot_storage[imf_count] = knots
                    # replace all vectors with reduced knot vector equivalent
                    intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef = \
                        basis.basis_function_approximation(knots, knot_time)
                    fluctuation_imf = Fluctuation(time=knot_time, time_series=intrinsic_mode_function_candidate)
                    imf_envelope_max, imf_envelope_max_coef = fluctuation_imf.envelope_basis_function_approximation(
                        knots, 'maxima', smooth, smoothing_penalty,
                        edge_effect, spline_method=spline_method, alpha=sym_alpha, nn_m=nn_m,
                        nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

                    imf_envelope_min, imf_envelope_min_coef = fluctuation_imf.envelope_basis_function_approximation(
                        knots, 'minima', smooth, smoothing_penalty,
                        edge_effect, spline_method=spline_method, alpha=sym_alpha, nn_m=nn_m,
                        nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

                    # calculate mean and coefficients
                    imf_envelope_mean = (imf_envelope_max + imf_envelope_min) / 2
                    imf_envelope_mean_coef = (imf_envelope_max_coef + imf_envelope_min_coef) / 2

                # recalculate optimal interpolation points at appropriate time for Enhanced EMD
                # optimal interpolation points were already calculated for first imf
                if imf_count > 1 and dft == 'enhanced':
                    derivative_of_lsq = np.matmul(intrinsic_mode_function_candidate_coef,
                                                  basis.derivative_cubic_b_spline(knots))
                    derivative_time = knot_time
                    derivative_knots = knots

                    emd_derivative = EMD(time=derivative_time, time_series=derivative_of_lsq)

                    # change (1) detrended_fluctuation_technique and (2) max_internal_iter and (3) debug
                    # (confusing with external debugging)
                    imf_1_of_derivative = \
                        emd_derivative.empirical_mode_decomposition(
                            derivative_knots, derivative_time, smooth=smooth, smoothing_penalty=smoothing_penalty,
                            edge_effect=edge_effect, stop_crit=stop_crit,
                            stop_crit_threshold=stop_crit_threshold, mft_theta_1=mft_theta_1,
                            mft_theta_2=mft_theta_2, mft_alpha=mft_alpha, mean_threshold=mean_threshold,
                            debug=False, verbose=False, spline_method=spline_method, dtht=dtht,
                            max_internal_iter=enhanced_iter, max_imfs=max_imfs,
                            matrix=matrix, initial_smoothing=initial_smoothing,
                            dft='envelopes',
                            order=order, increment=increment, preprocess=preprocess,
                            preprocess_window_length=preprocess_window_length,
                            preprocess_quantile=preprocess_quantile)[0][1, :]

                    utility_derivative = Utility(time=knot_time, time_series=imf_1_of_derivative)
                    optimal_maxima = \
                        np.r_[utility_derivative.derivative_forward_diff() < 0,
                              False] & np.r_[utility_derivative.zero_crossing() == 1]

                    optimal_minima = \
                        np.r_[utility_derivative.derivative_forward_diff() > 0,
                              False] & np.r_[utility_derivative.zero_crossing() == 1]

                    fluctuation_imf = Fluctuation(time=knot_time, time_series=intrinsic_mode_function_candidate)

                    # calculates maximum envelope & coefficients
                    imf_envelope_max, imf_envelope_max_coef = \
                        fluctuation_imf.envelope_basis_function_approximation_fixed_points(knots,
                                                                                           'maxima', optimal_maxima,
                                                                                           optimal_minima,
                                                                                           smooth, smoothing_penalty,
                                                                                           edge_effect='symmetric',
                                                                                           alpha=sym_alpha, nn_m=nn_m,
                                                                                           nn_k=nn_k,
                                                                                           nn_method=nn_method,
                                                                                           nn_learning_rate=nn_learning_rate,
                                                                                           nn_iter=nn_iter)
                    # calculates minimum envelope & coefficients
                    imf_envelope_min, imf_envelope_min_coef = \
                        fluctuation_imf.envelope_basis_function_approximation_fixed_points(knots,
                                                                                           'minima', optimal_maxima,
                                                                                           optimal_minima,
                                                                                           smooth, smoothing_penalty,
                                                                                           edge_effect='symmetric',
                                                                                           alpha=sym_alpha, nn_m=nn_m,
                                                                                           nn_k=nn_k,
                                                                                           nn_method=nn_method,
                                                                                           nn_learning_rate=nn_learning_rate,
                                                                                           nn_iter=nn_iter)

                    imf_envelope_mean = (imf_envelope_min + imf_envelope_max) / 2
                    # mean envelope
                    imf_envelope_mean_coef = (imf_envelope_min_coef + imf_envelope_max_coef) / 2
                    # mean envelope coefficients

                # FAIL-SAFE ADDENDUM - remove imf caused by numerical errors

                # while any maxima are negative or any minima are positive or mean envelope too far from zero
                # - essentially while not imf continue
                # also while internal iteration counter less than maximum internal iteration count
                # also while imf count is less than maximum allowed number of imfs
                # also while imf candidate amplitude larger than numerical error level
                while (any(intrinsic_mode_function_max_storage < 0) or any(intrinsic_mode_function_min_storage > 0)
                       or (np.sum(np.abs(imf_envelope_mean)) > mean_threshold)) \
                        and internal_iteration_count < max_internal_iter and imf_count < (max_imfs + 1) and \
                        np.any((intrinsic_mode_function_candidate -
                                np.mean(intrinsic_mode_function_candidate)) > num_err):

                    count_max = np.count_nonzero(intrinsic_mode_function_max_storage)  # count number of maximums
                    count_min = np.count_nonzero(intrinsic_mode_function_min_storage)  # count number of minimums
                    count_non_zero = count_max + count_min  # count total number of extrema

                    # uses Mean Value Theorem in a sense
                    # deals with both cases - as cant have 2 max without a min and vice versa
                    if count_non_zero > 2:

                        internal_iteration_count += 1

                        if debug:
                            plt.plot(knot_time, intrinsic_mode_function_candidate)
                            plt.scatter(imf_min_time_points, intrinsic_mode_function_min_storage)
                            plt.scatter(imf_max_time_points, intrinsic_mode_function_max_storage)
                            if dft == 'envelopes':
                                plt.plot(knot_time, imf_envelope_max)
                                plt.plot(knot_time, imf_envelope_min)
                            elif dft == 'inflection_points':
                                utility_imf = Utility(time=time, time_series=intrinsic_mode_function_candidate)
                                plt.scatter(knot_time[utility_imf.inflection_point()],
                                            intrinsic_mode_function_candidate[utility_imf.inflection_point()])
                            elif dft == 'binomial_average':
                                binomial_range = np.arange(len(intrinsic_mode_function_candidate))
                                binomial_bool = np.mod(binomial_range, increment) == 0
                                # boolean where indices are at correct locations
                                utility_bin_av = Utility(time=time[binomial_bool],
                                                         time_series=intrinsic_mode_function_candidate[binomial_bool])
                                binomial_average_points = utility_bin_av.binomial_average(order=order)
                                plt.scatter(knot_time[binomial_bool], intrinsic_mode_function_candidate[binomial_bool])
                                plt.scatter(knot_time[binomial_bool], binomial_average_points)
                            elif dft == 'enhanced':
                                plt.scatter(knot_time[optimal_maxima],
                                            intrinsic_mode_function_candidate[optimal_maxima],
                                            marker="x", c='r')
                                plt.scatter(knot_time[optimal_minima],
                                            intrinsic_mode_function_candidate[optimal_minima],
                                            marker="x", c='b')
                                plt.plot(knot_time, imf_envelope_max)
                                plt.plot(knot_time, imf_envelope_min)
                            plt.plot(knot_time, imf_envelope_mean)
                            plt.xlabel('t')
                            plt.ylabel('g(t)')
                            plt.title('IMF ({imf}, {iter})'.format(imf=imf_count, iter=internal_iteration_count))
                            plt.show()

                        if stop_crit == 'sd':  # Standard Deviation

                            imf_k_1 = intrinsic_mode_function_candidate  # IMF candidate m iteration (k-1)
                            imf_k = intrinsic_mode_function_candidate - imf_envelope_mean  # IMF candidate m iteration k

                            sd = sum(((imf_k_1 - imf_k) ** 2) / (imf_k_1 ** 2))  # calculate 'standard deviation'

                            if sd < stop_crit_threshold:

                                intrinsic_mode_function_storage, intrinsic_mode_function_candidate, \
                                    intrinsic_mode_function_storage_coef, intrinsic_mode_function_candidate_coef, \
                                    remainder, remainder_coef, imf_count, internal_iteration_count,\
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef, coef_storage = \
                                    stopping_criterion_fail_helper(
                                        intrinsic_mode_function_storage, intrinsic_mode_function_candidate,
                                        intrinsic_mode_function_storage_coef, intrinsic_mode_function_candidate_coef,
                                        remainder, remainder_coef, imf_count, internal_iteration_count, knot_time,
                                        debug, dft, matrix, b_spline_matrix_extended,
                                        knots, smooth, smoothing_penalty, edge_effect, spline_method,
                                        sym_alpha, order, increment, optimal_maxima, optimal_minima, verbose,
                                        stop_crit_threshold, stop_crit, coef_storage, optimise_knots, knots,
                                        calculated_threshold=sd,
                                        mean_fluctuation_theta_1=mft_theta_1, mft_alpha=mft_alpha, nn_m=nn_m, nn_k=nn_k,
                                        nn_method=nn_method, nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

                            # if not - removes mean and calculates all necessary values for next iteration of loop
                            else:

                                intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef = \
                                    stopping_criterion_pass_helper(
                                        intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef,
                                        imf_envelope_mean, imf_envelope_mean_coef,
                                        imf_count, internal_iteration_count, knot_time, debug,
                                        dft, matrix, b_spline_matrix_extended,
                                        knots, smooth, smoothing_penalty, edge_effect, spline_method, sym_alpha,
                                        order, increment, optimal_maxima, optimal_minima, verbose,
                                        stop_crit_threshold, stop_crit, calculated_threshold=sd,
                                        mean_fluctuation_theta_1=mft_theta_1, mft_alpha=mft_alpha,
                                        mean_fluctuation_theta_2=mft_theta_2, s_stoppage_count=s_stoppage_count,
                                        nn_m=nn_m, nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                        nn_iter=nn_iter)

                        if stop_crit == 'sd_11a':  # Standard Deviation from
                            # A review on Hilbert-Huang Transform: Method and its applications to geophysical studies

                            imf_k_1 = intrinsic_mode_function_candidate  # IMF candidate m iteration (k-1)
                            imf_k = intrinsic_mode_function_candidate - imf_envelope_mean  # IMF candidate m iteration k

                            sd = np.sum((imf_k_1 - imf_k) ** 2) / np.sum(imf_k_1 ** 2)  # calculate 'standard deviation'

                            if sd < stop_crit_threshold:

                                intrinsic_mode_function_storage, intrinsic_mode_function_candidate, \
                                    intrinsic_mode_function_storage_coef, intrinsic_mode_function_candidate_coef, \
                                    remainder, remainder_coef, imf_count, internal_iteration_count, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef, coef_storage = \
                                    stopping_criterion_fail_helper(intrinsic_mode_function_storage,
                                                                   intrinsic_mode_function_candidate,
                                                                   intrinsic_mode_function_storage_coef,
                                                                   intrinsic_mode_function_candidate_coef,
                                                                   remainder, remainder_coef, imf_count,
                                                                   internal_iteration_count,
                                                                   knot_time, debug, dft,
                                                                   matrix, b_spline_matrix_extended,
                                                                   knots, smooth, smoothing_penalty,
                                                                   edge_effect, spline_method, sym_alpha, order,
                                                                   increment, optimal_maxima, optimal_minima, verbose,
                                                                   stop_crit_threshold, stop_crit, coef_storage,
                                                                   optimise_knots, knots,
                                                                   calculated_threshold=sd,
                                                                   mean_fluctuation_theta_1=mft_theta_1,
                                                                   mft_alpha=mft_alpha, nn_m=nn_m, nn_k=nn_k,
                                                                   nn_method=nn_method,
                                                                   nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

                            # if not - removes mean and calculates all necessary values for next iteration of loop
                            else:

                                intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef = \
                                    stopping_criterion_pass_helper(
                                        intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef,
                                        imf_envelope_mean, imf_envelope_mean_coef,
                                        imf_count, internal_iteration_count, knot_time, debug,
                                        dft, matrix, b_spline_matrix_extended,
                                        knots, smooth, smoothing_penalty, edge_effect, spline_method, sym_alpha,
                                        order, increment, optimal_maxima, optimal_minima, verbose,
                                        stop_crit_threshold, stop_crit, calculated_threshold=sd,
                                        mean_fluctuation_theta_1=mft_theta_1, mft_alpha=mft_alpha,
                                        mean_fluctuation_theta_2=mft_theta_2, s_stoppage_count=s_stoppage_count,
                                        nn_m=nn_m, nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                        nn_iter=nn_iter)

                        if stop_crit == 'sd_11b':  # Standard Deviation from
                            # A review on Hilbert-Huang Transform: Method and its applications to geophysical studies

                            sd = np.sum(imf_envelope_mean ** 2) / np.sum(intrinsic_mode_function_candidate ** 2)
                            # calculate 'standard deviation'

                            if sd < stop_crit_threshold:

                                intrinsic_mode_function_storage, intrinsic_mode_function_candidate, \
                                    intrinsic_mode_function_storage_coef, intrinsic_mode_function_candidate_coef, \
                                    remainder, remainder_coef, imf_count, internal_iteration_count, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef, coef_storage = \
                                    stopping_criterion_fail_helper(intrinsic_mode_function_storage,
                                                                   intrinsic_mode_function_candidate,
                                                                   intrinsic_mode_function_storage_coef,
                                                                   intrinsic_mode_function_candidate_coef,
                                                                   remainder, remainder_coef, imf_count,
                                                                   internal_iteration_count,
                                                                   knot_time, debug, dft,
                                                                   matrix, b_spline_matrix_extended,
                                                                   knots, smooth, smoothing_penalty,
                                                                   edge_effect, spline_method, sym_alpha, order,
                                                                   increment, optimal_maxima, optimal_minima, verbose,
                                                                   stop_crit_threshold, stop_crit, coef_storage,
                                                                   optimise_knots, knots,
                                                                   calculated_threshold=sd,
                                                                   mean_fluctuation_theta_1=mft_theta_1,
                                                                   mft_alpha=mft_alpha, nn_m=nn_m, nn_k=nn_k,
                                                                   nn_method=nn_method,
                                                                   nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

                            # if not - removes mean and calculates all necessary values for next iteration of loop
                            else:

                                intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef = \
                                    stopping_criterion_pass_helper(
                                        intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef,
                                        imf_envelope_mean, imf_envelope_mean_coef,
                                        imf_count, internal_iteration_count, knot_time, debug,
                                        dft, matrix, b_spline_matrix_extended,
                                        knots, smooth, smoothing_penalty, edge_effect, spline_method, sym_alpha,
                                        order, increment, optimal_maxima, optimal_minima, verbose,
                                        stop_crit_threshold, stop_crit, calculated_threshold=sd,
                                        mean_fluctuation_theta_1=mft_theta_1, mft_alpha=mft_alpha,
                                        mean_fluctuation_theta_2=mft_theta_2, s_stoppage_count=s_stoppage_count,
                                        nn_m=nn_m, nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                        nn_iter=nn_iter)

                        elif stop_crit == 'mft':  # Mean Fluctuation Threshold

                            sd = 0  # avoids unnecessary errors

                            # calculate 'mode amplitude' for Mean Fluctuation Threshold stopping criteria
                            mode_amplitude = (imf_envelope_max - imf_envelope_min) / 2
                            # calculates Mean Fluctuation Threshold
                            mean_fluctuation_function = np.abs(imf_envelope_mean / mode_amplitude)

                            # calculates percentage of Mean Fluctuation above theta_1 threshold
                            mean_fluctuation_theta_1 = sum(mean_fluctuation_function < mft_theta_1) / len(
                                mean_fluctuation_function)
                            # calculates percentage of Mean Fluctuation above theta_2 threshold
                            mean_fluctuation_theta_2 = sum(mean_fluctuation_function < mft_theta_2) / len(
                                mean_fluctuation_function)

                            # if Mean Fluctuation criteria are met
                            # i.e. if mean_fluctuation_function is less than theta_1 for (1 - alpha)% of entire
                            # range and mean_fluctuation_function is less than theta_2 for entire range then:
                            if mean_fluctuation_theta_1 > (1 - mft_alpha) and mean_fluctuation_theta_2 == 1:

                                intrinsic_mode_function_storage, intrinsic_mode_function_candidate, \
                                    intrinsic_mode_function_storage_coef, intrinsic_mode_function_candidate_coef, \
                                    remainder, remainder_coef, imf_count, internal_iteration_count, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef, coef_storage = \
                                    stopping_criterion_fail_helper(intrinsic_mode_function_storage,
                                                                   intrinsic_mode_function_candidate,
                                                                   intrinsic_mode_function_storage_coef,
                                                                   intrinsic_mode_function_candidate_coef,
                                                                   remainder, remainder_coef, imf_count,
                                                                   internal_iteration_count,
                                                                   knot_time, debug, dft,
                                                                   matrix, b_spline_matrix_extended,
                                                                   knots, smooth, smoothing_penalty,
                                                                   edge_effect, spline_method, sym_alpha, order,
                                                                   increment, optimal_maxima, optimal_minima, verbose,
                                                                   stop_crit_threshold, stop_crit, coef_storage,
                                                                   optimise_knots, knots,
                                                                   calculated_threshold=sd,
                                                                   mean_fluctuation_theta_1=mean_fluctuation_theta_1,
                                                                   mft_alpha=mft_alpha, nn_m=nn_m, nn_k=nn_k,
                                                                   nn_method=nn_method,
                                                                   nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

                            else:

                                intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef = \
                                    stopping_criterion_pass_helper(
                                        intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef,
                                        imf_envelope_mean, imf_envelope_mean_coef,
                                        imf_count, internal_iteration_count, knot_time, debug,
                                        dft, matrix, b_spline_matrix_extended,
                                        knots, smooth, smoothing_penalty, edge_effect, spline_method, sym_alpha,
                                        order, increment, optimal_maxima, optimal_minima, verbose,
                                        stop_crit_threshold, stop_crit, calculated_threshold=sd,
                                        mean_fluctuation_theta_1=mean_fluctuation_theta_1, mft_alpha=mft_alpha,
                                        mean_fluctuation_theta_2=mean_fluctuation_theta_2,
                                        s_stoppage_count=s_stoppage_count, nn_m=nn_m, nn_k=nn_k, nn_method=nn_method,
                                        nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

                        elif stop_crit == 'edt':  # Energy Difference Tracking

                            # if remainder and intrinsic mode function candidate are the same
                            # step required otherwise criterion would be met for every IMF after the first one.
                            if not any(remainder - intrinsic_mode_function_candidate):

                                # subtracts current mean and then recalculates all relevant values
                                intrinsic_mode_function_candidate -= imf_envelope_mean

                                fluctuation_imf = Fluctuation(time=time,
                                                              time_series=intrinsic_mode_function_candidate)

                                if dft == 'envelopes':
                                    # calculate envelopes
                                    if matrix:
                                        imf_envelope_max, imf_envelope_max_coef = \
                                            fluctuation_imf.envelope_basis_function_approximation_matrix(
                                                b_spline_matrix_extended, knots, 'maxima', smooth,
                                                smoothing_penalty, edge_effect, alpha=sym_alpha, nn_m=nn_m,
                                                nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                                nn_iter=nn_iter)

                                        imf_envelope_min, imf_envelope_min_coef = \
                                            fluctuation_imf.envelope_basis_function_approximation_matrix(
                                                b_spline_matrix_extended, knots, 'minima', smooth,
                                                smoothing_penalty, edge_effect, alpha=sym_alpha, nn_m=nn_m,
                                                nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                                nn_iter=nn_iter)
                                    else:
                                        imf_envelope_max, imf_envelope_max_coef = \
                                            fluctuation_imf.envelope_basis_function_approximation(
                                                knots, 'maxima', smooth, smoothing_penalty, edge_effect,
                                                spline_method=spline_method, alpha=sym_alpha, nn_m=nn_m,
                                                nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                                nn_iter=nn_iter)

                                        imf_envelope_min, imf_envelope_min_coef = \
                                            fluctuation_imf.envelope_basis_function_approximation(
                                                knots, 'minima', smooth, smoothing_penalty, edge_effect,
                                                spline_method=spline_method, alpha=sym_alpha, nn_m=nn_m,
                                                nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                                nn_iter=nn_iter)

                                    # calculate means
                                    imf_envelope_mean = (imf_envelope_max + imf_envelope_min) / 2
                                    imf_envelope_mean_coef = (imf_envelope_max_coef + imf_envelope_min_coef) / 2

                                elif dft == 'inflection_points':

                                    imf_envelope_mean, imf_envelope_mean_coef = \
                                        fluctuation_imf.direct_detrended_fluctuation_estimation(
                                            knots, smooth=smooth, smoothing_penalty=smoothing_penalty,
                                            technique='inflection_points')

                                elif dft == 'binomial_average':

                                    imf_envelope_mean, imf_envelope_mean_coef = \
                                        fluctuation_imf.direct_detrended_fluctuation_estimation(
                                            knots, smooth=smooth, smoothing_penalty=smoothing_penalty,
                                            technique='binomial_average', order=order, increment=increment)

                                elif dft == 'enhanced':

                                    # calculates maximum envelope & coefficients
                                    imf_envelope_max, imf_envelope_max_coef = \
                                        fluctuation_imf.envelope_basis_function_approximation_fixed_points(
                                            knots, 'maxima', optimal_maxima, optimal_minima, smooth,
                                            smoothing_penalty, edge_effect='symmetric', alpha=sym_alpha, nn_m=nn_m,
                                            nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                            nn_iter=nn_iter)
                                    # calculates minimum envelope & coefficients
                                    imf_envelope_min, imf_envelope_min_coef = \
                                        fluctuation_imf.envelope_basis_function_approximation_fixed_points(
                                            knots, 'minima', optimal_maxima, optimal_minima, smooth,
                                            smoothing_penalty, edge_effect='symmetric', alpha=sym_alpha, nn_m=nn_m,
                                            nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                            nn_iter=nn_iter)

                                    imf_envelope_mean = (imf_envelope_min + imf_envelope_max) / 2
                                    # mean envelope
                                    imf_envelope_mean_coef = (imf_envelope_min_coef + imf_envelope_max_coef) / 2
                                    # mean envelope coefficients

                            # calculate imf candidate energy
                            utility_imf = Utility(time=knot_time, time_series=intrinsic_mode_function_candidate)
                            imf_candidate_energy = utility_imf.energy()
                            # calculate remainder minus candidate energy
                            utility_remainder = Utility(time=knot_time,
                                                        time_series=remainder - intrinsic_mode_function_candidate)
                            remainder_energy = utility_remainder.energy()
                            # calculate remainder energy
                            utility_prev_remainder = Utility(time=knot_time, time_series=remainder)
                            previous_remainder_energy = utility_prev_remainder.energy()
                            # calculate absolute energy difference
                            energy_difference = np.abs(
                                previous_remainder_energy - (imf_candidate_energy + remainder_energy))

                            # if energy difference tracking is met
                            if energy_difference < stop_crit_threshold:

                                intrinsic_mode_function_storage, intrinsic_mode_function_candidate, \
                                    intrinsic_mode_function_storage_coef, intrinsic_mode_function_candidate_coef, \
                                    remainder, remainder_coef, imf_count, internal_iteration_count, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef, coef_storage = \
                                    stopping_criterion_fail_helper(intrinsic_mode_function_storage,
                                                                   intrinsic_mode_function_candidate,
                                                                   intrinsic_mode_function_storage_coef,
                                                                   intrinsic_mode_function_candidate_coef,
                                                                   remainder, remainder_coef, imf_count,
                                                                   internal_iteration_count,
                                                                   knot_time, debug, dft,
                                                                   matrix, b_spline_matrix_extended,
                                                                   knots, smooth, smoothing_penalty,
                                                                   edge_effect, spline_method, sym_alpha, order,
                                                                   increment, optimal_maxima, optimal_minima, verbose,
                                                                   stop_crit_threshold, stop_crit, coef_storage,
                                                                   optimise_knots, knots,
                                                                   calculated_threshold=energy_difference,
                                                                   mean_fluctuation_theta_1=mft_theta_1,
                                                                   mft_alpha=mft_alpha, nn_m=nn_m, nn_k=nn_k,
                                                                   nn_method=nn_method,
                                                                   nn_learning_rate=nn_learning_rate,
                                                                   nn_iter=nn_iter)

                            # if not - remove mean and calculate all values to iterate over again
                            else:

                                intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef = \
                                    stopping_criterion_pass_helper(
                                        intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef,
                                        imf_envelope_mean, imf_envelope_mean_coef,
                                        imf_count, internal_iteration_count, knot_time, debug,
                                        dft, matrix, b_spline_matrix_extended,
                                        knots, smooth, smoothing_penalty, edge_effect, spline_method, sym_alpha,
                                        order, increment, optimal_maxima, optimal_minima, verbose,
                                        stop_crit_threshold, stop_crit,
                                        calculated_threshold=energy_difference,
                                        mean_fluctuation_theta_1=mft_theta_1, mft_alpha=mft_alpha,
                                        mean_fluctuation_theta_2=mft_theta_2, s_stoppage_count=s_stoppage_count,
                                        nn_m=nn_m, nn_k=nn_k, nn_method=nn_method,
                                        nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

                        elif stop_crit == 'S_stoppage':

                            sd = 0  # avoids unnecessary error

                            if initialize_stoppage == 0:
                                number_maxima_previous = 0
                                number_minima_previous = 0
                                number_zero_crossings_previous = 0
                                initialize_stoppage = 1

                            utility_imf = Utility(time=time, time_series=intrinsic_mode_function_candidate)

                            number_maxima_next = sum(np.asarray(utility_imf.max_bool_func_1st_order_fd()))
                            number_minima_next = sum(np.asarray(utility_imf.min_bool_func_1st_order_fd()))
                            number_zero_crossings_next = sum(np.asarray(utility_imf.zero_crossing()))

                            if (number_maxima_previous == number_maxima_next) and \
                                    (number_minima_previous == number_minima_next) and \
                                    (number_zero_crossings_previous == number_zero_crossings_next):
                                s_stoppage_count += 1

                            # reset
                            else:
                                s_stoppage_count = 0

                            if s_stoppage_count == stop_crit_threshold:

                                s_stoppage_count = 0

                                intrinsic_mode_function_storage, intrinsic_mode_function_candidate, \
                                    intrinsic_mode_function_storage_coef, intrinsic_mode_function_candidate_coef, \
                                    remainder, remainder_coef, imf_count, internal_iteration_count, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef, coef_storage = \
                                    stopping_criterion_fail_helper(intrinsic_mode_function_storage,
                                                                   intrinsic_mode_function_candidate,
                                                                   intrinsic_mode_function_storage_coef,
                                                                   intrinsic_mode_function_candidate_coef,
                                                                   remainder, remainder_coef, imf_count,
                                                                   internal_iteration_count,
                                                                   knot_time, debug, dft,
                                                                   matrix, b_spline_matrix_extended,
                                                                   knots, smooth, smoothing_penalty,
                                                                   edge_effect, spline_method, sym_alpha, order,
                                                                   increment, optimal_maxima, optimal_minima, verbose,
                                                                   stop_crit_threshold, stop_crit, coef_storage,
                                                                   optimise_knots, knots,
                                                                   calculated_threshold=stop_crit_threshold,
                                                                   mean_fluctuation_theta_1=mft_theta_1,
                                                                   mft_alpha=mft_alpha, nn_m=nn_m, nn_k=nn_k,
                                                                   nn_method=nn_method,
                                                                   nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)

                            # if not - removes mean and calculates all necessary values for next iteration of loop
                            else:

                                utility_imf = Utility(time=time, time_series=intrinsic_mode_function_candidate)

                                number_maxima_previous = sum(np.asarray(utility_imf.max_bool_func_1st_order_fd()))
                                number_minima_previous = sum(np.asarray(utility_imf.min_bool_func_1st_order_fd()))
                                number_zero_crossings_previous = sum(np.asarray(utility_imf.zero_crossing()))

                                intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef, \
                                    imf_envelope_mean, imf_envelope_mean_coef, imf_max_time_points, \
                                    imf_min_time_points, intrinsic_mode_function_max_storage, \
                                    intrinsic_mode_function_min_storage, imf_envelope_max, imf_envelope_max_coef, \
                                    imf_envelope_min, imf_envelope_min_coef = \
                                    stopping_criterion_pass_helper(
                                        intrinsic_mode_function_candidate, intrinsic_mode_function_candidate_coef,
                                        imf_envelope_mean, imf_envelope_mean_coef,
                                        imf_count, internal_iteration_count, knot_time, debug,
                                        dft, matrix, b_spline_matrix_extended,
                                        knots, smooth, smoothing_penalty, edge_effect, spline_method, sym_alpha,
                                        order, increment, optimal_maxima, optimal_minima, verbose,
                                        stop_crit_threshold, stop_crit, calculated_threshold=sd,
                                        mean_fluctuation_theta_1=mft_theta_1, mft_alpha=mft_alpha,
                                        mean_fluctuation_theta_2=mft_theta_2, s_stoppage_count=s_stoppage_count,
                                        nn_m=nn_m, nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                        nn_iter=nn_iter)

                    # if number of extrema (maxima and minima) is less than or equal to 3 then - continue
                    else:

                        internal_iteration_count += 1

                        # creates global mean vector
                        imf_envelope_mean = \
                            np.mean(intrinsic_mode_function_candidate) * np.ones_like(intrinsic_mode_function_candidate)
                        # creates global mean vector coefficients
                        imf_envelope_mean_coef = \
                            np.mean(intrinsic_mode_function_candidate) * np.ones_like(
                                intrinsic_mode_function_candidate_coef)

                        # removes global mean from imf candidate
                        intrinsic_mode_function_candidate = intrinsic_mode_function_candidate - imf_envelope_mean
                        # removes global mean from imf candidate coefficients
                        intrinsic_mode_function_candidate_coef = \
                            intrinsic_mode_function_candidate_coef - imf_envelope_mean_coef

                        # ensures loop is broken when extrema storage is checked if non-empty
                        intrinsic_mode_function_max_storage = np.array([])
                        intrinsic_mode_function_min_storage = np.array([])

                        # mean set to zero
                        imf_envelope_mean = np.zeros_like(intrinsic_mode_function_candidate)

                        if debug:
                            plt.plot(knot_time, intrinsic_mode_function_candidate)
                            plt.xlabel('t')
                            plt.ylabel('g(t)')
                            plt.title('remainder = IMF ({},{})'.format(imf_count, internal_iteration_count))
                            plt.show()

                        # print report
                        if verbose:
                            print(
                                f'IMF_{imf_count}{internal_iteration_count} TREND CONDITION MET with '
                                f'fewer than 3 extrema and global mean removed')

                # Code after escaping inner while loop

                # print report when maximum internal iteration count is reached
                if internal_iteration_count == max_internal_iter and verbose:
                    print(f'IMF_{imf_count}{internal_iteration_count} Internal iteration limit MET with '
                          f'limit = {internal_iteration_count}')

                # FAIL-SAFE ADDENDUM - remove imf caused by numerical errors

                # print 'ALL IMF CONDITIONS MET' if that is why inner while loop ended

                # essentially if imf and non-zero
                # only debugging step as imf candidate would have been stored within while loop if stopping criterion
                # met if maximum iteration count is met then dealt with in next if statement
                if (all(intrinsic_mode_function_max_storage > 0) and all(intrinsic_mode_function_min_storage < 0)) \
                        and (np.sum(np.abs(imf_envelope_mean)) < mean_threshold) and \
                        np.any((intrinsic_mode_function_candidate -
                                np.mean(intrinsic_mode_function_candidate)) > num_err):

                    # internal iteration count
                    internal_iteration_count += 1

                    # print report
                    if verbose:
                        print(f'IMF_{imf_count}{internal_iteration_count} ALL IMF CONDITIONS MET')

                    if debug:
                        plt.plot(knot_time, intrinsic_mode_function_candidate)
                        plt.scatter(imf_min_time_points, intrinsic_mode_function_min_storage)
                        plt.scatter(imf_max_time_points, intrinsic_mode_function_max_storage)
                        utility_imf = Utility(time=time, time_series=intrinsic_mode_function_candidate)
                        if dft == 'inflection_points':
                            plt.scatter(knot_time[utility_imf.inflection_point()],
                                        intrinsic_mode_function_candidate[utility_imf.inflection_point()])
                        elif dft == 'binomial_average':
                            binomial_range = np.arange(len(intrinsic_mode_function_candidate))
                            binomial_bool = np.mod(binomial_range,
                                                   increment) == 0  # boolean where indices are at correct locations
                            utility_bin_av = Utility(time=time[binomial_bool],
                                                     time_series=intrinsic_mode_function_candidate[binomial_bool])
                            binomial_average_points = utility_bin_av.binomial_average(order=order)
                            plt.scatter(knot_time[binomial_bool], intrinsic_mode_function_candidate[binomial_bool])
                            plt.scatter(knot_time[binomial_bool], binomial_average_points)
                        elif dft == 'enhanced':
                            plt.plot(knot_time[optimal_maxima], intrinsic_mode_function_candidate[optimal_maxima],
                                     marker="x", c='r')
                            plt.plot(knot_time[optimal_minima], intrinsic_mode_function_candidate[optimal_minima],
                                     marker="x", c='b')
                        plt.plot(knot_time, imf_envelope_mean)
                        plt.xlabel('t')
                        plt.ylabel('g(t)')
                        plt.title('IMF ({},{}) satisfies extrema and mean conditions'.format(imf_count,
                                                                                             internal_iteration_count))
                        plt.show()

                # set internal iteration to zero
                internal_iteration_count = 0

                # FAIL-SAFE ADDENDUM - remove imf caused by numerical errors

                # store imf if 'intrinsic_mode_function_candidate' is non-zero
                # - escaped inner while loop without storing imf if it satisfied all conditions,
                # but did not fail stopping criterion

                # if non-zero remove imf candidate when max internal iteration count has been met
                if not all(intrinsic_mode_function_candidate == np.zeros_like(intrinsic_mode_function_candidate)) and \
                        np.any((intrinsic_mode_function_candidate -
                                np.mean(intrinsic_mode_function_candidate)) > num_err):

                    intrinsic_mode_function_storage = np.vstack((intrinsic_mode_function_storage,
                                                                 intrinsic_mode_function_candidate))
                    # stores imf in storage matrix
                    if optimise_knots != 2:
                        intrinsic_mode_function_storage_coef = np.vstack((intrinsic_mode_function_storage_coef,
                                                                          intrinsic_mode_function_candidate_coef))
                    else:
                        coef_storage[imf_count] = intrinsic_mode_function_candidate_coef
                    # stores imf coefficient in storage matrix

                    # fixes coefficient length discrepancy when optimise_knots=2
                    remainder = remainder - intrinsic_mode_function_candidate
                    # removes imf from what remains of smoothed signal
                    if len(remainder_coef) != len(intrinsic_mode_function_candidate_coef):
                        basis_coef = Basis(time=knot_time, time_series=remainder)
                        remainder, remainder_coef = basis_coef.basis_function_approximation(knots=knots,
                                                                                            knot_time=knot_time)
                    else:
                        remainder_coef = remainder_coef - intrinsic_mode_function_candidate_coef

                    intrinsic_mode_function_candidate = remainder.copy()  # new imf candidate for next loop iteration
                    intrinsic_mode_function_candidate_coef = remainder_coef.copy()

                    utility_imf = Utility(time=time, time_series=intrinsic_mode_function_candidate)

                    intrinsic_mode_function_max_bool = utility_imf.max_bool_func_1st_order_fd()
                    # finds imf maxima boolean
                    intrinsic_mode_function_max_storage = intrinsic_mode_function_candidate[
                        intrinsic_mode_function_max_bool]
                    # extracts imf maximums

                    intrinsic_mode_function_min_bool = utility_imf.min_bool_func_1st_order_fd()
                    # finds imf minima boolean
                    intrinsic_mode_function_min_storage = intrinsic_mode_function_candidate[
                        intrinsic_mode_function_min_bool]
                    # extracts imf minima

                    if debug:
                        imf_max_time_points = knot_time[intrinsic_mode_function_max_bool]
                        imf_min_time_points = knot_time[intrinsic_mode_function_min_bool]

                    # FAIL-SAFE ADDENDUM - remove imf caused by numerical errors

                    # if both maxima and minima are non-empty calculate extrema envelopes
                    # for next iteration of loop
                    if any(intrinsic_mode_function_max_storage) and any(intrinsic_mode_function_min_storage) and \
                            np.any((intrinsic_mode_function_candidate -
                                    np.mean(intrinsic_mode_function_candidate)) > num_err):
                        # attempt to fix extrema propagation through small no zero errors

                        fluctuation_imf = Fluctuation(time=time,
                                                      time_series=intrinsic_mode_function_candidate)

                        if dft == 'envelopes':

                            if matrix:
                                imf_envelope_max, imf_envelope_max_coef = \
                                    fluctuation_imf.envelope_basis_function_approximation_matrix(
                                        b_spline_matrix_extended, knots, 'maxima', smooth,
                                        smoothing_penalty, edge_effect, alpha=sym_alpha, nn_m=nn_m,
                                        nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                        nn_iter=nn_iter)

                                imf_envelope_min, imf_envelope_min_coef = \
                                    fluctuation_imf.envelope_basis_function_approximation_matrix(
                                        b_spline_matrix_extended, knots, 'minima', smooth,
                                        smoothing_penalty, edge_effect, alpha=sym_alpha, nn_m=nn_m,
                                        nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                        nn_iter=nn_iter)
                            else:
                                imf_envelope_max, imf_envelope_max_coef = \
                                    fluctuation_imf.envelope_basis_function_approximation(
                                        knots, 'maxima', smooth, smoothing_penalty, edge_effect,
                                        spline_method=spline_method, alpha=sym_alpha, nn_m=nn_m,
                                        nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                        nn_iter=nn_iter)

                                imf_envelope_min, imf_envelope_min_coef = \
                                    fluctuation_imf.envelope_basis_function_approximation(
                                        knots, 'minima', smooth, smoothing_penalty, edge_effect,
                                        spline_method=spline_method, alpha=sym_alpha, nn_m=nn_m,
                                        nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                        nn_iter=nn_iter)

                            # calculate mean
                            imf_envelope_mean = (imf_envelope_max + imf_envelope_min) / 2
                            imf_envelope_mean_coef = (imf_envelope_max_coef + imf_envelope_min_coef) / 2

                        elif dft == 'inflection_points':

                            imf_envelope_mean, imf_envelope_mean_coef = \
                                fluctuation_imf.direct_detrended_fluctuation_estimation(
                                    knots, smooth=smooth, smoothing_penalty=smoothing_penalty,
                                    technique='inflection_points')

                        elif dft == 'binomial_average':

                            imf_envelope_mean, imf_envelope_mean_coef = \
                                fluctuation_imf.direct_detrended_fluctuation_estimation(
                                    knots, smooth=smooth, smoothing_penalty=smoothing_penalty,
                                    technique='binomial_average', order=order, increment=increment)

                        elif dft == 'enhanced':

                            # calculates maximum envelope & coefficients
                            imf_envelope_max, imf_envelope_max_coef = \
                                fluctuation_imf.envelope_basis_function_approximation_fixed_points(
                                    knots, 'maxima', optimal_maxima, optimal_minima, smooth,
                                    smoothing_penalty, edge_effect='symmetric', alpha=sym_alpha, nn_m=nn_m,
                                    nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                    nn_iter=nn_iter)
                            # calculates minimum envelope & coefficients
                            imf_envelope_min, imf_envelope_min_coef = \
                                fluctuation_imf.envelope_basis_function_approximation_fixed_points(
                                    knots, 'minima', optimal_maxima, optimal_minima, smooth,
                                    smoothing_penalty, edge_effect='symmetric', alpha=sym_alpha, nn_m=nn_m,
                                    nn_k=nn_k, nn_method=nn_method, nn_learning_rate=nn_learning_rate,
                                    nn_iter=nn_iter)

                            imf_envelope_mean = (imf_envelope_min + imf_envelope_max) / 2
                            # mean envelope
                            imf_envelope_mean_coef = (imf_envelope_min_coef + imf_envelope_max_coef) / 2
                            # mean envelope coefficients

                    # else set mean to zero
                    # else:
                    #     imf_envelope_mean = np.zeros_like(intrinsic_mode_function_candidate)

            # FAIL-SAFE ADDENDUM - remove imf caused by numerical errors

            # Remove last trend or residual if non-zero (within acceptable error margin)
            # or does not equal last extracted imf candidate - removed
            if not all(intrinsic_mode_function_candidate == np.zeros_like(intrinsic_mode_function_candidate)) \
                    and np.any((intrinsic_mode_function_candidate -
                                np.mean(intrinsic_mode_function_candidate)) > num_err):
                intrinsic_mode_function_storage = np.vstack((intrinsic_mode_function_storage,
                                                             intrinsic_mode_function_candidate))
                if optimise_knots != 2:
                    intrinsic_mode_function_storage_coef = np.vstack((intrinsic_mode_function_storage_coef,
                                                                      intrinsic_mode_function_candidate_coef))
                else:
                    coef_storage[imf_count] = intrinsic_mode_function_candidate_coef

            # All the code below relates to the meta-problem of what output to display
            # It is part debugging and part finding the optimal HT and IF representation

            # extend time and knot vectors and create relevant Booleans
            extended_knot_time = time_extension(knot_time)
            extended_knot_envelope = time_extension(knots)
            time_bool = np.r_[extended_knot_time >= knot_time[0]] & np.r_[extended_knot_time <= knot_time[-1]]
            knot_bool = np.r_[extended_knot_envelope >= knots[0]] & np.r_[
                extended_knot_envelope <= knots[-1]]

            # extend knot bool for implicit smoothing
            column_number = len(knots) - 1
            knot_bool[int(column_number - 1)] = True
            knot_bool[(- int(column_number))] = True
            knot_bool = knot_bool[2:-2]

            basis_extended = Basis(time=extended_knot_time, time_series=extended_knot_time)

            # create extended Hilbert martix for coefficients
            hilbert_matrix = basis_extended.hilbert_cubic_b_spline(extended_knot_envelope)
            hilbert_matrix = hilbert_matrix[:, time_bool]
            hilbert_matrix = hilbert_matrix[knot_bool, :]

            # optimised speed
            # fixing edge effects
            if not preprocess == 'downsample_decimate':
                hilbert_matrix_adjust = hilbert_matrix.copy()
                for row in [0, 1, 2]:
                    hilbert_dtht = Hilbert(time=time, time_series=b_spline_matrix_smooth[row, :])
                    hilbert_matrix_adjust[row, :] = hilbert_dtht.dtht_fft()
                for row in [-1, -2, -3]:
                    hilbert_dtht = Hilbert(time=time, time_series=b_spline_matrix_smooth[row, :])
                    hilbert_matrix_adjust[row, :] = hilbert_dtht.dtht_fft()
                hilbert_matrix = hilbert_matrix_adjust.copy()

            # create storage matrices of appropriate size for Hilbert transforms and instantaneous frequencies
            intrinsic_mode_function_storage_ht = np.zeros_like(intrinsic_mode_function_storage)
            intrinsic_mode_function_storage_if = np.zeros_like(intrinsic_mode_function_storage)
            intrinsic_mode_function_storage_if = intrinsic_mode_function_storage_if[:, 1:]

            # create storage matrices of appropriate size for DISCRETE-TIME Hilbert transforms and
            # instantaneous frequencies
            intrinsic_mode_function_storage_dt_ht = np.zeros_like(intrinsic_mode_function_storage)
            intrinsic_mode_function_storage_dt_if = np.zeros_like(intrinsic_mode_function_storage)
            intrinsic_mode_function_storage_dt_if = intrinsic_mode_function_storage_dt_if[:, 1:]

            # FAIL-SAFE ADDENDUM -  fix error with last 'imf'

            error_boundary = 1e-06

            if not any(np.abs(intrinsic_mode_function_storage[-1, :] -
                              np.mean(intrinsic_mode_function_storage[-1, :])) > error_boundary):
                intrinsic_mode_function_storage[-2, :] += intrinsic_mode_function_storage[-1, :]
                intrinsic_mode_function_storage = intrinsic_mode_function_storage[:-1, :]

                if optimise_knots != 2:
                    intrinsic_mode_function_storage_coef[-2, :] += intrinsic_mode_function_storage_coef[-1, :]
                    intrinsic_mode_function_storage_coef = intrinsic_mode_function_storage_coef[:-1, :]
                else:
                    coef_storage[np.asarray(coef_storage.keys())[-2]] += \
                        coef_storage[np.asarray(coef_storage.keys())[-1]]
                    coef_storage = coef_storage[coef_storage.keys()[:-1]]

            for coef in range(np.shape(intrinsic_mode_function_storage)[0]):
                # issue could be because of the last constant 'imf'

                ht = {}  # avoids unnecessary error
                dt_ht = {}  # avoids unnecessary error

                # Standard B-spline Hilbert transform calculation and storage
                # if initial smoothing is not done Hilbert transform of IMF must be calculated using
                # discrete-time Hilbert transform
                hilbert_dtht = Hilbert(time=time, time_series=intrinsic_mode_function_storage[coef, :])
                if spline_method == 'b_spline':
                    if (not initial_smoothing and (coef == 0 or coef == 1)) or preprocess == 'downsample_decimate':
                        if dtht_method == 'kak':
                            ht = hilbert_dtht.dtht_kak()
                        elif dtht_method == 'fft':
                            ht = hilbert_dtht.dtht_fft()
                    else:
                        if optimise_knots != 2:
                            ht = np.matmul(intrinsic_mode_function_storage_coef[coef, :], hilbert_matrix)
                        else:
                            # fix
                            ht = hilbert_dtht.dtht_fft()
                            # ht = np.matmul(coef_storage[coef], hilbert_matrix)
                else:  # i.e. if spline method is Cubic Hermite Spline or Akima Spline
                    if dtht_method == 'kak':
                        ht = hilbert_dtht.dtht_kak()
                    elif dtht_method == 'fft':
                        ht = hilbert_dtht.dtht_fft()
                intrinsic_mode_function_storage_ht[coef, :] = ht

                # Standard B-spline instantaneous frequency calculation and storage
                theta_calc = theta(intrinsic_mode_function_storage[coef, :],
                                   intrinsic_mode_function_storage_ht[coef, :])
                inst_freq = omega(knot_time, theta_calc)
                intrinsic_mode_function_storage_if[coef, :] = inst_freq

                hilbert_dtht = Hilbert(time=time, time_series=intrinsic_mode_function_storage[coef, :])

                # Uses discrete-time Hilbert transform
                if dtht:
                    if dtht_method == 'kak':
                        dt_ht = hilbert_dtht.dtht_kak()
                    elif dtht_method == 'fft':
                        dt_ht = hilbert_dtht.dtht_fft()
                    intrinsic_mode_function_storage_dt_ht[coef, :] = dt_ht
                    dt_ht_theta_calc = theta(intrinsic_mode_function_storage[coef, :],
                                             intrinsic_mode_function_storage_dt_ht[coef, :])
                    dt_ht_inst_freq = omega(knot_time, dt_ht_theta_calc)
                    intrinsic_mode_function_storage_dt_if[coef, :] = dt_ht_inst_freq

            try:
                intrinsic_mode_function_storage_coef
            except:
                intrinsic_mode_function_storage_coef = None
            try:
                intrinsic_mode_function_storage_dt_ht
            except:
                intrinsic_mode_function_storage_dt_ht = None
            try:
                intrinsic_mode_function_storage_dt_if
            except:
                intrinsic_mode_function_storage_dt_if = None
            if optimise_knots == 2:
                intrinsic_mode_function_storage_coef = coef_storage
                knots = knot_storage

            return intrinsic_mode_function_storage, intrinsic_mode_function_storage_ht, \
                intrinsic_mode_function_storage_if, intrinsic_mode_function_storage_coef, \
                knots, intrinsic_mode_function_storage_dt_ht, intrinsic_mode_function_storage_dt_if

        #########################################
        # Ensemble Empirical Mode Decomposition #
        #########################################

        elif ensemble:

            preprocess_ensemble = Preprocess(time=time, time_series=time_series)

            median_filtered_input_signal = preprocess_ensemble.median_filter(preprocess_window_length)[1]

            basis = Basis(time=time, time_series=median_filtered_input_signal)

            if matrix:  # use matrix constructed above to improve speed of algorithm - only B-splines
                median_filtered_input_signal = basis.basis_function_approximation_matrix(b_spline_matrix_signal,
                                                                                         b_spline_matrix_smooth)[0]
            else:  # construct matrix everytime
                median_filtered_input_signal = basis.basis_function_approximation(knots, knot_time,
                                                                                  spline_method=spline_method)[0]

            sd_of_signal = np.std(time_series - median_filtered_input_signal)

            ensemble_count = 0
            ensemble_storage = [np.nan] * ensemble_iter

            while ensemble_count < ensemble_iter:
                print("Sifting commencing on iteration {} of EEMD".format(ensemble_count + 1))
                noise = np.random.normal(0, ensemble_sd * sd_of_signal, len(time_series))
                time_series += noise

                emd = EMD(time=time, time_series=time_series)

                temp_storage = emd.empirical_mode_decomposition(
                    knots, knot_time, smooth=smooth, smoothing_penalty=smoothing_penalty,
                    edge_effect=edge_effect, sym_alpha=sym_alpha, stop_crit=stop_crit,
                    stop_crit_threshold=stop_crit_threshold, mft_theta_1=mft_theta_1,
                    mft_theta_2=mft_theta_2, mft_alpha=mft_alpha, mean_threshold=mean_threshold, debug=False,
                    verbose=verbose, spline_method=spline_method, dtht=dtht, dtht_method=dtht_method,
                    max_internal_iter=max_internal_iter, max_imfs=max_imfs, matrix=matrix,
                    initial_smoothing=initial_smoothing, dft=dft, order=order, increment=increment,
                    preprocess=preprocess, preprocess_window_length=preprocess_window_length,
                    preprocess_quantile=preprocess_quantile, preprocess_penalty=preprocess_penalty,
                    preprocess_order=preprocess_order, preprocess_norm_1=preprocess_norm_1,
                    preprocess_norm_2=preprocess_norm_2, ensemble=False, enhanced_iter=enhanced_iter,
                    output_coefficients=output_coefficients, optimise_knots=optimise_knots,
                    knot_method=knot_method, output_knots=output_knots,
                    knot_error=knot_error, knot_lamda=knot_lamda, knot_epsilon=knot_epsilon,
                    downsample_window=downsample_window, downsample_decimation_factor=downsample_decimation_factor,
                    downsample_window_factor=downsample_window_factor, nn_m=nn_m, nn_k=nn_k, nn_method=nn_method,
                    nn_learning_rate=nn_learning_rate, nn_iter=nn_iter)[:3]

                ensemble_storage[ensemble_count] = list(temp_storage)

                ensemble_count += 1

            # should incorporate mdlp algorithm here

            min_imfs = np.inf
            for i in range(len(ensemble_storage)):
                if np.shape(ensemble_storage[i][0])[0] < min_imfs:
                    min_imfs = np.shape(ensemble_storage[i][0])[0]

            for j in range(len(ensemble_storage)):
                ensemble_storage[j][0] = np.vstack((ensemble_storage[j][0][:int(min_imfs - 1), :],
                                                    ensemble_storage[j][0][-1, :]))
                ensemble_storage[j][1] = np.vstack((ensemble_storage[j][1][:int(min_imfs - 1), :],
                                                    ensemble_storage[j][1][-1, :]))
                ensemble_storage[j][2] = np.vstack((ensemble_storage[j][2][:int(min_imfs - 1), :],
                                                    ensemble_storage[j][2][-1, :]))

            imf_ensemble_storage = np.zeros_like(ensemble_storage[0][0])
            imf_ensemble_storage_ht = np.zeros_like(ensemble_storage[0][1])
            imf_ensemble_storage_if = np.zeros_like(ensemble_storage[0][2])
            for k in range(ensemble_iter):
                imf_ensemble_storage += ensemble_storage[k][0]
                imf_ensemble_storage_ht += ensemble_storage[k][1]
                imf_ensemble_storage_if += ensemble_storage[k][2]

            imf_ensemble_storage /= ensemble_iter
            imf_ensemble_storage_ht /= ensemble_iter
            imf_ensemble_storage_if /= ensemble_iter

            return imf_ensemble_storage, imf_ensemble_storage_ht, imf_ensemble_storage_if
