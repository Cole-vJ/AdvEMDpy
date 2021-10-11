
#     ________
#            /
#      \    /
#       \  /
#        \/

import time
import pytest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from AdvEMDpy import EMD
from emd_basis import Basis
from emd_hilbert import Hilbert, theta, omega, hilbert_spectrum, morlet_window, morlet_window_adjust
from emd_mean import Fluctuation
from emd_preprocess import Preprocess
from emd_utils import Utility, time_extension
from PyEMD import EMD as pyemd0215
import emd as emd040

sns.set(style='darkgrid')


class EMDUnitTests:

    def __init__(self, emd, basis, hilbert, mean, preprocess, utility):

        self.emd = emd
        self.basis = basis
        self.hilbert = hilbert
        self.mean = mean
        self.preprocess = preprocess
        self.utility = utility

    # test acceptable time series

    def test_all(self, print_all=False):

        test_1 = self.test_time_series_linear(test_all=True)
        test_2 = self.test_time_series_single_extreme(test_all=True)
        test_3 = self.test_time_series_double_extreme(test_all=True)

        test_4 = self.test_time_series_error_length(test_all=True)
        test_5 = self.test_smooth_bool(test_all=True)
        test_6 = self.test_smooth_penalty(test_all=True)
        test_7 = self.test_edge_effect(test_all=True)
        test_8 = self.test_sym_alpha_type(test_all=True)
        test_9 = self.test_sym_alpha_value(test_all=True)
        test_10 = self.test_stopping_criteria(test_all=True)
        test_11 = self.test_stopping_criteria_threshold(test_all=True)
        test_12 = self.test_mft_theta_types(test_all=True)
        test_13 = self.test_mft_theta_values(test_all=True)
        test_14 = self.test_mft_alpha_type(test_all=True)
        test_15 = self.test_mft_alpha_value(test_all=True)
        test_16 = self.test_mean_threshold_type(test_all=True)
        test_17 = self.test_mean_threshold_value(test_all=True)
        test_18 = self.test_debug_type(test_all=True)
        test_19 = self.test_text_type(test_all=True)
        test_20 = self.test_spline_method_value(test_all=True)
        test_21 = self.test_dtht_type(test_all=True)
        test_22 = self.test_dtht_method_value(test_all=True)
        test_23 = self.test_max_internal_iter_value(test_all=True)
        test_24 = self.test_matrix_type(test_all=True)
        test_25 = self.test_initial_smoothing_type(test_all=True)
        test_26 = self.test_dft_value(test_all=True)
        test_27 = self.test_order_value(test_all=True)
        test_28 = self.test_increment_value(test_all=True)
        test_29 = self.test_preprocess_value(test_all=True)
        test_30 = self.test_preprocess_window_length_value(test_all=True)
        test_31 = self.test_preprocess_quantile_value(test_all=True)
        test_32 = self.test_preprocess_penalty_value(test_all=True)
        test_33 = self.test_preprocess_order_value(test_all=True)
        test_34 = self.test_preprocess_norm_1_value(test_all=True)
        test_35 = self.test_preprocess_norm_2_value(test_all=True)
        test_36 = self.test_ensemble_type(test_all=True)
        test_37 = self.test_ensemble_sd_value(test_all=True)
        test_38 = self.test_ensemble_iter_value(test_all=True)
        test_39 = self.test_output_coefficient_type(test_all=True)
        test_40 = self.test_optimize_knots_type(test_all=True)
        test_41 = self.test_knot_method_value(test_all=True)
        test_42 = self.test_output_knots_value(test_all=True)
        test_43 = self.test_downsample_window_value(test_all=True)
        test_44 = self.test_downsample_decimation_factor_value(test_all=True)
        test_45 = self.test_downsample_window_factor_value(test_all=True)
        test_46 = self.test_downsample_decimation_and_window_factor_value(test_all=True)
        test_47 = self.test_nn_m_value(test_all=True)
        test_48 = self.test_nn_k_value(test_all=True)
        test_49 = self.test_nn_method_value(test_all=True)
        test_50 = self.test_nn_learning_rate_value(test_all=True)
        test_51 = self.test_nn_iter_value(test_all=True)
        test_52 = self.test_hp_preprocess_order_value(test_all=True)
        test_53 = self.test_hw_preprocess_order_value(test_all=True)

        # emd_basis
        test_54 = self.test_emd_basis(test_all=True)

        # emd_hilbert
        test_55 = self.test_emd_hilbert(test_all=True)

        # emd_mean
        test_56 = self.test_emd_mean(test_all=True)

        # emd_preprocess
        test_57 = self.test_emd_preprocess(test_all=True)

        # emd_utility
        test_58 = self.test_emd_utility(test_all=True)

        # try break AdvEMDpy
        test_59 = self.try_break_emd(test_all=True)

        tests = [test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_10,
                 test_11, test_12, test_13, test_14, test_15, test_16, test_17, test_18, test_19, test_20,
                 test_21, test_22, test_23, test_24, test_25, test_26, test_27, test_28, test_29, test_30,
                 test_31, test_32, test_33, test_34, test_35, test_36, test_37, test_38, test_39, test_40,
                 test_41, test_42, test_43, test_44, test_45, test_46, test_47, test_48, test_49, test_50,
                 test_51, test_52, test_53, test_54, test_55, test_56, test_57, test_58, test_59]

        if print_all:
            print(tests)
            print('Number of tests: {}'.format(len(tests)))

        if all(tests):
            print('ALL TESTS PASSED.')
        else:
            print('SOME TESTS FAILED.')
            for i, test in enumerate(tests):
                if not test:
                    print(f'TEST {int(i + 1)} FAILED')

    def test_time_series_linear(self, test_all=False):

        time_series_linear = np.linspace(0, 10, 101)
        knots = np.linspace(0, 100, 11)
        emd_linear_class = self.emd(time=np.arange(len(time_series_linear)),
                                    time_series=time_series_linear)
        emd_linear = emd_linear_class.empirical_mode_decomposition(knots=knots,
                                                                   knot_time=np.arange(len(time_series_linear)),
                                                                   initial_smoothing=False)
        if not test_all:
            print(all(emd_linear[0] == time_series_linear))
        else:
            return all(emd_linear[0] == time_series_linear)

    def test_time_series_single_extreme(self, test_all=False):

        time_series_quad = np.asarray((np.linspace(0, 10, 101) - 5) ** 2)
        knots = np.linspace(0, 100, 11)
        emd_quad_class = self.emd(time=np.arange(len(time_series_quad)),
                                  time_series=time_series_quad)
        emd_quad = emd_quad_class.empirical_mode_decomposition(knots=knots,
                                                               knot_time=np.arange(len(time_series_quad)),
                                                               initial_smoothing=False)
        if not test_all:
            print(all(emd_quad[0] == time_series_quad))
        else:
            return all(emd_quad[0] == time_series_quad)

    def test_time_series_double_extreme(self, test_all=False):

        time_series_cubic = np.asarray((np.linspace(0, 10, 101) - 2) *
                                       (np.linspace(0, 10, 101) - 5) *
                                       (np.linspace(0, 10, 101) - 8))
        knots = np.linspace(0, 100, 11)
        emd_cubic_class = self.emd(time=np.arange(len(time_series_cubic)),
                                   time_series=time_series_cubic)
        emd_cubic = emd_cubic_class.empirical_mode_decomposition(knots=knots,
                                                                 knot_time=np.arange(len(time_series_cubic)),
                                                                 initial_smoothing=False)
        if not test_all:
            print(all(emd_cubic[0] == time_series_cubic))
        else:
            return all(emd_cubic[0] == time_series_cubic)

    # test error messages

    def test_time_series_error_length(self, test_all=False):

        error_time = np.linspace(0, 100, 101)
        time_series = np.linspace(0, 100, 100)

        with pytest.raises(ValueError) as error_info:
            self.emd(time_series=time_series, time=error_time)
        if not test_all:
            print(error_info.type is ValueError and
                  error_info.value.args[0] == 'Input time series and input time are incompatible lengths.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   'Input time series and input time are incompatible lengths.'

    def test_smooth_bool(self, test_all=False):

        smooth = 'not_bool'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(smooth=smooth)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] == '\'smooth\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == '\'smooth\' must be boolean.'

    def test_smooth_penalty(self, test_all=False):

        smooth_penalty = 'not_bool_penalty'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(smoothing_penalty=smooth_penalty)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'smoothing_penalty\' must be float or integer.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'smoothing_penalty\' must be float or integer.'

    def test_edge_effect(self, test_all=False):

        edge_effect = 'not_edge_effect'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(edge_effect=edge_effect)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'edge_effect\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'edge_effect\' not an acceptable value.'

    def test_sym_alpha_type(self, test_all=False):

        sym_alpha = 'not_bool_penalty'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(sym_alpha=sym_alpha)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'sym_alpha\' value must be float or integer.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'sym_alpha\' value must be float or integer.'

    def test_sym_alpha_value(self, test_all=False):

        sym_alpha = 2
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(sym_alpha=sym_alpha)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'sym_alpha\' value must not be greater than 1')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'sym_alpha\' value must not be greater than 1'

    def test_stopping_criteria(self, test_all=False):

        stopping_criteria = 'not_stopping_criterion'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit=stopping_criteria)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'stop_crit\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'stop_crit\' not an acceptable value.'

    def test_stopping_criteria_threshold(self, test_all=False):

        stopping_criteria_threshold = 'not_stopping_criterion_threshold'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit_threshold=stopping_criteria_threshold)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'stop_crit_threshold\' must be float or integer.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'stop_crit_threshold\' must be float or integer.'

    def test_mft_theta_types(self, test_all=False):

        theta_1_string = 'not_theta_1'
        theta_2_string = 'not_theta_2'
        theta_1 = 1
        theta_2 = 10
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit='mft',
                                                                      mft_theta_1=theta_1_string,
                                                                      mft_theta_2=theta_2)
        with pytest.raises(TypeError) as error_info_2:
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit='mft',
                                                                      mft_theta_1=theta_1,
                                                                      mft_theta_2=theta_2_string)
        with pytest.raises(TypeError) as error_info_3:
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit='mft',
                                                                      mft_theta_1=theta_1_string,
                                                                      mft_theta_2=theta_2_string)
        if not test_all:
            print(all([error_info_1.type is TypeError and error_info_1.value.args[0] ==
                       '\'mft_theta_1\' and \'mft_theta_2\' must be floats of integers.',
                       error_info_2.type is TypeError and error_info_2.value.args[0] ==
                       '\'mft_theta_1\' and \'mft_theta_2\' must be floats of integers.',
                       error_info_3.type is TypeError and error_info_3.value.args[0] ==
                       '\'mft_theta_1\' and \'mft_theta_2\' must be floats of integers.']))
        else:
            return all([error_info_1.type is TypeError and error_info_1.value.args[0] ==
                        '\'mft_theta_1\' and \'mft_theta_2\' must be floats of integers.',
                        error_info_2.type is TypeError and error_info_2.value.args[0] ==
                        '\'mft_theta_1\' and \'mft_theta_2\' must be floats of integers.',
                        error_info_3.type is TypeError and error_info_3.value.args[0] ==
                        '\'mft_theta_1\' and \'mft_theta_2\' must be floats of integers.'])

    def test_mft_theta_values(self, test_all=False):

        theta_1_not_greater_than_zero = -1
        theta_1_greater_than_theta_2 = 100
        theta_1 = 1
        theta_2_not_greater_than_zero = -2
        theta_2 = 10
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit='mft',
                                                                      mft_theta_1=theta_1_not_greater_than_zero,
                                                                      mft_theta_2=theta_2)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit='mft',
                                                                      mft_theta_1=theta_1,
                                                                      mft_theta_2=theta_2_not_greater_than_zero)
        with pytest.raises(ValueError) as error_info_3:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit='mft',
                                                                      mft_theta_1=theta_1_greater_than_theta_2,
                                                                      mft_theta_2=theta_2)

        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'mft_theta_1\' and \'mft_theta_2\' not acceptable values.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'mft_theta_1\' and \'mft_theta_2\' not acceptable values.',
                       error_info_3.type is ValueError and error_info_3.value.args[0] ==
                       '\'mft_theta_1\' and \'mft_theta_2\' not acceptable values.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                        '\'mft_theta_1\' and \'mft_theta_2\' not acceptable values.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'mft_theta_1\' and \'mft_theta_2\' not acceptable values.',
                        error_info_3.type is ValueError and error_info_3.value.args[0] ==
                        '\'mft_theta_1\' and \'mft_theta_2\' not acceptable values.'])

    def test_mft_alpha_type(self, test_all=False):

        mft_alpha = 'not_mft_alpha'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit='mft', mft_alpha=mft_alpha)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'mft_alpha\' must be float or integer.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'mft_alpha\' must be float or integer.'

    def test_mft_alpha_value(self, test_all=False):

        mft_alpha_less_than_0 = -1
        mft_alpha_greater_than_1 = 2
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit='mft',
                                                                      mft_alpha=mft_alpha_less_than_0)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(stop_crit='mft',
                                                                      mft_alpha=mft_alpha_greater_than_1)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'mft_alpha\' must be a percentage.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'mft_alpha\' must be a percentage.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                        '\'mft_alpha\' must be a percentage.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'mft_alpha\' must be a percentage.'])

    def test_mean_threshold_type(self, test_all=False):

        mean_threshold = 'not_mean_threshold'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(mean_threshold=mean_threshold)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'mean_threshold\' must be a float or integer.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'mean_threshold\' must be a float or integer.'

    def test_mean_threshold_value(self, test_all=False):

        mean_threshold = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(mean_threshold=mean_threshold)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'mean_threshold\' must be greater than zero.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'mean_threshold\' must be greater than zero.'

    def test_debug_type(self, test_all=False):

        debug = 'not_debug'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(debug=debug)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'debug\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'debug\' must be boolean.'

    def test_text_type(self, test_all=False):

        text = 'not_text'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(verbose=text)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'verbose\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'verbose\' must be boolean.'

    def test_spline_method_value(self, test_all=False):

        spline_method = 'not_spline_method'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(spline_method=spline_method)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'spline_method\' is not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'spline_method\' is not an acceptable value.'

    def test_dtht_type(self, test_all=False):

        dtht = 'not_dtht'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(dtht=dtht)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'dtht\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'dtht\' must be boolean.'

    def test_dtht_method_value(self, test_all=False):

        dtht_method = 'not_dtht_method'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(dtht=True, dtht_method=dtht_method)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'dtht_method\' is not an acceptable method.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'dtht_method\' is not an acceptable method.'

    def test_max_internal_iter_value(self, test_all=False):

        max_internal_iter_string = 'max_internal_iter'
        max_internal_iter_less_than_1 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(max_internal_iter=max_internal_iter_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(max_internal_iter=max_internal_iter_less_than_1)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'max_internal_iter\' must be a non-negative integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'max_internal_iter\' must be a non-negative integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                        '\'max_internal_iter\' must be a non-negative integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'max_internal_iter\' must be a non-negative integer.'])

    def test_matrix_type(self, test_all=False):

        matrix = 'not_matrix'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(matrix=matrix)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'matrix\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'matrix\' must be boolean.'

    def test_initial_smoothing_type(self, test_all=False):

        initial_smoothing = 'not_initial_smoothing'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(initial_smoothing=initial_smoothing)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'initial_smoothing\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'initial_smoothing\' must be boolean.'

    def test_dft_value(self, test_all=False):

        dft = 'not_dft'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(dft=dft)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'dft\' not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'dft\' not an acceptable value.'

    def test_order_value(self, test_all=False):

        order_string = 'not_order'
        order_even = 2
        order_less_than_1 = -1
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(dft='binomial_average', order=order_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(dft='binomial_average', order=order_even)
        with pytest.raises(ValueError) as error_info_3:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(dft='binomial_average', order=order_less_than_1)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'order\' must be a positive, odd, integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'order\' must be a positive, odd, integer.',
                       error_info_3.type is ValueError and error_info_3.value.args[0] ==
                       '\'order\' must be a positive, odd, integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'order\' must be a positive, odd, integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'order\' must be a positive, odd, integer.',
                        error_info_3.type is ValueError and error_info_3.value.args[0] ==
                        '\'order\' must be a positive, odd, integer.'])

    def test_increment_value(self, test_all=False):

        increment_string = 'not_increment'
        increment_less_than_1 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(dft='binomial_average',
                                                                      increment=increment_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(dft='binomial_average',
                                                                      increment=increment_less_than_1)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'increment\' must be a positive integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'increment\' must be a positive integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'increment\' must be a positive integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'increment\' must be a positive integer.'])

    def test_preprocess_value(self, test_all=False):

        preprocess = 'not_preprocess'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess=preprocess)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'preprocess\' technique not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'preprocess\' technique not an acceptable value.'

    def test_preprocess_window_length_value(self, test_all=False):

        preprocess_window_length_string = 'not_preprocess_window_length'
        preprocess_window_length_even = 2
        preprocess_window_length_less_than_1 = -1
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.\
                empirical_mode_decomposition(preprocess='median_filter',
                                             preprocess_window_length=preprocess_window_length_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.\
                empirical_mode_decomposition(preprocess='median_filter',
                                             preprocess_window_length=preprocess_window_length_even)
        with pytest.raises(ValueError) as error_info_3:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.\
                empirical_mode_decomposition(preprocess='median_filter',
                                             preprocess_window_length=preprocess_window_length_less_than_1)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_window_length\' must be a positive, odd, integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'preprocess_window_length\' must be a positive, odd, integer.',
                       error_info_3.type is ValueError and error_info_3.value.args[0] ==
                       '\'preprocess_window_length\' must be a positive, odd, integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_window_length\' must be a positive, odd, integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'preprocess_window_length\' must be a positive, odd, integer.',
                        error_info_3.type is ValueError and error_info_3.value.args[0] ==
                        '\'preprocess_window_length\' must be a positive, odd, integer.'])

    def test_preprocess_quantile_value(self, test_all=False):

        preprocess_quantile_string = 'not_preprocess_quantile'
        preprocess_quantile_0 = 0
        preprocess_quantile_1 = 1
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='winsorize',
                                                                      preprocess_quantile=preprocess_quantile_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='winsorize',
                                                                      preprocess_quantile=preprocess_quantile_0)
        with pytest.raises(ValueError) as error_info_3:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='winsorize',
                                                                      preprocess_quantile=preprocess_quantile_1)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_quantile\' value must be a percentage.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'preprocess_quantile\' value must be a percentage.',
                       error_info_3.type is ValueError and error_info_3.value.args[0] ==
                       '\'preprocess_quantile\' value must be a percentage.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_quantile\' value must be a percentage.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'preprocess_quantile\' value must be a percentage.',
                        error_info_3.type is ValueError and error_info_3.value.args[0] ==
                        '\'preprocess_quantile\' value must be a percentage.'])

    def test_preprocess_penalty_value(self, test_all=False):

        preprocess_penalty_string = 'not_preprocess_penalty'
        preprocess_penalty_less_than_0 = -1
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='HP',
                                                                      preprocess_penalty=preprocess_penalty_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='HP',
                                                                      preprocess_penalty=preprocess_penalty_less_than_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_penalty\' must be a non-negative float or integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'preprocess_penalty\' must be a non-negative float or integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_penalty\' must be a non-negative float or integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'preprocess_penalty\' must be a non-negative float or integer.'])

    def test_preprocess_order_value(self, test_all=False):

        preprocess_order_string = 'not_preprocess_order'
        preprocess_order_less_than_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='HP',
                                                                      preprocess_order=preprocess_order_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='HP',
                                                                      preprocess_order=preprocess_order_less_than_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_order\' must be a positive integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'preprocess_order\' must be a positive integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_order\' must be a positive integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'preprocess_order\' must be a positive integer.'])

    def test_preprocess_norm_1_value(self, test_all=False):

        preprocess_norm_1_string = 'not_preprocess_norm_1'
        preprocess_norm_1_less_than_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='HP',
                                                                      preprocess_norm_1=preprocess_norm_1_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='HP',
                                                                      preprocess_norm_1=preprocess_norm_1_less_than_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_norm_1\' must be a positive integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'preprocess_norm_1\' must be a positive integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_norm_1\' must be a positive integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'preprocess_norm_1\' must be a positive integer.'])

    def test_preprocess_norm_2_value(self, test_all=False):

        preprocess_norm_2_string = 'not_preprocess_norm_1'
        preprocess_norm_2_less_than_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='HP',
                                                                      preprocess_norm_2=preprocess_norm_2_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='HP',
                                                                      preprocess_norm_2=preprocess_norm_2_less_than_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_norm_2\' must be a positive integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'preprocess_norm_2\' must be a positive integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'preprocess_norm_2\' must be a positive integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'preprocess_norm_2\' must be a positive integer.'])

    def test_ensemble_type(self, test_all=False):

        ensemble = 'not_ensemble'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(ensemble=ensemble)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'ensemble\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'ensemble\' must be boolean.'

    def test_ensemble_sd_value(self, test_all=False):

        ensemble_sd_string = 'not_ensemble_sd'
        ensemble_sd_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(ensemble_sd=ensemble_sd_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(ensemble_sd=ensemble_sd_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'ensemble_sd\' must be positive float or integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'ensemble_sd\' must be positive float or integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'ensemble_sd\' must be positive float or integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'ensemble_sd\' must be positive float or integer.'])

    def test_ensemble_iter_value(self, test_all=False):

        ensemble_iter_string = 'not_ensemble_iter'
        ensemble_iter_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(ensemble_iter=ensemble_iter_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(ensemble_iter=ensemble_iter_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'ensemble_iter\' must be a positive integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'ensemble_iter\' must be a positive integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'ensemble_iter\' must be a positive integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'ensemble_iter\' must be a positive integer.'])

    def test_output_coefficient_type(self, test_all=False):

        output_coefficient = 'not_output_coefficient'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(output_coefficients=output_coefficient)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'output_coefficients\' must be boolean.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'output_coefficients\' must be boolean.'

    def test_optimize_knots_type(self, test_all=False):

        optimise_knots = 'not_optimise_knots'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(optimise_knots=optimise_knots)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'optimise_knots\' is not an appropriate value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'optimise_knots\' is not an appropriate value.'

    def test_knot_method_value(self, test_all=False):

        knot_method = 'not_knot_method'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(optimise_knots=True, knot_method=knot_method)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'knot_method\' technique not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'knot_method\' technique not an acceptable value.'

    def test_output_knots_value(self, test_all=False):

        output_knots = 'not_output_knots_bool'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(TypeError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(output_knots=output_knots)
        if not test_all:
            print(error_info.type is TypeError and error_info.value.args[0] ==
                  '\'output_knots\' must be boolean.')
        else:
            return error_info.type is TypeError and error_info.value.args[0] == \
                   '\'output_knots\' must be boolean.'

    def test_downsample_window_value(self, test_all=False):

        downsample_window = 'not_downsample_window'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='downsample',
                                                                      downsample_window=downsample_window)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'downsample_window\' unknown value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'downsample_window\' unknown value.'

    def test_downsample_decimation_factor_value(self, test_all=False):

        downsample_decimation_factor_string = 'not_downsample_decimation_factor'
        downsample_decimation_factor_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.\
                empirical_mode_decomposition(preprocess='downsample', downsample_window='hann',
                                             downsample_decimation_factor=downsample_decimation_factor_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.\
                empirical_mode_decomposition(preprocess='downsample', downsample_window='hann',
                                             downsample_decimation_factor=downsample_decimation_factor_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'downsample_decimation_factor\' must be a positive integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'downsample_decimation_factor\' must be a positive integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                        '\'downsample_decimation_factor\' must be a positive integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'downsample_decimation_factor\' must be a positive integer.'])

    def test_downsample_window_factor_value(self, test_all=False):

        downsample_window_factor_string = 'not_downsample_window_factor'
        downsample_window_factor_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.\
                empirical_mode_decomposition(preprocess='downsample', downsample_window='hann',
                                             downsample_window_factor=downsample_window_factor_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.\
                empirical_mode_decomposition(preprocess='downsample', downsample_window='hann',
                                             downsample_window_factor=downsample_window_factor_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'downsample_window_factor\' must be a positive integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'downsample_window_factor\' must be a positive integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                        '\'downsample_window_factor\' must be a positive integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'downsample_window_factor\' must be a positive integer.'])

    def test_downsample_decimation_and_window_factor_value(self, test_all=False):

        downsample_decimation_factor = 3
        downsample_window_factor = 5
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.\
                empirical_mode_decomposition(preprocess='downsample', downsample_window='hann',
                                             downsample_decimation_factor=downsample_decimation_factor,
                                             downsample_window_factor=downsample_window_factor)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  'Product of \'downsample_decimation_factor\' and \'downsample_window_factor\' must be even.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   'Product of \'downsample_decimation_factor\' and \'downsample_window_factor\' must be even.'

    def test_nn_m_value(self, test_all=False):

        nn_m_string = 'not_nn_m'
        nn_m_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(edge_effect='neural_network', nn_m=nn_m_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(edge_effect='neural_network', nn_m=nn_m_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'nn_m\' training outputs must be a positive integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'nn_m\' training outputs must be a positive integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'nn_m\' training outputs must be a positive integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'nn_m\' training outputs must be a positive integer.'])

    def test_nn_k_value(self, test_all=False):

        nn_k_string = 'not_nn_k'
        nn_k_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(edge_effect='neural_network', nn_k=nn_k_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(edge_effect='neural_network', nn_k=nn_k_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'nn_k\' training inputs must be a positive integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'nn_k\' training inputs must be a positive integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'nn_k\' training inputs must be a positive integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'nn_k\' training inputs must be a positive integer.'])

    def test_nn_method_value(self, test_all=False):

        nn_method = 'not_nn_method'
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(edge_effect='neural_network', nn_method=nn_method)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  '\'nn_method\' technique not an acceptable value.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   '\'nn_method\' technique not an acceptable value.'

    def test_nn_learning_rate_value(self, test_all=False):

        nn_learning_rate_string = 'not_nn_learning_rate_string'
        nn_learning_rate_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(edge_effect='neural_network',
                                                                      nn_learning_rate=nn_learning_rate_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(edge_effect='neural_network',
                                                                      nn_learning_rate=nn_learning_rate_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'nn_learning_rate\' must be a positive float or integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'nn_learning_rate\' must be a positive float or integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                        '\'nn_learning_rate\' must be a positive float or integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'nn_learning_rate\' must be a positive float or integer.'])

    def test_nn_iter_value(self, test_all=False):

        nn_iter_string = 'not_nn_iter'
        nn_iter_0 = 0
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info_1:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(edge_effect='neural_network',
                                                                      nn_iter=nn_iter_string)
        with pytest.raises(ValueError) as error_info_2:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(edge_effect='neural_network',
                                                                      nn_iter=nn_iter_0)
        if not test_all:
            print(all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                       '\'nn_iter\' must be a positive integer.',
                       error_info_2.type is ValueError and error_info_2.value.args[0] ==
                       '\'nn_iter\' must be a positive integer.']))
        else:
            return all([error_info_1.type is ValueError and error_info_1.value.args[0] ==
                        '\'nn_iter\' must be a positive integer.',
                        error_info_2.type is ValueError and error_info_2.value.args[0] ==
                        '\'nn_iter\' must be a positive integer.'])

    def test_hp_preprocess_order_value(self, test_all=False):

        preprocess_order_5 = 5
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='HP',
                                                                      preprocess_order=preprocess_order_5)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  'Hodrick-Prescott order must be 1, 2, 3, or 4.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   'Hodrick-Prescott order must be 1, 2, 3, or 4.'

    def test_hw_preprocess_order_value(self, test_all=False):

        preprocess_order_4 = 4
        time_series = np.linspace(0, 100, 101)

        with pytest.raises(ValueError) as error_info:
            empirical_mode_decomposition = self.emd(time_series=time_series)
            empirical_mode_decomposition.empirical_mode_decomposition(preprocess='HW',
                                                                      preprocess_order=preprocess_order_4)
        if not test_all:
            print(error_info.type is ValueError and error_info.value.args[0] ==
                  'Henderson-Whittaker order must be odd.')
        else:
            return error_info.type is ValueError and error_info.value.args[0] == \
                   'Henderson-Whittaker order must be odd.'

    def test_emd_basis(self, test_all=False, plot=False):

        try:
            emd_basis_time = np.linspace(0, 10, 1001)
            emd_basis_time_series = np.cos(emd_basis_time) + np.cos(5 * emd_basis_time)
            emd_basis_knots = np.linspace(0, 10, 5)
            emd_basis = self.basis(time=emd_basis_time, time_series=emd_basis_time_series)
            single_cubic_b_spline = emd_basis.b(knots=emd_basis_knots, degree=3)
            single_hilbert_cubic_b_spline = emd_basis.hilbert_b(knots=emd_basis_knots, degree=3)
            if plot:
                plt.title('B-Spline and Hilbert Transform')
                plt.plot(single_cubic_b_spline, label='Cubic B-spline')
                plt.plot(single_hilbert_cubic_b_spline, label='HT cubic B-spline')
                plt.legend(loc='lower left', fontsize=6)
                plt.show()
            emd_basis_knots = np.linspace(0, 10, 21)
            quadratic_b_spline = emd_basis.quadratic_b_spline(knots=emd_basis_knots)
            if plot:
                plt.title('Quadratic B-Splines')
                plt.plot(quadratic_b_spline.T)
                plt.show()
            derivative_cubic_b_spline = emd_basis.derivative_cubic_b_spline(knots=emd_basis_knots)
            if plot:
                plt.title('Derivative Cubic B-Splines')
                plt.plot(derivative_cubic_b_spline.T)
                plt.show()
            cubic_b_spline = emd_basis.cubic_b_spline(knots=emd_basis_knots)
            hilbert_cubic_b_spline = emd_basis.hilbert_cubic_b_spline(knots=emd_basis_knots)
            if plot:
                plt.title('B-Spline Bases and Hilbert Transform Bases')
                plt.plot(cubic_b_spline.T)
                plt.plot(hilbert_cubic_b_spline.T)
                plt.show()
            basis_function_approximation = emd_basis.basis_function_approximation(knots=emd_basis_knots,
                                                                                  knot_time=emd_basis_time)[0]
            optimize_knot_points_bisect = emd_basis.optimize_knot_points(method='bisection')
            if plot:
                basis_function_approximation_bisect = \
                    emd_basis.basis_function_approximation(knots=optimize_knot_points_bisect,
                                                           knot_time=emd_basis_time)[0]
                plt.title('Bisection Knot Optimization')
                plt.plot(emd_basis_time, emd_basis_time_series, label='Time series')
                plt.plot(emd_basis_time, basis_function_approximation, label='Uniform knot spline estimate')
                plt.plot(emd_basis_time, basis_function_approximation_bisect, '--',
                         label='Bisection knot spline estimate')
                for i in np.asarray(optimize_knot_points_bisect):
                    plt.plot(i * np.ones(100), np.linspace(-1, 1, 100), '--')
                plt.legend(loc='lower left', fontsize=6)
                plt.show()
            optimize_knot_points_ser_bisect = emd_basis.optimize_knot_points(method='ser_bisect')
            if plot:
                basis_function_approximation_ser_bisect = \
                    emd_basis.basis_function_approximation(knots=optimize_knot_points_ser_bisect,
                                                           knot_time=emd_basis_time)[0]
                plt.title('Serial Bisection Knot Optimization')
                plt.plot(emd_basis_time, emd_basis_time_series, label='Time series')
                plt.plot(emd_basis_time, basis_function_approximation, label='Uniform knot spline estimate')
                plt.plot(emd_basis_time, basis_function_approximation_ser_bisect, '--',
                         label='Serial bisection knot spline estimate')
                for i in np.asarray(optimize_knot_points_ser_bisect):
                    plt.plot(i * np.ones(100), np.linspace(-1, 1, 100), '--')
                plt.legend(loc='lower left', fontsize=6)
                plt.show()
            chsi_basis = emd_basis.chsi_basis(knots=emd_basis_knots)
            if plot:
                plt.title('Cubic Hermite Spline Bases')
                plt.plot(chsi_basis.T)
                plt.show()
            chsi_basis_fit = emd_basis.chsi_fit(knots_time=emd_basis_time, knots=emd_basis_knots)
            asi_basis_fit = emd_basis.asi_fit(knots_time=emd_basis_time, knots=emd_basis_knots)
            if plot:
                plt.title('Different Bases')
                plt.plot(emd_basis_time, emd_basis_time_series, label='Time series')
                plt.plot(emd_basis_time, chsi_basis_fit, label='CHSI approximation')
                plt.plot(emd_basis_time, asi_basis_fit, label='ASI approximation')
                plt.plot(emd_basis_time, basis_function_approximation, label='Smoothed cubic B-spline approximation')
                plt.legend(loc='lower left', fontsize=6)
                plt.show()
            emd_basis_time_extended = np.linspace(0, 10, 2001)
            emd_basis_time_series_extended = np.cos(emd_basis_time_extended) + np.cos(5 * emd_basis_time_extended)
            emd_basis_knots_extended = np.linspace(-1.5, 11.5, 27)
            cubic_b_spline = emd_basis.cubic_b_spline(knots=emd_basis_knots_extended)
            emd_basis_extended = self.basis(time=emd_basis_time_extended, time_series=emd_basis_time_series_extended)
            cubic_b_spline_extended = emd_basis_extended.cubic_b_spline(knots=emd_basis_knots_extended)
            basis_function_approximation_matrix = \
                emd_basis.basis_function_approximation_matrix(b_spline_matrix_signal=cubic_b_spline,
                                                              b_spline_matrix_smooth=cubic_b_spline_extended)[0]
            if plot:
                plt.title('Time Series and Extended Time Series')
                plt.plot(emd_basis_time, basis_function_approximation, label='Time series original length')
                plt.plot(emd_basis_time_extended, basis_function_approximation_matrix, '--',
                         label='Time series extended length')
                plt.legend(loc='lower left', fontsize=6)
                plt.show()

            if not test_all:
                print('emd_basis.py all tests passed.')
            else:
                print('emd_basis.py all tests passed.')
                return True

        except:
            if not test_all:
                print('emd_basis.py some tests failed.')
            else:
                return False

    def test_emd_hilbert(self, test_all=False, plot=False):

        try:
            emd_hilbert_time = np.linspace(0, 10, 1001)
            emd_hilbert_real = np.cos(emd_hilbert_time)
            emd_hilbert_imag = np.sin(emd_hilbert_time)
            emd_theta = theta(emd_hilbert_real, emd_hilbert_imag)
            emd_omega = omega(emd_hilbert_time, emd_theta)
            if plot:
                plt.title('Signal, Hilbert Transform, and Instantaneous Frequency')
                plt.plot(emd_hilbert_real, label='Signal')
                plt.plot(emd_hilbert_imag, label='Hilbert transform')
                plt.plot(emd_omega, label='Instantaneous frequency')
                plt.legend(loc='lower left', fontsize=6)
                plt.show()
            imfs = np.vstack((np.cos(emd_hilbert_time) + np.cos(5 * emd_hilbert_time),
                              np.cos(5 * emd_hilbert_time), np.cos(emd_hilbert_time)))
            hts = np.vstack((np.sin(emd_hilbert_time) + np.sin(5 * emd_hilbert_time),
                             np.sin(5 * emd_hilbert_time), np.sin(emd_hilbert_time)))
            ifs = np.vstack((3 * np.ones_like(emd_hilbert_time[1:]), 5 * np.ones_like(emd_hilbert_time[1:]),
                             np.ones_like(emd_hilbert_time[1:])))
            hilbert_spectrum(emd_hilbert_time, imfs, hts, ifs, max_frequency=6, plot=plot)
            morlet = morlet_window(256, 2 * np.pi)
            morlet_adjust = morlet_window_adjust(256, 2 * np.pi, 50, 3)
            if plot:
                plt.title('Morlet Wavelet Packets')
                plt.plot(morlet, label='Standard wavelet')
                plt.plot(morlet_adjust, label='Frequency-dependant wavelet')
                plt.legend(loc='lower left', fontsize=6)
                plt.show()

            temp_time = np.linspace(0, 5 * np.pi, 1001)
            temp_time_series = np.sin(5 * temp_time)
            temp_time_series_hilbert = -np.cos(5 * temp_time)

            hilbert = self.hilbert(time=temp_time, time_series=temp_time_series)
            hilbert_kak = hilbert.dtht_kak()
            hilbert_fft = hilbert.dtht_fft()

            if plot:
                plt.title('Hilbert Transforms')
                plt.plot(temp_time, temp_time_series, label='Time series')
                plt.plot(temp_time, temp_time_series_hilbert, label='Hilbert transform')
                plt.plot(temp_time, hilbert_kak, '--', label='Basic Hilbert transform')
                plt.plot(temp_time, hilbert_fft, '--', label='FFT Hilbert transform')
                plt.legend(loc='lower left', fontsize=6)
                plt.show()

            t_custom, f_custom, z_custom = hilbert.stft_custom()
            if plot:
                plt.title('Short-Time Fourier Transform')
                plt.pcolormesh(t_custom, f_custom, np.abs(z_custom), vmin=0, vmax=np.max(np.max(np.abs(z_custom))))
                plt.plot(t_custom, 5 * np.ones_like(t_custom), '--', label='True frequency')
                plt.legend(loc='lower left', fontsize=6)
                plt.ylim(0, 10)
                plt.show()

            t_custom, f_custom, z_custom = hilbert.morlet_wavelet_custom()
            if plot:
                plt.title('Morlet Wavelet Transform')
                plt.pcolormesh(t_custom, f_custom, np.abs(z_custom), vmin=0, vmax=np.max(np.max(np.abs(z_custom))))
                plt.plot(t_custom, 5 * np.ones_like(t_custom), '--', label='True frequency')
                plt.legend(loc='lower left', fontsize=6)
                plt.ylim(0, 10)
                plt.show()

            if not test_all:
                print('emd_hilbert.py all tests passed.')
            else:
                print('emd_hilbert.py all tests passed.')
                return True

        except:
            if not test_all:
                print('emd_hilbert.py some tests failed.')
            else:
                return False

    def test_emd_mean(self, test_all=False, plot=False):

        try:
            mean_time = np.linspace(0, 5 * np.pi, 1001)
            mean_time_series = np.cos(mean_time) + np.cos(5 * mean_time)
            if plot:
                plt.title('Standard Envelope Technique')
                plt.plot(mean_time, mean_time_series, label='time series')
            fluctuation = self.mean(time=mean_time, time_series=mean_time_series)
            mean_knots = np.linspace(0, 5 * np.pi, 101)

            for technique in ['neural_network', 'symmetric', 'symmetric_anchor', 'symmetric_discard', 'anti-symmetric',
                              'characteristic_wave_Huang', 'characteristic_wave_Coughlin', 'slope_based_method',
                              'improved_slope_based_method', 'average', 'none']:
                maxima_envelope = \
                    fluctuation.envelope_basis_function_approximation(knots_for_envelope=mean_knots,
                                                                      extrema_type='maxima', smooth=True,
                                                                      smoothing_penalty=1, edge_effect=technique,
                                                                      spline_method='b_spline', alpha=0.1, nn_m=200,
                                                                      nn_k=100, nn_method='grad_descent',
                                                                      nn_learning_rate=0.01, nn_iter=100)[0]

                minima_envelope = \
                    fluctuation.envelope_basis_function_approximation(knots_for_envelope=mean_knots,
                                                                      extrema_type='minima', smooth=True,
                                                                      smoothing_penalty=1, edge_effect=technique,
                                                                      spline_method='b_spline', alpha=0.1, nn_m=200,
                                                                      nn_k=100, nn_method='grad_descent',
                                                                      nn_learning_rate=0.01, nn_iter=100)[0]

                if plot:
                    plt.plot(mean_time, maxima_envelope, '--', label=f'max {technique}')
                    plt.plot(mean_time, minima_envelope, '--', label=f'min {technique}')

            if plot:
                plt.legend(loc='lower left', fontsize=6)
                plt.show()

            if plot:
                plt.title('Different Spline Envelope Technique')
                plt.plot(mean_time, mean_time_series, label='time series')
            for spline in ['b_spline', 'chsi', 'asi']:
                maxima_envelope = \
                    fluctuation.envelope_basis_function_approximation(knots_for_envelope=mean_knots,
                                                                      extrema_type='maxima', smooth=True,
                                                                      smoothing_penalty=1, edge_effect='symmetric',
                                                                      spline_method=spline, alpha=0.1, nn_m=200,
                                                                      nn_k=100, nn_method='grad_descent',
                                                                      nn_learning_rate=0.01, nn_iter=100)[0]

                minima_envelope = \
                    fluctuation.envelope_basis_function_approximation(knots_for_envelope=mean_knots,
                                                                      extrema_type='minima', smooth=True,
                                                                      smoothing_penalty=1, edge_effect='symmetric',
                                                                      spline_method=spline, alpha=0.1, nn_m=200,
                                                                      nn_k=100, nn_method='grad_descent',
                                                                      nn_learning_rate=0.01, nn_iter=100)[0]

                if plot:
                    plt.plot(mean_time, maxima_envelope, '--', label=f'max {spline}')
                    plt.plot(mean_time, minima_envelope, '--', label=f'min {spline}')

            if plot:
                plt.legend(loc='lower left', fontsize=6)
                plt.show()

            temp_extended_knots = time_extension(mean_knots)
            extended_time = time_extension(mean_time)
            basis_extended = Basis(time=extended_time, time_series=extended_time)
            temp_extended_matrix = basis_extended.cubic_b_spline(temp_extended_knots)
            if plot:
                plt.title('Matrix Envelope Technique')
                plt.plot(mean_time, mean_time_series, label='time series')

            for technique in ['neural_network', 'symmetric', 'symmetric_anchor', 'symmetric_discard',
                              'anti-symmetric', 'characteristic_wave_Huang', 'characteristic_wave_Coughlin',
                              'slope_based_method', 'improved_slope_based_method', 'average', 'none']:

                maxima_envelope = \
                    fluctuation.envelope_basis_function_approximation_matrix(extended_matrix=temp_extended_matrix,
                                                                             knots_for_envelope=mean_knots,
                                                                             extrema_type='maxima', smooth=True,
                                                                             smoothing_penalty=1,
                                                                             edge_effect=technique,
                                                                             alpha=0.1, nn_m=200,
                                                                             nn_k=100, nn_method='grad_descent',
                                                                             nn_learning_rate=0.01, nn_iter=100)[0]

                minima_envelope = \
                    fluctuation.envelope_basis_function_approximation_matrix(extended_matrix=temp_extended_matrix,
                                                                             knots_for_envelope=mean_knots,
                                                                             extrema_type='minima', smooth=True,
                                                                             smoothing_penalty=1,
                                                                             edge_effect=technique,
                                                                             alpha=0.1, nn_m=200,
                                                                             nn_k=100, nn_method='grad_descent',
                                                                             nn_learning_rate=0.01, nn_iter=100)[0]

                if plot:
                    plt.plot(mean_time, maxima_envelope, '--', label=f'max {technique}')
                    plt.plot(mean_time, minima_envelope, '--', label=f'min {technique}')

            if plot:
                plt.legend(loc='lower left', fontsize=6)
                plt.show()

            utility_fixed = Utility(time=mean_time, time_series=mean_time_series)
            max_bool_fixed = utility_fixed.max_bool_func_1st_order_fd()
            min_bool_fixed = utility_fixed.min_bool_func_1st_order_fd()
            if plot:
                plt.title('Fixed Points Envelope Technique')
                plt.plot(mean_time, mean_time_series, label='time series')

            for technique in ['neural_network', 'symmetric', 'symmetric_anchor', 'symmetric_discard',
                              'anti-symmetric', 'characteristic_wave_Huang', 'characteristic_wave_Coughlin',
                              'slope_based_method', 'improved_slope_based_method', 'average', 'none']:

                maxima_envelope = \
                    fluctuation.envelope_basis_function_approximation_fixed_points(max_bool=max_bool_fixed,
                                                                                   min_bool=min_bool_fixed,
                                                                                   knots_for_envelope=mean_knots,
                                                                                   extrema_type='maxima', smooth=True,
                                                                                   smoothing_penalty=1,
                                                                                   edge_effect=technique,
                                                                                   alpha=0.1, nn_m=200,
                                                                                   nn_k=100,
                                                                                   nn_method='grad_descent',
                                                                                   nn_learning_rate=0.01,
                                                                                   nn_iter=100)[0]

                minima_envelope = \
                    fluctuation.envelope_basis_function_approximation_fixed_points(max_bool=max_bool_fixed,
                                                                                   min_bool=min_bool_fixed,
                                                                                   knots_for_envelope=mean_knots,
                                                                                   extrema_type='minima', smooth=True,
                                                                                   smoothing_penalty=1,
                                                                                   edge_effect=technique,
                                                                                   alpha=0.1, nn_m=200,
                                                                                   nn_k=100,
                                                                                   nn_method='grad_descent',
                                                                                   nn_learning_rate=0.01,
                                                                                   nn_iter=100)[0]

                if plot:
                    plt.plot(mean_time, maxima_envelope, '--', label=f'max {technique}')
                    plt.plot(mean_time, minima_envelope, '--', label=f'min {technique}')

            if plot:
                plt.legend(loc='lower left', fontsize=6)
                plt.show()

            if plot:
                plt.title('Direct Local Mean Estimation Technique')
                plt.plot(mean_time, mean_time_series, label='time series')
            for direct_technique in ['inflection_points', 'binomial_average']:
                direct = \
                    fluctuation.direct_detrended_fluctuation_estimation(knots=mean_knots, smooth=True,
                                                                        smoothing_penalty=1,
                                                                        technique=direct_technique, order=15,
                                                                        increment=10)[0]

                if plot:
                    plt.plot(mean_time, direct, '--', label=f'mean {direct_technique}')

            if plot:
                plt.legend(loc='lower left', fontsize=6)
                plt.show()

            if not test_all:
                print('emd_mean.py all tests passed.')
            else:
                print('emd_mean.py all tests passed.')
                return True

        except:
            if not test_all:
                print('emd_mean.py some tests failed.')
            else:
                return False

    def test_emd_preprocess(self, test_all=False, plot=False):

        try:
            preprocess_time = np.linspace(0, 5 * np.pi, 1001)
            preprocess_time_series = np.cos(preprocess_time) + np.cos(5 * preprocess_time)
            preprocess = self.preprocess(time=preprocess_time, time_series=preprocess_time_series)
            downsample = preprocess.downsample(decimate=False)
            downsample_decimate = preprocess.downsample()
            median = preprocess.median_filter()
            mean = preprocess.mean_filter()
            upper_quantile = preprocess.quantile_filter()
            lower_quantile = preprocess.quantile_filter(q=0.05)
            winsorize = preprocess.winsorize()
            winsorize_interpolate = preprocess.winsorize_interpolate()
            hp = preprocess.hp(smoothing_penalty=5, norm_2=1)
            hw = preprocess.hw()

            if plot:
                plt.title('Preprocessing Methods')
                plt.plot(preprocess_time, preprocess_time_series, label='time series')
                plt.plot(downsample[0], downsample[1], label='downsampled time series')
                plt.plot(downsample_decimate[0], downsample_decimate[1], label='downsampled and decimated time series')
                plt.plot(preprocess_time, median[1], label='median time series')
                plt.plot(preprocess_time, mean[1], label='mean time series')
                plt.plot(preprocess_time, upper_quantile[1], label='upper quantile time series*')
                plt.plot(preprocess_time, lower_quantile[1], label='lower quantile time series*')
                plt.plot(preprocess_time, winsorize[1], label='winsorize time series')
                plt.plot(preprocess_time, winsorize_interpolate[1], label='winsorize interpolate time series')
                plt.plot(preprocess_time, hp[1], label='HP time series')
                plt.plot(preprocess_time, hw[1], label='HW time series')
                plt.legend(loc='lower left', fontsize=6)
                plt.show()

            if not test_all:
                print('emd_preprocess.py all tests passed.')
            else:
                print('emd_preprocess.py all tests passed.')
                return True

        except:
            if not test_all:
                print('emd_preprocess.py some tests failed.')
            else:
                return False

    def test_emd_utility(self, test_all=False, plot=False):

        try:
            utility_time = np.linspace(0, 5 * np.pi, 1001)
            utility_time_series = np.cos(utility_time) + np.cos(5 * utility_time)
            utility = self.utility(time=utility_time, time_series=utility_time_series)
            utility_time_extension = time_extension(utility_time)
            utility_time_series_extension = np.cos(utility_time_extension) + np.cos(5 * utility_time_extension)
            if plot:
                plt.title('Utility Methods')
                plt.plot(utility_time, utility_time_series, label='time series')
                plt.plot(utility_time_extension, utility_time_series_extension, '--', label='extended time series')
            zero_crossing = utility.zero_crossing()
            zero_crossing_time = utility_time[zero_crossing]
            zero_crossing_time_series_plot = utility_time_series[zero_crossing]
            max_1 = utility.max_bool_func_1st_order_fd()
            max_1_time = utility_time[max_1]
            max_1_time_series = utility_time_series[max_1]
            min_1 = utility.min_bool_func_1st_order_fd()
            min_1_time = utility_time[min_1]
            min_1_time_series = utility_time_series[min_1]
            max_2 = utility.max_bool_func_2nd_order_fd()
            max_2_time = utility_time[max_2]
            max_2_time_series = utility_time_series[max_2]
            min_2 = utility.min_bool_func_2nd_order_fd()
            min_2_time = utility_time[min_2]
            min_2_time_series = utility_time_series[min_2]
            inflection = utility.inflection_point()
            inflection_time = utility_time[inflection]
            inflection_time_series = utility_time_series[inflection]
            derivative_time_series = utility.derivative_forward_diff()
            if plot:
                plt.scatter(zero_crossing_time, zero_crossing_time_series_plot, c='r', label='zero crossings')
                plt.scatter(max_1_time, max_1_time_series, c='magenta', label='1st order max')
                plt.scatter(min_1_time, min_1_time_series, c='cyan', label='1st order min')
                plt.scatter(max_2_time, max_2_time_series, c='gray', label='2nd order max')
                plt.scatter(min_2_time, min_2_time_series, c='black', label='2nd order min')
                plt.scatter(inflection_time, inflection_time_series + 0.05, c='green', label='inflection shifted')
                plt.plot(utility_time[:-1], derivative_time_series, label='forward difference')
                plt.plot(utility_time, -np.sin(utility_time) + -5 * np.sin(5 * utility_time), '--',
                         label='derivative')
                plt.legend(loc='lower left', fontsize=6)
                # print(f'Energy of time series: {utility.energy()}')
                plt.show()

            utility_time_series_bin_av = np.sin(utility_time) + np.random.normal(0, 0.5, len(utility_time))
            utility_bin = Utility(time=utility_time, time_series=utility_time_series_bin_av)
            if plot:
                plt.title('Binomial Averaging')
                plt.plot(utility_time, utility_time_series_bin_av, label='time series')
                plt.plot(utility_time, utility_bin.binomial_average(order=15), label='binomial average')
                plt.legend(loc='lower left', fontsize=6)
                plt.show()

            if not test_all:
                print('emd_utility.py all tests passed.')
            else:
                print('emd_utility.py all tests passed.')
                return True

        except:
            if not test_all:
                print('emd_utility.py some tests failed.')
            else:
                return False

    def try_break_emd(self, test_all=False, plot=False):

        try:
            # too many knots test
            time_test = np.linspace(0.5, 9.5, 1901)
            time_series_test = np.cos(2 * time_test) + np.cos(4 * time_test)
            emd_test = self.emd(time=time_test, time_series=time_series_test)
            knots = np.linspace(-2, 12, 31)
            emd_test_sift = emd_test.empirical_mode_decomposition(knots=knots, verbose=False)
            if plot:
                plt.title('Extended Knot Sequence Test')
                for i in np.asarray(knots):
                    plt.plot(i * np.ones(100), np.linspace(-1, 1, 100), '--')
                plt.plot(time_test, time_series_test, label='Time series')
                plt.plot(time_test, emd_test_sift[0][0, :], label='Smoothed time series')
                plt.plot(time_test, emd_test_sift[0][1:, :].T, label='IMFS')
                plt.show()

            # test stopping criteria
            if plot:
                plt.title('Stopping Criteria Test')
            for crit in ['sd', 'sd_11a', 'sd_11b', 'mft', 'edt', 'S_stoppage']:
                emd_test_sift_imfs = \
                    emd_test.empirical_mode_decomposition(knots=knots, stop_crit=crit)[0][1:, :]
                if plot:
                    plt.plot(emd_test_sift_imfs.T, label=f'{crit} stopping criterion')
                    plt.legend(loc='lower left', fontsize=6)
            if plot:
                plt.show()

            if not test_all:
                print('AdvEMDpy.py all tests passed.')
            else:
                print('AdvEMDpy.py all tests passed.')
                return True

        except:
            if not test_all:
                print('AdvEMDpy.py some tests failed.')
            else:
                return False


class EMDSpeedTests:

    def __init__(self, emd):

        self.emd = emd
        self.PyEMD0215 = pyemd0215
        self.emd040 = emd040

    def test_speed(self):

        time_series = np.cos(5 * np.linspace(0, 5 * np.pi, 1001)) + np.cos(np.linspace(0, 5 * np.pi, 1001))

        empirical_mode_decomposition = self.emd(time_series=time_series)
        start_advemdpy = time.time()
        for i in range(10):
            _, _, _, _, _, _, _ = empirical_mode_decomposition.empirical_mode_decomposition(matrix=True,
                                                                                            verbose=False,
                                                                                            stop_crit_threshold=1000000,
                                                                                            smooth=True,
                                                                                            edge_effect='none')
        end_advemdpy = time.time()
        print(f'AdvEMDpy 0.0.1 runtime: {np.round((end_advemdpy - start_advemdpy) / 10, 5)} seconds')

        pyemd = pyemd0215()
        start_py_emd = time.time()
        for i in range(10):
            _ = pyemd(time_series)
        end_py_emd = time.time()
        print(f'pyemd 0.2.10 runtime: {np.round((end_py_emd - start_py_emd) / 10, 5)} seconds and '
              f'is {np.round((end_advemdpy - start_advemdpy) / (end_py_emd - start_py_emd))} times faster')

        start_emd = time.time()
        for i in range(10):
            _ = emd040.sift.sift(time_series)
            emd040.spectra.frequency_transform(_, (1001 / (5 * np.pi)), 'hilbert')
        end_emd = time.time()
        print(f'emd 0.3.3 runtime: {np.round((end_emd - start_emd) / 10, 5)} seconds and '
              f'is {np.round((end_advemdpy - start_advemdpy) / (end_emd - start_emd))} times faster')


if __name__ == "__main__":

    test_emd = EMDUnitTests(EMD, Basis, Hilbert, Fluctuation, Preprocess, Utility)
    test_emd.test_all()
    # test_emd.test_emd_basis(plot=True)
    # test_emd.test_emd_hilbert(plot=True)
    # test_emd.test_emd_mean(plot=True)
    # test_emd.test_emd_preprocess(plot=True)
    # test_emd.test_emd_utility(plot=True)
    # test_emd.try_break_emd(plot=True)
    EMDSpeedTests(EMD).test_speed()
