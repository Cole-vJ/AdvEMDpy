
# Python EMD Package

EMD is a Python library that performs Empirical Mode Decomposition with numerous algorithmic variations and other optional additions.



## Installation

```bash
pip install AdvEMDpy
```



## Usage

```python
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
imfs, hts, ifs = emd.empirical_mode_decomposition(knots=sample_knots, knot_time=sample_knot_time)

plt.figure(1)
plt.title('Sample EMD of First Component')
plt.plot(sample_time, np.cos(5 * sample_time))
plt.plot(sample_time, imfs[1, :], '--')
plt.show()

plt.figure(2)
plt.title('Sample EMD of Second Component')
plt.plot(sample_time, np.cos(sample_time))
plt.plot(sample_time, imfs[2, :], '--')
plt.show()
```



## Output

### Text Output

```python
IMF_11 ALL IMF CONDITIONS MET
IMF_21 ALL IMF CONDITIONS MET

Process finished with exit code 0
```



### Figure Output

![](./README_Images/Figure_1.png)

![](./README_Images/Figure_2.png)



### Class EMD

#### Parameters

time_series : real ndarray (required)

​	Time series to be decomposed using EMD.



### Method empirical_mode_decomposition

#### Parameters

smooth : boolean
    Whether or not envelope smoothing takes place - Statistical Empirical Mode Decomposition (SEMD) from (1).

smoothing_penalty : float
    Penalty to be used when smoothing - Statistical Empirical Mode Decomposition (SEMD).

edge_effect : string
    What technique is used for the edges of the envelopes to not propagate errors:

- 'symmetric' : reflect extrema with no anchored extrema at reflection point.
- 'symmetric_anchor' : reflect extrema with forced extrema at reflection point depending on alpha value - alpha value of 1 is equivalent to (2) and (3).
- 'symmetric_discard' : reflect extrema about last extrema - discards ends of signal from (4) and (5).
- 'anti-symmetric' : reflects about end point on both axes - reflected about x = x_end_point and y = y_end_point - modified version of (2).
- 'characteristic_wave_Huang' : calculate characteristic wave (sinusoid) using first/last two maxima or minima from (6).
- 'charcteristic_wave_Coughlin' : calculate characteristic wave (sinusoid) using first/last maximum and minimum from (7).
- 'slope_based_method' : calculate extrema using slopes between extrema and time difference from (8).
- 'improved_slope_based_method' : calculate extrema using slopes betweeen extrema and time difference takes into account end points and possibly anchors them from (5).
- 'average' : averages the last two extrema and uses time difference to repeat pattern from (9).
- 'neural_network' : uses a single neuron neural network to explicitly extrapolate the whole time series to approximate next extrema from (10).
- 'none' : no edge-effect considered.

sym_alpha : float
    Value (alpha) applied to conditional symmetric edge effect.

stopping_criterion : string
    What stopping criterion to use in the internal sifting loop:

- 'sd' : standard deviation stopping criterion from (6).
- 'sd_11a' : standard deviation stopping criterion from (11).
- 'sd_11b' : standard deviation stopping criterion from (11).
- 'mft' : mean fluctuation threshold stopping criterion from (4) and (12).
- 'edt' : energy difference tracking stopping criterion from (13).
- 'S_stoppage' : stoppage criterion based on number of successive IMF candidates with the same number of extrema and zero-crossings from (11).

stopping_criterion_threshold : float
    What threshold is used for whatever stopping criterion we use in stopping_criterion.

mft_theta_1 : float
    Mean fluctuation threshold (theta_1) value used in stopping criterion.

mft_theta_2 : float
    Mean fluctuation threshold (theta_2) value used in stopping criterion.

mft_alpha : float
    Mean fluctuation threshold (alpha) value used in stopping criterion.

mean_threshold : float
    What threshold is used for the difference for the mean_envelope from zero.

debug : boolean
    If debugging, this displays every single incremental IMF with corresponding extrema, envelopes, and mean.

text : boolean
    Whether or not to print success or failure of various criteria - 2 IMF conditions, stopping criteria, etc.

spline_method : string
    Spline method to use for smoothing and envelope sifting.

smooth_if : boolean
    If True, removes undesirable discontinuities.

smooth_ht : boolean
    If True, removes undesirable discontinuities from Hilbert transform.

dtht : boolean
    If True, performs discrete-time Hilbert transform of IMFs.

dtht_method : string
    Whether to use Basic DTHT ('kak') or FFT DTHT ('fft').

max_internal_iter : integer (positive)
    Additional stopping criterion - hard limit on number of internal siftings.

max_imfs : integer (positive)
    Hard limit on number of external siftings.

matrix : boolean
    If true, constructs cubic-basis spline matrix once at outset - greatly increases speed. Important: Overrides 'spline_method' choice.

initial_smoothing : boolean
    If true, performs initial smoothing and associated co-efficient fitting. 
	If False, no initial smoothing is done - dtht is used to get Hilbert transform of IMF 1.

detrended_fluctuation_technique : string
	What technique is used to estimate local mean of signal:

- 'envelopes' : fits cubic spline to both maxima and minima and calculates mean by averaging envelopes from (6).
- 'inflection_points' : estimates local mean by fitting cubic spline through inflection points from (14).
- 'binomial_average' : estimates local mean by taking binmoial average of surrounding points and interpolating from (15). Important: difficult to extract mean accurately on lower frequency structures - replicates time series too closely.
- 'enhanced' : performs Enhanced EMD on derivative of signal (or IMF candidate) to approximate extrema locations of highest frequency component which are better interpolation points for extrema envelopes from (16).

order : integer (odd positive)
    The number of points to use in binomial averaging. If order=5, then weighting vector will be (1/16, 4/16, 6/16, 4/16, 1/16) centred on selected points.

increment : integer (positive)
    The incrementing of binomial averaging. If increment=10, then point used will have indices: 0, 10, 20, etc.

preprocess : string
    What preprocessing technique to use (if at all) - effective when dealing with heavy-tailed and mixed noise:

- 'median_filter' : impose a median filter on signal - very robust when dealing with outliers.
- 'mean_filter' : impose a mean filter on signal - more susceptible to outliers.
- 'winsorize' : use quantile filters for upper and lower confidence intervals and set time series values equal to upper or lower quantile values when time series exceeds these limits.
- 'winsorize_interpolate' : use quantile filters for upper and lower confidence intervals and interpolate time series values that are discarded when time series exceeds these limits.
- 'HP' : use generalised Hodrick-Prescott filtering.
- 'HW' : use Henderson-Whittaker smoothing.
- 'none' : perform no preprocessing.

preprocess_window_length : integer (odd positive)
    Window length to use when preprocessing signal - should be odd as requires original point at centre.

preprocess_quantile : float
    Confidence level to use when using 'winsorize' or 'winsorize_interpolate' preprocessing techniques.

preprocess_penalty : float
    Penalty to use in generalised Hodrick-Prescott filter. Original HP filter - preprocess_penalty = 1.

preprocess_order : integer
    if preprocess = 'HP':
        Order of smoothing to be used in generalised Hodrick-Prescott filter. Original HP filter -
        preprocess_order = 2.
    if preprocess = 'HW':
        Width of Henderson-Whittaker window used to calculate weights for smoothing.

preprocess_norm_1 : integer
    Norm to be used on curve fit for generalised Hodrick-Prescott filter. Original HP filter -
    preprocess_norm_1 = 2.

preprocess_norm_2 : integer
    Norm to be used on smoothing order for generalised Hodrick-Prescott filter. Original HP filter -
    preprocess_norm_2 = 2.

ensemble : boolean
    Whether or not to use Ensemble Empirical Mode Decomposition routine from (17).

ensemble_sd : float
    Fraction of standard deviation of detreneded signal to used when generated noise assisting noise.

ensemble_iter : integer (positve)
    Number of iterations to use when performing Ensemble Empirical Mode Decomposition.

enhanced_iter : integer (positve)
    Number of internal iterations to use when performing Enhanced Empirical Mode Decomposition.

output_coefficients : boolean
    Optionally output coefficients corresponding to B-spline IMFs. Increases outputs to 4.

optimise_knots : boolean
    Optionally optimise knots.

knot_optimisation_method : string
    Knot point optimisation method to use:
    	'bisection' - bisects until error condition is met,
    	'serial_bisection' - bisects until error condition met - extends, re-evaluates.

output_knots : boolean
    Optionally output knots - only relevant when optionally optimised.
    Increases ouputs to 4 or 5 depending if coefficients are outputted.

downsample_window : string
    Window to use when downsampling.

downsample_decimation_factor : integer (positive)
    Decimation level when downsampling. Product of downsample_decimation_factor and downsample_window_factor must be even.

downsample_window_factor : integer (positive)
    Downsampling level when downsampling. Product of downsample_decimation_factor and downsample_window_factor must be even.

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



#### Returns

intrinsic_mode_function_storage : real ndarray
    Matrix containing smoothed original (optional) signal in intrinsic_mode_function_storage[0, :] followed by IMFs and trend in successive rows.

intrinsic_mode_function_storage_ht : real ndarray
    Matrix containing HT of smoothed original signal in intrinsic_mode_function_storage_ht[0, :] (not used, but included for consistency) followed by HTs of 	IMFs and trend in successive rows.

intrinsic_mode_function_storage_if : real ndarray
    Matrix containing IF of smoothed original signal in intrinsic_mode_function_storage_if[0, :] (not used, but included for consistency) followed by IFs 
	of IMFs and trend in successive rows.

intrinsic_mode_function_storage_coef : real ndarray (optional)
    Matrix containing B-spline coefficients corresponding to spline curves in intrinsic_mode_function_storage.

knot_envelope : real ndarray (optional)
    Vector containing (possibly optimised) knots.

intrinsic_mode_function_storage_dt_ht : real ndarray (optional)
    Discrete-time Hilbert transform.

intrinsic_mode_function_storage_dt_if : real ndarray (optional)
    Instantaneous frequency using discrete-time Hilbert transform.



## Contributing

This project is by no means complete or exhaustive.



## Algorithm Summary


	(1) maxima non-empty
	(2) minima non-empty
	(3) total IMF count < max allowed IMF count
	(4) numerical error IMF > limit
	
	(5) any maxima are negative
	(6) any minima are positive
	(7) sum(abs(local mean)) > mean threshold
	
	(5) - (7) core non-IMF requirements
	
	(8) internal iteration count < max allowed internal count
	
	(9) all maxima are positive
	(10) all minima are negative
	(11) sum(abs(local mean)) < mean threshold
	
	(9) - (11) core IMF requirements
	
	if not ensemble:
	
	while (1) and (2) and (3) and (4):
	
		while [(5) or (6) or (7)] and (8) and (3) and (4):
	
			if extrema count > 2:
	
				check stopping criterion.
				STORE or recalculate
	
			else:
	
				global mean extracted,
				STORE IMF,
				maxima and minima empty,
				local mean set to zero.
		
		if (9) and (10) and (11) and (4):
		
			print('All conditions met.').
	
		if (4):
	
			STORE IMF and recalculate.
	
	if (4):
	
		STORE IMF.



## References

(1)   D. Kim, K. Kim, and H. Oh. Extending the scope of empirical mode decomposition by
		smoothing. EURASIP Journal on Advances in Signal Processing, 2012(168):1–17,
		2012.

(2)	K. Zeng and M. He. A Simple Boundary Process Technique for Empirical Mode
        Decomposition. In IEEE International Geoscience and Remote Sensing Symposium,
        volume 6, pages 4258–4261. IEEE, 2004.

(3)	J. Zhao and D. Huang. Mirror Extending and Circular Spline Function for Empirical
        Mode Decomposition Method. Journal of Zhejiang University - Science A, 2(3):
        247–252, 2001.

(4)	G. Rilling, P. Flandrin, and P. Goncalves. On Empirical Mode Decomposition and its
        Algorithms. In IEEE-EURASIP Workshop on Nonlinear Signal and Image Process-
        ing, volume 3, pages 8–11. NSIP-03, Grado (I), 2003.

(5)	F. Wu and L Qu. An improved method for restraining the end effect in empirical mode
        decomposition and its applications to the fault diagnosis of large rotating machinery.
        Journal of Sound and Vibration, 314(3-5):586–602, 2008. doi: 10.1016/j.jsv.2008.
        01.020.

(6)	N. Huang, Z. Shen, S. Long, M. Wu, H. Shih, Q. Zheng, N. Yen, C. Tung, and H. Liu.
        The Empirical Mode Decomposition and the Hilbert Spectrum for Nonlinear and
        Non-Stationary Time Series Analysis. Proceedings of the Royal Society of London
        A, 454:903–995, 1998.

(7)	K. Coughlin and K. Tung. 11-Year solar cycle in the stratosphere extracted by the
        empirical mode decomposition method. Advances in Space Research, 
		34(2):323–329, 2004. doi: 10.1016/j.asr.2003.02.045.

(8)	M. Dätig and T. Schlurmann. Performance and limitations of the hilbert-huang trans-
        formation (hht) with an application to irregular water waves. Ocean Engineering,
        31(14-15):1783–1834, 2004.

(9)    F. Chiew, M. Peel, G. Amirthanathan, and G. Pegram. Identification of oscillations in
        historical global streamflow data using empirical mode decomposition. In Regional
        Hydrological Impacts of Climatic Change - Hydroclimatic Variabiltiy, volume 296,
        pages 53–62. International Association of Hydrological Sciences, 2005.

(10) Y. Deng, W. Wang, C. Qian, Z. Wang, D. Dai (2001). “Boundary-Processing-Technique in 
		EMD Method and Hilbert Transform.”Chinese Science Bulletin,46(1), 954–960.

(11) N. Huang and Z. Wu. A review on Hilbert-Huang transform: Method and its appli-
        cations to geophysical studies. Reviews of Geophysics, 46(RG2006):1–23, 2008. doi:
        10.1029/2007RG000228.

(12) A. Tabrizi, L. Garibaldi, A. Fasana, and S. Marchesiello. Influence of Stopping Crite-
        rion for Sifting Process of Empirical Mode Decomposition (EMD) on Roller Bear-
        ing Fault Diagnosis. In Advances in Condition Monitoring of Machinery in Non-
        Stationary Operations, pages 389–398. Springer-Verlag, Berlin Heidelberg, 2014.

(13) C. Junsheng, Y. Dejie, and Y. Yu. Research on the Intrinsic Mode Function (IMF)
        Criterion in EMD Method. Mechanical Systems and Signal Processing, 20(4):817–
        824, 2006.

(14) Y. Kopsinis and S. McLaughlin. Investigation of the empirical mode decomposition
        based on genetic algorithm optimization schemes. In Proceedings of the 32nd IEEE
        International Conference on Acoustics, Speech and Signal Processing (ICASSP’07),
        volume 3, pages 1397–1400, Honolulu, Hawaii, United States of America, 2007.
        IEEE.

(15) Q. Chen, N. Huang, S. Riemenschneider, and Y. Xu. A B-spline Approach for Em-
        pirical Mode Decompositions. Advances in Computational Mathematics, 24(1-4):
        171–195, 2006.

(16) Y. Kopsinis and S. McLaughlin. Enhanced Empirical Mode Decomposition using a
        Novel Sifting-Based Interpolation Points Detection. In Proceedings of the IEEE/SP
        14th Workshop on Statistical Signal Processing (SSP’07), pages 725–729, Madison,
        Wisconsin, United States of America, 2007. IEEE Computer Society.

(17) Z. Wu and N. Huang. Ensemble Empirical Mode Decomposition: a noise-assisted data
		analysis method. Advances in Adaptive Data Analysis, 1(1):1–41, 2009.



## Licence

