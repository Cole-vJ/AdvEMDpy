
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

Preamble

The GNU General Public License is a free, copyleft license for software and other kinds of works.

The licenses for most software and other practical works are designed to take away your freedom to share and change the works. By contrast, the GNU General Public License is intended to guarantee your freedom to share and change all versions of a program--to make sure it remains free software for all its users. We, the Free Software Foundation, use the GNU General Public License for most of our software; it applies also to any other work released this way by its authors. You can apply it to your programs, too.

When we speak of free software, we are referring to freedom, not price. Our General Public Licenses are designed to make sure that you have the freedom to distribute copies of free software (and charge for them if you wish), that you receive source code or can get it if you want it, that you can change the software or use pieces of it in new free programs, and that you know you can do these things.

To protect your rights, we need to prevent others from denying you these rights or asking you to surrender the rights. Therefore, you have certain responsibilities if you distribute copies of the software, or if you modify it: responsibilities to respect the freedom of others.

For example, if you distribute copies of such a program, whether gratis or for a fee, you must pass on to the recipients the same freedoms that you received. You must make sure that they, too, receive or can get the source code. And you must show them these terms so they know their rights.

Developers that use the GNU GPL protect your rights with two steps: (1) assert copyright on the software, and (2) offer you this License giving you legal permission to copy, distribute and/or modify it.

For the developers' and authors' protection, the GPL clearly explains that there is no warranty for this free software. For both users' and authors' sake, the GPL requires that modified versions be marked as changed, so that their problems will not be attributed erroneously to authors of previous versions.

Some devices are designed to deny users access to install or run modified versions of the software inside them, although the manufacturer can do so. This is fundamentally incompatible with the aim of protecting users' freedom to change the software. The systematic pattern of such abuse occurs in the area of products for individuals to use, which is precisely where it is most unacceptable. Therefore, we have designed this version of the GPL to prohibit the practice for those products. If such problems arise substantially in other domains, we stand ready to extend this provision to those domains in future versions of the GPL, as needed to protect the freedom of users.

Finally, every program is threatened constantly by software patents. States should not allow patents to restrict development and use of software on general-purpose computers, but in those that do, we wish to avoid the special danger that patents applied to a free program could make it effectively proprietary. To prevent this, the GPL assures that patents cannot be used to render the program non-free.

The precise terms and conditions for copying, distribution and modification follow.

TERMS AND CONDITIONS
0. Definitions.

“This License” refers to version 3 of the GNU General Public License.

“Copyright” also means copyright-like laws that apply to other kinds of works, such as semiconductor masks.

“The Program” refers to any copyrightable work licensed under this License. Each licensee is addressed as “you”. “Licensees” and “recipients” may be individuals or organizations.

To “modify” a work means to copy from or adapt all or part of the work in a fashion requiring copyright permission, other than the making of an exact copy. The resulting work is called a “modified version” of the earlier work or a work “based on” the earlier work.

A “covered work” means either the unmodified Program or a work based on the Program.

To “propagate” a work means to do anything with it that, without permission, would make you directly or secondarily liable for infringement under applicable copyright law, except executing it on a computer or modifying a private copy. Propagation includes copying, distribution (with or without modification), making available to the public, and in some countries other activities as well.

To “convey” a work means any kind of propagation that enables other parties to make or receive copies. Mere interaction with a user through a computer network, with no transfer of a copy, is not conveying.

An interactive user interface displays “Appropriate Legal Notices” to the extent that it includes a convenient and prominently visible feature that (1) displays an appropriate copyright notice, and (2) tells the user that there is no warranty for the work (except to the extent that warranties are provided), that licensees may convey the work under this License, and how to view a copy of this License. If the interface presents a list of user commands or options, such as a menu, a prominent item in the list meets this criterion.

1. Source Code.

The “source code” for a work means the preferred form of the work for making modifications to it. “Object code” means any non-source form of a work.

A “Standard Interface” means an interface that either is an official standard defined by a recognized standards body, or, in the case of interfaces specified for a particular programming language, one that is widely used among developers working in that language.

The “System Libraries” of an executable work include anything, other than the work as a whole, that (a) is included in the normal form of packaging a Major Component, but which is not part of that Major Component, and (b) serves only to enable use of the work with that Major Component, or to implement a Standard Interface for which an implementation is available to the public in source code form. A “Major Component”, in this context, means a major essential component (kernel, window system, and so on) of the specific operating system (if any) on which the executable work runs, or a compiler used to produce the work, or an object code interpreter used to run it.

The “Corresponding Source” for a work in object code form means all the source code needed to generate, install, and (for an executable work) run the object code and to modify the work, including scripts to control those activities. However, it does not include the work's System Libraries, or general-purpose tools or generally available free programs which are used unmodified in performing those activities but which are not part of the work. For example, Corresponding Source includes interface definition files associated with source files for the work, and the source code for shared libraries and dynamically linked subprograms that the work is specifically designed to require, such as by intimate data communication or control flow between those subprograms and other parts of the work.

The Corresponding Source need not include anything that users can regenerate automatically from other parts of the Corresponding Source.

The Corresponding Source for a work in source code form is that same work.

2. Basic Permissions.

All rights granted under this License are granted for the term of copyright on the Program, and are irrevocable provided the stated conditions are met. This License explicitly affirms your unlimited permission to run the unmodified Program. The output from running a covered work is covered by this License only if the output, given its content, constitutes a covered work. This License acknowledges your rights of fair use or other equivalent, as provided by copyright law.

You may make, run and propagate covered works that you do not convey, without conditions so long as your license otherwise remains in force. You may convey covered works to others for the sole purpose of having them make modifications exclusively for you, or provide you with facilities for running those works, provided that you comply with the terms of this License in conveying all material for which you do not control copyright. Those thus making or running the covered works for you must do so exclusively on your behalf, under your direction and control, on terms that prohibit them from making any copies of your copyrighted material outside their relationship with you.

Conveying under any other circumstances is permitted solely under the conditions stated below. Sublicensing is not allowed; section 10 makes it unnecessary.

3. Protecting Users' Legal Rights From Anti-Circumvention Law.

No covered work shall be deemed part of an effective technological measure under any applicable law fulfilling obligations under article 11 of the WIPO copyright treaty adopted on 20 December 1996, or similar laws prohibiting or restricting circumvention of such measures.

When you convey a covered work, you waive any legal power to forbid circumvention of technological measures to the extent such circumvention is effected by exercising rights under this License with respect to the covered work, and you disclaim any intention to limit operation or modification of the work as a means of enforcing, against the work's users, your or third parties' legal rights to forbid circumvention of technological measures.

4. Conveying Verbatim Copies.

You may convey verbatim copies of the Program's source code as you receive it, in any medium, provided that you conspicuously and appropriately publish on each copy an appropriate copyright notice; keep intact all notices stating that this License and any non-permissive terms added in accord with section 7 apply to the code; keep intact all notices of the absence of any warranty; and give all recipients a copy of this License along with the Program.

You may charge any price or no price for each copy that you convey, and you may offer support or warranty protection for a fee.

5. Conveying Modified Source Versions.

You may convey a work based on the Program, or the modifications to produce it from the Program, in the form of source code under the terms of section 4, provided that you also meet all of these conditions:

a) 	The work must carry prominent notices stating that you modified it, and giving a relevant date.
b)	 The work must carry prominent notices stating that it is released under this License and any conditions added under section 7. This requirement 		modifies the requirement in section 4 to “keep intact all notices”.
c) 	You must license the entire work, as a whole, under this License to anyone who comes into possession of a copy. This License will therefore apply, 		along with any applicable section 7 additional terms, to the whole of the work, and all its parts, regardless of how they are packaged. This License 		gives no permission to license the work in any other way, but it does not invalidate such permission if you have separately received it.
d) 	If the work has interactive user interfaces, each must display Appropriate Legal Notices; however, if the Program has interactive interfaces that do 		not display Appropriate Legal Notices, your work need not make them do so.

A compilation of a covered work with other separate and independent works, which are not by their nature extensions of the covered work, and which are not combined with it such as to form a larger program, in or on a volume of a storage or distribution medium, is called an “aggregate” if the compilation and its resulting copyright are not used to limit the access or legal rights of the compilation's users beyond what the individual works permit. Inclusion of a covered work in an aggregate does not cause this License to apply to the other parts of the aggregate.

6. Conveying Non-Source Forms.

You may convey a covered work in object code form under the terms of sections 4 and 5, provided that you also convey the machine-readable Corresponding Source under the terms of this License, in one of these ways:

a) 	Convey the object code in, or embodied in, a physical product (including a physical distribution medium), accompanied by the Corresponding Source 		fixed on a durable physical medium customarily used for software interchange.
b) 	Convey the object code in, or embodied in, a physical product (including a physical distribution medium), accompanied by a written offer, valid for at 		least three years and valid for as long as you offer spare parts or customer support for that product model, to give anyone who possesses the object 		code either (1) a copy of the Corresponding Source for all the software in the product that is covered by this License, on a durable physical medium 		customarily used for software interchange, for a price no more than your reasonable cost of physically performing this conveying of source, or (2) 		access to copy the Corresponding Source from a network server at no charge.
c) 	Convey individual copies of the object code with a copy of the written offer to provide the Corresponding Source. This alternative is allowed only 		occasionally and noncommercially, and only if you received the object code with such an offer, in accord with subsection 6b.
d) 	Convey the object code by offering access from a designated place (gratis or for a charge), and offer equivalent access to the Corresponding Source 		in the same way through the same place at no further charge. You need not require recipients to copy the Corresponding Source along with the                   		object code. If the place to copy the object code is a network server, the Corresponding Source may be on a different server (operated by you or a 		third party) that supports equivalent copying facilities, provided you maintain clear directions next to the object code saying where to find the 			        		Corresponding Source. Regardless of what server hosts the Corresponding Source, you remain obligated to ensure that it is available for as long as 		needed to satisfy these requirements.
e) 	Convey the object code using peer-to-peer transmission, provided you inform other peers where the object code and Corresponding Source of the 		work are being offered to the general public at no charge under subsection 6d.

A separable portion of the object code, whose source code is excluded from the Corresponding Source as a System Library, need not be included in conveying the object code work.

A “User Product” is either (1) a “consumer product”, which means any tangible personal property which is normally used for personal, family, or household purposes, or (2) anything designed or sold for incorporation into a dwelling. In determining whether a product is a consumer product, doubtful cases shall be resolved in favor of coverage. For a particular product received by a particular user, “normally used” refers to a typical or common use of that class of product, regardless of the status of the particular user or of the way in which the particular user actually uses, or expects or is expected to use, the product. A product is a consumer product regardless of whether the product has substantial commercial, industrial or non-consumer uses, unless such uses represent the only significant mode of use of the product.

“Installation Information” for a User Product means any methods, procedures, authorization keys, or other information required to install and execute modified versions of a covered work in that User Product from a modified version of its Corresponding Source. The information must suffice to ensure that the continued functioning of the modified object code is in no case prevented or interfered with solely because modification has been made.

If you convey an object code work under this section in, or with, or specifically for use in, a User Product, and the conveying occurs as part of a transaction in which the right of possession and use of the User Product is transferred to the recipient in perpetuity or for a fixed term (regardless of how the transaction is characterized), the Corresponding Source conveyed under this section must be accompanied by the Installation Information. But this requirement does not apply if neither you nor any third party retains the ability to install modified object code on the User Product (for example, the work has been installed in ROM).

The requirement to provide Installation Information does not include a requirement to continue to provide support service, warranty, or updates for a work that has been modified or installed by the recipient, or for the User Product in which it has been modified or installed. Access to a network may be denied when the modification itself materially and adversely affects the operation of the network or violates the rules and protocols for communication across the network.

Corresponding Source conveyed, and Installation Information provided, in accord with this section must be in a format that is publicly documented (and with an implementation available to the public in source code form), and must require no special password or key for unpacking, reading or copying.

7. Additional Terms.

“Additional permissions” are terms that supplement the terms of this License by making exceptions from one or more of its conditions. Additional permissions that are applicable to the entire Program shall be treated as though they were included in this License, to the extent that they are valid under applicable law. If additional permissions apply only to part of the Program, that part may be used separately under those permissions, but the entire Program remains governed by this License without regard to the additional permissions.

When you convey a copy of a covered work, you may at your option remove any additional permissions from that copy, or from any part of it. (Additional permissions may be written to require their own removal in certain cases when you modify the work.) You may place additional permissions on material, added by you to a covered work, for which you have or can give appropriate copyright permission.

Notwithstanding any other provision of this License, for material you add to a covered work, you may (if authorized by the copyright holders of that material) supplement the terms of this License with terms:

a) 	Disclaiming warranty or limiting liability differently from the terms of sections 15 and 16 of this License; or
b) 	Requiring preservation of specified reasonable legal notices or author attributions in that material or in the Appropriate Legal Notices displayed by 		works containing it; or
c) 	Prohibiting misrepresentation of the origin of that material, or requiring that modified versions of such material be marked in reasonable ways as 		different from the original version; or
d) 	Limiting the use for publicity purposes of names of licensors or authors of the material; or
e) 	Declining to grant rights under trademark law for use of some trade names, trademarks, or service marks; or
f) 	Requiring indemnification of licensors and authors of that material by anyone who conveys the material (or modified versions of it) with contractual 		assumptions of liability to the recipient, for any liability that these contractual assumptions directly impose on those licensors and authors.

All other non-permissive additional terms are considered “further restrictions” within the meaning of section 10. If the Program as you received it, or any part of it, contains a notice stating that it is governed by this License along with a term that is a further restriction, you may remove that term. If a license document contains a further restriction but permits relicensing or conveying under this License, you may add to a covered work material governed by the terms of that license document, provided that the further restriction does not survive such relicensing or conveying.

If you add terms to a covered work in accord with this section, you must place, in the relevant source files, a statement of the additional terms that apply to those files, or a notice indicating where to find the applicable terms.

Additional terms, permissive or non-permissive, may be stated in the form of a separately written license, or stated as exceptions; the above requirements apply either way.

8. Termination.

You may not propagate or modify a covered work except as expressly provided under this License. Any attempt otherwise to propagate or modify it is void, and will automatically terminate your rights under this License (including any patent licenses granted under the third paragraph of section 11).

However, if you cease all violation of this License, then your license from a particular copyright holder is reinstated (a) provisionally, unless and until the copyright holder explicitly and finally terminates your license, and (b) permanently, if the copyright holder fails to notify you of the violation by some reasonable means prior to 60 days after the cessation.

Moreover, your license from a particular copyright holder is reinstated permanently if the copyright holder notifies you of the violation by some reasonable means, this is the first time you have received notice of violation of this License (for any work) from that copyright holder, and you cure the violation prior to 30 days after your receipt of the notice.

Termination of your rights under this section does not terminate the licenses of parties who have received copies or rights from you under this License. If your rights have been terminated and not permanently reinstated, you do not qualify to receive new licenses for the same material under section 10.

9. Acceptance Not Required for Having Copies.

You are not required to accept this License in order to receive or run a copy of the Program. Ancillary propagation of a covered work occurring solely as a consequence of using peer-to-peer transmission to receive a copy likewise does not require acceptance. However, nothing other than this License grants you permission to propagate or modify any covered work. These actions infringe copyright if you do not accept this License. Therefore, by modifying or propagating a covered work, you indicate your acceptance of this License to do so.

10. Automatic Licensing of Downstream Recipients.

Each time you convey a covered work, the recipient automatically receives a license from the original licensors, to run, modify and propagate that work, subject to this License. You are not responsible for enforcing compliance by third parties with this License.

An “entity transaction” is a transaction transferring control of an organization, or substantially all assets of one, or subdividing an organization, or merging organizations. If propagation of a covered work results from an entity transaction, each party to that transaction who receives a copy of the work also receives whatever licenses to the work the party's predecessor in interest had or could give under the previous paragraph, plus a right to possession of the Corresponding Source of the work from the predecessor in interest, if the predecessor has it or can get it with reasonable efforts.

You may not impose any further restrictions on the exercise of the rights granted or affirmed under this License. For example, you may not impose a license fee, royalty, or other charge for exercise of rights granted under this License, and you may not initiate litigation (including a cross-claim or counterclaim in a lawsuit) alleging that any patent claim is infringed by making, using, selling, offering for sale, or importing the Program or any portion of it.

11. Patents.

A “contributor” is a copyright holder who authorizes use under this License of the Program or a work on which the Program is based. The work thus licensed is called the contributor's “contributor version”.

A contributor's “essential patent claims” are all patent claims owned or controlled by the contributor, whether already acquired or hereafter acquired, that would be infringed by some manner, permitted by this License, of making, using, or selling its contributor version, but do not include claims that would be infringed only as a consequence of further modification of the contributor version. For purposes of this definition, “control” includes the right to grant patent sublicenses in a manner consistent with the requirements of this License.

Each contributor grants you a non-exclusive, worldwide, royalty-free patent license under the contributor's essential patent claims, to make, use, sell, offer for sale, import and otherwise run, modify and propagate the contents of its contributor version.

In the following three paragraphs, a “patent license” is any express agreement or commitment, however denominated, not to enforce a patent (such as an express permission to practice a patent or covenant not to sue for patent infringement). To “grant” such a patent license to a party means to make such an agreement or commitment not to enforce a patent against the party.

If you convey a covered work, knowingly relying on a patent license, and the Corresponding Source of the work is not available for anyone to copy, free of charge and under the terms of this License, through a publicly available network server or other readily accessible means, then you must either (1) cause the Corresponding Source to be so available, or (2) arrange to deprive yourself of the benefit of the patent license for this particular work, or (3) arrange, in a manner consistent with the requirements of this License, to extend the patent license to downstream recipients. “Knowingly relying” means you have actual knowledge that, but for the patent license, your conveying the covered work in a country, or your recipient's use of the covered work in a country, would infringe one or more identifiable patents in that country that you have reason to believe are valid.

If, pursuant to or in connection with a single transaction or arrangement, you convey, or propagate by procuring conveyance of, a covered work, and grant a patent license to some of the parties receiving the covered work authorizing them to use, propagate, modify or convey a specific copy of the covered work, then the patent license you grant is automatically extended to all recipients of the covered work and works based on it.

A patent license is “discriminatory” if it does not include within the scope of its coverage, prohibits the exercise of, or is conditioned on the non-exercise of one or more of the rights that are specifically granted under this License. You may not convey a covered work if you are a party to an arrangement with a third party that is in the business of distributing software, under which you make payment to the third party based on the extent of your activity of conveying the work, and under which the third party grants, to any of the parties who would receive the covered work from you, a discriminatory patent license (a) in connection with copies of the covered work conveyed by you (or copies made from those copies), or (b) primarily for and in connection with specific products or compilations that contain the covered work, unless you entered into that arrangement, or that patent license was granted, prior to 28 March 2007.

Nothing in this License shall be construed as excluding or limiting any implied license or other defenses to infringement that may otherwise be available to you under applicable patent law.

12. No Surrender of Others' Freedom.

If conditions are imposed on you (whether by court order, agreement or otherwise) that contradict the conditions of this License, they do not excuse you from the conditions of this License. If you cannot convey a covered work so as to satisfy simultaneously your obligations under this License and any other pertinent obligations, then as a consequence you may not convey it at all. For example, if you agree to terms that obligate you to collect a royalty for further conveying from those to whom you convey the Program, the only way you could satisfy both those terms and this License would be to refrain entirely from conveying the Program.

13. Use with the GNU Affero General Public License.

Notwithstanding any other provision of this License, you have permission to link or combine any covered work with a work licensed under version 3 of the GNU Affero General Public License into a single combined work, and to convey the resulting work. The terms of this License will continue to apply to the part which is the covered work, but the special requirements of the GNU Affero General Public License, section 13, concerning interaction through a network will apply to the combination as such.

14. Revised Versions of this License.

The Free Software Foundation may publish revised and/or new versions of the GNU General Public License from time to time. Such new versions will be similar in spirit to the present version, but may differ in detail to address new problems or concerns.

Each version is given a distinguishing version number. If the Program specifies that a certain numbered version of the GNU General Public License “or any later version” applies to it, you have the option of following the terms and conditions either of that numbered version or of any later version published by the Free Software Foundation. If the Program does not specify a version number of the GNU General Public License, you may choose any version ever published by the Free Software Foundation.

If the Program specifies that a proxy can decide which future versions of the GNU General Public License can be used, that proxy's public statement of acceptance of a version permanently authorizes you to choose that version for the Program.

Later license versions may give you additional or different permissions. However, no additional obligations are imposed on any author or copyright holder as a result of your choosing to follow a later version.

15. Disclaimer of Warranty.

THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM “AS IS” WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

16. Limitation of Liability.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

17. Interpretation of Sections 15 and 16.

If the disclaimer of warranty and limitation of liability provided above cannot be given local legal effect according to their terms, reviewing courts shall apply local law that most closely approximates an absolute waiver of all civil liability in connection with the Program, unless a warranty or assumption of liability accompanies a copy of the Program in return for a fee.
