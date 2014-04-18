import sys
import math
import re

import numpy
import scipy.stats
import statsmodels.stats.diagnostic
import statsmodels.api as sm
import pylab as plt

def ks(data):
	
	''' Perform special case of the Kolmogorov-Smirnov Test for normality known as Lilliefors Test
	    (i.e. case of unknown mean and variance -- the expected value and variance of the distribution
	    are not specified); null hypothesis does not specify which normal distribution. '''

	normal = 0
	#(D, p) = scipy.stats.kstest(data, 'norm')
	(D, p) = statsmodels.stats.diagnostic.lillifors(data)
	#print '\nN = {0}'.format(len(data))
	#print 'The test statistic is: {0}'.format(D)
	#print 'The p-value for the hypothesis test is: {0}'.format(p)

	if p > 0.05:
		#print 'The data tests normal'
		normal = 1
	else:
		#print 'The data does not test normal'
		normal = 0

	return normal			

def chisquare(data):
	
	''' Perform the Chi-Squared Test for normality. ''' 

	normal = 0
	(tstat, p) = scipy.stats.chisquare(data)
	#print '\nN = {0}'.format(len(data))
	#print 'The test statistic is: {0}'.format(tstat)
	#print 'The p-value for the hypothesis test is: {0}'.format(p)

	if p > 0.05:
		#print 'The data tests normal'
		normal = 1
	else:
		#print 'The data does not test normal'
		normal = 0

	return normal

def anderson(data):
	
	''' Perform the Anderson-Darling Test for normality. '''

	normal = 0
	(result) = scipy.stats.anderson(data, dist='norm')
	A2 = result[0]
	critvals  = result[1]
	siglevels = result[2]
	#print '\nN = {0}'.format(len(data))
	#print 'The test statistic is: {0}'.format(A2)
	#print 'Critical values: '
	#for i in critvals:
	#	print '{0} '.format(i)
	#print 'Significance levels: '
	#for i in siglevels:
	#	print '{0} '.format(i)
	if A2 < critvals[2]:
		#print 'The data tests normal at the 5% significance level ({0:.3g} < {1})'.format(A2, critvals[2])
		normal = 1
	else:
		#print 'The data does not test normal at the 5% significance level ({0:.3g} > {1})'.format(A2, critvals[2])
		normal = 0

	return normal

def wilks(data):
	
	''' Perform the Wilks-Shapiro Test for normality. Can only be used for sample sizes between 3 and 50. '''

	normal = 0
	(w, p) = scipy.stats.shapiro(data)
	#print '\nN = {0}'.format(len(data))
	#print 'The test statistic is: {0}'.format(w)
	#print 'The p-value for the hypothesis test is: {0}'.format(p)

	if p < 0.05:
		#print 'The data does not test normal'
		normal = 0
	else:
		#print 'Not enough evidence to say the data is not normal'
		normal = 1

	return normal

def dagostino(data):
	
	''' Perform D'Agostino and Pearson's Test for normality. Used as an omnibus test of normality.
	    Sample size must be greater than 20. '''

	normal = 0
	(k2, p) = scipy.stats.normaltest(data)
	#print '\nN = {0}'.format(len(data))
	#print 'The test statistic is: {0}'.format(k2)
	#print 'The p-value for the hypothesis test is: {0}'.format(p)

	if p < 0.05:
		#print 'The data does not test normal'
		normal = 0
	else:
		#print 'Not enough evidence to say the data is not normal'
		normal = 1

	return normal	

def normtest(a):

	''' Runs four normality tests given a 1-D array of data and determines the number of normality tests passed. '''

	count = []
	
	if len(a) <= 50:
		count.append(wilks(a))
	else:
		count.append(dagostino(a))
	count.append(anderson(a))
	count.append(chisquare(a))
	count.append(ks(a))

	sum = 0
	for each in count:
		sum += each

	return sum	

def normplot(a, outp):
	
	''' Plots histograms describing the PDF and CDF corresponding to a 1-D array of data and saves the figure as a PNG file. '''

	# Regular expression that determines the input filename minus the extension to use as the base for the figure filename.
	name_base = re.search(r'\b(.+)\b\.', outp)

	# Set the figure size and generate the PDF plot canvas.
	fig = plt.figure(figsize=(12, 6))
	p = plt.subplot(121)

	# Determine how many normality tests were passed.
	count = normtest(a)
	
	# If 2 or more of the 4 normality tests are passed, decide that data is normally distributed; otherwise, decide that it is not.
	if count >= 2:
		p.set_title('PDF\n(Data appears to be normal...)', fontsize=12)
	elif count < 2:
		p.set_title('PDF\n(Data doesn\'t appear to be normal...)', fontsize=12)

	# Add labels to the PDF plot.
	p.set_xlabel('K-eff')
	p.set_ylabel('Frequency')

	# Plot a histogram of the data (the PDF).
	count, bins, ignored = plt.hist(a, normed=True, histtype='stepfilled', alpha=0.5)

	mu = numpy.mean(a)
	sigma = numpy.std(a)
	plt.step(bins, 1/(sigma * numpy.sqrt(2 * numpy.pi)) * numpy.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='g')
	
	# Generate the CDF plot canvas and add labels.
	p = plt.subplot(122) 
	p.set_title('CDF', fontsize=12)
	p.set_xlabel('K-eff')
	p.set_ylabel('Frequency') 
	
	# Plot a histogram of the data (the CDF).
	#h = plt.hist(a, histtype='step', cumulative=True, normed=True, bins=100)
	ecdf = sm.distributions.ECDF(a)
	x = numpy.linspace(min(a), max(a))
	y = ecdf(x)
	plt.step(x, y)

	s = numpy.random.normal(mu, sigma, 10000)
	ecdf = sm.distributions.ECDF(s)
	x = numpy.linspace(min(a), max(a))
	y = ecdf(x)
	plt.step(x, y)

	#plt.show()

	# Save the figure of the PDF and CDF plots as a PNG file with the input file name.
	plt.savefig(name_base.group(1) + '_pdf_cdf.png', bbox_inches='tight')

def checknorm(data, sig, outp):

	''' Determine what analysis path is taken based on whether or not the data tests as normally distributed. '''

	# Determine the input file name for use as the base for the output file name; get ready to write some output to the ouput file.
	name_base = re.search(r'\b(.+)\b\.', outp)
	fname = open(name_base.group(1) + '.out', 'w')

	# Determine how many normality tests were passed.
	count = normtest(data)
	
	# Determine the weighted mean k-eff given a 1-D array of data and corresponding 1-D array of 1-sigma error values.
	k_bar = wtd_avg(data, sig)
	
	# Determine the one-sided lower tolerance limit.
	K_L = ss_tol_lim(data, sig)

	# If there is indication that the data is normal, determine the bias and bias uncertainty for the case assuming no trends in the data.
	if count >= 2:
		fname.write('\n -----------------------------------------------------------------\n')
		print '\n -----------------------------------------------------------------'
		fname.write('| {0}/4 Normality Tests Passed For The Dataset => Assume Normality |\n'.format(count))
		print '| {0}/4 Normality Tests Passed For The Dataset => Assume Normality |'.format(count)
		fname.write(' -----------------------------------------------------------------\n')
		print ' -----------------------------------------------------------------'
		fname.write('\nBias & Bias Uncertainty (Assumptions: No Trend, Normality, No Extrapolation)\n')
		print '\nBias & Bias Uncertainty (Assumptions: No Trend, Normality, No Extrapolation)'
		fname.write('----------------------------------------------------------------------------\n')
		print '----------------------------------------------------------------------------'
		fname.write('\nWARNING: A Positive Bias Is Non-Conservative!\n')
		print '\nWARNING: A Positive Bias Is Non-Conservative!'
		fname.write('\nBias:             {0:-.5f}\n'.format(k_bar - 1))
		print '\nBias:             {0:-.5f}'.format(k_bar - 1)
		fname.write('Bias Uncertainty:  {0:-.5f}\n'.format(k_bar - K_L))
		print 'Bias Uncertainty:  {0:-.5f}'.format(k_bar - K_L)
		fname.close()
		# Flag for use in the 'analyze' function; this means trending will be done.
		normal = 1
	# If there is no indication that the data is normal, print a warning.
	elif count < 2:
		fname.write('\n -----------------------------------------------------------------------\n')
		print '\n -----------------------------------------------------------------------'	
		fname.write('| {0}/4 Normality Tests Passed For The Dataset => DO NOT Assume Normality |\n'.format(count))
		print '| {0}/4 Normality Tests Passed For The Dataset => DO NOT Assume Normality |'.format(count)
		fname.write(' -----------------------------------------------------------------------\n')
		print ' -----------------------------------------------------------------------'
		fname.close()
		# Flag for use in the 'analyze' function; this means trending will not be done since it is likely that it is inappropriate.
		normal = 0

	return normal

def wtd_avg(data, sig):

	''' Perform a weighted average of a 1-D array of data using the variance of the data as the weighting function. '''

	top = 0
	bot = 0
	for i in range(0, len(data)):
		top += data[i]/sig[i]**2
		bot += 1/sig[i]**2
	avg = top/bot

	return avg

def var(data, sig):

	''' Determines the variance about the mean of a 1-D array of data. '''

	n = float(len(data))
	kbar = wtd_avg(data, sig)
	top = 0
	bot = 0	
	for i in range(0, len(data)):
		top += (1/sig[i]**2) * (data[i] - kbar)**2
		bot += 1/sig[i]**2

	var = (1/(n-1) * top) / (1/n * bot)

	return var

def unc_avg(data, sig):

	''' Determine the average total uncertainty of a 1-D array of data. '''

	n = float(len(data))
	res = 0
	for i in range(0, len(data)):
		res += 1/sig[i]**2	
	unc_bar = n / res

	return unc_bar

def bias_unc(data, sig):
	
	''' Determine the square root of the pooled variance. This is the mean bias uncertainty when applying the single-
	    sided tolerance limit (i.e. the S_p in Eqn. 20 of NUREG/CR-6698). '''

	vpool_root = math.sqrt(var(data, sig) + unc_avg(data, sig))

	return vpool_root

def tolfac(n):

	''' Determines the tolerance factor to use in the one-sided lower tolerance limit formulation; table taken from NUREG/CR-6698. '''

	# Each entry represents the number of data points.
	x = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 35, 40, 45, 50]
	
	# Each entry represents the tolerance factor for the associated number of data points.
	y = [2.911, 2.815, 2.736, 2.670, 2.614, 2.566, 2.523, 2.486, 2.453, 2.423, 2.396, 2.371, 2.350, 2.329, 2.309, 2.292, 2.220, 2.166, 2.126, 2.092, 2.065]
	
	# If there are greater than 50 data points, apply the minimum tolerance factor.
	if float(n) >= 50:
		U = 2.065
	# If there are less than 10 data points, the code quits; shouldn't except anything with this little data.
	elif float(n) < 10:
		sys.exit('\nToo few critical experiments! Get some more and come back...')
	# If there are between 10 and 50 data points, determine the appropriate tolerance factor; linear interpolation is used if needed.
	else:
		U = numpy.interp(n, x, y)

	return U

def ss_tol_lim(data, sig):

	''' Determines the one-sided lower tolerance limit as defined in NUREG/CR-6698. '''

	n = float(len(data))
	
	U = tolfac(n)
	S_p = bias_unc(data, sig)
	k_bar = wtd_avg(data, sig)
	K_L = k_bar - U * S_p
	
	return K_L 

def npm(n):

	''' Determine if there should be any non-parametric margin applied in the non-parametric statistical treatment.
	    Uses the guidance in NUREG/CR-6698. '''

	if n <= 0.4:
		sys.exit('\nToo few critical experiments! Get some more and come back...')
	elif n < 0.5 and n > 0.4:
		m = 0.05
	elif n < 0.6 and n > 0.5:
		m = 0.04
	elif n < 0.7 and n > 0.6:
		m = 0.03
	elif n < 0.8 and n > 0.7:
		m = 0.02
	elif n <= 0.9 and n > 0.8:
		m = 0.01
	else:
		m = 0.00

	return m

def nonparam(data, sig, outp):

	''' Perform a non-parametric statistical analysis based on the guidance in NUREG/CR-6698. '''

	# Determine the input file name for use as the base for the output file name; get ready to append some output to the ouput file.
	name_base = re.search(r'\b(.+)\b\.', outp)
	fname = open(name_base.group(1) + '.out', 'a')

	# Determine the number of elements in the 1-D array of data.
	n = len(data)
	
	# Calculated beta. There is beta confidence that 95% of the population lies above the smallest observed value...
	beta = 1 - 0.95**(n)
	
	# Write some stuff to the output file. 
	fname.write('\nNon-Parametric Analysis\n')
	print '\nNon-Parametric Analysis'
	fname.write('-----------------------\n')
	print '-----------------------'
	fname.write('\nBeta = {0:4.1f}% => Should Be > 90% for NPM = 0\n'.format(beta*100))
	print '\nBeta = {0:4.1f}% => Should Be > 90% for NPM = 0'.format(beta*100)
	
	# Determine the minimum value of the 1-D array of data.
	min_k = numpy.amin(data)
	
	# Determine what element corresponds to the minimum value of the 1-D array of data.
	min_k_loc = numpy.argmin(data)
	
	# Determine the corresponding 1-sigma error associated with the minimum value of the 1-D array of data.
	min_ksig = sig[min_k_loc]
	
	# Write the bias and bias uncertainty to the output file. The 'npm' function determines the non-parametric margin
	# as a function of beta, which comes into play in our case if beta is less than 0.9; see NUREG/CR-6698 for details.
	fname.write('\nBias:             {0:-.5f}\n'.format(min_k - 1 - npm(beta)))
	print '\nBias:             {0:-.5f}'.format(min_k - 1 - npm(beta))
	fname.write('Bias Uncertainty:  {0:-.5f}\n'.format(min_ksig))
	print 'Bias Uncertainty:  {0:-.5f}'.format(min_ksig)
	fname.close()

def k_lband(x, param, data, sig, conf):

	''' Calculate the one-sided lower tolerance band for use in trending analysis. Formulation is taken from NUREG/CR-6698.'''

	# The number of elements in the 1-D array of data.
	n = float(len(data))
	# The desired confidence.
	p = conf
	# The F distribution percentile with degree of fit, n-2 degrees of freedom. The degree of fit is 2 for a linear fit.
	F = scipy.stats.f.ppf(1-p, 2, n-2, loc=0, scale=1)
	# Perform a weighted average of the 1-D array of parameter values; uses the corresponding variance as the weighting function.
	x_bar = wtd_avg(param, sig)
	# The symmetric percentile of the normal distribution that contains the P fraction.
	z = scipy.stats.norm.ppf(p, loc=0, scale=1)
	gamma = (1 - p) / 2
	# The upper Chi-squared percentile.
	chi_sq = scipy.stats.chi2.ppf(1-gamma, n-2, loc=0, scale=1)

	# Perform a linear regression and build the regression line for plotting.
	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(param, data)
	polynomial = numpy.poly1d([slope, intercept])
	k_fit = polynomial(param)
	
	# Calculate variance about the fit.
	top = 0
	bot = 0	
	for i in range(0, len(data)):
		top += (1/sig[i]**2) * (data[i] - k_fit[i])**2
		bot += 1/sig[i]**2

	var_fit = (1/(n-2) * top) / (1/n * bot)
	
	# Calculate the average total variance of the data.
	var_bar = n / bot

	# Calculate the parameter-dependent variance of the fit.
	S_P_fit = math.sqrt(var_fit + var_bar)

	# Calculate the variance about the mean of the parameter values.
	x_diff_sq = 0
	for i in range(0, len(data)):
		x_diff_sq += (param[i] - x_bar)**2
 
	# Determine the data fit at the requested parameter value.
	K_fit = slope * x + intercept
	
	# Determine the one-sided lower tolerance band at the requested value.
	K_L_fit = K_fit - S_P_fit * ( ( math.sqrt(2 * F * ( (1/n) + ( (x-x_bar)**2 / (x_diff_sq) ) ) ) ) + ( z * math.sqrt( (n-2) / (chi_sq) ) ) )

	#rv = scipy.stats.f(2, n-2)
	#x = numpy.linspace(0.01, numpy.minimum(rv.dist.b, 3))
	#h = plt.plot(x, rv.pdf(x))
	#plt.show()

	return K_fit, K_L_fit 

def trendplot(x, y):

	''' Peforms a linear regression of a 1-D array of data given a corresponding 1-D array of parameter values.
	    The regression line fit is then plotted along with the 1-D array of data.  '''
	
	# Perform a linear regression. The p-value is the parameter of interest.
	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
	
	# Build the regression line for plotting.
	polynomial = numpy.poly1d([slope, intercept])
	line = polynomial(x)
	
	# Generate the canvas for plotting the regression line and data.
	p = plt.subplot(111)
	
	# Plot the regression line with the data and add labels to the plot.
	plt.plot(x, line,'r-', x, y, 'o')
	p.set_title('Regression Analysis with 95/95 One-Sided Lower Tolerance Band', fontsize=12)
	p.set_xlabel('Parameter')
	p.set_ylabel('K-eff') 
	plt.figtext(0.5, 0.15, 'R = {0:-.3g}'.format(r_value), fontsize=12, ha='left')
	plt.figtext(0.5, 0.18, 'y = {0:-.3g}x + {1:6.4f}'.format(slope, intercept), fontsize=12, ha='left')
	if p_value <= 0.05:
		plt.figtext(0.5, 0.12, '{0:-.3g} < 0.05 => Trend Indicated'.format(p_value), fontsize=12, ha='left')
	elif p_value > 0.05:
		plt.figtext(0.5, 0.12, '{0:-.3g} > 0.05 => No Trend Indicated'.format(p_value), fontsize=12, ha='left')
	#plt.show()

	# Determine if there is a statistically significant trend (with 95% confidence). 
	if p_value <= 0.05:
		trend_exists = True
	else:
		trend_exists = False

	# Return whether or not a trend was found and the associated p-value used for the determination.
	return trend_exists, p_value

def bandplot(x, y, ysig, outp):

	''' Plot the one-sided lower tolerance band alongside the regression fit (K_fit). '''

	# Regular expression that determines the input filename minus the extension to use as the base for the figure filename.
	name_base = re.search(r'\b(.+)\b\.', outp)

	# Determine if a trend is indicated based on the result of hypothesis testing after performing a regression analysis.
	(trend_exists, p_value) = trendplot(x, y)
	
	# Determine the range of x-values for plotting.
	x_min = numpy.amin(x)
	x_max = numpy.amax(x)
	
	# Get ready to plot 101 points.
	inc = (x_max - x_min) / float(100)

	# Build two 1-D arrays of corresponding data; one with the parameter values with the range sampled 101 times, the other with the 
	# one-sided lower tolerance band sampled 101 times.
	x1 = []
	y1 = []
	for i in range(0, 100):
		x1.append(x_min + i*inc)
		(K_fit, K_L_fit) = k_lband(x_min + i*inc, x, y, ysig, 0.95)
		y1.append(K_L_fit)

	# Add the plot of the one-sided lower tolerance band on the existing figure canvas containing the regression plot.
	plt.plot(x1, y1,'g-')
	
	#plt.show()
	
	# Save the figure of the regression and one-sided lower tolerance band plots as a PNG file with the input file name.
	plt.savefig(name_base.group(1) + '_regression.png', bbox_inches='tight')

	# Return whether or not a trend is indicated and the corresponding p-value for printing to output.
	return trend_exists, p_value

def analyze(param, data, sig, request, outp):
	
	''' This is basically a driver function. It calls all of the necessary functions that perform the actual calculations and prints
	    the results to output. '''

	# Perform normality tests on a 1-D array of data. 
	normal = checknorm(data, sig, outp)
	
	# Plot the 1-D array of data and save to a figure so a visual check can be performed.
	normplot(data, outp)

	# If the data does not test normal, only perform the non-parametric statistical analysis.
	if normal == 0:
		nonparam(data, sig, outp)
	# If the data does test normal, perform a regression, check if there is a trend based on the regression p-value, and then perform
	# the non-parametric statistical analysis for comparison purposes.
	elif normal == 1:
		# Determine the range of the 1-D array of parameter values for printing to output.
		p_min = numpy.amin(param)
		p_max = numpy.amax(param)
		# Perform regression; plot regression line [ K_fit(x) ] and one-sided lower tolerance band [ K_L_fit(x) ].
		name_base = re.search(r'\b(.+)\b\.', outp)
		fname = open(name_base.group(1) + '.out', 'a')
		fname.write('\nTrending Analysis\n')
		print '\nTrending Analysis'
		fname.write('-----------------\n')
		print '-----------------'
		fname.write('\nParameter Range: {0}-{1}\n'.format(p_min, p_max))
		print '\nParameter Range: {0}-{1}'.format(p_min, p_max)
		(trend_exists, p_value) = bandplot(param, data, sig, outp)
		if trend_exists:
			fname.write('\n{0:-.3g} < 0.05 => Trend Indicated\n'.format(p_value))
			print '\n{0:-.3g} < 0.05 => Trend Indicated'.format(p_value)
		else:
			fname.write('\n{0:-.3g} > 0.05 => No Trend Indicated\n'.format(p_value))
			print '\n{0:-.3g} > 0.05 => No Trend Indicated'.format(p_value)
		# Sample the one-sided lower tolerance band with the requested parameter value.
		x_sample = request
		fname.write('\nApplication Parameter: {0}\n'.format(x_sample))
		print '\nApplication Parameter: {0}'.format(x_sample)
		# Calculate bias and bias uncertainty at the requested parameter value, then print to output.
		(K_fit, K_L_fit) = k_lband(x_sample, param, data, sig, 0.95)
		fname.write('\nBias:             {0:-.5f}\n'.format(K_fit - 1))
		print '\nBias:             {0:-.5f}'.format(K_fit - 1)
		fname.write('Bias Uncertainty:  {0:-.5f}\n'.format(K_fit - K_L_fit))
		print 'Bias Uncertainty:  {0:-.5f}'.format(K_fit - K_L_fit)
		fname.close()
		# Perform non-parametric statistical analysis.
		nonparam(data, sig, outp)

##########################################################################

def main():

	''' The main function. Reads the input file and breaks it down in order to pass to the driver function 'analyze'.
	    The 'analyze' function will determine code bias and bias uncertainty following the guidance in NUREG/CR-6698. '''

	x = []
	y = []
	ysig = []

	# Read the input filename.
	fname = sys.argv[1]
	# Open the input file for reading.
	f = open(fname)

	# Read each line until the end of file (EOF).
	iter = 0
	while True:
		# Grab a line.
		line = f.readline()
		# If the length of the line is zero, the EOF has been reached, so exit the while loop.
		if len(line) == 0:
			break
		# If there is any leading whitespace at the beginning of the line, get rid of it before processing.
		line = re.sub(r'^\s+', '', line)
		# If there are any commas, replace them with spaces before processing.
		line = re.sub(',', ' ', line)
		# If there is a '#', or a newline return, or the EOF has been reached after pre-processing, start back
		# at the top of the loop.
		if re.search('#', line) or re.search(r'^\n', line) or len(line) == 0:
			continue
		# If it's the first pass through the loop, extract the requested value for use during trending analysis
		# (if a trend is indicated).
		if iter == 0:
			app_param = float(line)
			iter += 1
		# If it's not the first pass through the loop, split the data by whitespace and store in the 'data' list.
		else:
			data = line.split()
			# The first value should be the parameter value.
			x.append(float(data[0]))
			# The second value should be the data value.
			y.append(float(data[1]))
			# The third value should be the error value.
			ysig.append(float(data[2]))
	# Close the input file; finished reading.
	f.close()

	# Run the driver function with all necessary inputs to determine code bias and bias uncertainty following NUREG/CR-6698.
	analyze(x, y, ysig, app_param, fname)

if __name__ == '__main__':
  main()	