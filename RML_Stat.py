from RML import *
from rest import *
import math

def ArithmeticMean(li=[],lower=[],upper=[],frequency=[]):
	if li == []:
		for i in range(length(lower)):
			midpoint =(lower[i]+upper[i])/2
			li = appends(li,midpoint)
	else:
		del lower
		del upper
		del frequency
		frequency = []
	
	if frequency == []:
		for i in range(length(li)):
			frequency = appends(frequency,1)
	sumn = 0
	iteration = 0
	for i in range(length(li)):
		sumn += (li[i]*frequency[i])
		iteration  += frequency[i]
	return float(sumn)/iteration

def ArithmeticMeanSingle(li):
	sumn = 0
	iteration = 0

	while True:
		try:
			sumn += li[iteration]
		except IndexError:
			break

		iteration  += 1
	return float(sumn)/iteration

def GeometricMean(li=[],lower=[],upper=[],frequency=[]):
	if li == []:
		for i in range(length(lower)):
			midpoint =(lower[i]+upper[i])/2
			li = appends(li,midpoint)
	else:
		del lower
		del upper
		del frequency
		frequency = []

	
	if frequency == []:
		for i in range(length(li)):
			frequency = appends(frequency,1)

	sumn = 0
	iteration = 0
	for i in range(length(li)):
		#frequency[i] * log(midpint)
		sumn += frequency[i]*math.log10(li[i])
		iteration  += frequency[i]
	
	return 10**(float(sumn)/iteration)

def GeometricMeanSingle(li):
	sumn = 1
	iteration = 0

	while True:
		try:
		#if li[iteration] == 0: what to do
			sumn *= li[iteration]
		except IndexError:
			break

		iteration  += 1
	
	return nroot(iteration,float(sumn))

def HarmonicMean(li=[],lower=[],upper=[],frequency=[]):
	if li == []:
		for i in range(length(lower)):
			midpoint =(lower[i]+upper[i])/2
			li = appends(li,midpoint)
	else:
		del lower
		del upper
		del frequency
		frequency = []
	
	if frequency == []:
		for i in range(length(li)):
			frequency = appends(frequency,1)
	sumn = 0
	iteration = 0
	for i in range(length(li)):
		#F/X
		sumn += (frequency[i]/float(li[i]))
		iteration  += frequency[i]
	#S(F)/(S(F/X))
	return iteration/float(sumn)

def HarmonicMeanSingle(li):
	inverse_sum = 0
	iteration = 0
	while True:
		try:
			inverse_sum += 1/float(li[iteration])
		except IndexError:
			break

		iteration  += 1
	
	return iteration/float(inverse_sum)
	
def mode(li=[],lower=[],upper=[],frequency=[]):
	mids = li
	if li == []:
		for i in range(length(lower)):
			midpoint =(lower[i]+upper[i])/2
			li = appends(li,midpoint)
		it = 0
		for lk,l,u,f in sorted(zip(li,lower,upper,frequency)):
			li[it],lower[it],upper[it],frequency[it] = lk,l,u,f
			it += 1

	else:
		return modeSingle(li)
		del lower
		del upper
		del frequency
		frequency = []
		#return mediansingle(li)
	if frequency == []:
		for i in range(length(li)):
			frequency = appends(frequency,1)

	sum = 0
	it=0
	for i in frequency:
		sum += i
		it += 1
	total_frequency = sum
	if(total_frequency % 2 == 0):
		median_point = ( (total_frequency/float(2)) + (total_frequency/float(2) + 1) ) / float(2)
	else:
		median_point = total_frequency/float(2)

	if mids != []:
		return median_point
	else:		
		_ , it = 0,0
		for c in frequency:
			_ += c
			if( _ >= median_point):
				median_key = it
				fl = _ - c
				break
			it += 1

		width = upper[median_key] - lower[median_key] + 1
		Lb = (lower[median_key] + upper[median_key-1]) / float(2)
		#print("median class = %s median point = %s, Cumalative frequency before lower class = %s, frequency of median group = %s, width = %s, Median Lower class boundary = %s" % (median_key,median_point,fl, frequency[median_key], width, Lb ))
		mode = Lb + ( float((frequency[median_key] - frequency[median_key - 1])) /  float((frequency[median_key] - frequency[median_key - 1])  +  (frequency[median_key] - frequency[median_key + 1]))  ) * width
		return  mode

def modeSingle(li):
	return maxn(li)

def mediansingle(li):
	li=conv_type(li,"int")
	li = quicksort(li)
	leng = length(li)
	if(leng % 2 == 0):
		return float(( li[leng/2-1] + li[leng/2] ))/2
	elif(leng % 2 != 0):
		return  li[leng/2]

def median(li=[],lower=[],upper=[],frequency=[]):
	mids = li
	if li == []:
		for i in range(length(lower)):
			midpoint =(lower[i]+upper[i])/2
			li = appends(li,midpoint)
		it = 0
		for lk,l,u,f in sorted(zip(li,lower,upper,frequency)):
			li[it],lower[it],upper[it],frequency[it] = lk,l,u,f
			it += 1

	else:
		return mediansingle(li)
		del lower
		del upper
		del frequency
		frequency = []
		#return mediansingle(li)
	if frequency == []:
		for i in range(length(li)):
			frequency = appends(frequency,1)


	sum = 0
	it=0
	for i in frequency:
		sum += i
		it += 1
	total_frequency = sum
	if(total_frequency % 2 == 0):
		median_point = ( (total_frequency/float(2)) + (total_frequency/float(2) + 1) ) / float(2)
	else:
		median_point = total_frequency/float(2)

	if mids != []:
		return median_point
	else:		
		_ , it = 0,0
		for c in frequency:
			_ += c
			if( _ >= median_point):
				median_key = it
				fl = _ - c
				break
			it += 1

		width = upper[median_key] - lower[median_key] + 1
		Lb = (lower[median_key] + upper[median_key-1]) / float(2)
		#print("median class = %s median point = %s, Cumalative frequency before lower class = %s, frequency of median group = %s, width = %s, Median Lower class boundary = %s" % (median_key,median_point,fl, frequency[median_key], width, Lb ))
		med = Lb + ( (median_point - fl)/float(frequency[median_key]) ) * width
		return med

def sample_standard_deviation(li):
	return (sum([(i - ArithmeticMean(li))**2 for i in li])/(length(li)-1))**(1/float(2))

def sample_variance(li):
	#squared sample_standard_deviation
	return (sum([(i - ArithmeticMean(li))**2 for i in li])/(length(li)-1))

def co_standard_deviation(li,pi):#cross_standard_deviation
	#x_n = y_n must have to be the same
	return (sum([(i - ArithmeticMean(li))*(j - ArithmeticMean(pi)) for i,j in zip(li,pi)])/(length(li)-1))**(1/float(2))

def co_variance(li,pi):#cross_variance
	#squared cross_standard_deviation
	#x_n = y_n must have to be the same
	return (sum([(i - ArithmeticMean(li))*(j - ArithmeticMean(pi)) for i,j in zip(li,pi)])/(length(li)-1))

def pearson_correlation_coefficient(li,pi):
	return co_variance(li,pi)/(sample_standard_deviation(li)*sample_standard_deviation(pi))

def normalize(df):
	dfc = []
	for c in df:
		mean= ArithmeticMean(df[c]) 
		std = sample_standard_deviation(df[c])
		_ =[ (a - mean)/std for a in df[c]]
		dfc = appends(dfc,_)
	return dfc
def reference_reverse_normalize(ypure,y_pred):
	y_PRED = []
	ystd = sample_standard_deviation(ypure)
	ymean = ArithmeticMean(ypure) 
	for c in range(length(y_pred)):
		y_reverse = y_pred[c]* ystd + ymean
		y_PRED.append(y_reverse)
	return y_PRED
