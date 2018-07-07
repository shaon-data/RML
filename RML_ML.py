import matplotlib.pyplot as plt
from RML_Stat import *
from RML_DataFrame import *
from RML import *
import numpy as np

#Machine Learning
def estimate_coef(x, y):
	"""Estimate co-efficients m,c for straight line mx+c"""
	# estimating coefficients
 	b1 = co_variance(x,y)/float(sample_variance(x))
	b0 = ArithmeticMean(y) - b1*ArithmeticMean(x)
	return(b0, b1)


def linear_regression(x,y,title='',xlabel='X',ylabel='Y'):
	"""Does simple linear regression"""
	b = estimate_coef(x, y)
    	print("Estimated eqauation:\ny = %s + %s*x + e"%(b[0], b[1]))
    	x_a,y_a = give_time_series(x,y)
	plot_regression_line(x_a, y_a, b, title=title,xlabel=xlabel,ylabel=ylabel)

def plot_regression_line(x, y, b,title='',xlabel='X',ylabel='Y'):
    	"""ploting the prediction line using simple linear regression"""
    	# plotting the actual points as scatter plot
	plt.plot(x, y, color = "g", label='Real', marker = "." )#s = 30
	b0,b1 = b
	# predicted response vector
	y_pred = [b0 + b1*xi for xi in x]
	plot_error_distance(x,y_pred,y)
    # plotting the regression line
	plt.plot(x, y_pred, color = "r", marker = ".", label='Prediction')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend()
	plt.show()

def plot_eachpoint_connected(x,y):
	"""Plot connecting every point with each other from a distribution"""
	i=1
	for a,b in zip(x,y):
		for x1,y1 in zip(x[i:],y[i:]):
			plt.plot([a,x1],[b,y1],color='r')
		i+=1
	plt.show()

def plot_error_distance(x,y_pred,y_actual):
	"""Plot error distance or residual"""
	for [a,b,c] in [ [x,y_r,y_p] for x,y_r,y_p in zip(x,y_pred,y_actual) ]:
		plt.plot([a,a],[b,c],color='y',label='residual')

def max_min_rectangle(x,y):
	"""Plot a rectangle using max and min point from a distribution"""
	print(zip([minn(x),maxn(x),maxn(x),minn(x)],[minn(y),minn(y),maxn(y),maxn(y)]))
	plt.plot([minn(x),maxn(x),maxn(x),minn(x),minn(x)],[minn(y),minn(y),maxn(y),maxn(y),minn(y)])
	plt.fill_between([minn(x),maxn(x),maxn(x),minn(x),minn(x)],[minn(y),minn(y),maxn(y),maxn(y),minn(y)], hatch = '///')
	plt.show()

def compute_distance(p1,p2):
	"""returns distance between two point"""
	x[0],y[0] = p1
	x[1],y[1] = p2
	return ((y[1] - y[0])**2 + (x[1] - x[0])**2)**(1/float(2))

def get_max_rectangele():
	pass

def get_max_area():
	pass

def give_time_series(x,y):
	"""Rearrange X,Y value pairs or points according to X's order"""
	xall = []
	yall = []
	for x1,y1 in sorted(zip(x,y)):
		xall.append(x1)
		yall.append(y1)
	return (xall,yall)

def meanResidual(pure , pred):
	"""returns average error distance or residual"""
	E = []
	for c in range(length(pure)):
		absolute_distance = abs(pure[c] - pred[c])
		E.append(absolute_distance)
	import numpy as np
	return np.mean(E)

def maxResidual(pure , pred):
	"""returns maximum error distance or residual"""
	E = []
	for c in range(length(pure)):
		absolute_distance = abs(pure[c] - pred[c])
		E.append(absolute_distance)

	return max(E)

def minResidual(pure , pred):
	"""returns minimum error distance or residual"""
	E = []
	for c in range(length(pure)):
		absolute_distance = abs(pure[c] - pred[c])
		E.append(absolute_distance)

	return min(E)


def MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='',ylabel='',title='',alpha = 0.01,iters = 1000,plot=1):	
	"""Does Multivariant Linear Regression
	properties:
		XDATA = The Feature Dataframe
		YDATA = The Target Dataframe
		xreference = 1/0 -> The column index in XDATA for ploting graph
		xlabel = Label for X in Graph
		ylabel = Label for Y in Graph
		title = title for graph]
		alpha = Learning rate for model
		iters = the number of iteration to train the model
	"""
	XDATA.conv_type('float',change_self=True)
	xpure = XDATA[xreference]
	XDATA.normalize(change_self=True)

	YDATA.conv_type('float',change_self=True)
	ypure = YDATA.tolist[0]
	YDATA.normalize(change_self=True)

	X=XDATA
	y=YDATA

	df =DataFrame()
	ones = df.new(X.shape[0],1,elm=1.)
	X = df.concat(ones,X,axis=1)
	
	theta = DataFrame().new(1,length(X.columns),elm=0.)
	
	def computeCost(X,y,theta):
		dot_product = DataFrame().dot(X,theta.T)	
		return float(    (  (dot_product - y)**2  ).sum(axis=0)    )/(2 * X.shape[0])
	
	def gradientDescent(X,y,theta,iters,alpha):
		#cost = np.zeros(iters)
		cost = []
		for i in range(iters):			
			dot_product = DataFrame().dot(X,theta.T)
			derivative = DataFrame(dataframe = [[(alpha/X.shape[0])]])  *  ( X*(dot_product - y) ).sum(axis = 0 ) 
			theta = theta - derivative			
			cost.append( computeCost(X, y, theta) ) #cost[i] = computeCost(X, y, theta)
		return theta,cost

	def print_equation(g):
		stra = "Estimated equation, y = %s"%g[0]
		g0 = g[0]
		del g[0]
		for c in range(length(g)):
			stra += " + %s*x%s"%(g[c],c+1)
		print(stra)

	def predict_li(XDATA,g):
		g0 = g[0]
		del g[0]
		y_pred = []			
		for row in range(XDATA.shape[0]):
			suma = 0
			suma += sum(list_multiplication( g , XDATA.row(row) ) )
			yres = g0 + suma
			y_pred.append(yres)	
		return y_pred

	g,cost = gradientDescent(X,y,theta,iters,alpha)
	finalCost = computeCost(X,y,g)
	#g = g.T
	g = g.two2oneD()
	print("Thetas = %s"%g) #print("cost = ",cost)
	print("finalCost = %s" % finalCost)
	gN = g[:]
	print_equation(gN)

	gN = g[:]	
	y_pred = predict_li(XDATA,gN)	
	
	y_PRED = reference_reverse_normalize(ypure,y_pred)
	emin,emean,emax = minResidual(ypure , y_PRED),meanResidual(ypure , y_PRED),maxResidual(ypure , y_PRED)
	
	print("Min,Mean,Max residual = %s, %s, %s"%( emin,emean,emax ) )
	print("Residual Min - Max Range = %s"%(emax-emin))
	print("Residual range percentage = %s" %((emax-emin)/(max(ypure) - min(ypure))) )
	
	print("Residual mean percentage = %s" %(emean/ArithmeticMean(ypure)) )




	#-- If finalcost is lowest mean Residual or mean Error distance also will be lowest


	#y_pred = [g[0] + g[1]*my_data[0][c] + g[2]*my_data[1][c] for c in range(my_data.shape[0])]
	y_actual = YDATA.tolist[0]
	x = XDATA[xreference]

	if plot == 1:
		fig, ax = plt.subplots()  
		ax.plot(np.arange(iters), cost, 'r')  
		ax.set_xlabel('Iterations')  
		ax.set_ylabel('Cost')  
		ax.set_title('Error vs. Training Epoch')  
		plt.show()

		x_a, y_a = give_time_series(xpure,y_PRED)#give_time_series(x,y_pred)
		plt.plot(x_a,y_a,color='r',marker='.',label='Prediction')

		x_a, y_a = give_time_series(xpure,ypure)#give_time_series(x,y_actual)
		plt.plot(x_a,y_a,color='g',marker='.',label='Real')
		
		if residual == 1:
			plot_error_distance(xpure,y_PRED,ypure)

		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.legend()
		plt.show()
	else:
		print('plot off')
		
	return finalCost
def softmax(li):
	"""softmax - Logistic Sigmoid function
	"""
	from math import exp
	sumE = sum([exp(i) for i in li])
	return [exp(i)/sumE for i in li]
def stable_softmax(li):
    	"""
    	softmax normalization - Logistic Sigmoid function
	"""
    	
    	sumx = sum(li)
    	maxx = max(li)
    	x_norm = [i-sumx for i in li]
	return softmax(x_norm)
def stable_softmax2(li):
    	"""
    	softmax normalization - Logistic Sigmoid function
	"""
    	from math import exp
    	sumx = sum(li)
    	maxx = max(li)
    	x_norm = [i-maxx for i in li]
	return softmax(x_norm)
	
def softmax_normalization(li):
    	"""
    	softmax normalization - Logistic Sigmoid function
	"""
    	from math import exp
    	sumx = sum(li)
    	maxx = max(li)
    	x_norm = [(i-maxx)/maxx for i in li]
	return softmax(x_norm)

def softmax_normalization2(li):
    	"""
    	softmax normalization - Logistic Sigmoid function
	"""
    	from math import exp
    	sumx = sum(li)
    	maxx = max(li)
    	x_norm = [(i-sumx)/maxx for i in li]
	return softmax(x_norm)

def softmax_normalization3(li):
    	"""
    	softmax normalization - Logistic Sigmoid function
	"""
    	from math import exp
	sumx = sum(li)
    	maxx = max(li)
    	x_norm = [(i-maxx)/sumx for i in li]
	return softmax(x_norm)

def softmax_normalization4(li):
    	"""
    	softmax normalization - Logistic Sigmoid function
	"""
    	from math import exp
    	sumx = sum(li)
    	maxx = max(li)
    	x_norm = [(i-sumx)/sumx for i in li]
	return softmax(x_norm)