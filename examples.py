from OGL import *
from OGL_DataFrame import *
from OGL_ML import *
from OGL_Stat import *
import numpy as np
def math_functions():
	print(sum_all(2,3,4))
	print(nroot(2,4))
def utility_functions():
	print_dic(name="Shaon",Position="Data Scientist")
def DataFrame_Class_all_properties_example():
	dm1 = DataFrame().read_csv('sample_inputs/input1.txt')
	print('List presentation',dm1.tolist)
	print(dm1)
	print(dm1[1])
	print("accessing cell")
	print(dm1[1][0])
	print(dm1['lower'])
	print('Columns',dm1.columns)
	print('Shape',dm1.shape)
	print('renaming columns')
	dm1.columns=['low','up','freq']
	print(dm1)
	print('empty DataFrame of 2x3',dm1.new(2,3))
	print('element with 1 , DataFrame of 2x3',dm1.new(2,3,elm=1))

	print('Transposed and returned',dm1.T)
	print(dm1)
	print('Transposed itself and also returned',dm1.transpose(change_self = True))
	print(dm1)

def Class_DataFrame_datatype_conversion():
	dm1 = DataFrame().read_csv('sample_inputs/matrice.txt')
	print('List presentation')
	print(dm1.tolist)
	print('returned result of conversion')
	dm2 = dm1.conv_type('int')
	print(dm2)
	print('print actual dataframe')
	print(dm1)
	print('conversion,returned and changing itself')
	dm1.conv_type('int',change_self=True)
	print(dm1)

def Add_two_dataframe_by_rows():
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	my_data.conv_type('float',change_self=True)
	my_data.normalize(change_self=True)
	X = DataFrame(dataframe = my_data[0:2])
	ones = DataFrame().create_dataframe(X.framesize[0],1,elm=1.)
	X = DataFrame(dataframe = X.T)
	ones = DataFrame(dataframe = ones.T)
	print(ones.framesize,X.framesize)
	X = DataFrame(dataframe=DataFrame().concatenate(ones,X,axis=0))
	print(X.tolist)

def Add_two_dataframe_by_columns():
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	my_data.conv_type('float',change_self=True)
	my_data.normalize(change_self=True)
	
	X = DataFrame(dataframe= my_data[0:2])
	
	ones = DataFrame().create_dataframe(X.framesize[0],1,elm=1.)
	
	
	X = DataFrame(dataframe=DataFrame().concatenate(ones,X,axis=1))
	print(X)

def Summation_to_rows_or_column_of_dataframe():
	#adding rows
	liq = DataFrame().mat_dot( DataFrame(dataframe=[[1,4],[2,5],[3,6]]), DataFrame(dataframe=[[7,9,11],[8,10,12]]) )
	print(DataFrame(dataframe =  liq.tolist ))
	print(DataFrame(dataframe =  liq.tolist ).sum(axis=1))
	#adding columns
	liq = DataFrame().mat_dot( DataFrame(dataframe=[[1,4],[2,5],[3,6]]), DataFrame(dataframe=[[7,9,11],[8,10,12]]) )
	print(DataFrame(dataframe =  liq.tolist ))
	print(DataFrame(dataframe =  liq.tolist ).sum(axis=0))
	
def Statistics_on_Single_Column_Data_example():
	df3 = DataFrame().read_csv('sample_inputs/input2.txt')
	print(df3)
	print("ArithmeticMean = %s" % ArithmeticMean(conv_type(df3[0],"int")) )
	print("GeometricMean = %s" % GeometricMean(conv_type(df3[0],"int")))
	print("HarmonicMean = %s" % HarmonicMean(df3[0]))
	print("Mode = %s" % mode(df3[0]))
	print("Median = %s" % median(df3[0]))

def Statistics_on_Class_Distribution_or_Grouped_Data_example():
	df = DataFrame().read_csv('sample_inputs/input1.txt')
	print(df)
	lower, upper, frequency = conv_type(df['lower'],"int"),conv_type(df['upper'],"int"),conv_type(df['frequency'],"int")
	print("ArithmeticMean = %s" % ArithmeticMean(lower=lower,upper=upper,frequency=frequency))
	print("GeometricMean = %s" % GeometricMean(lower=lower,upper=upper,frequency=frequency))
	print("HarmonicMean = %s" % HarmonicMean(lower=lower,upper=upper,frequency=frequency))
	print("Mode = %s" % mode(lower=lower,upper=upper,frequency=frequency))
	print("Median = %s" % median(lower=lower,upper=upper,frequency=frequency))

def Simple_Linear_Regression_from_textfile_example():
	dm = DataFrame()
	dm.read_csv('sample_inputs/matrice2.txt')
	linear_regression(conv_type(dm[0],'int'),conv_type(dm[1],'int'))

def Simple_Linear_Regression_from_textfile_by_dataframe_representation_example():
	dm = DataFrame().read_csv('sample_inputs/matrice2.txt')
	linear_regression(conv_type(dm[0],'int'),conv_type(dm[1],'int'))

def multivariant_linear_regression_example():
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	
	XDATA = DataFrame(dataframe= my_data[0:2],columns=['size','bedroom'])
	YDATA = DataFrame(dataframe= [my_data[2]])
	
	MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = 0.01,iters = 1000)


def Matrice_dot_multiplication_from_textfile_example():
	dm1 = read_csv('sample_inputs/matrice.txt')
	dm2 = read_csv('sample_inputs/matrice3.txt')
	res = mat_dot(dm1,dm2)
	print(res)

def Matrice_dot_multiplication_from_textfile_by_Dataframe_representation_example():
	dm1 = DataFrame().read_csv('sample_inputs/matrice.txt')
	dm2 = DataFrame().read_csv('sample_inputs/matrice3.txt')
	print(  DataFrame().mat_dot(dm1,dm2)  )#creates another object
	print(  mat_dot(dm1.dataframe,dm2.dataframe)  )#efficient
	print(  mat_dot(dm1.tolist,dm2.tolist)  )#efficient

def Matrice_dot_multiplication_from_matrice_representation_example():
	dm1 = matrice( [[4,8],
	        [0,2],
	        [1,6]] )
	dm2 = matrice( [[5,2],
	        [9,4]] )
	res = mat_dot(dm1,dm2)
	print(res)

def Rectangle_with_Max_Min_point_example():
	dm1 = DataFrame().read_csv('sample_inputs/youtube.txt')
	max_min_rectangle(dm1['length(s)'],dm1['views'])

class clla:
	def __getitem__(self, *keys):
		print keys
def Subtract_two_dataframe_by_same_size():
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	my_data.conv_type('float',change_self=True)
	X = DataFrame(dataframe= my_data[0:2])
	Y = DataFrame(dataframe= my_data[1:3])
	print(DataFrame().substract(X, Y))
def n_power_of_dataframe():
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	my_data.conv_type('float',change_self=True)
	my_data.power(2)
	print(my_data)


def multivariant_linear_regression_lib():
	my_data = pd.read_csv('sample_inputs/home.txt',names=["size","bedroom","price"])

	#we need to normalize the features using mean normalization
	my_data = (my_data - my_data.mean())/my_data.std()



	#setting the matrixes
	X = my_data.iloc[:,0:2]
	ones = DataFrame().new(1,X.shape[0],elm=1)
	X = np.concatenate((ones.tolist,X),axis=1)# for making Xnot = 1 column
	y = my_data.iloc[:,2:3].values #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray

	
	#theta = DataFrame().new(3,1,elm=0.)
	theta = np.zeros([1,3])

	#computecost
	def computeCost(X,y,theta):
		tobesummed = np.power(( np.dot(X, theta.T) -y), 2 )
		#print('to be summed',np.sum(tobesummed))
		return np.sum(tobesummed)/(2 * len(X))
	
	def gradientDescent(X,y,theta,iters,alpha):
	    cost = np.zeros(iters)
	    for i in range(iters):
	        theta = theta - (alpha/len(X)) * np.sum(X * (np.dot(X, theta.T) - y), axis=0)
	        cost[i] = computeCost(X, y, theta)
	    
	    return theta,cost

	#set hyper parameters
	alpha = 0.01
	iters = 1000

	g,cost = gradientDescent(X,y,theta,iters,alpha)
	print(g)

	finalCost = computeCost(X,y,g)
	print(finalCost)

	fig, ax = plt.subplots()  
	ax.plot(np.arange(iters), cost, 'r')  
	ax.set_xlabel('Iterations')  
	ax.set_ylabel('Cost')  
	ax.set_title('Error vs. Training Epoch')  
	plt.show()

def model_comparison_for_youtube_view():
	dm = DataFrame()
	dm.read_csv('sample_inputs/youtube.txt')
	print(dm)
	linear_regression(conv_type(dm['length(s)'],'int'),conv_type(dm['views'],'int'),xlabel='length',ylabel='views',title='Youtube views SLR')

	my_data = DataFrame().read_csv('sample_inputs/youtube.txt')
	
	XDATA = DataFrame(dataframe= [my_data['length(s)'],my_data['serial']],columns=['size','bedroom'])
	YDATA = DataFrame(dataframe= [my_data['views']])
	

	MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = 0.02,iters = 1000)


def youtube():
	my_data = DataFrame().read_csv('sample_inputs/youtube.txt')
	
	X = [my_data['length(s)'],my_data['serial']]
	Y = [my_data['views']]
	XDATA = DataFrame(dataframe= X,columns=['length(s)','serial'])
	YDATA = DataFrame(dataframe= Y)

	a=0.01
	c = MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = a,iters = 1000,plot=1)
	print c

	my_data = DataFrame().read_csv('sample_inputs/youtube.txt')
	
	X = [my_data['length(s)'],my_data['serial']]
	Y = [my_data['views']]
	XDATA = DataFrame(dataframe= X,columns=['size','bedroom'])
	YDATA = DataFrame(dataframe= Y)

	
	def find_best_alpha():			
		a=0.001
		cost = []
		n=1
		it=0
		last_alpha = 0
		while n>0.0001:
			my_data = DataFrame().read_csv('sample_inputs/youtube.txt')
	
			X = [my_data['length(s)'],my_data['serial']]
			Y = [my_data['views']]
			XDATA = DataFrame(dataframe= X,columns=['length(s)','serial'])
			YDATA = DataFrame(dataframe= Y)

			c = MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = abs(n),iters = 1000,plot=0)
			last_alpha =abs(n)
			cost.append(c)
			n = n - a
			it+=1
			print("iteration=%s,n=%s"%(it,n))
			

		print(min(cost))
		i = cost.index(min(cost))
		alpha = i*a
		c = MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = alpha,iters = 1000,plot=0)
		print c,alpha,n,alpha
		print('last_alpha',last_alpha)
		return alpha

	a=0.67#find_best_alpha()
	print("Got best alpha = %s"%a)
	c = MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = a,iters = 1000,plot=1)
	print c


def house():
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	
	X = my_data[0:2]
	Y = [my_data[2]]
	XDATA = DataFrame(dataframe= X,columns=['size','bedroom'])
	YDATA = DataFrame(dataframe= Y)
	
	a=0.01
	c = MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = a,iters = 1000,plot=1)
	print c
	

	def find_best_alpha():
		a=0.001
		cost = []
		n=1
		it=0
		last_alpha = 0
		while n>0.0001:
			my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
		
			X = my_data[0:2]
			Y = [my_data[2]]
			XDATA = DataFrame(dataframe= X,columns=['size','bedroom'])
			YDATA = DataFrame(dataframe= Y)

			c = MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = abs(n),iters = 1000,plot=0)
			last_alpha =abs(n)
			cost.append(c)
			n = n - a
			it+=1
			print("iteration=%s,n=%s"%(it,n))
			

		print(min(cost))
		i = cost.index(min(cost))
		alpha = i*a
		c = MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = alpha,iters = 1000,plot=0)
		print c,alpha,n,alpha
		print('last_alpha',last_alpha)
		return alpha

	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	
	X = my_data[0:2]
	Y = [my_data[2]]
	XDATA = DataFrame(dataframe= X,columns=['size','bedroom'])
	YDATA = DataFrame(dataframe= Y)
	

	a=0.929#find_best_alpha()
	print("Got best alpha = %s"%a)
	c = MLVR(XDATA,YDATA,xreference=0,residual=1,xlabel='serial,length',ylabel='views',title='Youtube View MLVR',alpha = a,iters = 1000,plot=1)
	print c
	
	

if __name__ == '__main__':	
	'''
	dm1 = DataFrame(dataframe=matrice( [[1,2,3],[4,5,6]] ))
	dm2 = DataFrame(dataframe=matrice( [[7,8],[9,10],[11,12]] ))
	res = DataFrame().mat_dot(dm1,dm2)
	print(res)
	print(res.sum(axis=0))
	'''
	house()
	
