from RML import *
from RML_Stat import *
#For Dataframe
def columns(df):#obs
	return [key for key in df]

def read_csv(input_file,names = []):#obs
	with open(input_file,'r') as file:#Taking file input as stream, handly for iterators and reading stream object saves memory losses
		data = file.readlines()#reading line by line
	
	header = split( strip(data[0],[' ','\n']) ,",")

	if header[0].isdigit() == True:
		#no header
		le = length(header)
		header=[]
		for c in range(0,le):
			header = appends(header,c)
		data  = appends(data,'0\n',0)
	
		if names == []:
			pass
		else:			
			header = names
		
	del data[0]#Deleting header Row
	df = {}#Creating dictionary for database
	for d in header:
			df[d] = []

	for c in data:
		line = split( strip(c,[' ','\n']) ,',')
		
		it=0
		for d in header:
			df[d] = appends(df[d],line[it])
			it += 1

	return df

def df_size(df):#obs
	row = length(df[0])
	column = length(df)
	return row, column

def create_dataframe(m,n):#obs
	'''
	df = []
	
		
	df_r = []
	for i in range(m):
		df_r = appends(df_r,0)

	for c in range(n):
		df = appends(df,[df_r])
	'''	
	df=[]
	df = [[None]*m for _ in range(n)]
	return df

def transpose(dm):
	n,m = df_size(dm)
	dm_n = create_dataframe(m,n)
	for c in range(m):
		it = 0
		for b in dm[c]:
			#print("dm_n[%s][%s] = %s" % (it,c,b))
			dm_n[it][c] = b
			it += 1
	return dm_n
def matrice(df):
	#convert dataframe file representation to matrix
	return transpose(df)
def list_multiplication(dm1,dm2):
	if length(dm1) == length(dm2):
		return [b*c for b,c in zip(dm1,dm2)]
	elif length(dm1) == 1 or length(dm2) == 1:
		if length(dm1) == 1:
			r = dm1[0]
			c = dm2
		elif length(dm2) == 1:
			r = dm2[0]
			c = dm1
		return [i*r for i in c]
	else:
		print("shape is not same for list multiplication")
		raise ValueError

def mat_dot(dm1,dm2):
	a,b=df_size(dm1)
	m,n=df_size(dm2)
	if(b==m):
		dm_n = create_dataframe(a,n)
		it=0
		for i in transpose(dm1):
			ij=0
			for c in dm2:
				dm_n[ij][it] = sum(list_multiplication(conv_type(i,'int'),conv_type(c,'int'))) 
				ij+=1
			it+=1
		return dm_n
	else:
		raise ValueError

def numpyarray2dataframelist(np):
	return np.T.tolist()
def dataframe2numpyarray(df):
	import numpy as np
	return np.array(df.T)

class DataFrame(object):
	'''This is Onnorokom General Library'''    
	def __init__(self, columns=[], dataframe = ['0']):
		#sequence checked
		self.dataframe = dataframe
		self.shape = self.framesize
		#self.T = self.trans
		self.columns = columns
		
		if self.dataframe != ['0']:
			self.shape = self.framesize
			self.T = self.trans
			if columns == []:
				self.columns = self.erase_col_names()
			

	def __del__(self):
		classname = self.__class__.__name__
		#print ('%s class destroyed' % (classname))

	def __str__(self):
		#return str(self.columns) + str(self.dataframe)
		#print(self.dataframe) #List representation
		strs = "Dataframe Representation\n"
		
		def str_sp(strs,space=10):
			
			for c in range(space - length(strs)):
				strs += " "
			return strs

		for c in self.columns:
			strs += str_sp(c)
		strs += "\n"
		for c in range( length(self.dataframe[0]) ):
			for d in self.dataframe:
				strs += str_sp( str(d[c]) )
				
			strs += "\n"
		return strs
	def __getitem__(self,name):
		if type(name) == int:
			
			return self.dataframe[name]
		elif type(name) == str:
			if name.isdigit() == True:
				return self[int(name)]
			
			return self.dataframe[getIndex(self.columns,name)]
		elif isinstance(name, slice):
			return self.dataframe[name]
	def __iter__(self):
        	return iter(self.columns)
        
        def normalize(self,change_self=False):
        	#we need to normalize the features using mean normalization
        	df = []
		for c in self.dataframe:
			mean= ArithmeticMean(c) 
			std = sample_standard_deviation(c)
			_ =[ (a - mean)/std for a in c]
			df = appends(df,[_])
		if change_self == True:
			self.dataframe = df
		return df
        def conv_type(self,var_type,change_self=False):
        	callables = eval(var_type)
        	df = []
        	col = []
        	for c in self.dataframe:
        		for i in c:

        			col = appends(col,callables(i))
        		df = appends(df,[col])
        		col = []

        	if change_self == True:
        		self.dataframe = df
        	return df
        	
        def ix(self):
        	pass
        def iloc(self):
        	pass
        def row(self,rowindex):
        	row = []
        	for c in self.columns:
        		row.append(self[c][rowindex])
        	return row
	def new(self,m,n,elm=''):
		if elm == '':
			elm = None
		df=[]
		df = [[elm]*m for _ in range(n)]
		#if change_self == True:
		return self.set_object(df)
	def concat(self,dm1,dm2,axis=0):
		#axis
		#[0 - row merge]
		#[1 - column merge]
		dm = []
		m,n = dm1.framesize
		x,y = dm2.framesize
		b = dm1.tolist
		d = dm2.tolist
		if axis == 0:
			if n != y:
				print('ValueError: all the input array dimensions except for the concatenation axis must match exactly')
				raise ValueError
			for c,a in zip(b,d):
				dm = appends(dm,[c+a])
		elif axis == 1:
			if m != x:
				print('ValueError: all the input array dimensions except for the concatenation axis must match exactly')
				raise ValueError
			
			for c in b:###
				dm = appends(dm,[c])
			
			for c in d:###
				dm = appends(dm,[c])
		return self.set_object(dm)
	def transpose(self,change_self=False):
		selfs = self.__class__()
		dm = self.dataframe
		n,m = self.size()
		dm_n = create_dataframe(m,n)
		for c in range(m):
			it = 0
			for b in dm[c]:
				#print("dm_n[%s][%s] = %s" % (it,c,b))
				dm_n[it][c] = b
				it += 1
		if change_self == True:
			self.dataframe = dm_n
			self.erase_col_names()
			#self.T = dm_n #previous_code 5-4-18
			#self.T = self.trans #previous_code 5-4-18
		
		#sequence for returing self after dataframe calculation
		selfs.dataframe	= dm_n
		selfs.shape = selfs.framesize
		selfs.columns = selfs.erase_col_names()
		return selfs
		#sequence for returing self after dataframe calculation
	def erase_col_names(self):
		self.columns = [str(c) for c in range(length(self.dataframe))]
		return self.columns
	def columns(self):
		return [ c for c in self.columns]
	def size(self,ob=[]):
		if ob == []:
			return length(self.dataframe[0]),length(self.dataframe)
		else:
			return length(ob[0]),length(ob)
	def read_csv(self,input_file,columns=[]):
		with open(input_file,'r') as file:#Taking file input as stream, handly for iterators and reading stream object saves memory losses
			data = file.readlines()#reading line by line
	
		first_line = split( strip(data[0],[' ','\n']) ,",")


		header = [c for c in range(0,length(first_line))]			
		
		if first_line[0].isdigit() == False:
			self.columns = first_line
			del data[0]
		else:
			self.columns = conv_type(header,"str")

		df = [[] for d in header]

		for c in data:
			line = split( strip(c,[' ','\n']) ,',')
			for d in header:
				df[d] = appends(df[d],line[d])

		if columns == []:
			columns = self.columns
		else:
			self.columns = columns
		
		#sequence for returing self after dataframe assigning
		self.dataframe = df
		self.shape = self.framesize
		self.T = self.trans
		#sequence for returing self after dataframe assigning

		return self
	def __sub__(self,dm2):
		dm1 = self
		selfs = self.__class__()
		m,n = dm1.shape
		x,y = dm2.shape
		dm=[]
		a = dm1.tolist
		b = dm2.tolist
		
		if m == x and n == y:
			j = 0
			for c in range(n):
				i = 0
				col = []
				for r in range(m):
					si = a[j][i] - b[j][i]
					i+=1
					col.append(si)
				j+=1	
				dm.append(col)
			return self.set_object(dm)
		else:
			print("Matrice Shape is not same for substract",dm1.shape,dm2.shape)
			raise ValueError
	def __add__(self,dm2):
		dm1 = self
		selfs = self.__class__()
		m,n = dm1.shape
		x,y = dm2.shape
		dm=[]
		a = dm1.tolist
		b = dm2.tolist
		
		if m == x and n == y:
			j = 0
			for c in range(n):
				i = 0
				col = []
				for r in range(m):
					si = a[j][i] + b[j][i]
					i+=1
					col.append(si)
				j+=1	
				dm.append(col)
			return self.set_object(dm)
		else:
			print("Matrice Shape is not same for substract",dm1.shape,dm2.shape)
			raise ValueError
	def __gt__(self, other):
		pass
	def __lt__(self, other):
		pass
	def __ge__(self, other):
		pass
	def __le__(self, other):
		pass
	def two2oneD(self):
		if self.shape[0] == 1:
			return self.T[0]
		elif self.shape[1] == 1:
			return self[0]
		else:
			print("Column/row != 1, can not converted to 1D list")

	def dot(self,dm1,dm2):
		a,b=dm1.shape
		m,n=dm2.shape
		if(b==m):
			dm_n = []
			it = 0
			for c in dm2.tolist:
				col = []
				for i in (dm1.T).tolist:
					col = appends(  col,sum( list_multiplication(i,c) )  )
				dm_n = appends(dm_n,[col])
				it+=1
			return self.set_object(dm_n)
		else:
			print("Shape is not same for matrice multiplication -> ",dm1.shape,dm2.shape)
			raise ValueError
	def __mul__(self,dm2):
		dm1 = self
		#used inverse dataframe to convert numpy array representation
		#then used numpy broadcasting
		c1 = (dm1.shape[0]==dm2.shape[0]) and (dm1.shape[1] == 1 or dm2.shape[1] == 1)
		c2 = (dm1.shape[1]==dm2.shape[1]) and (dm1.shape[0] == 1 or dm2.shape[0] == 1)
		if dm1.shape == dm2.shape or c1:
			dm = []
			dm1 = (dm1.T).tolist
			dm2 = (dm2.T).tolist
			for r1,r2 in zip(dm1,dm2):
				dm.append(list_multiplication(r1,r2))
			#they are equal, or
			#one of them is 1
			return self.set_object(transpose(dm))
		elif c2:
			dm = []
			dm1 = (dm1.T).tolist
			dm2 = (dm2.T).tolist
			for r1 in dm1:
				dm = [list_multiplication(r1,r2) for r2 in dm2]
			#they are equal, or
			#one of them is 1
			return self.set_object(transpose(dm))
		else:
			print("cross is not allowed")
			raise ValueError
	def sum(self,axis=0):
		if axis == 0:
			dm = [[sum(c)] for c in self.tolist]
		elif axis == 1:
			dm = [[sum(self.row(d)) for d in range(self.shape[0])]]
		return self.set_object(dm)
		
	def sum_np(self,axis=0):
		return self.sum(axis).two2oneD()
	def __float__(self):
		if self.shape == (1,1):
			return self[0][0]
	def __pow__(self,power):
		dm = [ [r**power for r in c] for c in self.tolist]
		return self.set_object(dm)
	def dftolist(self):
		return self.dataframe
	def dataA(self,dataframe):
		self.dataframe = dataframe
		return self
	def set_object(self,dm):
		selfs = self.__class__()
		#sequence for returing self after dataframe calculation
		selfs.dataframe	= dm
		selfs.shape = selfs.framesize
		selfs.T = selfs.trans
		selfs.columns = selfs.erase_col_names()
		return selfs
		#sequence for returing self after dataframe calculation

	framesize = property(size)
	tolist = property(dftolist)
	trans = property(transpose)
	prop_var =property(set_object)
