#string
def strip(string,strip_chars = [" "]):
	iters = True
	iteration = 0
	result = ""
	rstat = False
	for c in string:
		
		for d in strip_chars:
			if c != d:
				rstat = True
			else:
				rstat = False
				break

		if rstat == True:
			result += c
			rstat = False
		

		iteration += 1
	return result

def split(string,spliting_char = ','):
	#Spliting Functions by comma, or character. Default spliting character is comma.
	word = ""
	li = []
	iteration = 0

	for c in string:
		if c != spliting_char:
			word += c
			#if c == '':
			#	c = null_string
		else:			
			li = appends(li,word)
			word = ""
		iteration += 1
	#if word != "":
	li = appends(li, word)
	return li

def appends(lst, obj, index = -2):
	if index == -2:# -2 for accepting 0 value to be passed as it represents begining of the array
		index = length(lst)

	if type(obj) is list:
		return lst[:index] + obj + lst[index:]
	else:
		return lst[:index] + [obj] + lst[index:]
def swap(li,p1,p2):
	#used when we don't want to change the passed string, rather we want the copy of string to be changed
	if(p1 == p2):
		return li
	else:
		po1,po2 = li[p1],li[p2]
	
		'''if p2 > p1:
			interval = p2-p1
		else:
			interval = p1-p2
		li = li[:p1] + [po2] + li[p1+1:interval+1]+ [po1] + li[p2+1: len(li)]'''
		li = li[:p1] + [po2] + li[p1+1:]
		li = li[:p2] + [po1] + li[p2+1:]	

		return li

def sum(li):
	_ = 0.0
	for i in li:
		_ += i
	return _

def length(li):
	iteration = 0
	for c in li:
		iteration += 1
	return iteration


def getIndex(li,val):
	#indexes = [i for i,x in enumerate(li) if x == val]
	#indexes[-1]
	iteration = 0
	for c in li:
		if(val == li[iteration]):
			return iteration
		iteration += 1
	return False
def getIndexes(val,li):
	indexes = [i for i,x in enumerate(li)]
	return indexes

#math Functions
def sum_all(*args):
	sum = 0
	for num in args:
		sum += num
	return sum
def nroot(n,number):
	return number**(1/float(n))

#Utility Functions
def auto_include(starts_with='RML',extension='py'):
	from os import listdir
	modules = [f[0:(length(f)-length('.'+extension))] for f in listdir('.') if f[-3:] == '.'+extension and f[0:length(starts_with)]==starts_with]
	return modules


    
def print_dic(**kwargs):#if kwargs is not None:
    for key, value in kwargs.items():
        print(key + ": " + value)

#Search and sorting
def binary_search(li, target, start = 0, end = None):
	#binary search
	if end is None:
		end = length(li) - 1

	if(end - start + 1 <= 0):
		return False
	else:
		midpoint = start + ( end - start )//2
		if li[midpoint] == target:
			return midpoint
		else:
			if li[midpoint] > target:
				return search(li, target, start, midpoint-1)
			else:
				return search(li, target, midpoint+1, end)

def quicksort(li, begin=0, end=None):
	if end is None:
		end = length(li)-1
	
	pivot = begin
	for i in xrange(begin+1, end+1):
		if li[i] <= li[begin]:
	    		pivot += 1		
			li = swap(li,i,pivot)
	li = swap(li,pivot,begin)
	
	if begin < end:
		li = quicksort(li, begin, pivot-1)
	if begin < end:
		li = quicksort(li, pivot+1, end)
	
    	return li


#iterables or str
def maxn(li):
	iteration = 0
	maxn = li[0]
	for c in li:
		if maxn < li[iteration]:
			maxn = li[iteration]
		iteration  += 1
	return maxn
def minn(li):
	iteration = 0
	minn = li[0]
	for c in li:
		if minn > li[iteration]:
			minn = li[iteration]
		#elif maxn >= li[iteration]:
		#	pass
		iteration  += 1
	
	return minn

def conv_type(obj,type_var):
	dts = ["int","float","str"]
	st = 0
	for c in dts:
		if type_var == c:
			st = 1
			break

	if st != 1:
		raise Exception('No avaiable conversion type passed')

	if type(obj) is list:
		#print("list")
		pass
	elif type(obj) is str:
		print("string")
	elif type(obj) is int:
		print("Integer")
	elif type(obj) is float:
		print("Float")
	else:
		print("else %s" % type(obj))

	lists = []
	callables = eval(type_var)
	for c in obj:
		try:
			lists = appends(lists,callables(c))
		except ValueError:
			lists = appends(lists,c)

	return lists

