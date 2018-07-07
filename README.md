# Robist Machine Library

A Library for handling general tasks regarding statistcis and ml with in depth detailing

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Requirements
```
Packages: Math
```

### Installing

A step by step series of examples that tell you have to get a development env running

To setup the project to your local machine simply type the command in git bash
`git clone https://github.com/ShaonMajumder/rml.git`

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo
### Function Manual

#### For Math
1. nroot(n,number) - return the nth root of a number
2. sum_all(*args) - return sum of all numbers seperated by comma
#### For Utility
1. auto_include(starts_with='RML',extension='py') - automatically include all py files starts with'RML'
2. print_dic(**kwargs) - print key="value" pairs
#### For Sorting and Searching
1. binary_search(itearble,target) - return the position of the target
2. quicksort(iterable) - return sorted iterable

#### For Iterable
1. maxn(iterable) - returns the max number of an iterable
2. minn(iterable) - returns the min number of an iterable
3. conv_type(iterable,"type") - converting iterable items to a certain data type
4. sum(iterable) - returns sum of an iterable which contains number
5. appends(iterable,object,index or empty) - appends an object on behind or specific position of an iterable , then returns
6. swap(iterable, position1, position2) - swaps items according to their index and returns the new iterable
7. length(iterable) - returns length of an iterable or string
8. getIndex(iterable,val) - returns Index of an item inside an iterable
9. getIndexes(iterable) - returns Indexes of iterable

#### For String

1. strip(string,strip_chars) - remove white spaces,newlines or any special chars contained in strip_chars iterable passed from string and returns
2. split(string,spliting_char = ',') - create list by dividing string with the spliting char and returns

#### For DataFrame
##### Functions
1. columns(df) - returns columns of a dataframe object
2. read_csv(input_file_url) - returns dataframe object from text file conversion
3. df_size(df) - returns the dataframe dimension as row , column
4. create_empty_dataframe(m,n) - returns an empty list with the given dimension
5. transpose(df) - returns transpose matrice orrientation of data list or integer indexed dictionary
6. matrice(df) -  assign matrice by matrice representation and convert to machine readable list
7. list_multiplication(dm1,dm2) - multiply two list of same size
8. mat_dot(dm1,dm2) - matrice dot multiplication between two dic or list
Machine Readable Dataframe list representation [column1=[c1row1,c1row2,c1row3]  column2=[c2row1,c2row2,c2row3]]
##### Class : (1)DataFrame
###### Creating Classobject
1. classobject = DataFrame(dataframe = [[2,3,4],[2,5,2]]) - assign dataframe by Machine Readable Dataframe list representation
2. classobject = DataFrame().read_csv(file_link) -  assign dataframe by reading text file of csv formate
###### Class Properties
3. classobject - get dataframe representation
4. classobject[colindex] - access a column with index
5. classobject[colindex][rowindex] - access a cell with 2d index
6. classobject['columnname'] - access a column with index_name or index_string
7. classobject.tolist - get list representation
8. classobject.shape - Shape of dataframe
9. classobject.columns - Get column names
10. classobject.columns=['low','up','freq'] - renaming columns
###### Class methods
10. classobject.create_dataframe(m,n,elm=None) - element with None , DataFrame of m x n
11. classobject.T - Transposed and returned
12. classobject.transpose(change_self = True) - Transposed itself and also returned
13. classobject.conv_type('int',change_self=True) - change datatype of dataframe
14. classobject.normalize() - normalizes the value of dataframe
15. classobject.concatenate(classobject1,classobject1,axis=0) - add two dataframe in x or y axist
16. classobject.sum(axis=0) - adding matrice to row/columns opposite to val
17. classobject.substract(dm1,dm2) - return by subtracting two matrice
18. classobject.power(n) - raise the dataframe to the power of n then return
19. classobject.mat_dot(dm1,dm2) - matrice dot multiplications for dataframe
20. classobject.cross(dm1,dm2) - multiply dataframes by broadcasting method
21. classobject.row(rowindex) - return a row by index
#### For Statistics
1. ArithmeticMean(iterable) - for single column
  ArithmeticMean(lower,upper,frequency) - for class distribution
2. GeometricMean(iterable) #for single column
  GeometricMean(lower,upper,frequency) #for class distribution
3. HarmonicMean(iterable) #for single column
  HarmonicMean(lower,upper,frequency) #for class distribution
4. mode(iterable) #for single column
  mode(lower,upper,frequency) #for class distribution
5. median(iterable) #for single column
  median(lower,upper,frequency) #for class distribution
6. sample_standard_deviation(iterator) #for single column
7. sample_variance(li)
8. co_standard_deviation(li,pi)
9. co_variance(li,pi)
10. pearson_correlation_coefficient(li,pi)
11. normalize(df) - scaling data for creating distributed dominance
12. reference_reverse_normalize(ypure,y_pred) - reverse scaling data by actual and normalized reference
#### For Machine Learning
1. MLVR(XDATA, YDATA, xreference=0, residual=1, xlabel='', ylabel='', title='', alpha=0.01, iters=1000, plot=1) -
        Does Multivariant Linear Regression
        properties:
                XDATA = The Feature Dataframe
                YDATA = The Target Dataframe
                xreference = 1/0 -> The column index in XDATA for ploting graph
                xlabel = Label for X in Graph
                ylabel = Label for Y in Graph
                title = title for graph]
                alpha = Learning rate for model
                iters = the number of iteration to train the model
2. compute_distance(p1, p2) - returns distance between two point
3. estimate_coef(x, y) - Estimate co-efficients m,c for straight line mx+c
4. get_max_area()  
5. get_max_rectangele()
6. give_time_series(x, y) - Rearrange X,Y value pairs or points according to X's order
7. linear_regression(x, y, title='', xlabel='X', ylabel='Y') - Does simple linear regression
8. maxResidual(pure, pred) - returns maximum error distance or residual
9. max_min_rectangle(x, y) - Plot a rectangle using max and min point from a distribution
10. meanResidual(pure, pred) - returns average error distance or residual
    
11. minResidual(pure, pred) - returns minimum error distance or residual
12. plot_eachpoint_connected(x, y) - Plot connecting every point with each other from a distribution    
13. plot_error_distance(x, y_pred, y_actual) - Plot error distance or residual  
14. plot_regression_line(x, y, b, title='', xlabel='X', ylabel='Y') - ploting the prediction line using simple linear regression
### Functionality
1.Statistical
2. DataFrame
3. Machine Learning
### Input and Output
Input Formats for DataFrame
Single Column
```
1921
1925
1945
1947
```
Multiple Column
```
serial, name,fair
11,20,34
21,30,4
31,40,33
41,50,7
```

### *Cautions*
1. Always conv the list or dataframe object by conv_type(var,type) to number more specificly float to avoid any error in using Statistical Functions.

### Example
Math functions
```
	print(sum_all(2,3,4))
	print(nroot(2,4))
```
Utility functions
```
	print_dic(name="Shaon",Position="Data Scientist")
```
DataFrame Class all properties
```
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
	print('empty DataFrame of 2x3',dm1.create_dataframe(2,3))
	print('element with 1 , DataFrame of 2x3',dm1.create_dataframe(2,3,elm=1))

	print('Transposed and returned',dm1.T)
	print(dm1)
	print('Transposed itself and also returned',dm1.transpose(change_self = True))
	print(dm1)
```
Class DataFrame datatype conversion
```
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
```
Add two dataframe by rows
```
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
```
Add two dataframe by columns
```
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	my_data.conv_type('float',change_self=True)
	my_data.normalize(change_self=True)
	
	X = DataFrame(dataframe= my_data[0:2])
	
	ones = DataFrame().create_dataframe(X.framesize[0],1,elm=1.)
	
	
	X = DataFrame(dataframe=DataFrame().concatenate(ones,X,axis=1))
	print(X)
```
Subtract two dataframe by same size
```
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	my_data.conv_type('float',change_self=True)
	X = DataFrame(dataframe= my_data[0:2])
	Y = DataFrame(dataframe= my_data[1:3])
	print(DataFrame().substract(X, Y))
```
n power of dataframe
```
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	my_data.conv_type('float',change_self=True)
	my_data.power(2)
	print(my_data)
```
Summation to rows or column of dataframe
```
	#adding rows
	liq = DataFrame().mat_dot( DataFrame(dataframe=[[1,4],[2,5],[3,6]]), DataFrame(dataframe=[[7,9,11],[8,10,12]]) )
	print(DataFrame(dataframe =  liq.tolist ))
	print(DataFrame(dataframe =  liq.tolist ).sum(axis=1))
	#adding columns
	liq = DataFrame().mat_dot( DataFrame(dataframe=[[1,4],[2,5],[3,6]]), DataFrame(dataframe=[[7,9,11],[8,10,12]]) )
	print(DataFrame(dataframe =  liq.tolist ))
	print(DataFrame(dataframe =  liq.tolist ).sum(axis=0))
```
Statistics on Single Column Data
```
	from RML_DataFrame import *
	from RML_Stat import *
	df3 = DataFrame().read_csv('sample_inputs/input2.txt')
	print(df3)
	print("ArithmeticMean = %s" % ArithmeticMean(conv_type(df3[0],"int")) )
	print("GeometricMean = %s" % GeometricMean(conv_type(df3[0],"int")))
	print("HarmonicMean = %s" % HarmonicMean(df3[0]))
	print("Mode = %s" % mode(df3[0]))
	print("Median = %s" % median(df3[0]))
```
Statistics on Class Distribution or Grouped Data
```
	from RML_DataFrame import *
	from RML_Stat import *
	df = DataFrame().read_csv('sample_inputs/input1.txt')
	print(df)
	lower, upper, frequency = conv_type(df['lower'],"int"),conv_type(df['upper'],"int"),conv_type(df['frequency'],"int")
	print("ArithmeticMean = %s" % ArithmeticMean(lower=lower,upper=upper,frequency=frequency))
	print("GeometricMean = %s" % GeometricMean(lower=lower,upper=upper,frequency=frequency))
	print("HarmonicMean = %s" % HarmonicMean(lower=lower,upper=upper,frequency=frequency))
	print("Mode = %s" % mode(lower=lower,upper=upper,frequency=frequency))
	print("Median = %s" % median(lower=lower,upper=upper,frequency=frequency))
```
Simple Linear Regression from textfile
```
	from RML_DataFrame import *
	from RML_ML import *
	dm = DataFrame()
	dm.read_csv('sample_inputs/matrice2.txt')
	linear_regression(conv_type(dm[0],'int'),conv_type(dm[1],'int'))
```
Simple Linear Regression from textfile by dataframe representation
```
	from RML_DataFrame import *
	from RML_ML import *
	dm = DataFrame().read_csv('sample_inputs/matrice2.txt')
	linear_regression(conv_type(dm[0],'int'),conv_type(dm[1],'int'))
```
Multivariant Linear Regression from textfile by dataframe representation
```
	my_data = DataFrame().read_csv('sample_inputs/home.txt',columns=["size","bedroom","price"])
	
	my_data.conv_type('float',change_self=True)
	my_data.normalize(change_self=True)
	XDATA = DataFrame(dataframe= my_data[0:2],columns=['size','bedroom'])
	YDATA = DataFrame(dataframe= [my_data[2]])
	
	multivariant_linear_regression(XDATA,YDATA,xreference=0,residual=0,xlabel='size',ylabel='price',title='Multivariant Linear regression')
```
Matrice dot multiplication from textfile
```
	dm1 = read_csv('sample_inputs/matrice.txt')
	dm2 = read_csv('sample_inputs/matrice3.txt')
	res = mat_dot(dm1,dm2)
	print(res)
```
Matrice dot multiplication from textfile by Dataframe representation
```
	from RML_DataFrame import *
	dm1 = DataFrame().read_csv('sample_inputs/matrice.txt')
	dm2 = DataFrame().read_csv('sample_inputs/matrice3.txt')
	print(  DataFrame().mat_dot(dm1,dm2)  )#creates another object
	print(  mat_dot(dm1.dataframe,dm2.dataframe)  )#efficient
	print(  mat_dot(dm1.tolist,dm2.tolist)  )#efficient
```
Matrice dot multiplication from matrice representation
```
	from RML_DataFrame import *
	dm1 = matrice( [[4,8],
		        [0,2],
	        	[1,6]] )
	dm2 = matrice( [[5,2],
		        [9,4]] )
	res = mat_dot(dm1,dm2)
	print(res)
```
Rectangle with Max, Min point
```
	from RML_DataFrame import *
	from RML_ML import *
	dm1 = DataFrame().read_csv('sample_inputs/youtube.txt')
	max_min_rectangle(dm1['length(s)'],dm1['views'])
```
### For Repository Managers
#### Configs
git config --global user.name "shaon"
git config --global user.email "smazoomder@gmail.com"

#### For adding the Remote Host
```
git remote add origin https://github.com/ShaonMajumder/rml.git
```
#### For sending updates from client to remote host
```
git add .
git commit -m "This a commit"
git push -u origin master
```
#### For receving updates from remote host to client
```
git pull origin master
```

## Troubleshoots
No module Found due to root acces
```
sudo jupyter notebook --allow-root
```

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```
## Rules
Library names must starts with RML , other file can not be start as RML

## Deployment

Add additional notes about how to deploy this on a live system


## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Shaon Majumder** - [Github](https://github.com/ShaonMajumder)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Sabbir Amin - For guideline for mentoring
* Inspiration
* etc
