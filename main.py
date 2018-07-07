#!/usr/bin/python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from OGL_Stat import *
from OGL_ML import *
from OGL_DataFrame import *


def cliid(depth):
	
	
	if depth == 1:
		print("Welcome OML..., type restart/exit anytime you want")
		print("How you want to give input?\nOption:\n1. For File Input -> type \"F\" \n2. For Text input -> type \"T\"\n and hit enter")
		
		input = input_keyboard()

		if(input == "F"):
			print("File Input")
			depth = 3

		elif(input == "T"):
			depth = 2
			return depth
		elif(input == False):
			depth = False
			return depth
	
	elif depth == 2:
		print("You selected - (T) Text Input")
		input = input_keyboard("Give the list. Input format: number1, number2, number3, ....:")
		print("your input: %s" % input)
		
		if input == "restart":
			depth = 1
			return depth
		elif input == False:
			depth = False
			return depth
		else:
			items = conv_type(   split(strip(input))  ,"int")
			if check_datatype(items) == True:	
				print("Your given list is : %s" % split(strip(input)))
				depth = 4
				return depth
			else:
				print("Illegal data entered\n")
				return depth

	elif depth == 3:
		pass
	elif depth == 4:
		print("\nWhat statistical Function you want to apply on your dataset?\n1. Press (1) for Arithmetic Mean or average\n2. Press (2) for Geometric Mean\n3. Press (3) for Harmonic Mean.\n4. Press (4) for Mode.\n5. Press (5) for Median.")
		input = input_keyboard("Your input:")

		if input == '1':
			print("\nArithmetic Mean is - %s" % ArithmeticMean(items))
		elif input == '2':
			print("\nGeometric Mean is - %s" % GeometricMean(items))
		elif input == '3':
			print("\nHarmonic Mean is - %s" % HarmonicMean(items))
		elif input == '4':
			print("You have selected for Mode")
			print("\nMode is - %s" % mode(items))
		elif input == '5':
			print("You have selected for Median")
			print("\nMedian is - %s" % median(items))
		elif input == "restart":
			depth = 1
			return depth
		elif input == False:
			depth = False
			return depth

		elif input == "restart":
			depth = 1
			return depth
		elif input == False:
			depth = False
			return depth
		else:
			print("Wrong Input")

	#if accidently depth is not returned
	return depth


def cli(depth):
	print("Welcome OML..., type restart/exit anytime you want")
	print("How you want to give input?\nOption:\n1. For File Input -> type \"F\" \n2. For Text input -> type \"T\"\n and hit enter")
	
	input = input_keyboard()
	
	
	if depth == 1:
		if(input == "F"):
			print("File Input")

		elif(input == "T"):
			print("You selected - (T) Text Input")
			input = input_keyboard("Give the list. Input format: number1, number2, number3, ....:")
			print("your input: %s" % input)
			
			if input == "restart":
				depth = 1
				return depth
			elif input == False:
				depth = False
				return depth
			else:
				items = conv_type(   split(strip(input))  ,"int")
				if check_datatype(items) == True:	
					print("Your given list is : %s" % split(strip(input)))
					
				else:
					print("Illegal data entered\n")
					return depth

			

			print("\nWhat statistical Function you want to apply on your dataset?\n1. Press (1) for Arithmetic Mean or average\n2. Press (2) for Geometric Mean\n3. Press (3) for Harmonic Mean.\n4. Press (4) for Mode.\n5. Press (5) for Median.")
			input = input_keyboard("Your input:")

			if input == '1':
				print("\nArithmetic Mean is - %s" % ArithmeticMean(items))
			elif input == '2':
				print("\nGeometric Mean is - %s" % GeometricMean(items))
			elif input == '3':
				print("\nHarmonic Mean is - %s" % HarmonicMean(items))
			elif input == '4':
				print("You have selected for Mode")
				print("\nMode is - %s" % mode(items))
			elif input == '5':
				print("You have selected for Median")
				print("\nMedian is - %s" % median(items))
			elif input == "restart":
				depth = 1
				return depth
			elif input == False:
				depth = False
				return depth

		elif input == "restart":
			depth = 1
			return depth
		elif input == False:
			depth = False
			return depth




		else:
			print("Wrong Input")
	elif(depth == 0):
		return False

	return depth

def main1():
	li = [3,6,2,54,43,65,76,87,89,23,90]
	depth = 1
	while True:
		depth = cli(depth)
		if depth == False:
			break
def execution_time(func):
	import time
	start_time = time.time()
	eval(func)
	print("--- %s seconds ---" % (time.time() - start_time))
def logistic_regression():
	pass
def main():
	dm = read_csv('sample_inputs/home.txt',names=["size","bedroom","price"])
	print(dm)
	#x = conv_type(dm['length(s)'],'float')
	#y = conv_type(dm['views'],'float')
	#linear_regression( x, y, title='Youtube View Prediction', xlabel='length', ylabel='views' )
	#plot_eachpoint_connected(x,y)
	#max_min_rectangle(x,y)
	#print(compute_distance([2,5],[3,8]))

if __name__ == "__main__":
	main()