# -*- coding: utf-8 -*-
from RML import *
def check_datatype(li):
	typ = type(li)
	if typ is list:

		for obj in li:
			typ = type(obj)

			
			if typ is str:
				if obj.isdigit():
					return True
				else:
					return False
			 
				obj = float(obj)
			elif typ is int:
				obj = float(obj)
			elif typ is float:
				pass
			elif typ is list:
				print("else type%s" % type(obj))
				return False
		return True

	else:
		return False


def check_datatyped(li):
	typ = type(li)
	if typ is list:
		for obj in li:
			typ = type(obj)
			
			if typ is str:
				print("string")
			elif typ is int:
				print("Integer")
			elif typ is float:
				print("Float")
			else:
				print("else %s" % type(obj))

	str = typ.__name__
	return str



def check_input(com,avaiable_commands):
	#usage: if check_input(com,['restart','exit',''])
	pass

				

def input_keyboard(stri="Your Option :"):
	input = raw_input("\n"+stri)
	if input == "exit":
		print("Exiting the program ...")
		return False
	#if input == "restart":
	#	goto(323)
	return input
def goto(line) :

	global depth

	depth = line

