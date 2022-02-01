# -*- coding: utf-8 -*-   
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
#from ium_data.config import *
import sys
sys.path.append('road_map2/bams/ium_data/')
from config import *
'''
Output the datetime list of BJ-RADAR-DATA

timeinfo.txt

'''

def dirs_name(file_dir):
	""" return all dirs name.
	:type file_dir: str
	:rtype dirs_all: list[]
	
	"""
	dirs_all = []
	for p in os.listdir(file_dir):
		dirs_all.append(p)
	return dirs_all

def gzfile_name(file_dir):
	""" return all files name.
	:type file_dir: str
	:rtype file_all: list[]
	
	"""
    
	file_all = []
	for p in os.listdir(file_dir):
		file_all.append(p)
	return file_all


def write_datetime_list(file_path, output_path, datetime_list = None, save_name="time_info.txt"):
	""" save "time_info.txt".
	:type file_path: str
	:type output_path: str
	:type datetime_list: None
	:type save_name: str
	
	"""
	if datetime_list is None:
		datetime_list = []
		dirs = sorted(dirs_name(file_path))
		print(dirs)
		for dir_name in dirs:
			files = sorted(gzfile_name(os.path.join(file_path, dir_name)))
			for i in range(len(files)):
				file_time = dir_name + files[i][0:6]
				datetime_list.append(file_time)

	print("file num: ", len(datetime_list))
	with open(os.path.join(output_path, save_name), 'w') as f:
		for time in datetime_list:
			f.write(time)
			f.write("\n")

def read_datetime_list(file_path, save_name="time_info.txt"):
	""" read "time_info.txt", return a list
	:type file_path: str
	:type save_name: str
	:rtype datetime_list: list[]
	
	"""
	datetime_list = []
	with open(os.path.join(file_path, save_name), 'r') as f:
		for line in f:
			datetime_list.append(line.rstrip('\n').rstrip('\r'))
	print("datetime_list number: ", len(datetime_list))
	return datetime_list


if __name__ == '__main__':

	write_datetime_list(nc_data_path, info_path)
	#read_datetime_list(info_path, "time_info.txt")
	




