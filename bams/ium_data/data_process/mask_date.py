# -*- coding: utf-8 -*-   
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np
from datetime import datetime, timedelta

from ium_data.config import *
from ium_data.data_process.get_date_list import read_datetime_list

"""
## read this at first ##
output datetime mask:
timedelta = next_frame_datetime - current_frame_datetime
if all $(seq_len) frames timedelta in specific range(i.g: 5min-7min),
then the start frame corresponding "1",
which means those $(seq_len) frames can make up a available sample

time_mask_$(seq_len)_timedelta_${lowmin}${highmin}.txt
"""


def mask_datetime_delta(datetime_list, low_min = 5, high_min = 7):
	"""
	:type datetime_list: list
	:type low_min: int
	:type high_min: int
	:rtype: list[]
	
 	"""
	mask_data = []
	num_true = 0
	num_false = 0
	for i in range(len(datetime_list) - 1):
		time1 = datetime.strptime(datetime_list[i], '%Y%m%d%H%M%S')
		time2 = datetime.strptime(datetime_list[i+1], '%Y%m%d%H%M%S')
		
		min_delta = (time2 - time1).total_seconds() / 60.0   # calculate min,  do not ignore year & day !!
		if low_min < min_delta and min_delta < high_min:
			num_true += 1
			mask_data.append(True)
		else:
			num_false += 1
			mask_data.append(False)

	mask_data.append(False); num_false += 1 # the last one
	assert len(mask_data) == len(datetime_list)
	print("num_false: ", num_false)
	print("num_true: ", num_true)
	return mask_data

def find_valid_datetime(mask_data, seq_len=25, invalid_frames=0):
	"""
	:type mask_data: list[]
	:type seq_len: int
	:type invalid_frames: int
	:rtype: list[]
	
	"""
	assert invalid_frames == 0, "no implement invalid frames threshold!"
	valid_data = []
	valid_len = len(mask_data) - (seq_len - 1)
	
	for i in range(valid_len):
		valid_frames = sum(mask_data[i:i+seq_len-1])  # sum (seq_len-1) frames
		if (valid_frames >= seq_len - 1 - invalid_frames):
			valid_data.append(True)
		else:
			valid_data.append(False)

	for i in range(seq_len - 1):
		valid_data.append(False)

	assert(len(mask_data) == len(valid_data))
	return valid_data

def write_valid_datetime(file_path, datetime_list, valid_data, seq_len, low_min=4, high_min=9):
	"""
	:type file_path: str
	:type datetime_list: list[]
	:type valid_data: list[]
	:type seq_len: int
	:type low_min: int
	:type high_min: int
	
	"""
	assert(len(datetime_list) == len(valid_data))
	file_path = os.path.join(file_path, "squence_len_{}".format(seq_len))
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	with open(os.path.join(file_path,\
		"time_mask_"+str(seq_len)+"_timedelta_"+str(low_min)+str(high_min)+".txt"), 'w') as f:
		for i in range(len(valid_data)):
			f.write(datetime_list[i])
			f.write("\t")
			f.write(str(int(valid_data[i])))
			f.write("\n")

def read_valid_datetime(file_path, file_name = "time_mask.txt"):
	"""
	:type file_path: str
	:type file_name: str
	:rtype: np.array
	
	"""
	valid_datetime = []
	with open(os.path.join(file_path, file_name), 'r') as f:
		for line in f:
			new_line = line.rstrip('\n').split('\t')
			valid_datetime.append([int(new_line[0]), int(new_line[1])])

	return np.array(valid_datetime)

def write_time_mask(seq_len=15, low_min=4, high_min=9):
	"""
	:type seq_len: int
	:type low_min: int
	:type high_min: int
	
	"""
	## 调用read_datetime_list(), 返回一包含所有样本时间信息的列表
	datetime_list = read_datetime_list(info_path, "time_info.txt")
	
	## 调用mask_datetime_delta(), 返回有效间隔及无效间隔
	mask_data = mask_datetime_delta(datetime_list, low_min, high_min)
	
	## 调用find_valid_datetime(), 根据有效间隔找到有效的完整 seq_len 帧样本
	valid_data = find_valid_datetime(mask_data, seq_len)
	
	## 调用write_valid_datetime()
	write_valid_datetime(info_path, datetime_list, valid_data, seq_len, low_min, high_min)
	

if __name__ == '__main__':
	write_time_mask(seq_len=35)
	#mask_datetime_delta(datetime_list, low_min=5, high_min=7)
	#valid_datetime = read_valid_datetime(info_path, "time_mask_raw.txt")
	#test_valid_datetime(valid_datetime)

