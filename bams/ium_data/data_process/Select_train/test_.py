'''
caculate the number of different day
 
'''
import numpy as np
import torch

number_40 = []
number_30 = []
number_20 = []
number_10 = []
number_0 = []
number = 0

file = open('Select_norainday.txt')

for line in file.readlines():
    number += 1
    curline = line.strip('\n').split("\t")
    curline = float(curline[1])

    if (curline>40 and curline<=50):
        number_40.append(curline)
    if (curline>30 and curline<=40):
        number_30.append(curline)
    if (curline>20 and curline<=30):
        number_20.append(curline)
    if (curline>10 and curline<=20):
        number_10.append(curline)
    if (curline>0 and curline<=10):
        number_0.append(curline)

print(num)
print(len(number_40))
print(len(number_30))
print(len(number_20))
print(len(number_10))
print(len(number_0))
    


