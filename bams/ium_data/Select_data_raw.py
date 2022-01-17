'''
Select data whose DBZ > 35
'''
import time
import sys
import numpy as np

sys.path.append("/media/4T/yangjitao/trajgru_pytorch_radar/ium_data")
from ium_data.bj_iterator import *


'''
select data 
'''

def Select_function(frame_data, Datetime_batch):
    
    frame_data[frame_data>=0.4375] = 1

    data = np.sum(frame_data[frame_data==1])
    data /= 35

    if data<50:
        with open('./no_rainday.txt', 'a') as file0:
                file0.write(Datetime_batch[0][0])
                file0.write("\t")
                file0.write(str(data))
                file0.write("\n")

    else:
        with open('./rainday.txt', 'a') as file1:
                file1.write(Datetime_batch[0][0])
                file1.write("\t")
                file1.write(str(data))
                file1.write('\n')

    return [data, Datetime_batch[0][0]]


if __name__=="__main__":

    train_bj_iter = BJIterator(datetime_set="/home/syao/Desktop/trajGRU/bams/ium_data/info/squence_len_35/bj_train_set.txt",
                               sample_mode="sequent", seq_len=35, width=600, height=600)


    start = time.time()
    data_list = []


    for i in range(155716):

        print("checking sequence  {} / 155716".format(i))
        frame_data, mask_data, Datetime_batch, _ = train_bj_iter.sample(batch_size=1)
        select_data = Select_function(frame_data, Datetime_batch)
        print(select_data)
        
    end = time.time()
    print("time used", end-start)




    
    
