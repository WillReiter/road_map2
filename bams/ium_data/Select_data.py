'''
Select data whose DBZ > 35
'''
import time
import sys
import numpy as np

sys.path.append("/home/syao/Desktop/trajGRU/bams/ium_data")
from ium_data.bj_iterator import *


'''
select data 
'''

def Select_function(frame_data, Datetime_batch):
    for i in range(frame_data.shape[1]):
        #print(frame_data.shape,frame_data[:,i,:,:,:].shape)
        frame_data[:,i,:,:,:][frame_data[:,i,:,:,:]>=0.4375] = 1
        data = np.sum(frame_data[:,i,:,:,:][frame_data[:,i,:,:,:]==1])
        data /= 35
        print(frame_data[0,i,:,:,:].max(), Datetime_batch[i][0])
        if data<50:
            with open('/home/syao/Desktop/trajGRU/bams/ium_data/info/no_rainday.txt', 'a') as file0:
                    file0.write(Datetime_batch[i][0])
                    file0.write("\t")
                    file0.write(str(data))
                    file0.write("\n")

        else:
            with open('/home/syao/Desktop/trajGRU/bams/ium_data/info/rainday.txt', 'a') as file1:
                    file1.write(Datetime_batch[i][0])
                    file1.write("\t")
                    file1.write(str(data))
                    file1.write('\n')

    return [data, Datetime_batch[0][0]]


if __name__=="__main__":
    path = "/home/syao/Desktop/trajGRU/bams/ium_data/info/squence_len_35/bj_test_set.txt"
    train_bj_iter = BJIterator(datetime_set=path,
                               sample_mode="sequent", seq_len=35, width=600, height=600)


    start = time.time()
    data_list = []
    batch_size = 100
    for i in range(477011):

        print("checking sequence  {} / 477011".format(i))
        frame_data, mask_data, Datetime_batch, _ = train_bj_iter.sample(batch_size=batch_size)
        select_data = Select_function(frame_data, Datetime_batch)
        print(select_data)
        
    end = time.time()
    print("time used", end-start)




    
    
