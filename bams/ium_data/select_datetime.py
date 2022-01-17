import os
import numpy as np
import subprocess
import sys
sys.path.append("/home/syao/Desktop/trajGRU/bams/ium_data")
from ium_data.bj_iterator import *

def save_valid_datetime(file_path):
    valid_datetime = []
    with open(file_path, 'r') as f:
        for line in f:
            valid_datetime.append(line[0:14])
    return valid_datetime
    # valid_datetime = np.array(valid_datetime)
    # np.save("valid_datetime.npy", valid_datetime)


def select_datetime(raw_path,dataset_path):
    valid_datetime = np.load("./valid_datetime.npy")
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    #创建dataset文件夹
    new_name = [i[0:8] for i in valid_datetime]
    for name in new_name:
        new_dir = os.path.join(dataset_path,name)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
    #构造base路径
    base_path = [i[0:8] +'/'+i[8:] + '.nc' for i in valid_datetime]
    for file in base_path:
        raw_filepath = os.path.join(raw_path,file)
        new_filepath = os.path.join(dataset_path,file[0:8])


        command = "mv " + raw_filepath + ' ' + new_filepath
        print(command)
        subprocess.call(str(command), shell=True)


if __name__ == '__main__':
    # valid_train_datetime = save_valid_datetime("/home/syao/Desktop/trajGRU/bams/ium_data/info/rainday_train.txt")
    # valid_valid_datetime = save_valid_datetime("/home/syao/Desktop/trajGRU/bams/ium_data/info/rainday_valid.txt")
    # valid_test_datetime = save_valid_datetime("/home/syao/Desktop/trajGRU/bams/ium_data/info/rainday_test.txt")
    # print(len(valid_train_datetime), len(valid_valid_datetime), len(valid_test_datetime))
    # valid_datetime = valid_train_datetime + valid_valid_datetime + valid_test_datetime
    # valid_datetime = np.array(valid_datetime)
    # print(valid_datetime.shape)
    # np.save("valid_datetime.npy", valid_datetime)

    # valid_datetime = np.load("./valid_datetime.npy")
    # with open('./datetime.txt', 'a') as file3:
    #     for i in valid_datetime:
    #         file3.write(i)
    #         file3.write("\n")
    # train_bj_iter = BJIterator(datetime_set="C:/Users/sssss/Desktop/trajGRU/bams/ium_data/datetime.txt",
    #                            sample_mode="sequent", seq_len=35, width=600, height=600)
    # _, _, Datetime_batch, _ = train_bj_iter.sample(batch_size=1)
    select_datetime("/home/syao/Desktop/trajGRU/nc_selected","/home/syao/Desktop/trajGRU/Hu_dataset")
    #
