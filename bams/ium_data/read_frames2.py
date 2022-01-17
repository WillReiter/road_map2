# Python plugin that supports loading batch of images in parallel
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import concurrent.futures
import netCDF4 as nc
import time
from concurrent.futures import ThreadPoolExecutor,wait

_imread_executor_pool = ThreadPoolExecutor(max_workers=16)



def read_hmw_frame(path, read_storage):        ### read satellite data
    
    assert(os.path.exists(path))
    disk_to_mem_start = time.time()
    sat = nc.Dataset(path)

    data = np.array(sat['DBZ'])
    #disk_to_mem_temp= time.time() - disk_to_mem_start
    normal_start = time.time()
    data = np.clip( data /80.0 , 0.0, 1.0)  ##  [ 0 , 80 ] 
    data = data[:, 0:700, 0:700]
    print(data.shape)
    read_storage[:] = data
    sat.close()#print(path)
    normal_temp = time.time()-normal_start
    #disk_to_mem.append(disk_to_mem_temp)
    #norm.append(normal_temp)
    #return disk_to_mem,normal


def quick_read_frames(path_list, im_w=350, im_h=250, resize=False, frame_size=None, grayscale=True, normalization=True):

    frame_num = len(path_list) ## batch_size * seq_len
    print(path_list)

    for path in path_list:
        if not os.path.exists(path):
            print(path)
            raise IOError

    if grayscale:
        read_storage = np.empty((frame_num, 1, im_h, im_w), dtype=np.float32)
    else:
        read_storage = np.empty((frame_num, 2, im_h, im_w), dtype=np.float32)
    disk_to_mem = []
    norm = []
    if not resize:
        future_objs = []
        for i in range(frame_num):
            #d2m,n = read_hmw_frame(path=path_list[i], read_storage = read_storage[i])
            obj = _imread_executor_pool.submit(read_hmw_frame,path_list[i],read_storage[i],disk_to_mem,norm)
            future_objs.append(obj)
        wait(future_objs)
        read_storage = read_storage.reshape((frame_num, 1, im_h, im_w))

        return sum(disk_to_mem),sum(norm),read_storage


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    path = '/home/syao/Desktop/trajGRU/nc_selected/20170826/000035.nc'
    read_storage = np.empty((1, 600, 600), dtype=np.float32)
    read_hmw_frame(path, read_storage)
    print(read_storage)
    plt.imshow(read_storage[0,:], cmap = 'gray')
    plt.show()

 
