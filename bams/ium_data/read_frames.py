# Python plugin that supports loading batch of images in parallel
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import concurrent.futures
import netCDF4 as nc
import time



def read_hmw_frame(path, read_storage):        ### read satellite data
    
    assert(os.path.exists(path))
    #print("read_frames")
    disk_to_mem_start = time.time()
    sat = nc.Dataset(path)

    data = np.array(sat['DBZ'])
    #disk_to_mem = time.time() - disk_to_mem_start
    #normal_start = time.time()
    data = np.clip( data /60.0 , 0.0, 1.0)  ##  [ 0 , 80 ]
    data = data[:, 0:600, 0:600]
    read_storage[:] = data
    sat.close()#print(path)
    #normal = time.time()-normal_start
    #return disk_to_mem,normal


def quick_read_frames(path_list, im_w=350, im_h=250, resize=False, frame_size=None, grayscale=True, normalization=True):

    frame_num = len(path_list) ## batch_size * seq_len

    for path in path_list:
        if not os.path.exists(path):
            #print(path_list)
            #print(path)
            raise IOError

    if grayscale:
        read_storage = np.empty((frame_num, 1, im_h, im_w), dtype=np.float32)
    else:
        read_storage = np.empty((frame_num, 2, im_h, im_w), dtype=np.float32)
    if not resize:
        for i in range(frame_num):
            read_hmw_frame(path=path_list[i], read_storage = read_storage[i])
        read_storage = read_storage.reshape((frame_num, 1, im_h, im_w))

        return read_storage


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    path = '/media/will/SSD/raw_data/nc_selected/20170401/001235.nc'
    read_storage = np.empty((1, 600, 600), dtype=np.float32)
    read_hmw_frame(path, read_storage)
    print(read_storage.shape,read_storage.max())
    plt.imshow(read_storage[0,:], cmap = 'gray')
    plt.show()

 
