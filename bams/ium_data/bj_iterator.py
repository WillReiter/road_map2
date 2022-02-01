from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import time
import sys

sys.path.append('./road_map2/bams/ium_data')
#from ium_data.config import *
#from ium_data.draw_frames import draw_all_frames, draw_frame
#from ium_data.read_frames import quick_read_frames
from ium_data.config import *

#COMMENTED THIS OUT, DRAW_ALL_FRAMES AND DRAW_FRAME IS NOT IN FRAW_FRAMES
#from draw_frames import draw_all_frames, draw_frame


from read_frames import quick_read_frames

sys.path.append('./road_map2/bams/ium_data/data_process')
#from ium_data.data_process.get_date_list import read_datetime_list
#from ium_data.data_process.split_dataset import read_datetime_set
#from ium_data.data_process.split_dataset import generate_datetime_set
from get_date_list import read_datetime_list
from split_dataset import read_datetime_set
from split_dataset import generate_datetime_set

class BJIterator(object):
    """
    Radar dataset iterator warpper
    size: 600*600(width*height)  
    resoluion: 1km*1km  6min
    equal-longtitude-latitude.
    Area: BEIJING  
    seq_len: in_len + out_len = 5 + 20 = 25   
 
    """

    def __init__(self, datetime_set, sample_mode, begin_idx=None, end_idx=None,
                 width=600, height=600, raw_width=600, raw_height=600,
                 seq_len=15, datetime_list_file="time_info.txt", stride=1):
        assert width <= raw_width and height <= raw_height
        self._raw_width = raw_width
        self._raw_height = raw_height
        self._width = width
        self._height = height
        self._stride = stride
        generate_datetime_set(seq_len)
        self._datetime_set = read_datetime_set(datetime_set, seq_len)
        self._length = len(self._datetime_set)
        self.set_begin_end(begin_idx, end_idx)
        self._seq_len = seq_len
        self._datetime_list = read_datetime_list(info_path, datetime_list_file)
        assert sample_mode in ["random", "sequent"]
        self._sample_mode = sample_mode
        if sample_mode == "sequent":
            self._current_idx = self._begin_idx

    def set_begin_end(self, begin_idx=None, end_idx=None):
        self._begin_idx = 0 if begin_idx is None else begin_idx
        self._end_idx = self._length - 1 if end_idx is None else end_idx
        assert self._begin_idx >= 0 and self._end_idx < self._length
        assert self._begin_idx <= self._end_idx

    def reset(self, begin_idx=None, end_idx=None):
        assert self._sample_mode == "sequent"
        self.set_begin_end(begin_idx=begin_idx, end_idx=end_idx)
        self._current_idx = self._begin_idx

    def random_reset(self):
        assert self.sample_mode == "sequent"
        self.set_begin_end(begin_idx=np.random.randint(self._begin_idx, self._end_idx + 1))
        self._current_idx = self._begin_idx

    @property
    def total_sample_num(self):
        return self._length

    @property
    def begin_time(self):
        return self._datetime_set[self._begin_idx]

    @property
    def end_time(self):
        return self._datetime_set[self._end_idx]

    @property
    def use_up(self):
        if self._sample_mode == "random":
            return False
        else:
            return self._current_idx > self._end_idx

    def get_frame_paths(self, datetime_batch):
        frame_paths = []
        #print('datetime_batch:',datetime_batch)   [93, 56, 359]??
        for dt in datetime_batch:
            batch = self._datetime_list[dt : dt + self._seq_len]
            batch = [os.path.join(nc_data_path,dt[0:8],dt[8:]+".nc") for dt in batch]
            frame_paths.append(batch)
        ##np.array(frame_paths).shape = (bath_size, seq_len) (3,22)//(2,22)//(1,22)
        return np.array(frame_paths)

    def get_real_datetime_batch(self, datetime_batch):
        datetime_batch_real = []
        for dt in datetime_batch:
            batch = self._datetime_list[dt: dt + self._seq_len]
            datetime_batch_real.append(batch)
        return datetime_batch_real

    #### load_frames()####
    def load_frames(self, datetime_batch, offset_height, offset_width, normalization):
        assert isinstance(datetime_batch, list)
        ## print(datetime_batch)  [93, 56, 359]??
        frame_paths = self.get_frame_paths(datetime_batch)   ## (3,22)
        #print("frame_paths", frame_paths)
        for dt in frame_paths:
            #print(len(dt),self._seq_len)
            assert len(dt) == self._seq_len   ### seq_len = 22
        batch_size = len(frame_paths)         ###  3 2 1 
        frame_dat = np.zeros((self._seq_len, batch_size, 1, self._height, self._width),
                                dtype=np.float32)
        mask_dat = np.ones((self._seq_len, batch_size, 1, self._height, self._width),
                                dtype=np.bool)
        if self._sample_mode == "random":
            paths = []
            hit_inds = []
            for i in range(self._seq_len):
                for j in range(batch_size):
                    paths.append(frame_paths[j,i])
                    hit_inds.append([i, j])
            hit_inds = np.array(hit_inds, dtype=np.int)
            all_frame_dat = quick_read_frames(path_list=paths,        ## path = 22*3=66
                                              im_h=self._raw_height,
                                              im_w=self._raw_width)
            #print(all_frame_dat.shape) #(66,16,600,600)
            frame_dat[hit_inds[:,0], hit_inds[:,1], :, :, :] =\
            all_frame_dat[:,:,offset_height:(offset_height+self._height), offset_width:(offset_width+self._width)]

        else:
            # np.unique(frame_paths)
            uniq_paths = set()
            for i in range(self._seq_len):
                for j in range(batch_size):
                    uniq_paths.add(frame_paths[j,i])
            uniq_paths = list(uniq_paths)
            uniq_paths.sort()
            all_frame_dat = quick_read_frames(path_list=uniq_paths,
                                              im_h=self._raw_height,
                                              im_w=self._raw_width)
            for i in range(self._seq_len):
                for j in range(batch_size):
                    idx = uniq_paths.index(frame_paths[j,i])
                    #print(idx, type(idx),offset_height,(offset_height+self._height),offset_width,(offset_width+self._width))
                    frame_dat[i,j,:,:,:] =\
                    all_frame_dat[idx,:,offset_height:(offset_height+self._height), offset_width:(offset_width+self._width)]

        return frame_dat, mask_dat

    #### sample()####
    def sample(self, batch_size, only_return_datetime=False, normalization=True):
        if self._sample_mode == "sequent":
            if self.use_up:
                raise ValueError("The BJIterator has been used up!")
            datetime_batch = []
            offset_width = (self._raw_width - self._width) // 2     # 0
            offset_height = (self._raw_height - self._height) // 2  # 0
            for i in range(batch_size):
                if not self.use_up:
                    frame_idx = self._datetime_set[self._current_idx, 1]
                    datetime_batch.append(frame_idx)
                    self._current_idx += self._stride


        if self._sample_mode == "random":
            datetime_batch = []
            offset_width = np.random.randint(0, self._raw_width - self._width + 1, 1)[0]    # 0
            offset_height = np.random.randint(0, self._raw_height - self._height + 1, 1)[0] # 0
            for i in range(batch_size):
                rand_idx = np.random.randint(self._begin_idx, self._end_idx + 1, 1)[0]
                frame_idx = self._datetime_set[rand_idx, 1]
                datetime_batch.append(frame_idx)
            '''
            rand_idx = np.random.randint(self._begin_idx, self._end_idx + 1 - self._seq_len * batch_size, 1)[0]
            for i in range(batch_size):
                frame_idx = self._datetime_set[rand_idx, 1]
                datetime_batch.append(frame_idx)
                rand_idx += self._seq_len
            '''
        # because "datetime_batch" only contain the start datetime of batch
        datetime_batch_real = self.get_real_datetime_batch(datetime_batch)
        if only_return_datetime:
            return datetime_batch_real, None
        print("datatime_batch:",datetime_batch)
        frame_dat, mask_dat = self.load_frames(datetime_batch, offset_height, offset_width, normalization)

        return frame_dat, mask_dat, datetime_batch_real, None



if __name__ == "__main__":
    np.random.seed(123344)
    import time
    import cProfile, pstats


    train_bj_iter = BJIterator(datetime_set="bj_valid_set.txt",
                               sample_mode="random", seq_len=15,
                               width=600, height=600)

    begin = time.time()
    frame_dat, mask_dat, datetime_batch, _ = train_bj_iter.sample(batch_size=3)
    print(frame_dat.shape)
    print(frame_dat)
    #print(mask_dat)
    print(np.array(datetime_batch).shape)
    '''
    for i in range(10):     ## read 100 frame data
        frame_dat, mask_dat, datetime_batch, _ = train_bj_iter.sample(batch_size=3)
        #if datetime_batch[0][0] ==" ":
          # np.save('1.npy',frame_dat)

        #print(datetime_batch[0][0])
        print(datetime_batch)
        print(np.array(datetime_batch).shape)  # 2dim  (3, 22)  
        print(frame_dat.shape)       # 5dim  (22, 3, 1, 600, 600)   
        ##.squeeze() delete single dimension
        #draw_all_frames((frame_dat[:24,0,...].squeeze()) / 255, 8)
        for frame in frame_dat:
          print(frame[0,0])
          #draw_frame(frame[0,0], True)
    '''
    end = time.time()   
    print('time:',end - begin)

