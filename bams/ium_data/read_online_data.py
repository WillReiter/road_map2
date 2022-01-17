import os
from datetime import datetime, timedelta
import numpy as np
import netCDF4 as nc

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class ReadBJDataOnline:
    def __init__(self, data_path):
        print("Read radar data for test!")
        self._read_frame_num = 5
        self._interval_upper_bound = 9*60  # seconds
        self._interval_lower_bound = 4*60  # seconds
        self._height = 600  #800
        self._width = 600  #800
        self._data_path = data_path

    def run(self):
        dir_list = self.get_dirs_name(self._data_path)
        file_path_list = self.get_input_path(dir_list)
        datetime_list = self.judge_legal_time_interval(file_path_list)
        frame_dat = self.read_frame(file_path_list)		
        return frame_dat, datetime_list

    def get_dirs_name(self, file_dir):
        dirs_all = []
        for p in os.listdir(file_dir):
            file_path = os.path.join(file_dir, p)
            #print(file_path)
            if os.path.isdir(file_path) and len(p)== 8 and is_number(p):
                dirs_all.append(file_path)
        dirs_all = sorted(dirs_all, key=lambda x: os.path.basename(x))
        #print(dirs_all)
        return dirs_all

    def get_input_path(self, dir_list):
        last_list = self.get_files_in_dir(dir_list[-1])
        print(last_list)
        if len(last_list) < self._read_frame_num and len(dir_list) > 1:
            last_list = self.get_files_in_dir(dir_list[-2]) + last_list
        
        if len(last_list) < self._read_frame_num:
            raise ValueError("input path less %d nc files!" % self._read_frame_num) 

        return last_list[-self._read_frame_num:]

    def get_files_in_dir(self, target_dir):
        cur_list = []
        for p in os.listdir(target_dir):
            file_path = os.path.join(target_dir, p)
            ## p : HHMMSS.nc
            if os.path.isfile(file_path) and len(p) == 9 and is_number(p[:7]):
                cur_list.append(file_path)
        cur_list = sorted(cur_list, key=lambda x: os.path.basename(x))
        return cur_list

    def judge_legal_time_interval(self, file_path_list):
        file_time_list = [datetime.strptime(ele[-18:], "%Y%m%d/%H%M%S.nc") for ele in file_path_list]
        for i in range(1, len(file_time_list)):
            time_diff = file_time_list[i] - file_time_list[i-1]
            if time_diff < timedelta(seconds=self._interval_lower_bound) or time_diff > timedelta(seconds=self._interval_upper_bound):
                raise ValueError("input nc file's interval is invalid!")

        return [dt.strftime("%Y%m%d%H%M%S") for dt in file_time_list]

    def read_frame(self, file_path_list):
        assert(len(file_path_list) == self._read_frame_num)
        frame_dat = np.zeros((self._read_frame_num, 1, 1, self._height, self._width), dtype=np.float) 

        for i in range(self._read_frame_num):    ## i in range(5): 
            print("reading nc: ", file_path_list[i])
            self.read_nc_file(file_path_list[i], frame_dat[i])
            #print(frame_dat[i].shape)          #(1, 1, 600, 600)

        return frame_dat
    
    def read_nc_file(self, path, read_storage):
        assert(os.path.exists(path))    
       
        sat = nc.Dataset(path)
        data = np.array(sat['DBZ'])
        data = np.clip( data /80.0 , 0.0, 1.0)  ##  [ 0 , 80 ] 
        data = data[:, 100:700, 100:700]
        read_storage[:] = data
        sat.close()

def main():
    input_path = '/media/4T/yangjitao/trajgru_pytorch_radar/experiments/data_online'
    frame_dat, date_time = ReadBJDataOnline(input_path).run()
    print(frame_dat)
    print(date_time)


if __name__ == "__main__":
    main()
