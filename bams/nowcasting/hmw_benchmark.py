import json
import os
import sys
sys.path.append('/home/syao@colostate.edu/trajGRU/bams')
import logging
import numpy as np
from nowcasting.config import cfg
from nowcasting.helpers.visualization import save_hmw_imgs
from nowcasting.hmw_evaluation import HMWEvaluation
from ium_data.bj_iterator import BJIterator

class HMWBenchmarkEnv(object):
    """The Benchmark environment for the HKO7 Dataset

    There are two settings for the Benchmark, the "fixed" setting and the "online" setting.
    In the "fixed" setting, pre-defined input sequences that have the same length will be
     fed into the model for prediction.
        This setting tests the model's ability to use the instant past to predict the future.
    In the "online" setting, M frames will be given each time and the forecasting model
     is required to predict the next K frames every stride steps.
        If the begin_new_episode flag is turned on, a new episode has begun, which means that the current received images have no relationship with the previous images.
        If the need_upload_prediction flag is turned on, the model is required to predict the
        This setting tests both the model's ability to adapt in an online fashion and
         the ability to capture the long-term dependency.
    The input frame will be missing in some timestamps.

    To run the benchmark in the fixed setting:

    env = HKOBenchmarkEnv(...)
    while not env.done:
        # Get the observation
        in_frame_dat, in_mask_dat, in_datetime_clips, out_datetime_clips, begin_new_episode =
         env.get_observation(batch_size=batch_size)
        # Running your algorithm to get the prediction
        prediction = ...
        # Upload prediction to the environment
        env.upload_prediction(prediction)

    """
    def __init__(self,
                 pd_path,
                 save_dir,
                 mode="fixed"):
        assert mode == "fixed" or mode == "online"
        self._pd_path = pd_path
        self._save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self._mode = mode
        self._out_seq_len = cfg.HKO.BENCHMARK.OUT_LEN
        self._stride = cfg.HKO.BENCHMARK.STRIDE   #5
        if mode == "fixed":
            self._in_seq_len = cfg.HKO.BENCHMARK.IN_LEN
        else:
            self._in_seq_len = cfg.HKO.BENCHMARK.STRIDE

        ## load data
        self._hmw_iter = BJIterator(datetime_set = pd_path,
                                    sample_mode = "sequent",
                                    width = cfg.HKO.ITERATOR.WIDTH,
                                    height = cfg.HKO.ITERATOR.HEIGHT,                                    
                                    seq_len = self._in_seq_len + self._out_seq_len)
        
        self._stat_dict = self._get_benchmark_stat()
        self._begin_new_episode = True
        self._received_pred_seq_num = 0
        self._need_upload_prediction = False
        #TODO Save some predictions

        #self._save_seq_inds = set(np.arange(1, cfg.HKO.BENCHMARK.VISUALIZE_SEQ_NUM + 1) * \
        #                      (self._stat_dict['pred_seq_num'] //
        #                       cfg.HKO.BENCHMARK.VISUALIZE_SEQ_NUM))
        #print(self._save_seq_inds)
        self._all_eval = HMWEvaluation(seq_len=self._out_seq_len, use_central = False)


        # Holder of the inner data
        self._in_frame_dat = None
        self._in_mask_dat = None
        self._in_datetime_clips = None
        self._out_frame_dat = None
        self._out_mask_dat = None
        self._out_datetime_clips = None

    def reset(self):
        self._hmw_iter.reset()
        self._all_eval.clear_all()
        self._begin_new_episode = True
        self._received_pred_seq_num = 0
        self._need_upload_prediction = False

    @property
    def _fingerprint(self):
        pd_file_name = os.path.splitext(os.path.basename(self._pd_path))[0]
        if self._mode == "fixed":
            fingerprint = pd_file_name + "_in" + str(self._in_seq_len)\
                          + "_out" + str(self._out_seq_len) + "_stride" + str(self._stride)\
                          + "_" + self._mode
        else:
            fingerprint = pd_file_name + "_out" + str(self._out_seq_len)\
                          + "_stride" + str(self._stride)\
                          + "_" + self._mode
        return fingerprint

    @property
    def _stat_filepath(self):
        filename = self._fingerprint + ".json"
        return os.path.join(cfg.HKO.BENCHMARK.STAT_PATH, filename)

    def _get_benchmark_stat(self):
        """Get the general statistics of the benchmark

        Returns
        -------
        stat_dict : dict
            'pred_seq_num' --> Total number of predictions the model needs to make
        """
        if os.path.exists(self._stat_filepath):
            stat_dict = json.load(open(self._stat_filepath))
        else:
            seq_num = 0
            episode_num = 0
            episode_start_datetime = []
            while not self._hmw_iter.use_up:
                if self._mode == "fixed":
                    datetime_clips, new_start =\
                        self._hmw_iter.sample(batch_size=1024, only_return_datetime=True)

                    print(datetime_clips,len(datetime_clips[0]))
                    if len(datetime_clips) == 0:
                        continue
                    seq_num += len(datetime_clips)
                    episode_num += len(datetime_clips)
                elif self._mode == "online":
                    datetime_clips, new_start = \
                        self._hmw_iter.sample(batch_size=1, only_return_datetime=True)

                    if len(datetime_clips) == 0:
                        continue
                    episode_num += new_start
                    if new_start:
                        episode_start_datetime.append(datetime_clips[0][0].strftime('%Y%m%d%H%M'))
                        if self._stride != 1:
                            seq_num += 1
                    else:
                        seq_num += 1
                print(self._fingerprint, seq_num, episode_num)
            self._hmw_iter.reset()
            stat_dict = {'pred_seq_num': seq_num,
                         'episode_num': episode_num,
                         'episode_start_datetime': episode_start_datetime}
            json.dump(stat_dict, open(self._stat_filepath, 'w'), indent=3)
        return stat_dict

    @property
    def done(self):
        print("env done info")
        print(self._received_pred_seq_num, self._stat_dict["pred_seq_num"])
        return self._received_pred_seq_num >= self._stat_dict["pred_seq_num"]

    def get_observation(self, batch_size=1):
        """

        Parameters
        ----------
        batch_size : int


        Returns
        -------
        in_frame_dat : np.ndarray
            Will be between 0 and 1
        in_datetime_clips : list
        out_datetime_clips : list
        begin_new_episode : bool
        need_upload_prediction : bool
        """
        assert not self._need_upload_prediction
        assert not self._hmw_iter.use_up,\
        "_received_pred_seq_num: {}, pre_seq_num: {}".format(self._received_pred_seq_num, self._stat_dict["pred_seq_num"])

        while True:
            frame_dat, mask_dat, datetime_clips, new_start =\
                self._hmw_iter.sample(batch_size=batch_size, only_return_datetime=False)
            if len(datetime_clips) == 0:
                continue
            else:
                break
        frame_dat = frame_dat.astype(np.float32)
        self._need_upload_prediction = True
        if self._mode == "online":
            self._begin_new_episode = new_start
            if new_start and self._stride == 1:
                self._need_upload_prediction = False
        else:
            self._begin_new_episode = True
        self._in_datetime_clips = [ele[:self._in_seq_len] for ele in datetime_clips]
        self._out_datetime_clips = [ele[self._in_seq_len:(self._in_seq_len + self._out_seq_len)]
                                    for ele in datetime_clips]
        self._in_frame_dat = frame_dat[:self._in_seq_len, ...]
        self._out_frame_dat = frame_dat[self._in_seq_len:(self._in_seq_len + self._out_seq_len), ...]
        self._in_mask_dat = mask_dat[:self._in_seq_len, ...]
        self._out_mask_dat = mask_dat[self._in_seq_len:(self._in_seq_len + self._out_seq_len), ...]

        return self._in_frame_dat,\
               self._in_datetime_clips,\
               self._out_datetime_clips,\
               self._begin_new_episode, \
               self._need_upload_prediction


    def upload_prediction(self, prediction, case_test = False, draw = False):
        """

        Parameters
        ----------
        prediction : np.ndarray

        """
        assert self._need_upload_prediction, "Must call get_observation first!" \
                                             " Also, check the value of need_upload_predction" \
                                             " after calling"
        self._need_upload_prediction = False
        received_seq_inds = range(self._received_pred_seq_num,
                                  self._received_pred_seq_num + prediction.shape[1])
        print(draw,self._in_datetime_clips[0][0][:8])
        # if draw and self._in_datetime_clips[0][0][:6] in ('201708'):
        # if draw and self._in_datetime_clips[0][0][:14] in ('20190214000035',"20190214101835", "20190214200635"):
        if draw and self._in_datetime_clips[0][0][:14] in ('20190215043635'):

            ###### file path named as the last input frame, format: 'yymmdd_HHSSMM'  ######
            index = self._in_seq_len - 1
            save_img_path = os.path.join(self._save_dir, self._in_datetime_clips[0][index][0:8], \
                                         self._in_datetime_clips[0][index][8:])

            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)
            print("Saving prediction images to %s" % save_img_path)  
            sys.stdout.flush()   
     
            ###### save images ######
            '''
            save_hmw_imgs(im_dat=self._in_frame_dat[:, 0, 0, ...],               ## in (10,1,1,300,300)>>(10,300,300)
        	         save_path=os.path.join(save_img_path, self._in_datetime_clips[0][0][:8]+ '_' + self._in_datetime_clips[0][0][8:]+ '_IN' ))
            '''

            save_hmw_imgs(im_dat=self._out_frame_dat[:, 0, 0, ...], \
        	         save_path=os.path.join(save_img_path, self._in_datetime_clips[0][index][8:] + '_GT'))

            save_hmw_imgs(im_dat=prediction[:, 0, 0, ...], \
        	         save_path=os.path.join(save_img_path, self._in_datetime_clips[0][index][8:] + '_F'))

        elif draw and case_test:

            # file path named as the last input frame, format: 'yymmdd_HHSSMM'
            index = self._in_seq_len - 1
            save_dir = '/media/4T/yangjitao/trajgru_pytorch_radar/experiments/case_test/Forecast'
            save_img_path = os.path.join(save_dir, self._in_datetime_clips[0][index][0:8], self._in_datetime_clips[0][index][8:])
            if not os.path.exists(save_img_path):
                os.makedirs(save_img_path)

            print("Saving prediction images to %s" % save_img_path)  
            sys.stdout.flush()  
          
            # save ground_truth and prediction images, file named as 'HHSSMM_F_xxmin'
            save_hmw_imgs(im_dat=self._out_frame_dat[:, 0, 0, ...], \
        	         save_path=os.path.join(save_img_path, self._in_datetime_clips[0][index][8:] + '_GT'))

            save_hmw_imgs(im_dat=prediction[:, 0, 0, ...],
        	         save_path=os.path.join(save_img_path, self._in_datetime_clips[0][index][8:] + '_F'))
        #print("mask:",self._out_mask_dat)
        self._received_pred_seq_num += prediction.shape[1]
        self._all_eval.update(gt = self._out_frame_dat,
        		      pred = prediction,
        		      mask = self._out_mask_dat,
        		      start_datetimes = [ele[0] for ele in self._out_datetime_clips])

    def print_stat_readable(self):
        self._all_eval.print_stat_readable(prefix="Received:%d " %self._received_pred_seq_num)

    def draw_roc(self,prediction, save_path):
        self._all_eval.plot_roc(gt= self._out_frame_dat[:, 0, 0, ...], pred= prediction[:, 0, 0, ...],save_path = save_path)

    def save_eval(self):
        logging.info("Saving evaluation result to %s" % self._save_dir)
        sys.stdout.flush()
        return self._all_eval.save(prefix=os.path.join(self._save_dir, "eval_all"))
        

if __name__ == '__main__':
   
    base_dir = "C:\\Users\\Think\\Desktop"
    env = HMWBenchmarkEnv(pd_path="hmw_valid_set.txt", save_dir=base_dir, mode="fixed")
    env._get_benchmark_stat()

