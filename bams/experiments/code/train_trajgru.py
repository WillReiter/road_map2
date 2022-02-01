#coding: utf-8
"""
File name: train_trajgru.py
Author: Yang
Date: 2020/06/15
Description: the main code of train trajgru

"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')
sys.path.append('./road_map2/bams')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 0: 1080ti  1: 1070
import datetime
import logging
import argparse
import shutil
import copy
import torch
import random
import numpy as np
from natsort import natsorted

#from nowcasting.config import cfg
#from nowcasting.hmw_benchmark import HMWBenchmarkEnv
sys.path.append('./road_map2/bams/nowcasting')
from config import cfg
from hmw_benchmark import HMWBenchmarkEnv

#from ium_data.bj_iterator import BJIterator
#from ium_data.plot_loss import plot_loss
sys.path.append('./road_map2/bams/ium_data')
from bj_iterator import BJIterator
from plot_loss import plot_loss

#from models.encoder import Encoder
#from models.forecaster import Forecaster
#from models.model import EF
from torch.optim import lr_scheduler
#from models.loss import Weighted_mse_mae
#from models.trajGRU import TrajGRU
#from models.net_params import *
sys.path.append('./road_map2/bams/models')
from encoder import Encoder
from forecaster import Forecaster
from model import EF
from loss import Weighted_mse_mae
from trajGRU import TrajGRU
from net_params import *





#### Random Seed Config ####
random.seed(123)
np.random.seed(92123)
torch.manual_seed(9302)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(9302)
    
logging.basicConfig(filename='train_trajgru{}.log'.format(str(IN_LEN + OUT_LEN)), level=logging.INFO)
logging.info('train trajgru begin...')

def parse_args():
    parser = argparse.ArgumentParser(description='Train the Nowcasting model')
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    parser.add_argument('--max_iterations', help='The max iterations', required=True, type=int)
    parser.add_argument('--valid_and_save_checkpoint_iterations', help='valid and save checkpoint', required=True, type=int)
    parser.add_argument('--batch_size', help='The batch size for train', required=True, type=int)
    parser.add_argument('--LR', help='The learning rate', required=True, type=float)
    args = parser.parse_args()

    return args

def write_csi(save_path, *args):
    """ save the "csi.txt"
    :type save_path: str
    :type *args: tuple()
    
    """
    with open(os.path.join(save_path, "csi.txt"), 'a') as f:
        for val in args:
            f.write("%s \t" % str(round(val, 4)))
        f.write("\n")

def train(args):
    """
    :type args: parser.parse_args()
    
    """
  
    ### dir Config   ###
    base_dir = args.save_dir
    save_dir = os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, base_dir)
    model_save_dir = os.path.join(save_dir, 'models')

    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    ### Train Config ###
    batch_size = args.batch_size   
    max_iterations = args.max_iterations
    valid_iteration_interval = args.valid_and_save_checkpoint_iterations    
    LR = args.LR
    #LR_step_size = 30000
    gamma = 0.5
    IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
    OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN
    

    ### Model Config ###
    criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)     ### loss
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
    forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
    net = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr = LR)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

    mult_step_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30000, 60000, 90000], gamma=gamma)

    #### from ium_data/bj_iterator.py import BJIterator 
    train_ium_iter = BJIterator(datetime_set=cfg.HKO_PD.RAINY_TRAIN,
                                sample_mode="random",
                                width=cfg.HKO.ITERATOR.WIDTH,      
                                height=cfg.HKO.ITERATOR.HEIGHT,                                     
                                seq_len=IN_LEN + OUT_LEN)
    train_loss = 0.0
    loss_dict = {}
    best_iter = 0
    min_mse_mae = 1e10

    ### logging info ###
    logging.basicConfig(filename='train_trajgru{}.log'.format(str(IN_LEN + OUT_LEN)), level=logging.INFO)
    logging.info('train trajgru begin...')
    print(f"train loss will be saved in train_trajgru{IN_LEN + OUT_LEN}.log")

    for i in range(1, max_iterations + 1):
        print("start!")

        ## sample a random minibatch data
        train_batch, train_mask, datetime_batch, _ = train_ium_iter.sample(batch_size=batch_size)

        train_batch = torch.from_numpy(train_batch.astype(np.float32)).to(cfg.GLOBAL.DEVICE)
        print("load data!")
        
        ## train data and train label
        train_data = train_batch[:IN_LEN, ...]
        train_label = train_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
        mask = torch.from_numpy(train_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)).to(cfg.GLOBAL.DEVICE)


        if i == 1:
            ### test input ### 
            logging.info(train_batch.shape)         # (15, 3, 1, 600, 600)
            logging.info(type(train_data))          # <class 'torch.Tensor'>
            logging.info(train_data.shape)          # torch.Size([5, 3, 1, 600, 600])
            logging.info(train_label.shape)         # torch.Size([10, 3, 1, 600, 600])
            logging.info(np.array(datetime_batch).shape)  #(3, 15)
            print("load saved model")
            # load_file = os.path.join("/home/syao/Desktop/trajGRU/bams/experiments/train_35_Houston/models",'encoder_forecaster_5001.pth')
            # net.load_state_dict(torch.load(load_file))
        net.train()
        optimizer.zero_grad()
        output = net(train_data)
        loss = criterion(output, train_label, mask)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=50.0)
        optimizer.step()
        mult_step_scheduler.step()   
        iter_loss = loss.item()
        train_loss += iter_loss 

          
        if i % valid_iteration_interval == 0:

            train_loss_ave = train_loss / valid_iteration_interval
            loss_dict[i] = train_loss_ave
            print('\n')
            print('{} iter, Ave_iter_loss: {} \n'.format(i, train_loss_ave))

            train_loss = 0
            img_dir = os.path.join(save_dir, 'iter_{}'.format(i))
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)

            ## when train iter+1 times, run benchmark and save csi
            env = HMWBenchmarkEnv(pd_path = cfg.HKO_PD.RAINY_VALID, save_dir = img_dir, mode = "fixed")
            with torch.no_grad():
                net.eval()
                while not env.done:
                    valid_start_load = time.time()
                    in_frame_dat, in_datetime_clips, out_datetime_clips, begin_new_episode, need_upload_prediction =\
                                        env.get_observation(batch_size=1)
                    in_frame_nd = torch.from_numpy(in_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE)
                    valid_end_load = time.time()-valid_start_load
                    valid_start_compute = time.time()
                    outputs = net(in_frame_nd)
                    output_numpy = np.clip(outputs.detach().cpu().numpy(), 0.0, 1.0)
                    env.upload_prediction(prediction = output_numpy, draw = False)
                
                ### save evaluation reault ### 
                mse, mae, csi = env.save_eval()
                if mse + mae < min_mse_mae:
                    min_mse_mae = mse + mae
                    best_iter = i
                write_csi(save_dir, i, best_iter, mse, mae, *csi)
                valid_end_compute = time.time()-valid_start_compute
                logging.info("load data:{}\tvalid:{}".format(valid_end_load ,valid_end_compute))
        
            torch.save(net.state_dict(), os.path.join(model_save_dir, 'encoder_forecaster_{}.pth'.format(i)))
    
        ### write and save loss ###
        if loss_dict:
            with open(os.path.join(model_save_dir, "iter_loss.txt"), 'a') as f:
                for itera, val in loss_dict.items():
                    f.write("%s\t%s\n" %(str(itera), str(round(val, 4)) ))
                f.write("\n")

            plot_loss(model_save_dir, list(loss_dict.values()), epoch = valid_iteration_interval)
        

if __name__ == "__main__":

    args = parse_args()
    train(args)
        
