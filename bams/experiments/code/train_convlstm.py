#coding: utf-8
"""
File name: train_convlstm.py
Author: Yang
Date: 2020/06/15
Description: the main code of train convlstm

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import datetime
import logging
import random
import torch
import shutil
import copy
import numpy as np

from nowcasting.config import cfg
from ium_data.bj_iterator import BJIterator
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from torch.optim import lr_scheduler
from models.loss import Weighted_mse_mae
from models.convLSTM import ConvLSTM
from models.net_params import convlstm_encoder_params, convlstm_forecaster_params
from nowcasting.hmw_benchmark import HMWBenchmarkEnv
from ium_data.plot_loss import plot_loss



### Train Config ###
random.seed(123)
np.random.seed(92123)
torch.manual_seed(9302)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(9302)

batch_size = 2
max_iterations = 150000
valid_iteration_interval = 5000
valid_and_save_checkpoint_iterations = 5000
LR_step_size = 30000
gamma = 0.7
LR = 1e-4


def parse_args():
    parser = argparse.ArgumentParser(description='Train the Nowcasting model')
    parser.add_argument('--batch_size', dest='batch_size', help="batchsize of the training process",
                        default=None, type=int)
    parser.add_argument('--cfg', dest='cfg_file', help='Optional configuration file', default=None, type=str)
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    parser.add_argument('--lr', dest='lr', help='learning rate', default=None, type=float)
    parser.add_argument('--wd', dest='wd', help='weight decay', default=None, type=float)
    parser.add_argument('--grad_clip', dest='grad_clip', help='gradient clipping threshold',
                        default=None, type=float)
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
    ### logging info ###
    logging.basicConfig(filename='train_convlstm02.log', level=logging.INFO)
    logging.info('train convlstm begin...')
    print("train loss will be saveed in train_convlstm02.log")
    
    ### dir Config   ###
    base_dir = args.save_dir
    save_dir = os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, base_dir)
    model_save_dir = os.path.join(save_dir, 'models')

    if os.path.exists(model_save_dir):
        shutil.rmtree(model_save_dir)
    os.mkdir(model_save_dir)

    ### Model Config ###
    IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
    OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
    net = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

    criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)    ### loss
    optimizer = torch.optim.Adam(net.parameters(), lr = LR)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

    train_loss = 0.0
    loss_dict = {}
    best_iter = 0
    min_mse_mae = 1e10

    #### from ium_data/bj_iterator.py import BJIterator 
    train_ium_iter = BJIterator(datetime_set=cfg.HKO_PD.RAINY_TRAIN,
                                sample_mode="random",
                                width=cfg.HKO.ITERATOR.WIDTH,      
                                height=cfg.HKO.ITERATOR.HEIGHT,                                     
                                seq_len=cfg.MODEL.IN_LEN + cfg.MODEL.OUT_LEN)

  
    for i in range(1, max_iterations+1):

        ## sample a random minibatch data
        train_batch, train_mask, datetime_batch, _ = train_ium_iter.sample(batch_size=1)
        train_batch = torch.from_numpy(train_batch.astype(np.float32)).to(cfg.GLOBAL.DEVICE)

        ## train data and train label
        train_data = train_batch[:IN_LEN, ...]
        train_label = train_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
        mask = torch.from_numpy(train_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)).to(cfg.GLOBAL.DEVICE)

        #print(train_batch.shape)         # (22, 3, 1, 250, 350)
        #print(type(train_data))          # <class 'torch.Tensor'>
        #print(train_data.shape)          # torch.Size([10, 2, 1, 250, 350])
        #print(train_label.shape)         # torch.Size([12, 2, 1, 250, 350])
        #print(np.array(datetime_batch).shape)  #(3, 22)
    
        net.train()
        optimizer.zero_grad()
        output = net(train_data)
        loss = criterion(output, train_label, mask)
        loss.backward()
        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=50.0)
        optimizer.step()
        exp_lr_scheduler.step()
        iter_loss = loss.item()
        train_loss += iter_loss
        
        logging.info("{} iter:{}/{}, iter_loss:{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),\
              i, max_iterations, iter_loss))
          
        if i % valid_iteration_interval == 0:
            ### print loss ###
            train_loss = train_loss/valid_iteration_interval
            loss_dict[i] = train_loss
            print('{} iter, Ave_iter_loss: {}'.format(i, train_loss) )
            train_loss = 0

            img_dir = os.path.join(save_dir, 'iter_{}'.format(i))
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)

            ## every iter+1, run_benchmark and write csi
            env = HMWBenchmarkEnv(pd_path = cfg.HKO_PD.RAINY_VALID, save_dir = img_dir, mode = "fixed")
            with torch.no_grad():
                net.eval()
                while not env.done:
                    in_frame_dat, in_datetime_clips, out_datetime_clips, begin_new_episode, need_upload_prediction =\
                         env.get_observation(batch_size=1)

                    in_frame_nd = torch.from_numpy(in_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE)
                    outputs = net(in_frame_nd)
                    
                    output_numpy = np.clip(outputs.detach().cpu().numpy(), 0.0, 1.0)
                    env.upload_prediction(prediction = output_numpy, draw = False)

                ### save evaluation reault ### 
                mse, mae, csi = env.save_eval()
                if mse + mae < min_mse_mae:
                    min_mse_mae = mse + mae
                    best_iter = i
                write_csi(base_dir, i, best_iter, mse, mae, *csi)
        
        if i % valid_and_save_checkpoint_iterations == 0:
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

