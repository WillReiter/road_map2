# coding: utf-8
"""
File name: train_trajgru.py
Author: Yang
Date: 2020/06/15
Description: the main code of train trajgru

"""

import os
import time
import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.append('/home/syao/Desktop/trajGRU/bams')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# 0: 1080ti  1: 1070
import argparse
import datetime
import logging

import numpy as np
import torch
from nowcasting.config import cfg
from ium_data.bj_iterator import BJIterator
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from torch.optim import lr_scheduler
from models.loss import Weighted_mse_mae
from models.trajGRU import TrajGRU
from models.convLSTM import ConvLSTM
from models.net_params import *
from nowcasting.hmw_benchmark import HMWBenchmarkEnv
from ium_data.plot_loss import plot_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Test the HMW nowcasting model')
    parser.add_argument('--load_dir', help='The directory to load the model', default=None, type=str)
    parser.add_argument('--load_iter', help='The iterator to load', default=-1, type=int)
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    parser.add_argument('--mode', dest='mode', help='Whether to used fixed setting or online setting',
                        default=None, type=str)
    parser.add_argument('--dataset', dest='dataset', help='Whether to used the test set or the validation set',
                        default="test", type=str)
    parser.add_argument('--lr', dest='lr', help='learning rate', default=None, type=float)
    parser.add_argument('--net_name', dest='net_name', help='choose model', default=None, type=str)
    args = parser.parse_args()

    return args


def write_csi(save_path, *args):
    """ save the "csi.txt"
    :type save_path: str
    :type *args: tuple()

    """
    with open(os.path.join(save_path, "csi.txt"), 'a') as f:
        for val in args:
            f.write("%s\t" % str(round(val, 4)))
        f.write("\n")


def valid_or_test(num, dataset='test', net_name='trajgru'):
    """
    :type num: int
    :type mode: str
    :type net_name: str

    """

    if net_name == 'trajgru':
        encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
    elif net_name == 'convlstm':
        encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
        forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

    base_dir = args.save_dir
    save_dir = os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, base_dir)
    net = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
    load_file = os.path.join(save_dir, 'encoder_forecaster_{}.pth'.format(num))
    net.load_state_dict(torch.load(load_file))

    save_dir = os.path.join(save_dir, 'iter_{}'.format(num))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if dataset == 'valid':
        pd_path = cfg.HKO_PD.RAINY_VALID
    else:
        pd_path = cfg.HKO_PD.RAINY_TEST

    ### logging info
    if os.path.exists('test_trajgru.log'):
        os.remove('test_trajgru.log')

    logging.basicConfig(filename="test_trajgru.log", level=logging.INFO)
    logging.info('test trajgru begin...')

    ### run benchmark and save csi
    env = HMWBenchmarkEnv(pd_path, save_dir, mode="fixed")
    counter = 1
    with torch.no_grad():
        net.eval()
        while not env.done:
            in_frame_dat, _, _, _, _ = env.get_observation(batch_size=1)
            print(in_frame_dat.shape)
            logging.info("{}: test{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), counter))
            counter += 1
            in_frame_nd = torch.from_numpy(in_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE)
            outputs = net(in_frame_nd)
            output_numpy = np.clip(outputs.detach().cpu().numpy(), 0.0, 1.0)
            env.upload_prediction(prediction=output_numpy, draw=False)
            env.draw_roc(prediction=output_numpy,save_path=save_dir)
        env.save_eval()


if __name__ == '__main__':
    # train(400)
    args = parse_args()
    num = args.load_iter
    dataset = args.dataset
    net_name = args.net_name
    valid_or_test(num, dataset, net_name)

