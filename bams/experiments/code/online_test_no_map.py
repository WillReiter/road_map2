#coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import datetime
import time
import copy

import numpy as np
import torch
import matplotlib as mpl
mpl.use('Agg')
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as colors
#from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
#from matplotlib.colors import LinearSegmentedColormap
from nowcasting.config import cfg
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from models.trajGRU import TrajGRU
#from models.convLSTM import ConvLSTM
from models.net_params import *
from ium_data.read_online_data import ReadBJDataOnline



def parse_args():
    parser = argparse.ArgumentParser(description='Test the HKO nowcasting model')
    parser.add_argument('--cfg', dest='cfg_file', help='Optional configuration file', type=str)
    parser.add_argument('--in_data_dir', help='The directory to read input data', default=None, type=str)
    parser.add_argument('--load_dir', help='The directory to load the model', default=None, type=str)
    parser.add_argument('--load_iter', help='The iterator to load', default=-1, type=int)
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    parser.add_argument('--ctx', dest='ctx', help='Running Context. E.g `--ctx gpu` or `--ctx gpu0,gpu1` or `--ctx cpu`',
                        type=str, default='gpu')
    parser.add_argument('--finetune', dest='finetune', help='Whether to do online finetuning',
                        default=None, type=int)
    parser.add_argument('--finetune_min_mse', dest='finetune_min_mse', help='Minimum error for finetuning',
                        default=None, type=float)
    parser.add_argument('--mode', dest='mode', help='Whether to used fixed setting or online setting',
                        required=True, type=str)
    parser.add_argument('--lr', dest='lr', help='learning rate', default=None, type=float)
    parser.add_argument('--wd', dest='wd', help='weight decay', default=None, type=float)
    parser.add_argument('--grad_clip', dest='grad_clip', help='gradient clipping threshold',
                        default=None, type=float)
    args = parser.parse_args()

    return args

def pixel_to_dbz(img):  
    """

    Parameters
    ----------
    img : np.ndarray or float

    Returns
    -------

    """
    return img * 80.0


def save_frame(data, save_path, normalization=False):

	if not normalization:
		data = pixel_to_dbz(data)
	#data = np.rot90(frame_dat.transpose(1,0))
	data[data<0] = 0
	data[data>60] = 60
	fig = plt.figure()
	ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
	color = ['whitesmoke','dodgerblue','cyan','limegreen','green',\
		     'yellow','goldenrod','orange','red','firebrick', 'darkred']
	clevs = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
	norm = colors.BoundaryNorm(clevs, 24)
	my_cmap = colors.ListedColormap(color)

	cs = plt.contourf(x, y, data, clevs, colors=color)
	cbar = plt.colorbar(cs, location='right', pad="5%")
	cbar.set_label('Radar reflectivity (dBZ)', size=14)
	#plt.imshow(data, cmap=my_cmap, norm=norm, extent=(113, 120, 37, 42))
	#plt.imshow(data, cmap=my_cmap, extent=(113, 120, 37, 42))
	plt.savefig(save_path, dpi=100)
	plt.close()
	ax.clear()

def save_nc(in_data, outputs, path):
    
    in_data = in_data
    pred = outputs     
    for idx, frame in enumerate(in_data):
        #print("in_data.shape:",frame.shape)         #(1, 1, 300, 300)
        frame = np.squeeze(frame)                    #(300, 300)
        minutes = (idx + 1) * 6
        save_frame(frame, os.path.join(path, "in-%dmins.png"%minutes))

    for idx, frame in enumerate(pred):
        #print("frame.shape2:",frame.shape)         #(1, 1, 300, 300)
        frame = np.squeeze(frame)                   #(300, 300)
        minutes = (idx + 1) * 6
        save_frame(frame, os.path.join(path, "pred-%dmins.png"%minutes))

def online_test(num, mode = 'test'):

    time_node1 = time.time()
    input_path = args.in_data_dir
    base_dir = args.save_dir
    save_dir = os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, base_dir)
    
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
    forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
    net = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
    model_load_path = os.path.join(save_dir, 'encoder_forecaster_{}.pth'.format(num))
    net.load_state_dict(torch.load(model_load_path))
    
    ## Save dir
    save_dir = os.path.join(save_dir, 'Prediction_{}_iter'.format(num))
    if not os.path.exists(save_dir):
        os.mkdirs(save_dir)

    with torch.no_grad():

        net.eval()
        in_frame_dat, in_datetime = ReadBJDataOnline(input_path).run()
        print(in_frame_dat.shape)                           ##(5, 1, 1, 500, 700)
        
        """
        convert (5,1,1,500,700) to (5,1,1,600,600)*2
        """
        
        in_frame_nd_1 = torch.from_numpy(in_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE)
        in_frame_nd_2 = torch.from_numpy(in_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE)
        # output dir
        if not os.path.exists(os.path.join(save_dir, in_datetime[-1][:8])):
            os.mkdir(os.path.join(save_dir, in_datetime[-1][:8]))
        output_dir = os.path.join(save_dir, in_datetime[-1][:8], in_datetime[-1][8:])
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        outputs_1 = net(in_frame_nd_1)
        outputs_2 = net(in_frame_nd_1)
        output_numpy_1 = np.clip(outputs_1.detach().cpu().numpy(), 0.0, 1.0)
        output_numpy_2 = np.clip(outputs_2.detach().cpu().numpy(), 0.0, 1.0)
        
        """
        convert (10,1,1,600,600)*2 to (10,1,1,500,700)
        """
        
        time_node2 = time.time()
        print('time2: ', time_node2 - time_node1)
        
        save_nc(in_frame_dat, output_numpy, output_dir)
        time_node3 = time.time()
        print('time3: ', time_node3 - time_node1)

              
if __name__=='__main__':
    #train(400)
    args = parse_args()
    num = args.load_iter
    mode = args.mode
    online_test(num, mode)
