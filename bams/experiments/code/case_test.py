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
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap

from nowcasting.config import cfg
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.model import EF
from models.trajGRU import TrajGRU
from models.convLSTM import ConvLSTM
from models.net_params import *
from nowcasting.hmw_benchmark import HMWBenchmarkEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Test the HKO nowcasting model')
    parser.add_argument('--in_data_dir', help='The directory to read input data', default=None, type=str)
    parser.add_argument('--load_dir', help='The directory to load the model', default=None, type=str)
    parser.add_argument('--load_iter', help='The iterator to load', default=-1, type=int)
    parser.add_argument('--save_dir', help='The saving directory', required=True, type=str)
    args = parser.parse_args()

    return args

def pixel_to_dbz(img):  ##### [0~1] >>  temperature(C) 
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
	m = Basemap(projection='cyl', llcrnrlon=113.2, llcrnrlat=37.1, urcrnrlon=119.8, urcrnrlat=42.5, resolution='l')
	m.readshapefile(r"/media/4T/yangjitao/ChinaMap/Province_Beijing", 'Province_Beijing', drawbounds=True)
	lons, lats = m.makegrid(600, 600) #change here
	x, y = m(lons, lats)
	parallels = np.arange(0., 90, 1)
	m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
	meridians = np.arange(80., 130., 1)
	m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

	color = ['whitesmoke','dodgerblue','cyan','limegreen','green',\
							'yellow','goldenrod','orange','red','firebrick', 'darkred']
	clevs = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
	norm = colors.BoundaryNorm(clevs, 24)
	my_cmap = colors.ListedColormap(color)

	cs = m.contourf(x, y, data, clevs, colors=color)
	cbar = m.colorbar(cs, location='right', pad="5%")
	cbar.set_label('Radar reflectivity (dBZ)', size=14)
	#plt.imshow(data, cmap=my_cmap, norm=norm, extent=(113, 120, 37, 42))
	#plt.imshow(data, cmap=my_cmap, extent=(113, 120, 37, 42))
	plt.savefig(save_path, dpi=100)
	plt.close()
	ax.clear()

def save_nc(in_data, outputs, path):
    
    in_data = in_data
    data = outputs
    ''' 
    for idx, frame in enumerate(in_data):
        #print("in_data.shape:",frame.shape)         #(1, 1, 300, 300)
        frame = np.squeeze(frame)                    #(300, 300)
        minutes = (idx + 1) * 6
        save_frame(frame, os.path.join(path, "in-%dmins.png"%minutes))
    '''

    for idx, frame in enumerate(data):
        #print("frame.shape2:",frame.shape)         #(1, 1, 300, 300)
        frame = np.squeeze(frame)                   #(300, 300)
        minutes = (idx + 1) * 6
        save_frame(frame, os.path.join(path, "pred-%dmins.png"%minutes))

def online_test(num):

    node1 = time.time()
    input_path = args.in_data_dir
    base_dir = args.save_dir
    model_load_dir = os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, base_dir)
    
    '''
    encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
    forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
    net = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
    '''
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
    net = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
    load_file = os.path.join(model_load_dir, 'encoder_forecaster_{}.pth'.format(num))
    net.load_state_dict(torch.load(load_file))

    save_dir = '/media/4T/yangjitao/trajgru_pytorch_radar/experiments/case_test/Forecast'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    env = HMWBenchmarkEnv(pd_path = '/media/4T/yangjitao/trajgru_pytorch_radar/experiments/case_test/case3.txt', \
                          save_dir = save_dir, mode = "fixed")
    with torch.no_grad():
        net.eval()
        while not env.done:
            in_frame_dat, in_datetime_clips, out_datetime_clips, begin_new_episode, need_upload_prediction =\
                     env.get_observation(batch_size=1)

            in_frame_nd = torch.from_numpy(in_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE)
            outputs = net(in_frame_nd)       
            output_numpy = np.clip(outputs.detach().cpu().numpy(), 0.0, 1.0)
            env.upload_prediction(prediction = output_numpy, case_test = True, draw = True)

              
if __name__=='__main__':
    #train(400)
    args = parse_args()
    num = args.load_iter
    online_test(num)
