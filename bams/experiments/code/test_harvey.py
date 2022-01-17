import os
import warnings
import sys
warnings.filterwarnings('ignore')
sys.path.append('/home/syao/Desktop/trajGRU/bams')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from models.net_params import *
import matplotlib.pyplot as plt
from ium_data import read_frames as rd
encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

data_path = '/home/syao/Downloads/dataset/20170826'
save_dir = '/home/syao/Downloads/dataset/test'
net = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
load_file = '/home/syao/Downloads/dataset/encoder_forecaster_9800.pth'
net.load_state_dict(torch.load(load_file))
save_dir = '/home/syao/Downloads/dataset/iter_9800'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
all_files = os.listdir(data_path)
all_files.sort()
path_list = []
for i in all_files:
    if i in ["170635.nc","171235.nc","171835.nc","172435.nc","173035.nc"]:
        path_list.append(os.path.join(data_path,i))

with torch.no_grad():

    in_frame_dat = rd.quick_read_frames(path_list,600,600)
    in_frame_dat = in_frame_dat.reshape(5,1,1,600,600)
    print(in_frame_dat.shape)
    in_frame_nd = torch.from_numpy(in_frame_dat.astype(np.float32)).to(cfg.GLOBAL.DEVICE)
    outputs = net(in_frame_nd)
    output_numpy = np.clip(outputs.detach().cpu().numpy(), 0.0, 1.0)
    output_numpy = output_numpy.reshape(30,600,600)
    plt.imshow(output_numpy[0])
    plt.show()