import sys
#sys.path.append('/media/data3/zr/bams/models/')

import torch
from models.encoder import Encoder
from models.forecaster import Forecaster
from models.convLSTM import ConvLSTM
from models.model import EF
from models.net_params import convlstm_encoder_params, convlstm_forecaster_params


encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)
forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)
net = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

print(net)
