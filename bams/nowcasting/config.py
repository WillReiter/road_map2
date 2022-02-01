#from nowcasting.helpers.ordered_easydict import OrderedEasyDict as edict
import sys
sys.path.append('./road_map2/bams/nowcasting/helpers')
from ordered_easydict import OrderedEasyDict as edict

import torch
import numpy as np
import os
from collections import OrderedDict


__C = edict()
cfg = __C
__C.GLOBAL = edict()
#__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.DEVICE = torch.device("cuda")
assert __C.GLOBAL.DEVICE is not None

__C.GLOBAL.BATCH_SIZE = 4
__C.GLOBAL.MODEL_SAVE_DIR=''
#__C.GLOBAL.MODEL_SAVE_DIR = './road_map2/bams/train_35'
assert __C.GLOBAL.MODEL_SAVE_DIR is not None

__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
__C.HKO_DATA_BASE_PATH = os.path.join(__C.ROOT_DIR, 'ium_data')

__C.HKO_PD = edict()
__C.HKO_PD.RAINY_TRAIN = "bj_train_set.txt"
__C.HKO_PD.RAINY_VALID = "bj_valid_set.txt"
__C.HKO_PD.RAINY_TEST = "bj_test_set.txt"

__C.HKO = edict()
__C.HKO.EVALUATION = edict()
__C.HKO.EVALUATION.THRESHOLDS = (10, 20, 30, 35, 40,50)
__C.HKO.EVALUATION.BALANCING_WEIGHTS = (1, 1, 5,10, 10, 30, 32)        # The corresponding balancing weights
#__C.HKO.EVALUATION.N0T_BALANCING_WEIGHTS = (1, 1, 1, 1, 1, 1, 1)    
__C.HKO.EVALUATION.CENTRAL_REGION = (0, 0, 600, 600)
__C.HKO.EVALUATION.VALID_DATA_USE_UP = True



__C.HKO.BENCHMARK = edict()
__C.HKO.BENCHMARK.STAT_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'benchmark_stat')
if not os.path.exists(__C.HKO.BENCHMARK.STAT_PATH):
    os.makedirs(__C.HKO.BENCHMARK.STAT_PATH)
__C.HKO.BENCHMARK.VISUALIZE_SEQ_NUM = 30  # Number of sequences that will be plotted and saved to the benchmark directory
__C.HKO.BENCHMARK.IN_LEN = 5   # The maximum input length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.OUT_LEN = 30  # The maximum output length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.STRIDE = 5   # The stride


__C.HKO.ITERATOR = edict()
__C.HKO.ITERATOR.WIDTH = 600
__C.HKO.ITERATOR.HEIGHT = 600

__C.MODEL = edict()
#from nowcasting.models.model import activation
sys.path.append('./road_map2/bams/nowcasting/models')
from model import activation

__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)
__C.MODEL.IN_LEN = 5              # Size of the input
__C.MODEL.OUT_LEN = 30             # Size of the output

