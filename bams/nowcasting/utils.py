import numpy as np
from torch import nn
from collections import OrderedDict
from nowcasting.config import cfg
#import cv2
import os.path as osp
import os
from nowcasting.hko.mask import read_mask_file

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:   ## no pool
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))

def count_pixels(name=None):
    png_dir = cfg.HKO_PNG_PATH
    mask_dir = cfg.HKO_MASK_PATH
    counts = np.zeros(256, dtype=np.float128)
    for root, dirs, files in os.walk(png_dir):
        for file_name in files:
            if not file_name.endswith('.png'):
                continue
            tmp_dir = '/'.join(root.split('/')[-3:])
            png_path = osp.join(png_dir, tmp_dir, file_name)
            mask_path = osp.join(mask_dir, tmp_dir, file_name.split('.')[0]+'.mask')
            label, count = np.unique(cv2.cvtColor(cv2.imread(png_path), cv2.COLOR_BGR2GRAY)[read_mask_file(mask_path)], return_counts=True)
            counts[label] += count
    if name is not None:
        np.save(name, counts)
    return counts



    return np.power(10, (dBZ - 10 * np.log10(a))/(10*b))

