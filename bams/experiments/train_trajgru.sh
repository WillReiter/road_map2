#!/bin/bash

if [ ! -d ./${1} ];then
	echo "can't find dir: "$1
	exit -1
fi

#batch_size=4
batch_size=4
max_iterations=100000
valid_and_save_checkpoint_iterations=1
LR=1e-4


nohup python ./code/train_trajgru.py --save_dir $1 \
                                     --batch_size ${batch_size} \
                                     --max_iterations ${max_iterations}\
                                     --valid_and_save_checkpoint_iterations ${valid_and_save_checkpoint_iterations}\
                                     --LR ${LR} \
                                     >./${1}/train_trajgru.log 2>&1 &


#--save_dir ./train_model --batch_size 4 --max_iterations 30000 --valid_and_save_checkpoint_iterations 5 --LR 1e-4
