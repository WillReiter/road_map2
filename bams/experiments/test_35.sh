#!/bin/bash

save_dir=test_35
iter=60000
dataset='test'         ## choose from [test, valid]
net_name='trajgru'    ## choose from [trajgru, convlstm]

if [ ! -d ./${save_dir} ];then
	echo "can't find dir: "$1
	exit -1
fi


nohup python ./road_map2/bams/code/valid_and_test.py \
       --load_iter ${iter} \
       --dataset   ${dataset} \
       --save_dir  ${save_dir} \
       --net_name  ${net_name} \
       >./${save_dir}/test_${iter}.log 2>&1 &

