#!/bin/bash

save_dir=test_1000_1600epoch
iter=600000
dataset='test'         ## choose from [test, valid]
net_name='trajgru'    ## choose from [trajgru, convlstm]

if [ ! -d ./${save_dir} ];then
        echo "can't find dir: "$1
        exit -1
fi


nohup python ./code/test.py \
       --load_iter ${iter} \
       --dataset   ${dataset} \
       --save_dir  ${save_dir} \
       --net_name  ${net_name} \
       >./${save_dir}/test_${iter}.log 2>&1 &

