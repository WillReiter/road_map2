#!/bin/bash

cd "/home/syao/Desktop/trajGRU/bams/experiments"


save_dir=online_test
iter=600000
mode='test'
in_data_dir=online_data/test

python ./code/online_test.py \
        --load_iter ${iter} \
        --mode ${mode} \
        --save_dir ${save_dir} \
	--in_data_dir ${in_data_dir} \
	>./${save_dir}/run_example.log 2>&1 &
