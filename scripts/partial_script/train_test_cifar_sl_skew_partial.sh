#!/bin/bash

n=${1-"10"}
i=${2-"1"}


time python src/decentralized_sl_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 224 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 20 --frac 1  --y_partition_skew --y_partition_ratio 0.3 --log_directory  "partial_scripts"   


time python src/decentralized_sl_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 224 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.7  --y_partition_skew --y_partition_ratio  0.3 --log_directory  "partial_scripts"  
 

time python src/decentralized_sl_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 224 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.5    --y_partition_skew --y_partition_ratio 0.3 --log_directory  "partial_scripts"   


time python src/decentralized_sl_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 224 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.3 --y_partition_skew --y_partition_ratio 0.3 --log_directory  "partial_scripts" 
wait 

time python src/decentralized_sl_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 224 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.2   --y_partition_skew --y_partition_ratio 0.3 --log_directory  "partial_scripts" 
 

time python src/decentralized_sl_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 224 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.1   --y_partition_skew --y_partition_ratio 0.3  --log_directory  "partial_scripts"  


time python src/decentralized_sl_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 224 --local_ep 5 --epochs=500 --log_file_name "dir_partial_sl" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 20 --frac 0.05 --y_partition_skew --y_partition_ratio 0.3 --log_directory  "partial_scripts"   
wait