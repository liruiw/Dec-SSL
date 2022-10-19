#!/bin/bash

n=${1-"10"}
i=${2-"1"}

time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --epochs=1000 --local_ep 50 --log_file_name "skew_partition_mae"  --log_directory "noniid_scripts" \
									  --lr 0.0002 --ssl_method mae --optimizer adam --local_bs 256   --batch_size 256 --num_users 5 --frac 1   --y_partition_skew --y_partition_ratio 0.    &

time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --epochs=1000 --local_ep 50 --log_file_name "skew_partition_mae"  --log_directory "noniid_scripts" \
									  --lr 0.0002 --ssl_method mae --optimizer adam --local_bs 256   --batch_size 256 --num_users 5 --frac 1   --y_partition_skew --y_partition_ratio 0.7     
 
time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --epochs=1000 --local_ep 50 --log_file_name "skew_partition_mae"  --log_directory "noniid_scripts" \
									  --lr 0.0002 --ssl_method mae --optimizer adam --local_bs 256   --batch_size 256 --num_users 5 --frac 1   --y_partition_skew --y_partition_ratio 0.1   &

time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --epochs=1000 --local_ep 50 --log_file_name "skew_partition_mae"  --log_directory "noniid_scripts" \
									  --lr 0.0002 --ssl_method mae --optimizer adam --local_bs 256   --batch_size 256 --num_users 5 --frac 1   --y_partition_skew --y_partition_ratio 0.3    

time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --epochs=1000 --local_ep 50 --log_file_name "skew_partition_mae"  --log_directory "noniid_scripts" \
									  --lr 0.0002 --ssl_method mae --optimizer adam --local_bs 256   --batch_size 256 --num_users 5 --frac 1   --y_partition_skew --y_partition_ratio 1    
wait