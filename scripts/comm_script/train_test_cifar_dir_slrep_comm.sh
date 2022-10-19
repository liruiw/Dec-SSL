#!/bin/bash

n=${1-"10"}
i=${2-"1"}


time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 500 --epochs=500 --log_file_name "dir_sl_comm" \
									  --lr 0.001 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 5 --frac 1   --dirichlet --dir_beta 0.02   --log_directory "comm_scripts"   

time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 250 --epochs=500 --log_file_name "dir_sl_comm" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 5 --frac 1      --dirichlet --dir_beta 0.02   --log_directory "comm_scripts"  
 
 
time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 100 --epochs=500 --log_file_name "dir_sl_comm" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 5 --frac 1  --dirichlet --dir_beta 0.02    --log_directory "comm_scripts"  

wait 

time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 20 --epochs=500 --log_file_name "dir_sl_comm" \
									  --lr 0.001 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 5 --frac 1   --dirichlet --dir_beta 0.02  --log_directory "comm_scripts" 

time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 10 --epochs=500 --log_file_name "dir_sl_comm" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 5 --frac 1      --dirichlet --dir_beta 0.02   --log_directory "comm_scripts"   
 
time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 5 --epochs=500 --log_file_name "dir_sl_comm" \
									  --lr 0.001 --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 5 --frac 1 --dirichlet --dir_beta 0.02  --log_directory "comm_scripts"   

time python src/decentralized_sl_repr_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 50 --epochs=500 --log_file_name "dir_sl_comm" \
									  --lr 0.001 --optimizer adam --backbone resnet18 --batch_size 256 --num_users 5 --frac 1  --dirichlet --dir_beta 0.02    --log_directory "comm_scripts"  
wait

 