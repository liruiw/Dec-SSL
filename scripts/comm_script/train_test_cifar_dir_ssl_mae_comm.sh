#!/bin/bash

n=${1-"10"}
i=${2-"1"}

time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 500 --epochs=1000 --log_file_name "dir_ssl_mae_comm" \
									  --lr 0.0002 --ssl_method mae --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 5 --frac 1   --dirichlet --dir_beta 0.02   --log_directory "comm_mae_scripts"  &  

time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 250 --epochs=1000 --log_file_name "dir_ssl_mae_comm" \
									  --lr 0.0002 --ssl_method mae --optimizer adam --backbone resnet18 --batch_size 256 --num_users 5 --frac 1      --dirichlet --dir_beta 0.02   --log_directory "comm_mae_scripts"   
 

time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 100 --epochs=1000 --log_file_name "dir_ssl_mae_comm" \
									  --lr 0.0002 --ssl_method mae --optimizer adam --backbone resnet18 --batch_size 256 --num_users 5 --frac 1  --dirichlet --dir_beta 0.02    --log_directory "comm_mae_scripts" & 
 
wait

time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 20 --epochs=1000 --log_file_name "dir_ssl_mae_comm" \
									  --lr 0.0002 --ssl_method mae --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 5 --frac 1   --dirichlet --dir_beta 0.02  --log_directory "comm_mae_scripts"     

time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 10 --epochs=1000 --log_file_name "dir_ssl_mae_comm" \
									  --lr 0.0002 --ssl_method mae --optimizer adam --backbone resnet18 --batch_size 256 --num_users 5 --frac 1      --dirichlet --dir_beta 0.02   --log_directory "comm_mae_scripts"  & 
 
time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 5 --epochs=1000 --log_file_name "dir_ssl_mae_comm" \
									  --lr 0.0002 --ssl_method mae --optimizer adam   --backbone resnet18 --batch_size 256 --num_users 5 --frac 1 --dirichlet --dir_beta 0.02  --log_directory "comm_mae_scripts"      

time python src/decentralized_ssl_mae_main.py --model=resnet --dataset=cifar  --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 50 --epochs=1000 --log_file_name "dir_ssl_mae_comm" \
									  --lr 0.0002 --ssl_method mae --optimizer adam --backbone resnet18 --batch_size 256 --num_users 5 --frac 1  --dirichlet --dir_beta 0.02    --log_directory "comm_mae_scripts"  
wait
 
 