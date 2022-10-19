#!/bin/bash
n=${1-"10"}
i=${2-"1"}

time python src/decentralized_sl_main.py --model=resnet --dataset=cifar  --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 50 --epochs=500  --log_directory "noniid_scripts" \
									  --lr 0.001 --optimizer adam --local_bs 256 --batch_size 256 --num_users 5 --frac 1  --local_ep 50 --x_shift_dirichlet --imagenet_based_cluster --dir_beta 0.01  --log_file_name "dirichlet_alpha_sl_featt"

time python src/decentralized_sl_main.py --model=resnet --dataset=cifar  --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 50 --epochs=500  --log_directory "noniid_scripts" \
									  --lr 0.001 --optimizer adam --local_bs 256 --batch_size 256 --num_users 5 --frac 1  --local_ep 50 --x_shift_dirichlet --imagenet_based_cluster --dir_beta 0.1  --log_file_name "dirichlet_alpha_sl_featt"

time python src/decentralized_sl_main.py --model=resnet --dataset=cifar   --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 50 --epochs=500  --log_directory "noniid_scripts" \
									  --lr 0.001 --optimizer adam --local_bs 256 --batch_size 256 --num_users 5 --frac 1  --local_ep 50 --x_shift_dirichlet --imagenet_based_cluster  --dir_beta 1  --log_file_name "dirichlet_alpha_sl_featt"

time python src/decentralized_sl_main.py --model=resnet --dataset=cifar  --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 50 --epochs=500  --log_directory "noniid_scripts" \
									  --lr 0.001 --optimizer adam --local_bs 256 --batch_size 256 --num_users 5 --frac 1  --local_ep 50 --x_shift_dirichlet --imagenet_based_cluster  --dir_beta 5  --log_file_name "dirichlet_alpha_sl_featt"
 