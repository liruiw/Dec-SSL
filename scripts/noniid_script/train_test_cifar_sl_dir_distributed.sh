#!/bin/bash
n=${1-"10"}
i=${2-"1"}

OMPI_COMM_WORLD_SIZE=1 OMPI_COMM_WORLD_RANK=0 python -m torch.distributed.run --nproc_per_node=1  --nnodes=1 --node_rank=0   src/decentralized_sl_repr_main.py \
--model=resnet --dataset=cifar --gpu=0 --iid=0 --batch_size=256 --local_bs 256 --local_ep 1 --epochs=1 --log_file_name "dir_sl_lesscomm3" \
 --lr 0.001 --optimizer adam   --backbone resnet18  --num_users 5 --frac 1   --dirichlet --dir_beta 0.2   --log_directory "comm_scripts" --local_ep 50   --finetuning_epoch 1  --distributed_training