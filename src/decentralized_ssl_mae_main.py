#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import random
import csv
import numpy as np
from tqdm import tqdm
import torch

from tensorboardX import SummaryWriter
from options import args_parser
from models import *
from utils import *
from datetime import datetime
from update import LocalUpdate, test_inference
from pprint import pprint
import IPython

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import Process
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import RandomSampler
import socket
from mae_model import *

if __name__ == "__main__":
    start_time = time.time()

    # define paths
    path_project = os.path.abspath("..")
    args = args_parser()
    exp_details(args)

    if args.distributed_training:
        global_rank, world_size = get_dist_env()
        hostname = socket.gethostname()
        print("initing distributed training")
        dist.init_process_group(
            backend="nccl",
            rank=global_rank,
            world_size=world_size,
            init_method=args.dist_url,
        )
        args.world_size = world_size
        args.batch_size *= world_size

    device = "cuda" if args.gpu else "cpu"

    # load dataset and user groups
    set_seed(args.seed)
    (
        train_dataset,
        test_dataset,
        user_groups,
        memory_dataset,
        test_user_groups,
    ) = get_dataset(args)
    batch_size = args.batch_size
    pprint(args)

    model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S") + "_{}".format(
        str(os.getpid())
    )
    model_output_dir = "save/" + model_time
    args.model_time = model_time
    save_args_json(model_output_dir, args)
    logger = SummaryWriter(model_output_dir + "/tensorboard")
    args.start_time = datetime.now()

    # build model
    start_epoch = 0
    global_model = get_global_model(args, train_dataset).to(device)
    global_weights = global_model.state_dict()

    if args.distributed_training:
        global_model = DDP(global_model)
    else:
        global_model = torch.nn.DataParallel(global_model)
    global_model.train()

    # Training
    train_loss, train_accuracy, global_model_accuracy = [], [], []
    print_every = 200
    local_models = [copy.deepcopy(global_model) for _ in range(args.num_users)]

    optimizer = torch.optim.AdamW(
        global_model.parameters(), lr=args.lr, weight_decay=0.05
    )

    total_epochs = int(args.epochs / args.local_ep)  # number of rounds
    schedule = [
        int(total_epochs * 0.3),
        int(total_epochs * 0.6),
        int(total_epochs * 0.9),
    ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=schedule, gamma=0.3
    )
    print("output model:", model_output_dir)
    print(
        "number of users per round: {}".format(max(int(args.frac * args.num_users), 1))
    )
    print("total number of rounds: {}".format(total_epochs))
    local_update_clients = [
        LocalUpdate(
            args=args,
            dataset=train_dataset,
            idx=idx,
            idxs=user_groups[idx],
            logger=logger,
            output_dir=model_output_dir,
        )
        for idx in range(args.num_users)
    ]

    for client in local_update_clients:
        client.init_model(global_model)

    lr = optimizer.param_groups[0]["lr"]

    for epoch in tqdm(range(start_epoch, total_epochs)):

        local_weights, local_losses = [], []
        print(f"\n | Global Training Round : {epoch+1} | Model : {model_time}\n")

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # update each local model
        for idx in idxs_users:
            local_model = local_update_clients[idx]
            w, loss = local_model.update_ssl_weights(
                model=local_models[idx],
                global_round=epoch,
                lr=lr,
            )
            local_models[idx] = local_model.model
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)
        if args.average_without_bn:
            for i in range(args.num_users):
                local_models[i] = load_weights_without_batchnorm(
                    local_models[i], global_weights
                )
        else:
            for i in range(args.num_users):
                local_models[i] = load_weights(local_models[i], global_weights)
        global_model.load_state_dict(global_weights)

        # print global training loss after every 'i' rounds
        if (int(epoch * args.local_ep) + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Local Client Loss : {np.mean(np.array(train_loss))}")

        scheduler.step()
        lr = scheduler._last_lr[0]
        global_model.module.save_model(model_output_dir, step=epoch)

    global_model.module.save_model(model_output_dir, step=epoch)

    # evaluate representations
    print("evaluating representations: ", model_output_dir)
    test_acc = global_repr_global_classifier(args, global_model, args.finetuning_epoch)

    print(f" \n Results after {args.epochs} global rounds of training:")
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

    # PLOTTING (optional)
    pprint(args)
    suffix = "{}_{}_{}_{}_dec_ssl_mae_{}".format(
        args.model, args.batch_size, args.epochs, args.save_name_suffix, args.ssl_method
    )
    write_log_and_plot(model_time, model_output_dir, args, suffix, test_acc)
