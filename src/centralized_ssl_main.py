#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from options import args_parser
from utils import *
from tensorboardX import SummaryWriter
import numpy as np

from datetime import datetime
from pprint import pprint
import os
from models import *
import os
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler


if __name__ == "__main__":
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
    device = "cuda"
    model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S") + "_{}".format(
        str(os.getpid())
    )
    args.model_time = model_time

    model_output_dir = os.path.join("save/", model_time)
    save_args_json(model_output_dir, args)
    # set_seed(args.seed)
    logger = SummaryWriter(model_output_dir + "/tensorboard")
    print("output dir:", model_output_dir)

    # load datasets
    train_dataset, test_dataset, _, memory_dataset, _ = get_dataset(args)
    batch_size = args.batch_size

    dist_sampler = (
        DistributedSampler(test_dataset)
        if args.distributed_training
        else RandomSampler(test_dataset)
    )
    test_loader = DataLoader(
        test_dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=False,
        drop_last=True,
    )
    dist_sampler = (
        DistributedSampler(memory_dataset)
        if args.distributed_training
        else RandomSampler(memory_dataset)
    )
    memory_loader = DataLoader(
        memory_dataset,
        sampler=dist_sampler,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=False,
        drop_last=True,
    )

    dist_sampler = (
        DistributedSampler(train_dataset)
        if args.distributed_training
        else RandomSampler(train_dataset)
    )

    # change the proportion of the trainloader
    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=dist_sampler,
        num_workers=16,
        pin_memory=False,
        drop_last=True,
    )

    suffix = "{}_{}_{}_{}_baseline_ssl_{}".format(
        args.model, args.batch_size, args.epochs, args.save_name_suffix, args.ssl_method
    )

    # BUILD MODEL
    global_model = get_global_model(args, train_dataset)
    if args.distributed_training:
        global_model = DDP(global_model.to(device))
    global_model.train()
    start_epoch = 0
    print_every = 50

    # Training
    # Set optimizer and criterion
    if args.optimizer == "sgd":
        args.lr = args.lr * (args.batch_size / 256)
        optim_params = global_model.parameters()
        optimizer = torch.optim.SGD(
            optim_params, lr=args.lr, momentum=0.9, weight_decay=5e-4
        )

    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            global_model.parameters(), lr=args.lr, weight_decay=1e-6
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 90], gamma=0.3
        )

    epoch_loss = []
    epoch_accuracy = []
    global_step = 0
    max_steps = len(trainloader) * args.epochs

    if hasattr(global_model, "net"):
        global_model.f = global_model.net

    scaler = GradScaler()

    for epoch in tqdm(range(0, args.epochs + 1)):
        # scheduler.step()
        global_model.train()
        if args.optimizer == "sgd":
            adjust_learning_rate(optimizer, args.lr, epoch, args)

        lr = optimizer.param_groups[0]["lr"]
        batch_loss = []

        for batch_idx, data in enumerate(trainloader):

            (pos_1, pos_2, labels) = data
            optimizer.zero_grad()
            loss = global_model(
                pos_1.to(device, non_blocking=True),
                pos_2.to(device, non_blocking=True),
            )
            loss.backward()
            optimizer.step()

            if args.ssl_method == "byol":
                global_model.update_moving_average(global_step, max_steps)  #

            if batch_idx % 50 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch + 1,
                        batch_idx * len(pos_1),
                        len(trainloader.dataset),
                        100.0 * batch_idx / len(trainloader),
                        loss.item(),
                    )
                )
            batch_loss.append(loss.item())
            global_step += 1

        loss_avg = sum(batch_loss) / len(batch_loss)
        print("\nTrain Epoch: {} loss: {} lr: {}".format(epoch, loss_avg, lr))
        epoch_loss.append(loss_avg)
        global_model.module.save_model(model_output_dir, step=epoch)

    test_acc = global_repr_global_classifier(args, global_model, args.finetuning_epoch)
    suffix = "{}_{}_{}_{}_baseline_ssl_{}".format(
        args.model, batch_size, args.epochs, args.save_name_suffix, args.ssl_method
    )
    write_log_and_plot(
        model_time,
        model_output_dir,
        args,
        suffix,
        test_acc,
    )
