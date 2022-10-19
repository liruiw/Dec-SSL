#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from update import LocalUpdate, test_inference

from options import args_parser
from models import *
from utils import *
import numpy as np
import random
import csv
from datetime import datetime
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel as DP
from pprint import pprint

if __name__ == "__main__":
    start_time = time.time()

    # define paths
    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = "cuda" if args.gpu else "cpu"

    # load dataset and user groups
    # set_seed(args.seed)
    (
        train_dataset,
        test_dataset,
        user_groups,
        memory_dataset,
        test_user_groups,
    ) = get_dataset(args)

    batch_size = args.batch_size
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    memory_loader = DataLoader(
        memory_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )
    suffix = "{}_{}_{}_{}_featarc_dec_ssl_{}".format(
        args.model, args.batch_size, args.epochs, args.save_name_suffix, args.ssl_method
    )
    model_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S") + "_{}".format(
        str(os.getpid())
    )  # to avoid collision
    args.model_time = model_time
    model_output_dir = "save/" + model_time
    save_args_json(model_output_dir, args)
    logger = SummaryWriter(model_output_dir + "/tensorboard")

    # build model
    num_clusters = args.num_clusters
    global_models = [get_global_model(args, train_dataset) for _ in range(num_clusters)]
    for global_model in global_models:
        global_model.to(device)
        global_model.train()

    # training
    train_loss, train_accuracy = [], []
    print_every = 50
    val_loss_pre, counter = 0, 0
    epoch_accuracy = []
    global_weights = [global_model.state_dict() for global_model in global_models]
    global_classifier_model_acc, global_model_loss = [], []

    cluster_assignments = [i % num_clusters for i in range(args.num_users)]
    local_models = [
        copy.deepcopy(global_models[cluster_assignments[i]])
        for i in range(args.num_users)
    ]

    sl_train_dataset, _ = get_classifier_dataset(args)
    print("output model:", model_output_dir)

    # training loop
    for epoch in tqdm(range(int(args.epochs / args.local_ep))):
        # use local user feature bank to identify cluster
        local_knn_model_acc, list_loss = [], []
        local_classifier_model_acc, local_model_loss = [], []
        local_feature_banks = []
        global_feature_banks = []
        for global_model in global_models:
            global_feature_model = global_model.f
            feature_bank = knn_monitor(
                global_feature_model,
                memory_loader,
                test_loader,
                device,
                k=200,
                hide_progress=False,
                feature_only=True,
            )
            global_feature_banks.append(feature_bank)

        for c in range(args.num_users):  # full participant
            local_model = LocalUpdate(
                args=args,
                dataset=sl_train_dataset,
                idx=c,
                idxs=user_groups[c],
                logger=logger,
                test_dataset=test_dataset,
                memory_dataset=memory_dataset,
            )

            feature_bank = knn_monitor(
                local_models[c].f,
                local_model.memory_loader,
                local_model.test_loader,
                device,
                k=200,
                hide_progress=False,
                feature_only=True,
            )
            local_feature_banks.append(feature_bank)
            # local_knn_model_acc.append(acc)
            list_loss.append(0.0)

            # evaluate cluster feature bank
            feature_bank_alignment = []
            for j in range(args.num_clusters):
                feature_bank_j = global_feature_banks[j][:, user_groups[c]][
                    :, : int(0.8 * len(user_groups[c]))
                ]
                feature_bank_alignment.append(
                    (feature_bank_j * feature_bank).sum(0).mean().item()
                )  # inner product

            assigned_id = np.argmax(feature_bank_alignment)
            cluster_assignments[c] = assigned_id
            print(
                "assign user {} to cluster {} alignment score: {}".format(
                    c, assigned_id, feature_bank_alignment
                )
            )

        # update local models
        local_weights, local_losses = [], []
        print(f"\n | Global Training Round : {epoch+1} |\n")

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            global_model = global_models[cluster_assignments[idx]]
            global_model.train()
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idx=idx,
                idxs=user_groups[idx],
                logger=logger,
            )
            w, loss = local_model.update_ssl_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                additionl_feature_banks=global_feature_banks[cluster_assignments[idx]],
            )
            local_models[idx] = local_model.model
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update cluster weights
        for i in range(args.num_clusters):
            selected_idx = np.where(np.array(cluster_assignments) == i)[0]
            if len(selected_idx) > 0:
                global_weights[i] = average_weights(
                    [local_weights[idx] for idx in selected_idx]
                )
                global_models[i].load_state_dict(global_weights[i])

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # knn monitor
        if (epoch + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Local Client Loss : {np.mean(np.array(train_loss))}")

        global_model.save_model(model_output_dir, step=epoch)

    for idx, global_model in enumerate(global_models):
        global_model.save_model(model_output_dir, "cluster_{}".format(idx))

    # evaluate representations
    test_acc = np.max(
        [
            global_repr_global_classifier(args, global_model, args.finetuning_epoch)
            for global_model in global_models
        ]
    )

    print(f" \n Results after {args.epochs} global rounds of training:")
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

    # PLOTTING (optional)
    pprint(args)
    write_log_and_plot(
        model_time,
        model_output_dir,
        args,
        suffix,
        test_acc,
    )
