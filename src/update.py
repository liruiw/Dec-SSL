#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import *
from torchvision import transforms
import numpy as np
import IPython
import copy
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import time


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(
        self, dataset, idxs, idx=0, noniid=False, noniid_prob=1.0, xshift_type="rot"
    ):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.idx = idx
        self.noniid = noniid
        self.classes = self.dataset.classes
        self.targets = np.array(self.dataset.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset[self.idxs[item]]


class LocalUpdate(object):
    def __init__(
        self,
        args,
        dataset,
        idx,
        idxs,
        logger=None,
        test_dataset=None,
        memory_dataset=None,
        output_dir="",
    ):
        self.args = args
        self.logger = logger
        self.id = idx  # user id
        self.idxs = idxs  # dataset id
        self.reg_scale = args.reg_scale
        self.output_dir = output_dir

        if dataset is not None:
            self.trainloader, self.validloader, self.testloader = self.train_val_test(
                dataset, list(idxs), test_dataset, memory_dataset
            )

        self.device = "cuda" if args.gpu else "cpu"
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def get_model(self):
        return self.model

    def init_dataset(self, dataset):
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs), test_dataset, memory_dataset
        )

    def init_model(self, model):
        """Initialize local models"""
        train_lr = self.args.lr
        self.model = model

        if self.args.optimizer == "sgd":
            train_lr = self.args.lr * (self.args.batch_size / 256)
            if self.args.distributed_training:
                train_lr = train_lr * self.args.world_size
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=train_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )

        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_lr, weight_decay=1e-6
            )

        total_epochs = self.args.local_ep * self.args.epochs
        self.schedule = [
            int(total_epochs * 0.3),
            int(total_epochs * 0.6),
            int(total_epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.schedule, gamma=0.3
        )
        self.scheduler = scheduler
        self.optimizer = optimizer

    def train_val_test(self, dataset, idxs, test_dataset=None, memory_dataset=None):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes. split indexes for train, validation, and test (80, 10, 10)
        """
        idxs_train = idxs[: int(0.8 * len(idxs))]
        self.idxs_train = idxs_train
        idxs_val = idxs[int(0.8 * len(idxs)) : int(0.9 * len(idxs))]
        idxs_test = idxs[int(0.9 * len(idxs)) :]

        train_dataset = DatasetSplit(
            dataset,
            idxs_train,
            idx=self.id,
        )

        if not self.args.distributed_training:
            trainloader = DataLoader(
                train_dataset,
                batch_size=self.args.local_bs,
                shuffle=True,
                num_workers=16,
                pin_memory=True,
                drop_last=True if len(train_dataset) > self.args.local_bs else False,
            )
        else:
            self.dist_sampler = DistributedSampler(train_dataset, shuffle=True)
            trainloader = DataLoader(
                train_dataset,
                sampler=self.dist_sampler,
                batch_size=self.args.local_bs,
                num_workers=16,
                pin_memory=True,
                drop_last=True,
            )

        validloader = DataLoader(
            DatasetSplit(
                dataset,
                idxs_val,
                idx=self.id,
            ),
            batch_size=self.args.local_bs,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        testloader = DataLoader(
            DatasetSplit(
                dataset,
                idxs_test,
                idx=self.id,
            ),
            batch_size=64,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        if test_dataset is not None:
            # such that the memory loader is the original dataset without pair augmentation
            memoryloader = DataLoader(
                DatasetSplit(
                    memory_dataset,
                    idxs_train,
                    idx=self.id,
                ),
                batch_size=64,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )

        else:
            memoryloader = DataLoader(
                DatasetSplit(
                    dataset,
                    idxs_train,
                    idx=self.id,
                ),
                batch_size=self.args.local_bs,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                drop_last=False,
            )

        self.memory_loader = memoryloader
        self.test_loader = testloader

        return trainloader, validloader, testloader

    def update_fc_weights(self, model, global_round, train_dataset=None):
        """Train the linear layer with the encode frozen"""
        model.train()
        epoch_loss = []
        if train_dataset is not None:
            trainloader = DataLoader(
                train_dataset,
                batch_size=self.args.local_bs,
                shuffle=True,
                num_workers=16,
                pin_memory=True,
            )
        else:
            trainloader = self.trainloader

        # only adam
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-6)
        for param in model.f.parameters():
            param.requires_grad = False

        for iter in range(int(self.args.local_ep)):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print(
                        "Update FC || User : {} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            self.id,
                            global_round,
                            self.args.local_ep * global_round + iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                if self.logger is not None:
                    self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_ssl_weights(
        self,
        model,
        global_round,
        additionl_feature_banks=None,
        lr=None,
        epoch_num=None,
        vis_feature=False,
    ):
        """Train the local model with self-superivsed learning"""
        epoch_loss = [0]
        global_model_copy = copy.deepcopy(model)
        global_model_copy.eval()

        # Set optimizer for the local updates
        train_epoch = epoch_num if epoch_num is not None else self.args.local_ep

        if self.args.optimizer == "sgd":
            train_lr = self.args.lr * (self.args.batch_size / 256)

            if self.args.distributed_training:
                train_lr = train_lr * self.args.world_size

            train_lr = lr if lr is not None else train_lr
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=train_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )

        elif self.args.optimizer == "adam":
            train_lr = lr if lr is not None else self.args.lr
            optimizer = torch.optim.Adam(
                model.parameters(), lr=train_lr, weight_decay=1e-6
            )

        if self.args.ssl_method == "mae":
            train_lr = lr if lr is not None else self.args.lr
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=train_lr, weight_decay=0.05
            )

        if self.args.model_continue_training and hasattr(self, "optimizer"):
            optimizer.load_state_dict(self.optimizer.state_dict())

        schedule = [
            int(self.args.local_ep * self.args.epochs * 0.3),
            int(self.args.local_ep * self.args.epochs * 0.6),
            int(self.args.local_ep * self.args.epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule, gamma=0.3
        )
        global_step = 0
        max_steps = len(self.trainloader) * self.args.local_ep
        if additionl_feature_banks is not None:
            # hack: append the global model features to target for later usage
            targets = (
                np.array(self.trainloader.dataset.dataset.targets).reshape(-1).copy()
            )
            self.trainloader.dataset.dataset.target_copy = targets.copy()
            self.trainloader.dataset.dataset.targets = np.concatenate(
                (targets[:, None], additionl_feature_banks.detach().cpu().numpy().T),
                axis=1,
            )

        train_epoch_ = int(np.ceil(train_epoch))
        max_iter = int(train_epoch * len(self.trainloader))
        epoch_start_time = time.time()

        for iter in range(train_epoch_):
            model.train()
            local_curr_ep = self.args.local_ep * global_round + iter

            if self.args.optimizer == "sgd":
                adjust_learning_rate(
                    optimizer,
                    train_lr,
                    local_curr_ep,
                    self.args.epochs * self.args.local_ep,
                    iter,
                )

            batch_loss = []
            batch_size = self.args.local_bs
            temperature = self.args.temperature
            start_time = time.time()

            if self.args.distributed_training:
                self.dist_sampler.set_epoch(int(local_curr_ep))

            for batch_idx, data in enumerate(self.trainloader):
                data_time = time.time() - start_time
                start_time = time.time()

                if additionl_feature_banks is not None:
                    (pos_1, pos_2, labels) = data
                    labels, addition_features = (
                        labels[:, [0]],
                        labels[:, 1:].to(self.device),
                    )

                    loss, feat = model(
                        pos_1.to(self.device),
                        pos_2.to(self.device),
                        addition_features,
                        self.reg_scale,
                        return_feat=True,
                    )
                else:
                    if self.args.ssl_method == "mae":
                        images, labels = data
                        images, labels = images.to(self.device), labels.to(self.device)
                        predicted_img, mask = model(images)
                        feat = mask
                        mask_ratio = 0.75
                        loss = (
                            torch.mean((predicted_img - images) ** 2 * mask)
                            / mask_ratio
                        )
                    else:
                        (pos_1, pos_2, labels) = data
                        loss, feat = model(
                            pos_1.to(self.device, non_blocking=True),
                            pos_2.to(self.device, non_blocking=True),
                            return_feat=True,
                        )

                loss = loss.mean()
                optimizer.zero_grad()
                if not loss.isnan().any():
                    loss.backward()
                    optimizer.step()

                model_time = time.time() - start_time
                start_time = time.time()

                if batch_idx % 10 == 0:
                    print(
                        "Update SSL || User : {} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f} \
                        LR: {:.4f}  Feat: {:.3f} Epoch Time: {:.3f} Model Time: {:.3f} Data Time: {:.3f} Model: {}".format(
                            self.id,
                            global_round,
                            local_curr_ep,
                            batch_idx * len(labels),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                            optimizer.param_groups[0]["lr"],
                            feat.mean().item(),
                            time.time() - epoch_start_time,
                            model_time,
                            data_time,
                            self.args.model_time,
                        )
                    )
                if self.logger is not None:
                    self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())
                data_start_time = time.time()
                scheduler.step(int(local_curr_ep))

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        if additionl_feature_banks is not None:
            self.trainloader.dataset.dataset.targets = (
                self.trainloader.dataset.dataset.target_copy
            )

        self.model = model
        self.optimizer = optimizer
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def update_weights(
        self, model, global_round, vis_feature=False, lr=None, epoch_num=None
    ):
        """Train the local model with superivsed learning"""
        self.model = model
        model.train()
        epoch_loss = []

        if self.args.optimizer == "sgd":
            train_lr = self.args.lr * (self.args.batch_size / 256)
            if self.args.distributed_training:
                train_lr = train_lr * self.args.world_size

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=train_lr,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay,
            )

        elif self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.args.lr, weight_decay=1e-6
            )
        if self.args.model_continue_training and hasattr(self, "optimizer"):
            optimizer.load_state_dict(self.optimizer.state_dict())

        schedule = [
            int(self.args.local_ep * self.args.epochs * 0.3),
            int(self.args.local_ep * self.args.epochs * 0.6),
            int(self.args.local_ep * self.args.epochs * 0.9),
        ]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=schedule, gamma=0.3
        )

        for iter in range(int(self.args.local_ep)):
            local_curr_ep = self.args.local_ep * global_round + iter
            batch_loss = []
            feature_bank, label_bank, image_bank = [], [], []

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                label_bank.append(labels.detach().cpu().numpy())
                optimizer.zero_grad()
                log_probs, feat = model(images, return_feat=True)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    print(
                        "Inference || User : {} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            self.id,
                            global_round,
                            self.args.local_ep * global_round + iter,
                            batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100.0 * batch_idx / len(self.trainloader),
                            loss.item(),
                        )
                    )
                self.logger.add_scalar("loss", loss.item())
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss) / (len(batch_loss) + 1e-4))
            scheduler.step(int(local_curr_ep))
        self.optimizer = optimizer
        self.model = model
        return model.state_dict(), sum(epoch_loss) / (len(epoch_loss) + 1e-4)

    def inference(self, model, test_dataset=None, test_user=None):
        """Returns the inference accuracy and loss for a local client."""
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        if test_dataset is not None:
            self.testloader = DataLoader(
                DatasetSplit(test_dataset, test_user, idx=self.id),
                batch_size=64,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
            )

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """Returns the test accuracy and loss."""

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = "cuda" if args.gpu else "cpu"
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=False
    )

    test_bar = tqdm((testloader), desc="Linear Probing", disable=False)

    for (images, labels) in test_bar:
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        test_bar.set_postfix({"Accuracy": correct / total * 100})

    accuracy = correct / total
    return accuracy, loss
