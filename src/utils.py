#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import shutil

import copy
import torch
from torchvision import datasets, transforms
from sampling import *
from PIL import Image

import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import *
from tqdm import tqdm
import IPython
from torch.utils.data import Dataset

import os
import json
import random
import csv
import math
import time
import glob
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader
from mae_model import *

plt_param = {
    "legend.fontsize": 65,
    "figure.figsize": (54, 36),  # (72, 48)
    "axes.labelsize": 80,
    "axes.titlesize": 80,
    "font.size": 80,
    "xtick.labelsize": 80,
    "ytick.labelsize": 80,
    "lines.linewidth": 10,
    "lines.color": (0, 0, 0),
}

plt.rcParams.update(plt_param)

modified_cifar_data = None
modified_cifar_test_data = None


def get_classifier_dataset(args):
    if args.dataset.endswith("ssl"):
        args.dataset = args.dataset[:-3]  # remove the ssl
    train_dataset, test_dataset, _, _, _ = get_dataset(args)
    return train_dataset, test_dataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def get_dist_env():
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        world_size = int(os.getenv("OMPI_COMM_WORLD_SIZE"))
    else:
        world_size = int(os.getenv("SLURM_NTASKS"))

    if "OMPI_COMM_WORLD_RANK" in os.environ:
        global_rank = int(os.getenv("OMPI_COMM_WORLD_RANK"))
    else:
        global_rank = int(os.getenv("SLURM_PROCID"))
    return global_rank, world_size


def global_repr_global_classifier(args, global_model, test_epoch=60):
    # global representation, global classifier
    from models import ResNetCifarClassifier
    from update import test_inference

    device = "cuda"

    train_dataset, test_dataset = get_classifier_dataset(args)
    if args.ssl_method == "mae":
        global_model_classifer = ViT_Classifier(
            global_model.encoder, num_classes=10
        ).to(device)
        global_model_classifer = global_model_classifer.cuda()
        optimizer = torch.optim.AdamW(
            global_model_classifer.head.parameters(), lr=3e-4, weight_decay=0.05
        )

    else:
        print("begin training classifier...")
        global_model_classifer = ResNetCifarClassifier(args=args)
        if hasattr(global_model, "module"):
            global_model = global_model.module
        global_model_classifer.load_state_dict(
            global_model.state_dict(), strict=False
        )  #
        global_model_classifer = global_model_classifer.cuda()
        for param in global_model_classifer.f.parameters():
            param.requires_grad = False

        # train only the last layer
        optimizer = torch.optim.Adam(
            global_model_classifer.fc.parameters(), lr=1e-3, weight_decay=1e-6
        )

    # remove the ssl in the training dataset name
    dist_sampler = (
        DistributedSampler(train_dataset)
        if args.distributed_training
        else RandomSampler(train_dataset)
    )
    trainloader = DataLoader(
        train_dataset,
        sampler=dist_sampler,
        batch_size=256,
        num_workers=16,
        pin_memory=False,
    )
    criterion = (
        torch.nn.NLLLoss().to(device)
        if args.ssl_method != "mae"
        else torch.nn.CrossEntropyLoss().to(device)
    )
    best_acc = 0

    # train global model on global dataset
    for epoch_idx in tqdm(range(test_epoch)):
        batch_loss = []
        if args.distributed_training:
            dist_sampler.set_epoch(epoch_idx)

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = global_model_classifer(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(
                    "Downstream Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch_idx + 1,
                        batch_idx * len(images),
                        len(trainloader.dataset),
                        100.0 * batch_idx / len(trainloader),
                        loss.item(),
                    )
                )
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        test_acc, test_loss = test_inference(args, global_model_classifer, test_dataset)
        if test_acc > best_acc:
            best_acc = test_acc
        print("\n Downstream Train loss: {} Acc: {}".format(loss_avg, best_acc))
    return best_acc


def write_log_and_plot(
    model_time, model_output_dir, args, suffix, test_acc, intermediate=False
):
    # mkdir_if_missing('save')
    if not os.path.exists("save/" + args.log_directory):
        os.makedirs("save/" + args.log_directory)

    log_file_name = (
        args.log_file_name + "_intermediate" if intermediate else args.log_file_name
    )
    elapsed_time = (
        (datetime.now() - args.start_time).seconds if hasattr(args, "start_time") else 0
    )
    with open(
        "save/{}/best_linear_statistics_{}.csv".format(
            args.log_directory, log_file_name
        ),
        "a+",
    ) as outfile:
        writer = csv.writer(outfile)

        res = [
            suffix,
            "",
            "",
            args.dataset,
            "acc: {}".format(test_acc),
            "num of user: {}".format(args.num_users),
            "frac: {}".format(args.frac),
            "epoch: {}".format(args.epochs),
            "local_ep: {}".format(args.local_ep),
            "local_bs: {}".format(args.local_bs),
            "lr: {}".format(args.lr),
            "backbone: {}".format(args.backbone),
            "dirichlet {}: {}".format(args.dirichlet, args.dir_beta),
            "imagenet_based_cluster: {}".format(args.imagenet_based_cluster),
            "partition_skew: {}".format(args.y_partition_skew),
            "partition_skew_ratio: {}".format(args.y_partition_ratio),
            "iid: {}".format(args.iid),
            "reg scale: {}".format(args.reg_scale),
            "cont opt: {}".format(args.model_continue_training),
            model_time,
            "elapsed_time: {}".format(elapsed_time),
        ]
        writer.writerow(res)

        name = "_".join(res).replace(": ", "_")
        print("writing best results for {}: {} !".format(name, test_acc))


def get_global_model(args, train_dataset):
    from models import SimCLR
    from simsiam import SimSiam

    if args.ssl_method == "simclr":
        global_model = SimCLR(args=args)
    elif args.ssl_method == "simsiam":
        global_model = SimSiam(args=args)
    elif args.ssl_method == "mae":
        global_model = MAE_ViT(mask_ratio=0.75)
    return global_model


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_args_json(path, args):
    mkdir_if_missing(path)
    arg_json = os.path.join(path, "args.json")
    with open(arg_json, "w") as f:
        args = vars(args)
        json.dump(args, f, indent=4, sort_keys=True)


def print_and_write(file_handle, text):
    print(text)
    if file_handle is not None:
        file_handle.write(text + "\n")
    return text


def adjust_learning_rate(optimizer, init_lr, epoch, full_epoch, local_epoch):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1.0 + math.cos(math.pi * epoch / full_epoch))
    for param_group in optimizer.param_groups:
        if "fix_lr" in param_group and param_group["fix_lr"]:
            param_group["lr"] = init_lr
        else:
            param_group["lr"] = cur_lr


def get_backbone(pretrained_model_name="resnet50", pretrained=False, full_size=False):
    f = []
    model = eval(pretrained_model_name)(pretrained=pretrained)

    for name, module in model.named_children():

        if name == "conv1" and not full_size:  # add not full_size
            module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        if full_size:
            print(name)
            if name != "fc":
                f.append(module)
        else:
            if not isinstance(module, nn.Linear) and not isinstance(
                module, nn.MaxPool2d
            ):
                f.append(module)

    # encoder
    f = nn.Sequential(*f)
    feat_dim = 2048 if "resnet50" in pretrained_model_name else 512
    print("feat dim:", feat_dim)
    return f, feat_dim


def get_dataset(args, **kwargs):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    global modified_cifar_data, modified_cifar_test_data
    start = time.time()
    print("sampling for dataset: {}".format(args.dataset))

    if "cifar" in args.dataset:
        data_dir = "data/cifar/"
        dataset_name = "CIFAR10"
        train_dataset_func = (
            eval(dataset_name + "Pair")
            if "ssl" in args.dataset
            else getattr(datasets, dataset_name)
        )
        img_size = 32
        train_transform_ = get_transform(img_size)
        test_transform_ = test_transform
        if args.ssl_method == "mae":
            train_transform_ = test_transform_ = get_transform_mae(img_size)

        train_dataset = train_dataset_func(
            root=data_dir,
            train=True,
            download=True,
            transform=train_transform_,
        )

        print("dataset sample num:", train_dataset.data.shape)
        test_dataset = getattr(datasets, dataset_name)(
            data_dir, train=False, download=True, transform=test_transform_
        )
        memory_dataset = getattr(datasets, dataset_name)(
            data_dir, train=True, download=True, transform=test_transform_
        )

    elif "imagenet100" in args.dataset:  # tiny imagenet
        data_dir = "data/imagenet100_v2"
        if "ssl" in args.dataset:
            train_dataset = ImageFolderPair(
                root=os.path.join(data_dir, "train"),
                transform=get_transform_imagenet(224),
                rescale=False,
            )
        else:
            # ImageFolderInstance
            train_dataset = ImageFolderInstance(
                root=os.path.join(data_dir, "train"),
                transform=linear_transform_imagenet,
                rescale=False,
                save_data=args.load_dataset_to_memory,
            )
        test_dataset = ImageFolderInstance(
            os.path.join(data_dir, "val"),
            transform=test_transform_imagenet,
            rescale=False,
            save_data=args.load_dataset_to_memory,
        )
        memory_dataset = ImageFolderInstance(
            os.path.join(data_dir, "train"),
            transform=test_transform_imagenet,
            rescale=False,
            save_data=args.load_dataset_to_memory,
        )

    print("get dataset time: {:.3f}".format(time.time() - start))
    start = time.time()

    # sample training data among users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(train_dataset, args.num_users)
        test_user_groups = cifar_iid(test_dataset, args.num_users)

    else:
        if args.dirichlet:
            print("Y dirichlet sampling")
            user_groups = cifar_noniid_dirichlet(
                train_dataset, args.num_users, args.dir_beta, vis=True
            )
            test_user_groups = cifar_noniid_dirichlet(
                test_dataset, args.num_users, args.dir_beta
            )

        elif args.imagenet_based_cluster:
            print("Feature Clustering dirichlet sampling")
            user_groups = cifar_noniid_x_cluster(
                train_dataset, args.num_users, "img_feature_load", args, vis=True
            )
            test_user_groups = cifar_noniid_x_cluster(
                test_dataset, args.num_users, "img_feature_load", args, test=True
            )

        elif args.y_partition_skew or args.y_partition:
            print("Y partition skewness sampling")
            user_groups = cifar_partition_skew(
                train_dataset, args.num_users, args.y_partition_ratio, vis=True
            )
            test_user_groups = cifar_partition_skew(
                test_dataset, args.num_users, args.y_partition_ratio
            )
        else:
            print("Use i.i.d. sampling")
            user_groups = cifar_iid(train_dataset, args.num_users)
            test_user_groups = cifar_iid(test_dataset, args.num_users)

    print("sample dataset time: {:.3f}".format(time.time() - start))
    print(
        "user data samples:", [len(user_groups[idx]) for idx in range(len(user_groups))]
    )
    return train_dataset, test_dataset, user_groups, memory_dataset, test_user_groups


def average_weights(w, avg_weights=None):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w[0].keys():
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key]

        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def load_weights_without_batchnorm(model, w):
    """
    Returns the average of the weights.
    """
    model.load_state_dict(
        {k: v for k, v in w.items() if "bn" not in k and "running" not in k},
        strict=False,
    )
    return model


def load_weights(model, w):
    """
    Returns the average of the weights.
    """
    model.load_state_dict({k: v for k, v in w.items()}, strict=False)
    return model


def exp_details(args):
    print("\nExperimental details:")
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.epochs}\n")
    print(f"    Fraction of users  : {args.frac}")
    print(f"    Local Batch size   : {args.local_bs}")
    print(f"    Local Epochs       : {args.local_ep}\n")
    return


class CIFAR10Pair(datasets.CIFAR10):
    """CIFAR10 Dataset."""

    def __init__(
        self,
        class_id=None,
        tgt_class=None,
        sample_num=10000,
        imb_factor=1,
        imb_type="",
        with_index=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sample_num = sample_num

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return pos_1, pos_2, target


class CIFAR100Pair(CIFAR10Pair):
    """CIFAR100 Dataset."""

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }


# https://github.com/tjmoon0104/pytorch-tiny-imagenet
class ImageFolderInstance(datasets.ImageFolder):
    """Folder datasets which returns the index of the image (for memory_bank)"""

    def __init__(
        self, root, transform=None, target_transform=None, rescale=True, save_data=True
    ):
        super(ImageFolderInstance, self).__init__(root, transform, target_transform)
        self.num = self.__len__()

        # load into memory
        self.data = []
        self.targets = []
        self.rescale = rescale
        self.tiny = "tiny" in root
        self.test = "test" in root
        data_path = os.path.join(root, "data.npy")
        target_path = os.path.join(root, "targets.npy")
        self.save_data = save_data

        s = time.time()
        if os.path.exists(data_path) and save_data:
            self.data = np.load(data_path, allow_pickle=True)
            self.targets = np.load(target_path, allow_pickle=True)

        else:
            print("start caching dataset")
            for i in range(len(self.imgs)):
                path, target = self.imgs[i]
                if save_data:
                    self.data.append(
                        cv2.resize(np.asarray(self.loader(path)), (256, 256))
                    )  # resize
                self.targets.append(target)
            #  print(target)
            print("finish caching dataset {:.3f}".format(time.time() - s))
            s = time.time()

            if save_data:
                self.data = np.array(self.data)
                np.save(data_path, self.data)

            self.targets = np.array(self.targets)
            np.save(target_path, self.targets)

    def __getitem__(self, index):
        if not self.save_data:
            path, target = self.imgs[index]
            image = self.loader(path)

        else:
            image, target = self.data[index], self.targets[index]  # self.imgs[index]
            image = Image.fromarray(image)

        if self.rescale:
            image = image.resize((256, 256))

        if self.transform is not None:
            image = self.transform(image)
        return image, target


class ImageFolderPair(ImageFolderInstance):
    """Folder datasets which returns the index of the image (for memory_bank)"""

    def __init__(
        self, root, transform=None, target_transform=None, rescale=True, save_data=True
    ):
        super(ImageFolderPair, self).__init__(root, transform, target_transform)
        self.num = self.__len__()
        self.rescale = rescale
        self.save_data = save_data

    def __getitem__(self, index):
        if not self.save_data:
            path, target = self.imgs[index]
            image = self.loader(path)

        else:
            image, target = self.data[index], self.targets[index]  # self.imgs[index]
            image = Image.fromarray(image)
        if self.rescale:
            image = image.resize((256, 256))

        # image
        if self.transform is not None:
            pos_1 = self.transform(image)
            pos_2 = self.transform(image)

        return pos_1, pos_2, target


def knn_monitor(
    net,
    memory_data_loader,
    test_data_loader,
    epoch,
    k=200,
    t=0.1,
    hide_progress=False,
    vis_tsne=False,
    save_fig_name="model_tsne.png",
    feature_only=False,
):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []

    with torch.no_grad():
        # generate feature bank
        for data, _ in tqdm(
            memory_data_loader,
            desc="Feature extracting",
            leave=False,
            disable=hide_progress,
        ):
            feature = net(data.cuda(non_blocking=True))
            feature = torch.flatten(feature, start_dim=1)
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)

        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        if feature_only:
            net.train()
            return feature_bank

        feature_labels = torch.tensor(
            memory_data_loader.dataset.targets, device=feature_bank.device
        )
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc="kNN", disable=hide_progress)

        for data in test_bar:
            if len(data) == 2:
                data, target = data
            else:
                data, data2, target = data

            # with autocast():
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            if type(feature) is tuple:  # mae model
                feature = feature[0]
            feature = torch.flatten(feature, start_dim=1)
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(
                feature, feature_bank, feature_labels, classes, k, t
            )
            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({"Accuracy": total_top1 / total_num * 100})
    net.train()
    return total_top1 / total_num * 100, feature_bank


# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


get_transform_mae = lambda s: transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

get_transform = lambda s: transforms.Compose(
    [
        transforms.RandomResizedCrop(s),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        # transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p = 0.2), # added gaussian blur
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ]
)

get_transform_imagenet = lambda s: transforms.Compose(
    [
        transforms.RandomResizedCrop(s),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomApply(
            [transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2
        ),  # added gaussian blur
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

linear_transform_imagenet = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform_imagenet = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
