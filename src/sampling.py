#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import IPython
import matplotlib.pyplot as plt
import torch
import time
import os


def mkdir_if_missing(dst_dir):
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_partition(dataset, num_users, shard_per_user=1):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_imgs = len(dataset) // num_users // shard_per_user
    num_shards = num_users
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, shard_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def cifar_partition_skew(dataset, num_users, beta=1, vis=False, labels=None):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    skew_ratio = 1 - beta
    print("partition skew: {} {} ".format(num_users, skew_ratio))
    if labels is None:
        labels = np.array(dataset.targets)
    skewed_data = []
    unskewed_data = []
    data_num_per_user = len(dataset) // num_users
    data_num_per_user_skew = int(data_num_per_user * skew_ratio)
    data_num_per_user_unskew = int(data_num_per_user * (1 - skew_ratio))
    print(data_num_per_user, data_num_per_user_skew, data_num_per_user_unskew)

    K = len(np.unique(labels))
    dict_users = {i: np.array([]) for i in range(num_users)}

    for i in range(K):
        index = np.where(labels == i)[0]
        np.random.shuffle(index)
        split = int(len(index) * skew_ratio)
        skewed_data.append(index[:split])
        unskewed_data.append(index[split:])

    skewed_data = np.concatenate(skewed_data)
    unskewed_data = np.concatenate(unskewed_data)
    np.random.shuffle(unskewed_data)  # uniform
    print(
        "len of skewed: {} len of unskewed: {} data_num_per_user_skew: {}".format(
            len(skewed_data), len(unskewed_data), data_num_per_user_skew
        )
    )

    # divide and assign
    print(data_num_per_user, split, data_num_per_user_skew)
    for i in range(num_users):
        skew_base_idx = i * data_num_per_user_skew
        unskew_base_idx = i * data_num_per_user_unskew
        dict_users[i] = np.concatenate(
            (
                skewed_data[skew_base_idx : skew_base_idx + data_num_per_user_skew],
                unskewed_data[
                    unskew_base_idx : unskew_base_idx + data_num_per_user_unskew
                ],
            ),
            axis=0,
        )

    return dict_users


def cifar_noniid(dataset, num_users, vis=True):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def dirichlet_sampling(labels, num_users, alpha, vis=False, fig_name="cluster"):
    """
    Sort labels and use dirichlet resampling to split the labels
    :param dataset:
    :param num_users:
    :return:
    """
    K = len(np.unique(labels))
    N = labels.shape[0]
    threshold = 0.5
    min_require_size = N / num_users * (1 - threshold)
    max_require_size = N / num_users * (1 + threshold)
    min_size, max_size = 0, 1e6
    iter_idx = 0

    while (
        min_size < min_require_size or max_size > max_require_size
    ) and iter_idx < 1000:
        idx_batch = [[] for _ in range(num_users)]
        plt.clf()
        for k in range(K):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))

            # avoid adding over
            proportions = np.array(
                [
                    p * (len(idx_j) < N / num_users)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]

            min_size = min([len(idx_j) for idx_j in idx_batch])
            max_size = max([len(idx_j) for idx_j in idx_batch])

        iter_idx += 1

    # divide and assign
    dict_users = {i: idx for i, idx in enumerate(idx_batch)}
    return dict_users


def cifar_noniid_x_cluster(
    dataset, num_users, cluster_type="pixels", args=None, vis=False, test=False
):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    dict_users = {i: np.array([]) for i in range(num_users)}
    import scipy.cluster
    import sys
    from models import SimCLR

    data = np.load("save/CIFAR10_tuned_features.npz")
    X = data["features_training"] if not test else data["features_testing"]

    from sklearn import decomposition

    pca = decomposition.PCA(n_components=30, whiten=True)
    # IPython.embed()
    X = pca.fit_transform(X)
    features = np.array(X)

    clusters, dist = scipy.cluster.vq.kmeans(features, num_users)
    center_dists = np.linalg.norm(clusters[:, None] - features[None], axis=-1)
    center_dists_argmin = np.argmin(center_dists, axis=0)
    print("{} clustering distortion: {}".format(cluster_type, dist))

    for i in range(num_users):
        dict_users[i] = np.nonzero(center_dists_argmin == i)[0]
        print("cluster {} size: {}".format(i, len(dict_users[i])))

    labels = dataset.targets
    for i in range(num_users):
        cls_cnt = np.array(labels)[dict_users[i]]
        print("dominant class:", np.bincount(cls_cnt))

    # could do resampling and run dirichlet after this for the degree
    labels = np.zeros_like(labels)
    for i in range(num_users):
        labels[dict_users[i]] = i

    # divide and assign
    dict_users = dirichlet_sampling(
        labels, num_users, args.dir_beta, vis=vis, fig_name=cluster_type
    )
    return dict_users


def cifar_noniid_x(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_imgs : (rand + 1) * num_imgs]), axis=0
            )
    return dict_users


def cifar_noniid_dirichlet(dataset, num_users, beta=0.4, labels=None, vis=False):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if labels is None:
        labels = np.array(dataset.targets)

    dict_users = dirichlet_sampling(
        labels, num_users, beta, vis=vis, fig_name="y_shift"
    )
    return dict_users
