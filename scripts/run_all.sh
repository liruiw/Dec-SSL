#!/bin/bash
LLsub scripts/comm_script/train_test_cifar_dir_sl_comm.sh -s 12 -g volta:1 -o logs/batch;
LLsub scripts/comm_script/train_test_cifar_dir_slrep_comm.sh -s 12 -g volta:1 -o logs/batch;

LLsub scripts/comm_script/train_test_cifar_dir_ssl_comm.sh -s 12 -g volta:1 -o logs/batch;
LLsub scripts/comm_script/train_test_cifar_skew_sl_comm.sh -s 12 -g volta:1 -o logs/batch;
LLsub scripts/comm_script/train_test_cifar_skew_slrep_comm.sh -s 12 -g volta:1 -o logs/batch;

LLsub scripts/comm_script/train_test_cifar_skew_ssl_comm.sh -s 12 -g volta:1 -o logs/batch;

LLsub scripts/partial_script/train_test_cifar_ssl_skew_partial.sh -s 12 -g volta:1 -o logs/batch;
LLsub scripts/partial_script/train_test_cifar_slrep_skew_partial.sh -s 12 -g volta:1 -o logs/batch;

LLsub scripts/partial_script/train_test_cifar_sl_skew_partial.sh -s 12 -g volta:1 -o logs/batch;
LLsub scripts/noniid_script/train_test_cifar_slrep_dir.sh -s 12 -g volta:1 -o logs/batch;

LLsub scripts/noniid_script/train_test_cifar_sl_dir.sh -s 12 -g volta:1 -o logs/batch;
LLsub scripts/noniid_script/train_test_cifar_ssl_dir.sh -s 12 -g volta:1 -o logs/batch;
LLsub scripts/noniid_script/train_test_cifar_ssl_alg_dir_reprod.sh -s 12 -g volta:1 -o logs/batch;
LLsub scripts/noniid_script/train_test_cifar_ssl_skewpartition_ray.sh -s 12 -g volta:1 -o logs/batch;

LLsub scripts/noniid_script/train_test_cifar_ssl_skewpartition.sh -s 12 -g volta:1 -o logs/batch;
LLsub scripts/noniid_script/train_test_cifar_slrep_skewpartition.sh -s 12 -g volta:1 -o logs/batch;

LLsub scripts/noniid_script/train_test_cifar_sl_skewpartition.sh -s 12 -g volta:1 -o logs/batch;
LLsub scripts/noniid_script/train_test_cifar_ssl_dir_simsiam.sh -s 12 -g volta:1 -o logs/batch;

sbatch scripts/imagenet_script/train_test_dec_slrep_imagenet100.sh  1 2 21132 d-12-10-1 s 5 1 1 0 200;
sbatch scripts/imagenet_script/train_test_dec_sl_imagenet100.sh  1 2 21132 d-12-10-1 s 5 1 1 0 200;
sbatch scripts/imagenet_script/train_test_dec_ssl_imagenet100_simclr.sh  1 2 21132 d-12-10-1 s 5 1 1 0 200;


sbatch scripts/imagenet_script/train_test_dec_slrep_imagenet100.sh  1 2 21132 d-12-10-1 s 5 1 1 1 200;
sbatch scripts/imagenet_script/train_test_dec_sl_imagenet100.sh  1 2 21132 d-12-10-1 s 5 1 1 1 200;
sbatch scripts/imagenet_script/train_test_dec_ssl_imagenet100_simclr.sh  1 2 21132 d-12-10-1 s 5 1 1 1 200;

# LLsub scripts/noniid_script/train_test_cifar_sl_skewpartition.sh -s 12 -g volta:1 -o logs/batch;
# LLsub scripts/noniid_script/train_test_cifar_ssl_dir_simsiam.sh -s 12 -g volta:1 -o logs/batch;
