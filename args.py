# -*- coding:utf-8 -*-
import argparse
import torch
from config import *
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--E', "--num-local-epochs", type=int, default=NUM_LOCAL_EPOCES)
    parser.add_argument('--r', "--num-total-epochs", type=int, default=NUM_TOTAL_EPOCHS)
    parser.add_argument('--K', "--num-clients", type=int, default=NUM_CLIENTS)
    parser.add_argument('--LAE', "--local-adaptation-epochs", type=int, default=LOCAL_ADAPTATION_EPOCHS)
    parser.add_argument('--per_algo_type', type=str, default='MAML_FO', help='choose MAML_FO or MAML_HF')
    parser.add_argument('--dt', "--data-type", type=str, default=DATA_TYPE)
    parser.add_argument('--input_dim', "--input-dim", type=int, default=INPUT_DIM)
    parser.add_argument('--lr', "--learning rate", type=float, default=LEARNING_RATE)
    parser.add_argument('--alpha', type=float, default=ALPHA, help='learning rate for each task')
    parser.add_argument('--beta', type=float, default=BETA, help='learning rate for meta update')
    parser.add_argument('--delta', type=float, default=DELTA, help='learning rate for hessian approximation')
    parser.add_argument('--C', "--sampling-rate", type=float, default=SAMPLING_RATE)
    parser.add_argument('--B', "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument('--optimizer', type=str, default='sgd', help='type of optimizer')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--algorithm', type=str, default=ALGORITHM, help='Per-fl or Fedavg')
    parser.add_argument('--clients', default=clients)
    args = parser.parse_args()
    return args

