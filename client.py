# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch import nn
import copy
from tqdm import tqdm
from data_process_cifar10_ import nn_seq

def train(args, model, ind, round):
    """
    Client training.
    :param args: hyperparameters
    :param model: server model
    :param ind: client id
    :param round: round
    :return: client model after training
    """
    model.train()
    Dtr, Dte = nn_seq(args.clients[ind], args.B)
    model.len = len(Dtr)
    print('training...')
    data = [x for x in iter(Dtr)]
    # "data" is a batch of data to train.
    for epoch in tqdm(range(args.E), desc='round' + str(round) + ' client' + str(ind) + ' local updating'):
        # origin_model = copy.deepcopy(model)
        final_model = copy.deepcopy(model)
        # step1
        model = one_step(args, data, model, lr=args.alpha)
        # step2
        model = get_grad(args, data, model)
        # step3
        # hessian_params = get_hessian(args, data, origin_model)
        hess_free = get_hessian_free(args, data, model, final_model)
        # step 4
        cnt = 0

        if args.per_algo_type == "MAML-FO":
            for param, param_grad in zip(final_model.parameters(), model.parameters()):
                # hess = hessian_params[cnt]
                cnt += 1
                # I = torch.ones_like(param.data)
                # grad = (I - args.alpha * hess) * param_grad.grad.data
                param.data = param.data - args.beta * param_grad.grad.data
        if args.per_algo_type == "MAML-HF":
            for param, param_grad in zip(final_model.parameters(), model.parameters()):
                cnt += 1
                # grad = (I - args.alpha * hess) * param_grad.grad.data
                param.data = param.data - args.beta * (param_grad.grad.data - args.alpha * hess_free)

        model = copy.deepcopy(final_model)

    return model

def one_step(args, data, model, lr):
    """
    :param args: hyperparameters
    :param data: a batch of data
    :param model: original client model
    :param lr: learning rate
    :return: model after one step gradient descent
    """
    ind = np.random.randint(0, high=len(data), size=None, dtype=int)
    seq, label = data[ind]
    seq = seq.to(args.device)
    label = label.to(args.device)

    y_pred = model(seq)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    loss = loss_function(y_pred, label.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model


def get_grad(args, data, model):
    """
    :param args: hyperparameters
    :param data: a batch of data
    :param model: model after one step gradient descent
    :return: gradient
    """
    ind = np.random.randint(0, high=len(data), size=None, dtype=int)
    seq, label = data[ind]
    seq = seq.to(args.device)
    label = label.to(args.device)
    y_pred = model(seq)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    loss = loss_function(y_pred, label.long())
    loss.backward()

    return model

def get_hessian_free(args, data, grad, w_k):
    """
    :param args: hyperparameters
    :param data: a batch of data
    :param model: gradient of model after one step gradient descent
    :param w_k: original client model in this round
    :return hess_free: an approximation for the term of hessian*grad
    """
    # w1 = w_k + args.delta * grad
    # w2 = w_k - args.delta * grad
    # grad_param.data = w_param.data - args.delta * grad_param.grad.data
    # grad1 = get_grad(args, data, w1)
    # grad2 = get_grad(args, data, w2)
    # hess_free = 0.5 / args.delta * (grad1 - grad2)

    w_k_ = copy.deepcopy(w_k)
    for w_param, grad_param in zip(w_k.parameters(), grad.parameters()):
        w_param.data = w_param.data + args.delta * grad_param.grad.data
    grad1 = get_grad(args, data, w_k)

    for w_param_, grad_param in zip(w_k_.parameters(), grad.parameters()):
        w_param_.data = w_param_.data - args.delta * grad_param.grad.data
    grad2 = get_grad(args, data, w_k_)

    for grad1_grad, grad2_grad in zip(grad1.parameters(), grad2.parameters()):
        grad1_grad.grad.data = 0.5 / args.delta * (grad1_grad.grad.data - grad2_grad.grad.data)

    # hess_free = copy.deepcopy(grad1)
    return grad1

def local_adaptation(args, clients_id, model):
    """
    Adaptive training.
    :param args:hyperparameters
    :param model: federated global model
    :return:final model after adaptive training
    """
    model.train()
    total = 0
    Dtr, Dte = nn_seq(clients_id, 50)
    model.len = len(Dtr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    # loss = 0
    # one step
    correct = 0
    if args.algorithm == "Fedavg":
        num_epoch = args.E
    elif args.algorithm == "Per-fl":
        num_epoch = args.LAE
    for epoch in range(num_epoch):
        for seq, label in Dtr:
            seq, label = seq.to(args.device), label.to(args.device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += label.size(0)
            _, predicted = torch.max(y_pred.data, 1)
            correct += (predicted.data.cpu().numpy() == label.data.cpu().numpy()).sum()

        '''local = 'local.txt'
        with open(local, "a") as file1:
            file1.write("acc" + str(100 * correct / total) + "%" + "\n")

        loss_txt = "loss.txt"
        with open(loss_txt, "a") as file2:
           file2.write("loss" + str(loss.item()) + "%" + "\n")'''

    return model

def test(args, clients_id, Model):
    Model.eval()
    Dtr, Dte = nn_seq(clients_id, args.B)
    print("client_id", clients_id)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    loss_ = []
    correct = 0
    total = 0

    for (seq, target) in tqdm(Dte):
        with torch.no_grad():
            seq, target = seq.to(args.device), target.to(args.device)
            y_pred = Model(seq)
            loss = loss_function(y_pred, target.long())
            _, predicted = torch.max(y_pred.data, 1)
            # print('predictetd', predicted)
            # print('target', target)
            total += target.size(0)
            print('total', total)
            '''target = target.cpu()
            target = target.detach().numpy()'''
            correct += (predicted.data.cpu().numpy() == target.data.cpu().numpy()).sum()

            # loss_.append(loss)
            '''for i in range(len(y_pred.data)):
                temp = torch.argmax((y_pred.data[i]))
                cnt += 1
                if temp == target.data[i]:
                    correct += 1'''

    acc_txt = "acc.txt"
    print("acc " + str(100 * correct / total) + "%" + "\n")
    # loss_txt = "loss.txt"
    # Loss = sum(loss_)/cnt
    with open(acc_txt, "a") as file1:
        file1.write("acc " + str(100 * correct / total) + "%" + "\n")

    # with open(loss_txt, "a") as file2:
    #    file2.write("loss" + str(Loss) + " " + "\n")
