# -*- coding:utf-8 -*-
import numpy as np
import torch
from torch import nn
import copy
from tqdm import tqdm
import random
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

    for epoch in tqdm(range(args.E), desc='round' + str(round) + ' client' + str(ind) + ' local updating'):
        final_model = copy.deepcopy(model)
        # step1
        model = one_step(args, data, model, lr=args.alpha)
        # step2
        model = get_grad(args, data, model)
        # step3
        hessian_model = get_hessian(args, data, final_model)
        # step 4
        for param, param_grad, hess in zip(final_model.parameters(), model.parameters(), hessian_model.parameters()):
            I = torch.ones_like(param.data)
            grad = (I - args.alpha * hess.grad.data) * param_grad.grad.data
            param.data = param.data - args.beta * grad

        '''if args.per_algo_type == "MAML-FO":
            for param, param_grad in zip(final_model.parameters(), model.parameters()):
                # hess = hessian_params[cnt]
                cnt += 1
                # I = torch.ones_like(param.data)
                # grad = (I - args.alpha * hess) * param_grad.grad.data
                param.data = param.data - args.beta * param_grad.grad.data
        '''

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
    params = weight_flatten(model)
    optimizer.zero_grad()
    loss.backward()
    '''for v in model.parameters():
        print("grad:", v.grad.data)
        break'''
    optimizer.step()

    return model


def get_grad(args, data, model):
    """
    :param args: hyperparameters
    :param data: a batch of data
    :param model: model after one step gradient descent
    :return: model
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

def get_hessian(args, data, model):
    """
    :param args: hyperparameters
    :param data: a batch of data
    :param model: original model
    :return: hessian matrix
    """
    ind = np.random.randint(0, high=len(data), size=None, dtype=int)
    seq, label = data[ind]
    seq = seq.to(args.device)
    label = label.to(args.device)
    y_pred = model(seq)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    loss = loss_function(y_pred, label.long())
    grad_params = torch.autograd.grad(loss, model.parameters(),
                                      retain_graph=True, create_graph=True)
    grad_norm = 0
    for grad in grad_params:
        grad_norm += grad.pow(2).sum()
    grad_norm = grad_norm.sqrt()
    grad_norm.backward()

    return model

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
            local = 'local.txt'
            with open(local, "a") as localfile:
                localfile.write(str(clients_id) + " epoch " + str(epoch) + " acc " + str(100 * correct / total) + "%" + "\n")
                localfile.write(str(clients_id) + " epoch " + str(epoch) + " loss " + str(loss.item()) + "\n")

    return model

def test(args, clients_id, Model):
    Model.eval()
    Dtr, Dte = nn_seq(clients_id, args.B)
    print("client_id", clients_id)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    loss_ = []
    correct = 0
    total = 0
    cnt = 0

    for (seq, target) in tqdm(Dte):
        with torch.no_grad():
            seq, target = seq.to(args.device), target.to(args.device)
            y_pred = Model(seq)
            loss = loss_function(y_pred, target.long())
            _, predicted = torch.max(y_pred.data, 1)
            total += target.size(0)
            '''target = target.cpu()
            target = target.detach().numpy()'''
            correct += (predicted.data.cpu().numpy() == target.data.cpu().numpy()).sum()

            loss_.append(loss)
            cnt += 1

    acc_txt = "log.txt"

    Loss = sum(loss_)/cnt
    with open(acc_txt, "a") as file1:
        file1.write(str(clients_id) + " test acc " + str(100 * correct / total) + "%" + "\n")
        file1.write(str(clients_id) + " test loss " + str(Loss) + " " + "\n")


def weight_flatten(model):
    params = []
    for u in model.parameters():
        params.append(u.view(-1))
    params = torch.cat(params)
    return params
