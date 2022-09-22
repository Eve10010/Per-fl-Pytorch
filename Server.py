# -*- coding:utf-8 -*-
import torch
import numpy as np
import random
from client import train, test, local_adaptation
from cnn import Model
import copy
from tqdm import tqdm

class Server_:
    def __init__(self, args):
        self.args = args
        self.nn = Model().to(args.device)
        self.nns = []
        # init
        # build K models for clients
        for i in range(self.args.K):
            temp = copy.deepcopy(self.nn)
            temp.name = self.args.clients[i]
            self.nns.append(temp)

    def server(self):
        for t in tqdm(range(self.args.r), desc='round'):
            # print('round', t + 1, ':')
            m = np.max([int(self.args.C * self.args.K), 1])
            index = random.sample(range(0, self.args.K), m)
            # dispatch parameters
            self.dispatch(index)
            # local updating
            self.client_update(index, t)
            # aggregation parameters
            self.aggregation(index)

        return self.nn

    def aggregation(self, index):
        s = 0
        for j in index:
            # normal
            s += self.nns[j].len

        params = {}
        for k, v in self.nns[0].named_parameters():
            params[k] = torch.zeros_like(v.data)

        for j in index:
            for k, v in self.nns[j].named_parameters():
                params[k] += v.data / len(index)
                # params[k] += v.data * (self.nns[j].len / s)

        for k, v in self.nn.named_parameters():
            v.data = params[k].data.clone()

    def dispatch(self, index):
        for j in index:
            for old_params, new_params in zip(self.nns[j].parameters(), self.nn.parameters()):
                old_params.data = new_params.data.clone()

    def client_update(self, index, t):  # update nn
        if self.args.algorithm == "Per-fl":
            for k in index:
                self.nns[k] = train(self.args, self.nns[k], k, t)
        elif self.args.algorithm == "Fedavg":
            for i in tqdm(index, desc='round' + str(round) + ' client' + ' local updating'):
                # model = copy.deepcopy(self.nn)
                self.nns[i].name = self.args.clients[i]
                self.nns[i] = local_adaptation(self.args, self.nns[i].name, self.nns[i])

    def cluster(self, index):
        

    def global_test(self):
        if self.args.algorithm == "Per-fl":
            for j in tqdm(range(self.args.K), 'Per-fl global test'):
                model = copy.deepcopy(self.nn)
                model.name = self.args.clients[j]
                model = local_adaptation(self.args, model.name, model)
                test(self.args, model.name, model)
        elif self.args.algorithm == "Fedavg":
            for j in tqdm(range(self.args.K), 'Fedavg global test'):
                model = copy.deepcopy(self.nn)
                model.name = self.args.clients[j]
                # model = local_adaptation(self.args, model.name, model)
                test(self.args, model.name, model)
