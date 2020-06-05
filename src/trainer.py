# coding: utf-8

import os
import sys
import torch
import datetime
import subprocess as sb
import torch.nn.functional as F
from progress import ProgressMeter, Average, Accuracy


class Trainer(object):
    """
    Ported from
    https://github.com/narumiruna/pytorch-distributed-example/blob/master/mnist/main.py
    """
    def __init__(self, config, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        # self.dataloaders = dataloaders
        self.device = device
        self.config = config

    def _record_cmd(self, cmd):
        if not os.path.isdir(self.config.out_path + 'CMDs'):
            os.mkdir(self.config.out_path + 'CMDs')
        else:
            with open('{}/CMDs/{}.cmd'.format(self.config.out_path, cmd), 'a') as f:
                f.write('-----------Mode: {}-------------\n'.format(cmd))
                f.write("Git Commit: {}\n".format(sb.check_output(
                    ["git", "rev-parse", "--short", "HEAD"]
                ).decode('ascii').strip()))
                f.write("Date: {}\n".format(datetime.datetime.now()))
                f.write("Command: {}\n".format(' '.join(sys.argv)))
                f.write('--------------------------------\n')

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.evaluate()

    def train(self, epoch):
        train_loss = Average()
        train_acc = Accuracy()
        # progress = ProgressMeter(len(self.config.data['train']),
        #                          [train_loss, train_acc],
        #                          prefix="Epoch: [{}]".format(epoch))
        self._record_cmd(cmd='train')
        self.model.train()
        for data, target in self.config.data['train']:
            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)
            loss = F.cross_entropy(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, target)

        # if epoch % 10 == 0:
        #     progress.display(epoch)
        return train_loss, train_acc

    def evaluate(self):
        test_loss = Average()
        test_acc = Accuracy()

        self._record_cmd(cmd='test')
        self.model.eval()
        with torch.no_grad():
            for data, target in self.config.data['valid']:

                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = F.cross_entropy(output, target)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, target)

        return test_loss, test_acc
