'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
import torch.distributed as dist
from myutils import progress_bar
import numpy as np
import pandas as pd
class Average(object):

    def all_reduce(self):
        value = torch.Tensor([self.sum, self.count]).cuda()/dist.get_world_size()
        dist.all_reduce(value)
        self.sum, self.count = value[0], value[1]
        return self

    def __init__(self):
        self.sum = 0
        self.count = 0

    def __str__(self):
        # self.all_reduce()
        return '{:.6f}'.format(self.average)

    @property
    def average(self):
        return self.sum / self.count

    def update(self, value, number):
        self.sum += value * number
        self.count += number
        return self


class Accuracy(object):

    def __init__(self):
        self.correct = 0.0
        self.count = 0.0

    def all_reduce(self):
        value = torch.Tensor([self.correct, self.count]).cuda()/dist.get_world_size()
        dist.all_reduce(value)
        self.correct, self.count = value[0], value[1]
        return self

    def __str__(self):
        return '{:.2f}%'.format(self.accuracy * 100)

    @property
    def accuracy(self):
        return self.correct / self.count*100

    def update(self, output, target):
        with torch.no_grad():
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()

        self.correct += correct
        self.count += output.size(0)
        return self


class Trainer(object):

    def __init__(self, model, optimizer, train_loader, test_loader, device, val_loader):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.record = [] 

    def adjust_learning_rate(self, epoch, args):
        lr = args.lr * (0.1 ** (epoch // 50))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def fit(self, best_acc, start_epoch, epochs, args):
        record = []
        train_loss, train_acc = self.evaluate(0, args, False)
        test_loss, test_acc = self.evaluate(0, args)
        record.append([float(train_loss.average), float(train_acc.accuracy), float(test_loss.average), float(test_acc.accuracy)])
        for epoch in range(start_epoch+1, start_epoch+epochs + 1):
            train_loss, train_acc = self.train(epoch, args, False)
            train_loss, train_acc = self.evaluate(epoch, args, False)
            test_loss, test_acc = self.evaluate(epoch, args)
            acc = test_acc.accuracy
            if args.rank==0 and epoch == start_epoch+epochs:
                    # print('Saving..')
                    state = {
                        'net': self.model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(state, './checkpoint/'+args.model+".pth")
            if acc > best_acc:
                if args.rank==0:
                    print('Saving..')
                    state = {
                        'net': self.model.state_dict(),
                        'acc': acc,
                        'epoch': epoch,
                    }
                    if not os.path.isdir('checkpoint'):
                        os.mkdir('checkpoint')
                    torch.save(state, './checkpoint/'+args.model+".pth")
                best_acc = acc
            record.append([float(train_loss.average), float(train_acc.accuracy), float(test_loss.average), float(test_acc.accuracy)])
            if args.rank == 0 and epoch%10==0:
                record_csv = pd.DataFrame(record)
                record_csv.to_csv(self.get_filename(args)+"_"+str(epoch)+".csv", index=False, header=None)
        
    def get_filename(self, args):
        alg_name = "minisgd"
        if args.period > 1 and args.vrl:
            alg_name = "vrlsgd"
        elif args.period > 1:
            alg_name = "localsgd"
        filename = "record/{}_{}_{}_local_{}_vrl_{}_eta_{}_b_{}_cluster_data_{}".format(alg_name, args.model, args.dataset, args.period, int(args.vrl) , args.lr, args.batch_size, args.cluster_data)
        return filename

    def warm_up(self, optimizer, lr_grow):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] + lr_grow

    def train(self, epoch, args, fake_acc=False):
        self.model.train()
        # self.model.eval()
        if args.rank==0:
            print('\nEpoch: %d' % epoch)
        train_loss = Average()
        train_acc = Accuracy()
        world_size = args.world_size
        target_set = []
        data_set = []
        update_cnt  = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            loss = F.cross_entropy(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

            train_loss.update(loss.item(), data.size(0)).all_reduce()
            train_acc.update(output, target).all_reduce()
            if args.rank ==0:
                progress_bar(batch_idx, len(self.train_loader), 'Loss: %.5f  | Acc: %.3f%% (%d/%d) | update %d'
                    % (train_loss.average, train_acc.accuracy, train_acc.correct*world_size, train_acc.count*world_size, update_cnt), fake_acc=fake_acc)

        self.model._sync_period()
        return train_loss, train_acc

    def evaluate(self, epoch, args, use_test=True):
        # print('\nEpoch: %d' % epoch)
        self.model.eval()

        test_loss = Average()
        test_acc = Accuracy()
        # world_size = args.world_size
        test_loader = self.test_loader
        if not use_test:
            test_loader = self.val_loader 
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = F.cross_entropy(output, target)
                test_loss.update(loss.item(), data.size(0))#.all_reduce()
                test_acc.update(output, target)#.all_reduce()

                if args.rank ==0:
                    progress_bar(batch_idx, len(test_loader), 'Loss: %.5f  | Acc: %.3f%% (%d/%d)'
                        % (test_loss.average,test_acc.accuracy, test_acc.correct, test_acc.count))

        return test_loss, test_acc
