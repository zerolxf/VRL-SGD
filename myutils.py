from models import *
import os
import sys
import time
import math
import torch.nn.init as init
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.dataset import *
from torchvision.datasets.vision import *
from numpy import *
from textCNN import *

class MLP(nn.Module):
    def __init__(self, input_size=2048, out_size=200):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, out_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DB_Pedia(VisionDataset):
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None):
        super(DB_Pedia, self).__init__(root)
        training_data_file = 'db_pedia_train_data.npy'
        training_label_file = 'db_pedia_train_label.npy'
        test_data_file = 'db_pedia_test_data.npy'
        test_label_file = 'db_pedia_test_label.npy'
        self.train = train  # training set or test set

        if self.train:
            self.data = np.load(training_data_file, allow_pickle=True)
            self.targets = np.load(training_label_file, allow_pickle=True)
        else:
            self.data = np.load(test_data_file, allow_pickle=True)
            self.targets = np.load(test_label_file, allow_pickle=True)
        self.data = [x.reshape(50,50) for x in self.data]
        self.targets =self.targets-1
        self.targets = [x for x in self.targets]
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def __len__(self):
        return len(self.data)

class TinyImageNet(VisionDataset):
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None):
        super(TinyImageNet, self).__init__(root)
        training_data_file = 'tiny_imagenet_train_x.npy'
        training_label_file = 'tiny_imagenet_train_y.npy'
        test_data_file = 'tiny_imagenet_train_x.npy'
        test_label_file = 'tiny_imagenet_train_y.npy'
        self.train = train  # training set or test set

        if self.train:
            self.data = np.load(training_data_file, allow_pickle=True)
            self.targets = np.load(training_label_file, allow_pickle=True)
        else:
            self.data = np.load(test_data_file, allow_pickle=True)
            self.targets = np.load(test_label_file, allow_pickle=True)
        # self.data = [x.reshape(50,50) for x in self.data]
        self.targets = [x for x in self.targets]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def __len__(self):
        return len(self.data)

def get_dataset(dataset_name, args):
    dataset_name = dataset_name.lower()
    print("dataset_name",dataset_name)
    if dataset_name == "tiny_imagenet":
        train_dataset = TinyImageNet(args.root, train=True)
        test_dataset = TinyImageNet(args.root, train=False)
    elif dataset_name == "db_pedia":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = DB_Pedia(args.root, train=True, transform=transform)
        test_dataset = DB_Pedia(args.root, train=False, transform=transform)
    elif dataset_name ==  "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST(args.root, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(args.root, train=False, transform=transform, download=True)
    return train_dataset, test_dataset



def get_model(args, num_classes=10):
    model_name = args.model.lower()
    if model_name == "text_cnn":
        return CNN_Text(class_num=14).cuda("cuda:"+str(args.rank))
    elif model_name == "lenet5":
        return LeNet5().cuda("cuda:"+str(args.rank))
    elif model_name == "mlp":
        return MLP().cuda("cuda:"+str(args.rank))
    return None



_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None, fake_acc=False):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        if not fake_acc:
            sys.stdout.write('\n')
        else:
            sys.stdout.write('\r')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
