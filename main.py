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
import torch.multiprocessing as mp
import os
import argparse
from myutils import *
from trainer import *
from torch import distributed, nn
from restarted import DistributedDataParallel
from DistributedSgd import DSGD
from UnshuffleSampler import UnshuffleDistributedSampler

def work_process(gpu, ngpus_per_node, args):
    args.rank = args.rank * ngpus_per_node + gpu 
    torch.cuda.set_device(args.rank+args.st) 
    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
    )
    device = torch.device("cuda:"+str(args.rank))
    model = get_model(args)
    model = DistributedDataParallel(model, device_ids=[args.rank+args.st], local=args.local, update_period=args.period)
    cudnn.benchmark = True
    best_acc = 0  # best test accuracy
    start_epoch = 0 
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+args.model+"_init.pth")
        model.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    optimizer = DSGD(model.parameters(), local=args.local, update_period=args.period, model=model, lr=args.lr, momentum=0, weight_decay=1e-4, vrl=args.vrl)

    train_dataset, test_dataset = get_dataset(args.dataset, args)
    val_dataset, test_dataset = get_dataset(args.dataset, args)

    train_sampler = UnshuffleDistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, cluster_data=args.cluster_data)
    val_loader = None
    val_loader = data.DataLoader(val_dataset, args.batch_size*args.world_size, shuffle=True)
    train_loader = data.DataLoader(train_dataset, args.batch_size, shuffle= (train_sampler is None),sampler=train_sampler)
    test_loader = data.DataLoader(test_dataset, args.batch_size*args.world_size, shuffle=True, num_workers=2)

    trainer = Trainer(model, optimizer, train_loader, test_loader, device, val_loader)
    trainer.fit(best_acc, start_epoch, args.epochs, args)



def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--model', type=str, default='vgg16', help='Name of the model to use.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Name of the dataset to use.')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--root', type=str, default='./data')
    
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('--period', default=10, type=int, help='update period')
    parser.add_argument('--st', default=0, type=int, help='gpu st')
    parser.add_argument('--port', default=23456, type=int, help='port')
    parser.add_argument('-s', '--world-size', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--gpu-num', default=2, type=int, help='gpu num')
    parser.add_argument('-v', '--vrl', dest='vrl', action='store_true')
    parser.set_defaults(vrl=False)
    parser.add_argument('--local', dest='local', action='store_true')
    parser.set_defaults(local=False)
    parser.add_argument('--cluster-data', dest='cluster_data', action='store_true')
    parser.set_defaults(cluster_data=False)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--backend', type=str, default='nccl', help='Name of the backend to use.')
    args = parser.parse_args()
    if not args.local:
        args.period = 1
    print(args)
    args.init_method = 'tcp://127.0.0.1:'+str(args.port)
    ngpus_per_node = torch.cuda.device_count()
    ngpus_per_node = args.gpu_num
    args.world_size = ngpus_per_node * args.world_size
    args.batch_size = int(args.batch_size//args.world_size)
    mp.spawn(work_process, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

if __name__ == "__main__":
    main()