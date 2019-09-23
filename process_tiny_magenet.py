import os
import torchvision.models as models
import sys
import time
import math
from torch import nn
import torch.nn.init as init
from torchvision import datasets, transforms
import numpy as np
import torch.utils.data as data
from torch.utils.data.dataset import *
from torchvision.datasets.vision import *


DATA_PATH = "train/"
BATCH_SIZE = 64
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
test_data = datasets.ImageFolder(root=DATA_PATH, transform=TRANSFORM_IMG)
data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
model_pre = models.inception_v3(pretrained=True).cuda()
num_ftrs = model_pre.fc.in_features
model_pre.fc = nn.Sequential()
for param in model_pre.parameters():
    param.requires_grad = False
input_size = 299

model_pre.eval()
cnt = 0
res = []
data_x = []
data_y = []
for x,y in data_loader:
    cnt += x.shape[0]
    xx = model_pre(x.cuda())
    data_x.extend(list(xx.cpu()))
    data_y.extend(y.tolist())
    
    
train_data_x = [x.numpy() for x in data_x]
train_data_x = np.array(train_data_x)
train_data_y = np.array(data_y)
print(train_data_x.shape)
np.save("tiny_imagenet_train_x", train_data_x,)
np.save("tiny_imagenet_train_y", train_data_y)