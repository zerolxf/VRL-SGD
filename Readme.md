# Code for ICLR 2020 paper: "Variance Reduced Local SGD with Lower Communication Complexity"


## Dependencies and Setup

All code runs on Python 3.6.7 using PyTorch version 1.1.0.

In addition, you will need to install

- torchvision
- torchtext
- numpy
- pandas

## Preprocess Data

### Db Pedia

- Download the data from [link](https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbQ2Vic1kxMmZZQ1k) and extract it to the current directory. Then you can get two files: `train.csv` and `test.csv`.
- Modify the data path in `process_text.py` and execute `process_text.py`.

### Tiny ImageNet

- Download the data from [link](https://tiny-imagenet.herokuapp.com/) and extract it to the current directory. 
- Modify the data path in `process_tiny_magenet.py` and execute `process_tiny_magenet.py`.

## Running Experiments
There are two main scripts:
- `train.sh` for training using S-SGD, Local SGD and VRL-SGD.
- `plot_all.sh` for plotting figure.

### Description of main parameters
- `--lr` learning rate
- `--model` model name, model: `lenet5`, `text_cnn`, `mlp`.
- `--data-set` dataset name, model: `mnist`, `DB_Pedia`, `tiny_imagenet`.
- `--epochs` the number of epochs for running.
- `--gpu-num` the number of GPUs.
- `--batch-size` batch size for each machine.
- `-r` resume the training.
- `local` whether to communicate periodically.
- `--period` the communication period. If `--local` is not set, then it will always be 1.
- `--cluster-data`  each worker only accesses a sub of data.
- `--vrl` whether to execute the VRLSGD algorithm.

### Warm Up
We recommend performing 2 epoch SGD to initialize the weights. If not, the `-r` parameter cannot be used. After the initialization is completed, modify the file name, for example, change the file `lenet5.pth` to `lenet5_init.pth`.

### LeNet on MNIST

#### Non-Identical Case

``` 
# S-SGD
python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6632 --cluster-data
# Local-SGD
python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6633 --cluster-data --local --period 20
# VRL-SGD
python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6634 --cluster-data --local --period 20 --vrl
```

#### Identical Case

``` 
# S-SGD
python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6632
# Local-SGD
python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6633 --local --period 20
# VRL-SGD
python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6634 --local --period 20 --vrl
```

### TextCNN on on DBPedia

#### Non-Identical Case

``` 
# S-SGD
python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r --cluster-data
# Local-SGD
python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r --cluster-data --local --period 50
# VRL-SGD
python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r --cluster-data --local --period 50 --vrl
```

#### Identical Case

``` 
# S-SGD
python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r 
# Local-SGD
python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r  --local --period 50
# VRL-SGD
python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r  --local --period 50 --vrl
```

### Transfer Learning on tiny ImageNet

#### Non-Identical Case

``` 
# S-SGD
python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6632 --batch-size 256 -r  --cluster-data 
# Local-SGD
python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6633 --batch-size 256 -r  --local  --period 20  --cluster-data
# VRL-SGD
python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6634 --batch-size 256 -r  --local  --period 20 --vrl--cluster-data
```

#### Identical Case

``` 
# S-SGD
python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6632 --batch-size 256 -r 
# Local-SGD
python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6633 --batch-size 256 -r  --local  --period 20  
# VRL-SGD
python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6634 --batch-size 256 -r  --local  --period 20 --vrl
```
