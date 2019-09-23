
# LeNet
## non-identical case
#  python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6632 --cluster-data
#  python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6633 --cluster-data --local --period 20
#  python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6634 --cluster-data --local --period 20 --vrl
 
## identical case
#  python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6632
#  python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6633 --local --period 20
#  python main.py --lr 0.005 --model lenet5 --dataset mnist --epochs 100  --st 0 -s 1 --gpu-num 8 -r --port 6634 --local --period 20 --vrl

# Text CNN
## non-identical case
# python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r --cluster-data

# python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r --cluster-data --local --period 50

# python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r --cluster-data --local --period 50 --vrl


## identical case
# python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r 

# python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r  --local --period 50

# python main.py --lr 0.01 --model text_cnn --dataset DB_Pedia --epochs 100 --st 0 -s 1 --gpu-num 8 --port 6632 --batch-size 512 -r  --local --period 50 --vrl




# Transfer Learnging
## non-identical case
# python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6632 --batch-size 256 -r  --cluster-data 

# python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6633 --batch-size 256 -r  --local  --period 20  --cluster-data

# python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6634 --batch-size 256 -r  --local  --period 20 --vrl --cluster-data

## identical case
# python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6632 --batch-size 256 -r  

# python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6633 --batch-size 256 -r  --local  --period 20  

# python main.py --lr 0.025 --model mlp --dataset tiny_imagenet --epochs 300 -s 1 --gpu-num 8 --port 6634 --batch-size 256 -r  --local  --period 20 --vrl 