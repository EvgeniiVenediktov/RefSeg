# Multi-Expressions for Transformer-based Referring Image Segmentation

This repo is the implementation of "Multi-Expressions for Transformer-based Referring Image Segmentation" and is organized as follows: 

* `./train.py` is implemented to train the model.
* `./test.py` is implemented to evaluate the model.
* `./refer` contains data pre-processing manual and code.
* `./data/dataset_.py` is where the dataset class is defined.
* `./lib` contains codes implementing vision encoder and segmentation decoder.
* `./bert` contains codes migrated from Hugging Face, which implement the BERT model.
* `./utils.py` defines functions that track training statistics and setup functions for `Distributed DataParallel`.

## Installation and Setup
### **Environment**

This repo requires Pytorch v 1.13.0 and Python 3.9.
Install Pytorch v 1.13.0 with a CUDA version that works on your cluster. We used CUDA 11.7 in this repo:
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Then, install the packages in `requirements.txt` via pip:
```
pip3 install -r requirements.txt
```

### **Datasets**

Follow `README.md` in the `./refer` directory to set up subdirectories and download annotations.
Download 2014 Train images [83K/13GB] from [COCO](https://cocodataset.org/#download), and extract the downloaded `train_2014.zip` file to `./refer/data/images/mscoco/images`. 

## Training

We use `DistributedDataParallel` from PyTorch. Our MERES were trained using 4 x 24G RTX4090 cards.
To run on multi GPUs (4 GPUs is used in this example) on a single node:
```
mkdir ./models

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --dataset refcoco --swin_type base --lr 0.00003 --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --dataset refcoco+ --swin_type base --lr 0.00003 --epochs 40 --img_size 480 2>&1 | tee ./models/refcoco+

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 train.py --dataset refcocog --splitBy umd --swin_type base --lr 0.00003 --epochs 40 --img_size 480 2>&1 | tee ./models/refcocog
```
To store the training logs, we need to manually create the `./models` directory via `mkdir` before running `train.py`.
* *--dataset* is the dataset name. One can choose from `refcoco`, `refcoco+`, and `refcocog`.
* *--splitBy* needs to be specified if and only if the dataset is G-Ref (which is also called RefCOCOg).
* *--swin_type* specifies the version of the Swin Transformer. One can choose from `tiny`, `small`, `base`, and `large`. The default is `base`.

## Testing

To evaluate, run one of:
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --swin_type base --dataset refcoco --split val --resume ./checkpoints/best_refcoco.pth --workers 1 --img_size 480

CUDA_VISIBLE_DEVICES=0 python3 test.py --swin_type base --dataset refcoco+ --split val --resume ./checkpoints/best_refcoco+.pth --workers 1 --img_size 480

CUDA_VISIBLE_DEVICES=0 python3 test.py --swin_type base --dataset refcocog --splitBy umd --split val --resume ./checkpoints/best_refcocog.pth --workers 1 --img_size 480
```
* *--split* is the subset to evaluate. One can choose from `val`, `testA`, and `testB` for RefCOCO/RefCOCO+, and `val` and `test` for G-Ref (RefCOCOg).
* *--resume* is the path to the weights of a trained model.
