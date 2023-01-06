# Pay Attention to Your Positive Pairs: Positive Pair Aware Contrastive Knowledge Distillation

## 环境依赖
所有实验都在一台配备了ubuntu 18.04操作系统、Intel(R) Xeon(R) Gold 6148 处理器和  3090ti显卡的8卡服务器上进行，相关依赖如下（见code/requirements.txt）：
- python 3.6.9
- torch 1.5.0+cuda90_cudnn7.6.3_lms

## 数据集

论文中主要涉及到以下四个数据集，所有数据均已公开，具体如下：
- CIFAR100: [下载地址](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), 解压到code/data/cifar100/cifar-100-python/
- STL10: [下载地址](http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz), 解压到code/data/stl10_binary/ 
- Tiny-ImageNet-200: [下载地址](http://cs231n.stanford.edu/tiny-imagenet-200.zip) , 解压到code/data/TinyImageNet200/, 运行   
    ```
    cd code/data/ 
    python preprocess_tinyimagenet.py
    ```

    最终目录结构为：
    ```
    TinyImageNet200
    ├── test
    ├── train
    ├── val
    ├── val_original
    ├── wnids.txt
    └── words.txt
    ```
- Imagenet1K: [下载地址](https://www.image-net.org/download.php) , 将数据集按如下目录目录结构解压并放置在 `code/data/images`, meta目录已提前处理好：
    ```
    images
    ├── train
        ├── n01440764
        │   ├── n01440764_10026.JPEG
        │   ├── n01440764_10027.JPEG
        │   ├── n01440764_10029.JPEG
        │   ├── n01440764_10040.JPEG
        ······
    ├── val
        ├── ILSVRC2012_val_00000001.JPEG
        ├── ILSVRC2012_val_00000002.JPEG
        ├── ILSVRC2012_val_00000003.JPEG
        ├── ILSVRC2012_val_00000004.JPEG
        ······
    └── metas
        ├── train.txt
        └── val.txt
    ```
## 代码运行

### 安装环境依赖

```
cd code
pip3 install -r requirements.txt --user
```

### 下载预训练模型：
```
wget https://github.com/smuelpeng/PACKD/releases/download/checkpoints/save_t.tar 
tar -xf save_t.tar
wget https://download.pytorch.org/models/resnet34-333f7ec4.pth
mv resnet34-333f7ec4.pth pretrained_models/
```

### 运行

#### CIFAR-100
```shell
bash scripts/run_distill.sh
```
#### ImageNet1k
Imagenet1K用于验证所提出方法在大数据集上的效果,

```shell
bash scripts/train_student_imagenet.sh
```
#### STL-10 and Tiny-ImageNet-200
STL-10 和 Tiny-ImageNet-200 主要是用作验证CIFAR100特征泛化能力，训练完cifar100之后运行
```shell
bash scripts/eval_rep_TinyImageNet200.sh
bash scripts/eval_rep_STL10.sh
```
#### Few-shot 
验证少量数据时所提出算法的有效性，运行命令如下：
```shell
bash scripts/run_few_shot.sh
```

## 实验结果
**基本要求：** 

实验过程中会进行日志保存，包括每个epoch的训练、验证、测试的损失及相应指标。
可以看出实验结果与论文中基本一致。