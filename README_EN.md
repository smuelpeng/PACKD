# Pay Attention to Your Positive Pairs: Positive Pair Aware Contrastive Knowledge Distillation
This is an official implementation with PyTorch, and we run our code on Ubuntu 18.04.1 server. More experimental details can be found in our paper.

## Dependencies
All the experiments are carried out on a ubuntu 18.04.1 server equipped with Intel(R) Xeon(R) Gold 6148 CPU and a V100 GPU. In addition, our experiments require
- python 3.6+
- pytorch 1.5+

## Datasets

In our paper, we evaluate our proposed method on three benchmark datasets, including:
- CIFAR-100: included in code/data/cifar100
- STL-10: included in code/data/stl10_binary
- Tiny-ImageNet-200: -included in code/data/TinyImageNet200
- ImageNet1k: [Download here](https://image-net.org/download.php ) and save at "code/data". 

## How to Run
- First of all, you need to install the dependencies by 
    ```
    pip3 install -r requirements.txt
    ```

#### CIFAR-100
```shell
sh scripts/run_distill.sh
```
#### Imagenet
```shell
sh scripts/train_student_imagenet.sh
```
#### STL-10 and Tiny-ImageNet-200
```shell
sh eval_rep_TinyImageNet200.sh
sh eval_rep_STL10.sh
```
#### Few-shot 
```shell
sh scripts/run_few_shot.sh
```

## Experiments
The results show the effectiveness of our proposed knowledge distillation method.