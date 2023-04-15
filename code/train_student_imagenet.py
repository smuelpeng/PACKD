import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from distiller_zoo import DistillKL
from dataset.imagenet_dataset import ImageNetPACKDDataset, ImageNetDataset

import os
import shutil
import argparse
import numpy as np

import models
import torchvision
import torchvision.transforms as transforms
from utils import cal_param_size, cal_multi_adds

from helper.util import AverageMeter, accuracy

from bisect import bisect_right
import time
import math

# PACKD Loss
from packd.packd import PACKDConLoss



parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--root_dir', default='./data/train/', type=str, help='Dataset directory')
parser.add_argument('--dataset', default='imagenet', type=str, help='Dataset name')
parser.add_argument('--arch', default='resnet18_imagenet', type=str, help='student network architecture')
parser.add_argument('--tarch', default='resnet34_imagenet', type=str, help='teacher network architecture')
parser.add_argument('--tcheckpoint', default='resnet34_imagenet.pth.tar', type=str, help='pre-trained weights of teacher')
parser.add_argument('--init-lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--milestones', default=[30,60,90], type=list, help='milestones for lr-multistep')
parser.add_argument('--warmup-epoch', default=0, type=int, help='warmup epoch')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=256, help='batch size')
parser.add_argument('--num-workers', type=int, default=4, help='the number of workers')
parser.add_argument('--gpu-id', type=str, default='0')
parser.add_argument('--manual_seed', type=int, default=0)
parser.add_argument('--kd_T', type=float, default=3, help='temperature for KD distillation')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--evaluate', '-e', action='store_true', help='evaluate model')
parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='directory fot storing checkpoints')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')                    

parser.add_argument('--n_data', default=50000, type=int, help='memory size')
parser.add_argument('--mixup_num', default=1, type=int, help='mixpos number')
parser.add_argument('--distill', type=str, default='PACKDCONLoss')
parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
parser.add_argument('--ops_eps', type=float, default=0.1)
parser.add_argument('-a', '--alpha', type=float,
                    default=1.0, help='weight balance for KD')
parser.add_argument('-b', '--beta', type=float,
                    default=0.8, help='weight balance for other losses')
parser.add_argument('--ops_err_thres', type=float, default=0.1)

def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    args.log_txt = 'log_PACKD_imagenet/'+ str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'arch' + '_' +  args.arch + '_'+\
            'dataset' + '_' +  args.dataset + '_'+\
            'seed'+ str(args.manual_seed) +'.txt'


    args.log_dir = str(os.path.basename(__file__).split('.')[0]) + '_'+\
            'arch'+ '_' + args.arch + '_'+\
            'dataset' + '_' +  args.dataset + '_'+\
            'seed'+ str(args.manual_seed)


    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    torch.set_printoptions(precision=4)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.log_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    with open(args.log_txt, 'a+') as f:
        f.write("==========\nArgs:{}\n==========".format(args) + '\n')

    print('==> Building model..')
    num_classes = 1000

    net = getattr(models, args.tarch)(num_classes=num_classes)
    net.eval()
    print('Teacher Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.tarch, cal_param_size(net)/1e6, cal_multi_adds(net, (1, 3, 224, 224))/1e9))
    del(net)

    net = getattr(models, args.arch)(num_classes=num_classes)
    net.eval()
    print('Student Arch: %s, Params: %.2fM, Multi-adds: %.2fG'
        % (args.arch, cal_param_size(net)/1e6, cal_multi_adds(net, (1, 3, 224, 224))/1e9))
    del(net)

    args.distributed = args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print('ngpus_per_node', ngpus_per_node)
    args.world_size = ngpus_per_node * args.world_size
    print('multiprocessing_distributed')
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    args.gpu = gpu
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                        world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)
    num_classes = 1000
    model = getattr(models, args.arch)
    net = model(num_classes=num_classes).cuda(args.gpu)
    data = torch.randn(2,3,224,224).cuda(args.gpu)

    feat_s, _ = net(data, is_feat=True)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
    net.eval()

    tmodel = getattr(models, args.tarch)
    tnet = tmodel(num_classes=num_classes)
    print('load pre-trained teacher weights from: {}'.format(args.tcheckpoint))     
    checkpoint = torch.load(args.tcheckpoint, map_location='cpu')    
    tnet.load_state_dict(checkpoint)
    feat_t, _ = net(data, is_feat=True)
    tnet.cuda(args.gpu)
    tnet = torch.nn.parallel.DistributedDataParallel(tnet, device_ids=[args.gpu])
    tnet.eval()

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    args.s_dim = feat_s[-1].shape[1]
    args.t_dim = feat_t[-1].shape[1]
    assert args.s_dim == args.t_dim 
    args.feat_dim = args.s_dim

    cudnn.benchmark = True

    trainable_list = nn.ModuleList([])
    trainable_list.append(net)

    if args.distill in ['PACKDCONLoss']:
        criterion_kd = PACKDConLoss(args)
        # module_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    else:
        raise NotImplementedError(args.distill)

    criterion_cls = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_div = DistillKL(args.kd_T).cuda(args.gpu)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)   # kd Loss
    criterion_list.cuda()

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=args.weight_decay)

    if args.resume:
        print('load intermediate weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar')))
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    # Build Train dataset
    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        # transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                    ])
    
    train_set = ImageNetPACKDDataset(
        args.root_dir + '/train/',
        args.root_dir + '/meta/train.txt',
        transform = transform_train,
        mixnum = args.mixup_num
    )

    # Build Test Dataset
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
    test_set = ImageNetDataset(
        args.root_dir + '/val/',
        args.root_dir + '/meta/val.txt',
        transform = transform_test,
    )

    # Build Dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    
    def train(epoch, criterion_list, optimizer):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_cls = AverageMeter()
        losses_div = AverageMeter()
        losses_kd = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        if epoch >= args.warmup_epoch:
            lr = adjust_lr(optimizer, epoch, args)

        start_time = time.time()

        criterion_cls = criterion_list[0]
        criterion_div = criterion_list[1]
        criterion_kd  = criterion_list[2]

        net.train()
        end = time.time()
        time1 = time.time()
        for batch_idx, batch in enumerate(trainloader):
            data_time.update(time.time() - end)
            input = batch['image']    
            target = batch['label']
            image_idx = batch['image_id']

            batch_start_time = time.time()
            input = input.float().cuda()
            target = target.long().cuda()
            size = input.shape[1:]

            if epoch < args.warmup_epoch:
                lr = adjust_lr(optimizer, epoch, args, batch_idx, len(trainloader))

            optimizer.zero_grad()
            batch_size,c,h,w = input.shape
            mix_num = c // 3
            input = input.reshape(batch_size*mix_num, 3, h, w)
            target = target.reshape(batch_size*mix_num)

            feat_s, logits_s = net(input, is_feat=True)

            with torch.no_grad():
                feat_t, logits_t = tnet(input, is_feat=True)
            loss_cls = criterion_cls(logits_s, target)  
            loss_div = criterion_div(logits_s, logits_t.detach())
            loss_kd = criterion_kd(feat_s[-1], feat_t[-1],
                                    labels=[target, image_idx],
                                )
            loss = loss_cls + args.alpha * loss_div + args.beta * loss_kd
            loss.backward()
            optimizer.step()
          
            acc1, acc5 = accuracy(logits_s, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            losses_cls.update(loss_cls.item(), input.size(0))
            losses_div.update(loss_div.item(), input.size(0))
            losses_kd.update(loss_kd.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx%100==0 and args.rank == 0:
                print('Epoch:{}\t batch_idx:{}/{} \t lr:{:.5f}\t duration:{:.3f} '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    ' loss:{:.5f}\t loss_cls:{:.5f}\t loss_div:{:.5f}'
                    ' loss_kd:{:.5f} Train Top-1 : {:.5f} \t Train Top-5 : {:.5f}'
                    .format(epoch, batch_idx, 
                            len(trainloader), lr, 
                            time.time() - start_time,
                            losses.avg,
                            losses_cls.avg,
                            losses_div.avg, losses_kd.avg,
                            top1.avg, top5.avg,
                            batch_time=batch_time,
                            data_time=data_time,), flush=True)

        with open(args.log_txt, 'a+') as f:
            f.write('Epoch:{}\t lr:{:.5f}\t duration:{:.3f}'
                    '\n train_loss:{:.5f}\t train_loss_cls:{:.5f}\t train_loss_div:{:.5f}'
                    '\n train_loss_kd:{:.5f} '
                    'Train Top-1 class_accuracy: {}\nTrain Top-5 class_accuracy: {}\n'
                    .format(epoch, lr, time.time() - start_time,
                            losses.avg, losses_cls.avg, losses_div.avg,
                            losses_kd.avg,
                            top1.avg, top5.avg))


    def test(epoch, criterion_cls, net):
        test_loss_cls = 0.
        top1_num = 0
        top5_num = 0
        total = 0
        
        net.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(testloader):
                input = batch['image']    
                target = batch['label'] 
                batch_start_time = time.time()
                input, target = input.cuda(), target.cuda()
                logits = net(input)
                loss_cls = criterion_cls(logits, target)
                test_loss_cls += loss_cls.item()/ len(testloader)
                batch_size = logits.size(0) 
                top1, top5 = correct_num(logits, target, topk=(1, 5))
                top1_num += top1
                top5_num += top5
                total +=  target.size(0)                
                if batch_idx%100==0 and args.rank == 0:
                    print('Epoch:{}, batch_idx:{}/{}, Duration:{:.2f}, Top-1 Acc:{:.4f}'.format(
                        epoch, batch_idx, len(testloader), time.time()-batch_start_time, (top1_num/(total)).item()), flush=True)
            class_acc1 = [round((top1_num/(total)).item(), 4)] 
            class_acc5 = [round((top5_num/(total)).item(), 4)]
            with open(args.log_txt, 'a+') as f:
                f.write('test epoch:{}\t test_loss_cls:{:.5f}\nTop-1 class_accuracy: {}\nTop-5 class_accuracy: {}\n'
                        .format(epoch, test_loss_cls, str(class_acc1), str(class_acc5)))
            if args.rank == 0:                        
                print('test epoch:{} \nTest Top-1 class_accuracy: {}\nTop-5 class_accuracy: {}\n'.format(
                    epoch, str(class_acc1), str(class_acc5)))
        return class_acc1[-1]

    if args.evaluate: 
        print('load pre-trained weights from: {}'.format(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))     
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        test(start_epoch, criterion_cls, net) 
    else:
        #print('Evaluate Teacher:')
        #acc = test(0, criterion_cls, tnet)
        #print('teacher accuracy:{}'.format(acc))
        #with open(args.log_txt, 'a+') as f:
        #    f.write('teacher accuracy:{}'.format(acc))
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            train(epoch, criterion_list, optimizer) 
            acc = test(epoch, criterion_cls, net)
            if args.rank == 0:
                state = {
                    'net': net.module.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'))
                is_best = False
                if best_acc < acc:
                    best_acc = acc
                    is_best = True
                if is_best:
                    shutil.copyfile(os.path.join(args.checkpoint_dir, str(model.__name__) + '.pth.tar'),
                                    os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'))

        print('Evaluate the best model:')
        print('load pre-trained weights from: {}'.format(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar')))
        args.evaluate = True
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, str(model.__name__) + '_best.pth.tar'),
                                map_location='cuda:{}'.format(args.gpu))
        net.module.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        top1_acc = test(start_epoch, criterion_cls, net)
        with open(args.log_txt, 'a+') as f:
            f.write('best_accuracy: {} \n'.format(best_acc))
        print('best_accuracy: {} \n'.format(best_acc))
        os.system('cp ' + args.log_txt + ' ' + args.checkpoint_dir)


def correct_num(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].float().sum()
        res.append(correct_k)
    return res


def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0, eta_min=0.):
    cur_lr = 0.
    if epoch < args.warmup_epoch:
        cur_lr = args.init_lr * float(1 + step + epoch*all_iters_per_epoch)/(args.warmup_epoch *all_iters_per_epoch)
    else:
        epoch = epoch - args.warmup_epoch
        cur_lr = args.init_lr * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr

if __name__ == '__main__':
    main()
