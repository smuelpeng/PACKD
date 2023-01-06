from __future__ import print_function
import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders_sample
from helper.util import adjust_learning_rate

from packd.packd import PACKDConLoss

from helper.loops import validate as validate
from helper.pretrain import init
from helper.util import AverageMeter, accuracy
from models.wrapper import wrapper
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from distiller_zoo import DistillKL


def parse_option():

    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int,
                        default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int,
                        default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int,
                        default=40, help='save frequency')
    parser.add_argument('--save_dir', type=str,
                        default='./save', help='save dir')
    parser.add_argument('--batch_size', type=int,
                        default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240,
                        help='number of training epochs')
    # optimization
    parser.add_argument('--learning_rate', type=float,
                        default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str,
                        default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float,
                        default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar100', 'cifar100_aug'], help='dataset')
    parser.add_argument('--ncls', type=int, default=100, help='class number')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None,
                        help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('-a', '--alpha', type=float,
                        default=None, help='weight balance for KD')
    parser.add_argument('--alpha_aug', type=float,
                        default=0, help='weight balance for mixup KD')
    parser.add_argument('-b', '--beta', type=float,
                        default=None, help='weight balance for other losses')
    parser.add_argument('--pr_w', type=float,
                        default=10, help='weight balance for pos relation losses')
    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4,
                        help='temperature for KD distillation')
    # NCE distillation
    parser.add_argument('--feat_dim', default=128,
                        type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact',
                        type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int,
                        help='number of negative samples for NCE')
    parser.add_argument('--loader_type', default="PACKD", type=str,
                        help='sskdloader')
    parser.add_argument('--mixup_num', default=0, type=int,
                        help='number of positive samples for mixup HNRT')
    parser.add_argument('--mixup_rotate', default=3, type=int,
                        help='using rotate mixup pos img')
    parser.add_argument('--mixup_ratio', default=0.0, type=float,
                        help='mixup ratio for pos img')
    parser.add_argument('--nce_m', default=0.5, type=float,
                        help='momentum for non-parametric updates')
    parser.add_argument('--pos_k', default=-1, type=int,
                        help='number of positive samples for NCE')

    # hint layer
    parser.add_argument('--hint_layer', default=2,
                        type=int, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--t_gamma', type=float, default=0.1)
    parser.add_argument('--t-milestones', type=int,
                        nargs='+', default=[30, 45])
    parser.add_argument('--t_momentum', type=float, default=0.9)
    parser.add_argument('--t_weight-decay', type=float, default=5e-4)
    parser.add_argument('--t-lr', type=float, default=0.05)
    parser.add_argument('--t-epoch', type=int, default=60)
    parser.add_argument('--t_save_folder', type=str, default='save_finetune')
    parser.add_argument('--few-ratio', type=float, default=1.0)
    parser.add_argument('--ops_err_thres', type=float, default=0.1)
    parser.add_argument('--ops_eps', type=float, default=0.1)
    parser.add_argument('--loss_margin', type=float, default=0.0)

    opt = parser.parse_args()
    print(opt)
    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = f'{opt.save_dir}/student_{opt.distill}_model'
    opt.tb_path = f'{opt.save_dir}/student_{opt.distill}_tensorboards'
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.model_t = get_teacher_name(opt.path_t)
    opt.model_name = 'S:{}_T:{}_{}_{}_a:{}_ag:{}_b:{}_{}'.format(opt.model_s,
                                                                opt.model_t, opt.dataset, opt.distill,
                                                                opt.alpha, opt.alpha_aug, opt.beta,
                                                                opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    print('==> done')
    return model


def save_train_teacher(model_t, optimizer_t, model_path, best_acc):
    state = {
        'epoch': 60,
        'model': model_t.state_dict(),
        't_acc': best_acc,
        'optimizer': optimizer_t.state_dict(),
    }
    print('saving the teacher model!')
    torch.save(state, model_path)
    return


def main():
    best_acc = 0
    opt = parse_option()
    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    # dataloader
    train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                       num_workers=opt.num_workers,
                                                                       k=opt.nce_k,
                                                                       mode=opt.mode,
                                                                       loader_type=opt.loader_type,
                                                                       opt=opt
                                                                       )
    n_cls = opt.ncls  # default  100 for cifar100
    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)
    data = torch.randn(2, 3, 32, 32)  # default (3,32,32) for cifar100
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    opt.s_dim = feat_s[-1].shape[1]
    opt.t_dim = feat_t[-1].shape[1]
    mixup_num = max(1, opt.mixup_num + 1)
    model_t = wrapper(model_t, opt.feat_dim).cuda()
    model_t_path = f'save_packd_t/{get_teacher_name(opt.path_t)}_embed.pth'
    if os.path.exists(model_t_path):
        model_t.load_state_dict(torch.load(
            model_t_path, map_location='cpu')['model'])
        model_t.eval()
    else:
        t_optimizer = optim.SGD([{'params': model_t.backbone.parameters(), 'lr': 0.0},
                                {'params': model_t.proj_head.parameters(),
                                 'lr': opt.t_lr},
                                {'params': model_t.classifier.parameters(), 'lr': opt.t_lr}],
                                momentum=opt.t_momentum,
                                weight_decay=opt.t_weight_decay)
        model_t.eval()
        t_scheduler = MultiStepLR(
            t_optimizer, milestones=opt.t_milestones, gamma=opt.t_gamma)
        # train ssp_head
        for epoch in range(opt.t_epoch):
            model_t.eval()
            loss_record = AverageMeter()
            acc_record = AverageMeter()
            start = time.time()
            for idx, data in enumerate(train_loader):
                x, target, _, _, _ = data
                x = x.cuda()
                target = target.cuda()
                t_optimizer.zero_grad()
                c, h, w = x.size()[-3:]
                x = x.view(-1, c, h, w)
                out, feat, proj_x, proj_logit = model_t(x, bb_grad=False)
                batch = int(x.size(0) / mixup_num)
                target = target.unsqueeze(1).expand(
                    batch, mixup_num).reshape(-1)
                loss = F.cross_entropy(proj_logit, target)
                loss.backward()
                t_optimizer.step()
                batch_acc = accuracy(proj_logit, target, topk=(1,))[0]
                loss_record.update(loss.item(), batch)
                acc_record.update(batch_acc.item(), batch)
            run_time = time.time() - start
            info = f'teacher_train_Epoch:{epoch}/{opt.t_epoch}\t run_time:{run_time:.3f}\t t_loss:{loss_record.avg:.3f}\t t_acc:{acc_record.avg:.2f}\t'
            print(info, flush=True)
        save_train_teacher(model_t, t_optimizer, model_t_path, 99)
    opt.t_dim = opt.feat_dim
    opt.n_data = n_data
    if opt.distill in ['PACKDCONLoss']:
        criterion_kd = PACKDConLoss(opt)
        module_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_s)
    else:
        raise NotImplementedError(opt.distill)
    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)
    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL-divergence loss
    criterion_list.append(criterion_kd)     # other knowledge distillation loss
    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc, flush=True)

    # routine
    for epoch in range(1, opt.epochs + 2):
        # set modules as train()
        for module in module_list:
            module.train()
        # set teacher as eval()
        module_list[-1].eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_cls = AverageMeter()
        losses_div_nor = AverageMeter()
        losses_div_aug = AverageMeter()
        losses_kd = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        time1 = time.time()
        for idx, data in enumerate(train_loader):
            data_time.update(time.time() - end)
            input, target, index, contrast_idx, mixup_indexes = data
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            mixup_indexes = mixup_indexes.cuda()
            if isinstance(contrast_idx, list):
                contrast_idx = [sub_idx.cuda()
                                for sub_idx in contrast_idx]
            else:
                contrast_idx = contrast_idx.cuda()
            c, h, w = input.size()[-3:]
            input = input.view(-1, c, h, w).cuda()
            target = target.cuda()
            batch = int(input.size(0) // mixup_num)
            nor_index = (torch.arange(mixup_num * batch) %
                         mixup_num == 0).cuda()
            aug_index = (torch.arange(mixup_num * batch) %
                         mixup_num != 0).cuda()
            # extract feature
            feat_s, logit_s = model_s(input, is_feat=True)

            f_s = feat_s[-1]
            with torch.no_grad():
                feat_t, logit_t, proj_x, proj_logit = model_t(input)
                f_t = proj_x.detach()
            # loss kd
            if opt.beta == 0:
                loss_kd = torch.Tensor([0]).cuda()
            else:
                loss_kd = criterion_kd(f_s, f_t,
                                       labels=[target, index],
                                       mask=None,
                                       contrast_idx=contrast_idx,
                                       mixup_indexes=mixup_indexes)
            # get all loss

            logit_s_nor = logit_s[nor_index]
            logit_t_nor = logit_t[nor_index]
            logit_s_aug = logit_s[aug_index]
            logit_t_aug = logit_t[aug_index]
            target = target.unsqueeze(1).expand(
                batch, mixup_num).reshape(-1)

            loss_cls = criterion_cls(logit_s, target)

            if opt.alpha > 0:
                loss_div_nor = criterion_div(logit_s_nor, logit_t_nor)
            else:
                loss_div_nor = torch.Tensor([0]).cuda()

            if opt.alpha_aug > 0:
                loss_div_aug = criterion_div(logit_s_aug, logit_t_aug)
            else:
                loss_div_aug = torch.Tensor([0]).cuda()

            loss = loss_cls + opt.alpha * loss_div_nor + opt.alpha_aug * loss_div_aug \
                 + opt.beta * loss_kd

            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            losses_cls.update(loss_cls.item(), input.size(0))
            losses_div_nor.update(loss_div_nor.item(), input.size(0))
            losses_div_aug.update(loss_div_aug.item(), input.size(0))
            losses_kd.update(loss_kd.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            # ===================backward=====================
            if epoch > 1:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================meters=====================
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg:.4f})\t'
                      'Loss_div_nor {loss_div_nor.val:.4f} ({loss_div_nor.avg:.4f})\t'
                      'Loss_div_aug {loss_div_aug.val:.4f} ({loss_div_aug.avg:.4f})\t'
                      'Loss_kd {loss_kd.val:.4f} ({loss_kd.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          epoch, idx, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, loss_cls=losses_cls,
                          loss_div_nor=losses_div_nor, loss_div_aug=losses_div_aug,
                          loss_kd=losses_kd, top1=top1, top5=top5), flush=True)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        time2 = time.time()
        logger.log_value('train_acc', top1.avg, epoch)
        logger.log_value('train_loss', losses.avg, epoch)

        test_acc, tect_acc_top5, test_loss = validate(
            val_loader, model_s, criterion_cls, opt)
        # save the best model
        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(
                opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)
    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
        'criterion_kd': criterion_kd.state_dict(),
    }
    save_file = os.path.join(
        opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
