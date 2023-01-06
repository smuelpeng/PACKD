import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from utils import AverageMeter, accuracy

def train_wrapper(t_model,train_loader,val_loader,opt):

    t_optimizer = optim.SGD([{'params':t_model.backbone.parameters(), 'lr':0.0},
                            {'params':t_model.proj_head.parameters(), 'lr':args.t_lr}],
                            momentum=args.momentum, weight_decay=args.weight_decay)
    t_model.eval()
    t_scheduler = MultiStepLR(t_optimizer, milestones=args.t_milestones, gamma=args.gamma)
    for epoch in range(opt.t_epoch):
        t_model.eval()
        loss_record = AverageMeter()
        acc_record = AverageMeter()

        start = time.time()
        for x, _ in train_loader:

            t_optimizer.zero_grad()

            x = x.cuda()
            c,h,w = x.size()[-3:]
            x = x.view(-1, c, h, w)

            _, rep, feat = t_model(x, bb_grad=False)
            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
            aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

            nor_rep = rep[nor_index]
            aug_rep = rep[aug_index]
            nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
            simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
            target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            loss = F.cross_entropy(simi, target)

            loss.backward()
            t_optimizer.step()

            batch_acc = accuracy(simi, target, topk=(1,))[0]
            loss_record.update(loss.item(), 3*batch)
            acc_record.update(batch_acc.item(), 3*batch)

        logger.add_scalar('train/teacher_ssp_loss', loss_record.avg, epoch+1)
        logger.add_scalar('train/teacher_ssp_acc', acc_record.avg, epoch+1)

        run_time = time.time() - start
        info = 'teacher_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\t'.format(
            epoch+1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
        print(info)

        t_model.eval()
        acc_record = AverageMeter()
        loss_record = AverageMeter()
        start = time.time()
        for x, _ in val_loader:

            x = x.cuda()
            c,h,w = x.size()[-3:]
            x = x.view(-1, c, h, w)

            with torch.no_grad():
                _, rep, feat = t_model(x)
            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
            aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

            nor_rep = rep[nor_index]
            aug_rep = rep[aug_index]
            nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
            simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
            target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            loss = F.cross_entropy(simi, target)

            batch_acc = accuracy(simi, target, topk=(1,))[0]
            acc_record.update(batch_acc.item(),3*batch)
            loss_record.update(loss.item(), 3*batch)

        run_time = time.time() - start
        logger.add_scalar('val/teacher_ssp_loss', loss_record.avg, epoch+1)
        logger.add_scalar('val/teacher_ssp_acc', acc_record.avg, epoch+1)

        info = 'ssp_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\n'.format(
                epoch+1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
        print(info)

        t_scheduler.step()

