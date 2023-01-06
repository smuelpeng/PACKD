from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from .auto_augmentation import CIFAR10Policy, Cutout
# import torch

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """

    data_folder = 'data/cifar100/'
    if not os.path.exists(data_folder):
        data_folder = 'data/cifar100/'

    return data_folder


class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar100_dataloaders(batch_size=128, num_workers=3, is_instance=False,
                             autoaugment=False, cutout=False, basic_transform=False, basic_sampler=False):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    # ])

    aug = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if autoaugment:
        aug.append(CIFAR10Policy())
    aug.append(transforms.ToTensor())
    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    aug.append(
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    )

    train_transform = transforms.Compose(aug)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    if basic_transform:
        train_transform = test_transform

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    if basic_sampler:
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)

    else:
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact',
                 is_sample=True, percent=1.0, 
                 pos_k=-1,
                 few_ratio=1.0
	):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample
        if few_ratio <1.0:
            from sklearn.model_selection import StratifiedShuffleSplit
            ss = StratifiedShuffleSplit(n_splits=1, test_size=1-few_ratio, random_state=0)
            train_indices, valid_indices = list(ss.split(np.array(self.targets)[:, np.newaxis],self.targets))[0]
            self.data = self.data[train_indices]
            self.targets = [self.targets[i] for i in train_indices]

        num_classes = 100
        if self.train:
            num_samples = len(self.data)
            label = self.targets
        else:
            num_samples = len(self.data)
            label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)
        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i])
                             for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i])
                             for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)
        self.pos_k = pos_k

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(
                self.cls_negative[target]) else False
            neg_idx = np.random.choice(
                self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            if self.pos_k > 0:
                replace = True if self.pos_k > len(
                    self.cls_positive[target]) else False
                pos_idx = np.random.choice(
                    self.cls_positive[target], self.pos_k, replace=replace)
                return img, target, index, [sample_idx, pos_idx]
            else:
                return img, target, index, sample_idx

class CIFAR100InstanceSamplePACKD(CIFAR100InstanceSample):
    """
    CIFAR100Instance+Sample+mixpos Dataset
    return one sample and mixup_num mixpos samples
    """

    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False, k=4096, mode='exact',
                 is_sample=True, percent=1.0, 
                 pos_k=-1,
                 mixup_num=1,
                 opt=None):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform, k=k, mode=mode,
                         is_sample=True, percent=percent,
                         pos_k=pos_k,few_ratio=opt.few_ratio)
        self.opt = opt

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # -1 mixup img
        # 0 normal img
        # n normal img + n mixup img

        if self.opt.mixup_num == -1:
            mixup_indexes = np.array([index])
            img = Image.fromarray(img)
            img = self.transform(img)
            m_idx = np.random.choice(self.cls_positive[target], 1)[0]
            img_midx = self.data[m_idx]
            if self.opt.mixup_rotate>0:
                img_midx = np.rot90(img_midx, np.random.randint(4)).copy()
            img_midx = Image.fromarray(img_midx)
            img_midx = self.transform(img_midx)
            lam = np.random.rand()
            img_midx = lam * img + (1-lam) * img_midx
            imgs = [img_midx]

        else:
            mixup_indexes = np.random.choice(
                self.cls_positive[target], self.opt.mixup_num)
            img = Image.fromarray(img)
            imgs = [self.transform(img)]
            for i, m_idx in enumerate(mixup_indexes):
                img_midx, target_midx = self.data[m_idx], self.targets[m_idx]
                if self.opt.mixup_rotate>0:
                    img_midx = np.rot90(img_midx, np.random.randint(4)).copy()
                img_midx = Image.fromarray(img_midx)
                img_midx = self.transform(img_midx)
                img_copy = img.copy()
                img_copy = self.transform(img_copy)
                lam = np.random.rand()
                if self.opt.mixup_rotate <-1:
                    lam = 0
                else:
                    lam = self.opt.mixup_ratio + (1-self.opt.mixup_ratio)*lam
                img_midx = lam * img_copy + (1-lam) * img_midx
                imgs.append(img_midx)
            mixup_indexes = np.append(np.array([index]), mixup_indexes)

        img = np.stack(imgs)
        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(
                self.cls_negative[target]) else False
            neg_idx = np.random.choice(
                self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            if self.pos_k > 0:
                replace = True if self.pos_k + 1  > len(
                    self.cls_positive[target]) else False
                cls_positive = self.cls_positive[target]
                cls_positive = np.delete(
                    cls_positive, np.where(cls_positive == index))
                pos_idxes = np.random.choice(
                    cls_positive, self.pos_k, replace=replace)
                if index in pos_idxes:
                    import pdb
                    pdb.set_trace()
                return img, target, index, [sample_idx, pos_idxes], mixup_indexes
            else:
                return img, target, index, sample_idx, mixup_indexes


def get_cifar100_dataloaders_sample(batch_size=128, num_workers=3, k=4096,
                                    mode='exact',
                                    is_sample=True, percent=1.0,
                                    autoaugment=False, cutout=False,
                                    pos_k=-1,
                                    loader_type=None,
                                    opt=None
                                    ):
    """
    cifar 100
    """
    data_folder = get_data_folder()
    aug = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if autoaugment:
        aug.append(CIFAR10Policy())
    aug.append(transforms.ToTensor())
    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    aug.append(
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    )

    train_transform = transforms.Compose(aug)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    if loader_type == "PACKD":
        train_set = CIFAR100InstanceSamplePACKD(root=data_folder,
                                               download=True,
                                               train=True,
                                               transform=train_transform,
                                               k=k,
                                               mode=mode,
                                               is_sample=is_sample,
                                               percent=percent,
                                               pos_k=pos_k,
                                               opt=opt)
    elif loader_type == "normal":
        train_set = CIFAR100InstanceSample(root=data_folder,
                                           download=True,
                                           train=True,
                                           transform=train_transform,
                                           k=k,
                                           mode=mode,
                                           is_sample=is_sample,
                                           percent=percent,
                                           pos_k=pos_k)
    else:
        raise NotImplementedError

    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data
