import torch
from torch.utils.data.sampler import Sampler
try:
    import linklink as link
except:    
    import spring.linklink as link
import math
import numpy as np
from tqdm import tqdm

class DistributedMixPosSampler(Sampler):
    def __init__(self, sampler, 
                cls_positives= None,
                metas= None):
        self.sampler = sampler
        self.last_iter = sampler.last_iter
        self.batch_size = sampler.batch_size
        self.cls_positives = cls_positives
        self.metas = metas

        self.indices = self.gen_new_list()
        self.call = 0


    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[self.last_iter * self.batch_size:])
        else:
            raise RuntimeError(
                "this sampler is not designed to be called more than once!!")

    def gen_new_list(self):
        new_indices = []
        # import pdb
        # pdb.set_trace()

        for idx, indice in  tqdm(enumerate(self.sampler.indices)):
            target = int(self.metas[indice]['label'])
            new_indice = np.random.choice(
                    self.cls_positives[target], 1, replace=False)
            new_indices.append(new_indice)

        indices = np.concatenate(new_indices,axis=0).reshape(-1)
        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        return self.total_size


class DistributedGivenIterationSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size,
                 world_size=None, rank=None, last_iter=0):

        if world_size is None:
            world_size = link.get_world_size()
        if rank is None:
            rank = link.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size

        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[self.last_iter * self.batch_size:])
        else:
            raise RuntimeError(
                "this sampler is not designed to be called more than once!!")

    def gen_new_list(self):
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)

        beg = self.total_size * self.rank 
        indices = indices[beg:beg + self.total_size]

        assert len(indices) == self.total_size 

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        return self.total_size

class DistributedEpochSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, world_size=None, rank=None, last_iter=0):
        if world_size is None:
            world_size = link.get_world_size()
        if rank is None:
            rank = link.get_rank()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.all_size_single = self.total_iter * self.batch_size

        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call == 0:
            self.call = 1
            return iter(self.indices[self.last_iter * self.batch_size:])
        else:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

    def get_one_epoch_self_part(self):
        num = len(self.dataset)
        indices = np.arange(num)
        extra_indices = np.random.choice(num, self.extra_per_epoch, replace=False)
        indices = np.concatenate((indices, extra_indices))
        np.random.shuffle(indices)
        assert len(indices) % (self.world_size * self.batch_size) == 0
        num_single = len(indices) // self.world_size
        return indices[self.rank * num_single:(self.rank + 1) * num_single]

    def gen_new_list(self):
        np.random.seed(0)

        self.all_num = self.total_iter * self.batch_size * self.world_size
        iter_per_epoch = (len(self.dataset) - 1) // (self.batch_size * self.world_size) + 1
        self.num_per_epoch = iter_per_epoch * self.batch_size * self.world_size
        self.extra_per_epoch = self.num_per_epoch - len(self.dataset)
        repeat = (self.all_num - 1) // self.num_per_epoch + 1
        indices = []
        for i in range(repeat):
            indice = self.get_one_epoch_self_part()
            indices.append(indice)

        indices = np.concatenate(indices)
        indices = indices[:self.all_size_single]

        assert len(indices) == self.all_size_single

        return indices

    def __len__(self):
        return self.all_size_single

def build_sampler_PACKD(dataset, batch_size, world_size, 
                        rank, max_epoch, mix_num=1
                       ):
    iter_per_epoch = (len(dataset) - 1) // (batch_size * world_size) + 1
    total_iter = max_epoch * iter_per_epoch
    metas = dataset.metas
    cls_positives = dataset.cls_positives
    base_sampler = DistributedEpochSampler(dataset, total_iter, batch_size, world_size, rank, last_iter=0)
    samplers = [base_sampler]
    for mix_i in range(mix_num):
        mix_sampler = DistributedMixPosSampler(base_sampler, cls_positives, metas)
        samplers.append(mix_sampler)
    return samplers

