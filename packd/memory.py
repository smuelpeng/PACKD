import torch
from torch import nn
from torch.nn import functional as F
import math


class SimpleMemory(nn.Module):
    def __init__(self,opt):
        super(SimpleMemory, self).__init__()
        self.opt = opt
        stdv = 1. / math.sqrt(opt.feat_dim / 3)
        self.register_buffer('memory_bank', torch.rand(
            self.opt.n_data, self.opt.feat_dim).mul_(2 * stdv).add_(-stdv))
        mem_norm = self.memory_bank.pow(2).sum(1, keepdim=True).pow(0.5)
        self.memory_bank = self.memory_bank.div(mem_norm)
        self.memory_change_num = 0

    def get_weights(self, idx):
        weight_mem = torch.index_select(self.memory_bank, 0, idx.view(-1))
        return weight_mem

    def forward(self, feature, y, idx=None, update=False,):
        '''
        '''
        momentum = self.opt.nce_m
        pos_K = self.opt.pos_k
        batchSize = feature.size(0)
        outputSize = self.opt.n_data
        inputSize = self.opt.feat_dim
        # update memory
        with torch.no_grad():
            weight_mem = torch.index_select(self.memory_bank, 0, y.view(-1))
            # check memory bank diff
            if update:
                if self.memory_change_num // 782 == 0:
                    momentum = 0
                if self.memory_change_num % 782 == 0 or self.memory_change_num % 100 == 0:
                    memory_change_diff = torch.einsum(
                        'ij,ij->i', (weight_mem, feature)).mean().cpu().numpy()
                    print(f'memory_change_diff: epoch {self.memory_change_num // 782}',
                          f'/{self.memory_change_num %782}, {memory_change_diff}',
                          flush=True)
                self.memory_change_num += 1
                weight_mem.mul_(momentum)
                weight_mem.add_(torch.mul(feature, 1 - momentum))
                mem_norm = weight_mem.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_weight = weight_mem.div(mem_norm)
                self.memory_bank.index_copy_(0, y.view(-1), updated_weight)

        weight = torch.index_select(
            self.memory_bank, 0, idx.contiguous().view(-1)).detach()
        weight = weight.view(batchSize, -1, inputSize)
        output = torch.bmm(weight, feature.view(batchSize, inputSize, 1))
        return output
