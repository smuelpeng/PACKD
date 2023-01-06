import os.path as osp
import json
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import io
import torch
import os
import time

def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    try:
        with Image.open(buff) as img:
            img = img.convert('RGB')
    except IOError:
        logger.info('Failed in loading {}'.format(filepath))
    return img

class ImageNetDataset(Dataset):
    """
    ImageNet Dataset.
    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - image_reader (:obj:`str`): reader type 'pil' or 'ks'

    Metafile example::
        "n01440764/n01440764_10026.JPEG 0\n"
    """
    def __init__(self, root_dir, meta_file, transform=None,
                 mixnum=3):

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
        self.image_reader = pil_loader
        self.mixnum = mixnum

        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        self.data_dict = {}
        self.num_classes = 1000
        self.cls_positives = [[] for i in range(self.num_classes)]
        self.read_from = 'fs'
        self.initialized = False

        for i,line in enumerate(lines):
            filename, label = line.rstrip().split()
            self.metas.append({'filename': filename, 'label': label})
            label = int(label)
            self.cls_positives[label].append(i)

    def __len__(self):
        return self.num

    def read_file(self, meta_dict):
        if self.read_from == 'fs':
            filename = osp.join(self.root_dir, meta_dict['filename']) 
            filebytes = np.fromfile(filename, dtype=np.uint8)

        img = self.image_reader(filebytes, meta_dict['filename'])
        return img

    def __getitem__(self, idx):
        curr_meta = self.metas[idx]
        label = int(curr_meta['label'])
        img = self.read_file(curr_meta)
        img = self.transform(img)
        imgs = img
        labels = label
        item = {
            'image': imgs,
            'label': labels,
            'image_id': idx,
            'filename': curr_meta['filename']
        }
        return item

class ImageNetPACKDDataset(Dataset):
    """
    ImageNet Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - image_reader (:obj:`str`): reader type 'pil' or 'ks'

    Metafile example::
        "n01440764/n01440764_10026.JPEG 0\n"
    """
    def __init__(self, root_dir, meta_file, transform=None,
                 mixnum=3):

        self.root_dir = root_dir

        self.meta_file = meta_file
        self.transform = transform
        self.image_reader = pil_loader
        self.mixnum = mixnum

        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        self.data_dict = {}
        self.num_classes = 1000
        self.cls_positives = [[] for i in range(self.num_classes)]
        self.read_from = 'fs'
        self.initialized = False
        for i,line in enumerate(lines):
            filename, label = line.rstrip().split()
            self.metas.append({'filename': filename, 'label': label})
            label = int(label)
            self.cls_positives[label].append(i)
        self.cls_positives = [np.asarray(self.cls_positives[i])
                             for i in range(self.num_classes)]

    def __len__(self):
        return self.num

    def read_file(self, meta_dict):
        if self.read_from == 'fs':
            filename = osp.join(self.root_dir, meta_dict['filename']) 
            filebytes = np.fromfile(filename, dtype=np.uint8)
        img = self.image_reader(filebytes, meta_dict['filename'])
        return img

    def __getitem__(self, idx):
        end = time.time()
        curr_meta = self.metas[idx].copy()
        label = int(curr_meta['label'])
        # add root_dir to filename
        img = self.read_file(curr_meta)
        base_read_time = time.time()- end
        mixup_indexes = np.random.choice(
            self.cls_positives[label], self.mixnum)
        img = self.transform(img)
        imgs = [img]
        labels = [label]
        base_time = time.time()-end
        for i, m_idx in enumerate(mixup_indexes):
            curr_meta_midx = self.metas[m_idx]
            label_mix = int(curr_meta_midx['label'])
            assert label == label_mix
            img_midx = self.read_file(curr_meta_midx)
            img_midx = img_midx.rotate(np.random.randint(4) * 90)
            img_midx = self.transform(img_midx)
            lam = np.random.rand()
            lam = 0.1 + (0.9) * lam
            img_midx = lam * img + (1-lam) * img_midx
            if self.transform is not None:
                imgs.append(img_midx)
                labels.append(label)
        try:
            imgs = torch.cat(imgs)
            labels = torch.Tensor(labels)
        except:
            import pdb
            pdb.set_trace()

        item = {
            'image': imgs,
            'label': labels,
            'image_id': idx,
            'filename': curr_meta['filename']
        }
        mix_time = time.time() - end
        return item

