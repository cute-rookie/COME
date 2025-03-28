import os
import random

import numpy as np
import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import models
from utils import datasets
from models.deformable_retrieval import build_retrieval_model
from models.deformable_transformer import build_deforamble_transformer
default_workers = os.cpu_count()

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
    
mean = (0.5000, 0.5000, 0.5000)
std = (0.5000, 0.5000, 0.5000)

normalize = transforms.Normalize(mean=mean, std=std)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,
])

# def nclass(config):
#     r = {
#         'utkface': 2,
#         'utkface_multicls': 5,
#         'utkface_multicls2': 5,
#         'celeba': 2
#     }[config['dataset']]
#
#     return r

def reasonableness_judgment(ds, nclass, ta, sa):
    # target attribute_sensitive attribute
    if ds == 'utkface':
        utkface_nclass2 = ['gender_ethnicity']
        utkface_nclass5 = ['ethnicity_age', 'age_ethnicity']
        if nclass == 2:
            assert ta + '_' + sa in utkface_nclass2, f'when the dataset is utkface and the number of classes is 2, the target attribute and the sensitive attribute should be {utkface_nclass2}'
            return 0 # class_type
        if nclass == 5:
            assert ta + '_' + sa in utkface_nclass5, f'when the dataset is utkface and the number of classes is 5, the target attribute and the sensitive attribute should be {utkface_nclass5}'
            if ta == 'ethnicity': return 1
            elif ta == 'age': return 2
    elif ds == 'celeba':
        assert ta in range(1, 41) and sa in range(1, 41), 'the target attribute and the sensitive attribute should be in range(21)'
        return 0

def R(config):
    r = {
        'utkface_multicls2': 5000,
        'utkface': 5000,
        'celeba': 5000
    }[config['dataset'] + {2: '_2'}.get(config['dataset_kwargs']['evaluation_protocol'], '')]

    return r


def arch(config, **kwargs):
    if config['arch'] in models.network_names:
        net = models.network_names[config['arch']](**config['arch_kwargs'], **kwargs)
    else:
        raise ValueError(f'Invalid Arch: {config["arch"]}')

    codebook = kwargs['codebook']
    nclass = config['arch_kwargs']['nclass']
    transformer = build_deforamble_transformer(config)
    model = build_retrieval_model(
        config,
        net,
        transformer,
        codebook=codebook,
        num_classes=nclass,
    )

    return model


def optimizer(config, params):
    o_type = config['optim']
    kwargs = config['optim_kwargs']

    if o_type == 'sgd':
        o = SGD(params,
                lr=kwargs['lr'],
                momentum=kwargs.get('momentum', 0.9),
                weight_decay=kwargs.get('weight_decay', 0.0005),
                nesterov=kwargs.get('nesterov', False))
    else:  # adam
        o = Adam(params,
                 lr=kwargs['lr'],
                 betas=kwargs.get('betas', (0.9, 0.999)),
                 weight_decay=kwargs.get('weight_decay', 0))

    return o


def scheduler(config, optimizer):
    s_type = config['scheduler']
    kwargs = config['scheduler_kwargs']

    if s_type == 'step':
        return lr_scheduler.StepLR(optimizer,
                                   kwargs['step_size'],
                                   kwargs['gamma'])
    elif s_type == 'mstep':
        return lr_scheduler.MultiStepLR(optimizer,
                                        [int(float(m) * int(config['epochs'])) for m in
                                         kwargs['milestones'].split(',')],
                                        kwargs['gamma'])
    else:
        raise Exception('Scheduler not supported yet: ' + s_type)


def compose_transform(mode='train', resize=0, crop=0, norm=0,
                      augmentations=None):
    """

    :param mode:
    :param resize:
    :param crop:
    :param norm:
    :param augmentations:
    :return:
    if train:
      Resize (optional, usually done in Augmentations)
      Augmentations
      ToTensor
      Normalize

    if test:
      Resize
      CenterCrop
      ToTensor
      Normalize
    """
    # norm = 0, 0 to 1
    # norm = 1, -1 to 1
    # norm = 2, standardization
    mean, std = {
        0: [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        1: [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        2: [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    }[norm]

    compose = []

    if resize != 0:
        resize1 = (resize, resize)
        compose.append(transforms.Resize(resize1))

    if mode == 'train' and augmentations is not None:
        compose += augmentations

    if mode == 'test' and crop != 0 and resize != crop:
        compose.append(transforms.CenterCrop(crop))

    compose.append(transforms.ToTensor())

    if norm != 0:
        compose.append(transforms.Normalize(mean, std))

    return transforms.Compose(compose)


def dataset(config, filename, transform_mode):
    dataset_name = config['dataset']
    nclass = config['arch_kwargs']['nclass']

    resize = config['dataset_kwargs'].get('resize', 0)
    crop = config['dataset_kwargs'].get('crop', 0)
    norm = config['dataset_kwargs'].get('norm', 2)
    reset = config['dataset_kwargs'].get('reset', False)

    if dataset_name == 'celeba':
            resizec = resize
            cropc = crop

            if transform_mode == 'train':
                
                mean = (0.5000, 0.5000, 0.5000)
                std = (0.5000, 0.5000, 0.5000)

                normalize = transforms.Normalize(mean=mean, std=std)

                train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])
                transform=train_transform
                
                # transform = compose_transform('train', resizec, 0, norm, [
                #     transforms.RandomHorizontalFlip(),
                #     transforms.ColorJitter(brightness=0.05, contrast=0.05),
                # ])
            else:
                transform = compose_transform('test', resizec, cropc, norm)
            d = datasets.celeba(transform=transform, filename=filename, reset=reset, config=config, transform_mode=transform_mode)


    elif dataset_name == 'utkface':
        resizec = resize
        cropc = crop

        if transform_mode == 'train':

            mean = (0.5000, 0.5000, 0.5000)
            std = (0.5000, 0.5000, 0.5000)

            normalize = transforms.Normalize(mean=mean, std=std)

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            transform = train_transform
            # transform = compose_transform('test', resizec, cropc, norm)
            # ])
        else:
            transform = compose_transform('test', resizec, cropc, norm)
        d = datasets.utk_multicls2(transform=transform, filename=filename, reset=reset, config=config, transform_mode=transform_mode)
        
    elif dataset_name == 'utkface_multicls':
        resizec = resize
        cropc = crop

        if transform_mode == 'train':
            mean = (0.5000, 0.5000, 0.5000)
            std = (0.5000, 0.5000, 0.5000)

            normalize = transforms.Normalize(mean=mean, std=std)

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            transform=TwoCropTransform(train_transform)
            transform = compose_transform('test', resizec, cropc, norm)
            # transform = compose_transform('train', resizec, 0, norm, [
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ColorJitter(brightness=0.05, contrast=0.05),
            # ])
        else:
            transform = compose_transform('test', resizec, cropc, norm)
        d = datasets.utk_multicls(transform=transform, filename=filename, reset=reset)

    elif dataset_name == 'utkface_multicls2':
        resizec = resize
        cropc = crop

        if transform_mode == 'train':
            mean = (0.5000, 0.5000, 0.5000)
            std = (0.5000, 0.5000, 0.5000)

            normalize = transforms.Normalize(mean=mean, std=std)

            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            # transform=TwoCropTransform(train_transform)
            transform = train_transform

        else:
            transform = compose_transform('test', resizec, cropc, norm)
        d = datasets.utk_multicls2(transform=transform, filename=filename, reset=reset, transform_mode=transform_mode)
        
    else:  # cifar10/ cifar100
        resizec = 0 if resize == 32 else resize
        cropc = 0 if crop == 32 else crop

        if transform_mode == 'train':
            transform = compose_transform('train', resizec, 0, norm, [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
            ])
        else:
            transform = compose_transform('test', resizec, cropc, norm)
        ep = config['dataset_kwargs'].get('evaluation_protocol', 1)
        d = datasets.cifar(nclass, transform=transform, filename=filename, evaluation_protocol=ep, reset=reset)

    return d


def dataloader(d, bs=256, shuffle=True, workers=-1, drop_last=True, transform_mode='train'):
    # if len(d) // 5 % 2 != 0:
    #     sub_bs = 128
    # else:
    #     sub_bs = len(d) // 5
    # bs = min(bs, sub_bs)
    # if workers < 0:
    #     workers = 8
    # l = DataLoader(d,
    #                bs,
    #                shuffle,
    #                drop_last=drop_last,
    #                num_workers=workers)
    # return l
    def custom_collate_fn(batch):
        # 如果批次的样本数为奇数，则丢弃最后一个样本
        if len(batch) % 2 == 1:
            batch = batch[:-1]
        return torch.utils.data.dataloader.default_collate(batch)

    if transform_mode == 'train':
        if len(d) // 5 % 2 != 0:
            sub_bs = 128
        else:
            sub_bs = len(d) // 5
        bs = min(bs, sub_bs)
    if workers < 0:
        workers = 8
    l = DataLoader(d,
                   bs,
                   shuffle,
                   drop_last=drop_last,
                   num_workers=workers,
                   collate_fn=custom_collate_fn)
    return l


def seeding(seed):
    #print('seed:',type(seed))
    if seed != -1:
        # 设置Python、NumPy、PyTorch的随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用GPU

        # 设置CUDNN的确定性算法（适用于GPU）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def tensor_to_dataset(tensor, transform=None):
    class TransformTensorDataset(Dataset):
        def __init__(self, tensor, ts=None):
            super(TransformTensorDataset, self).__init__()
            self.tensor = tensor
            self.ts = ts

        def __getitem__(self, index):
            if self.ts is not None:
                return self.ts(self.tensor[index])
            return self.tensor[index]

        def __len__(self):
            return len(self.tensor)

    ttd = TransformTensorDataset(tensor, transform)
    return ttd
