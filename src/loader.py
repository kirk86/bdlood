# coding: utf-8

import os
import torch
import logging
import numpy as np
import torchvision as tv


class Dataset(object):
    """
    Abstract class including operations that should be extended
    by subclasses
    """
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.args = args

    def split_in_out_domain(self):
        pass

    def additional_data(self):
        pass


class Cifar10(Dataset):

    """
    CIFAR-10 dataset [Kri09]_.
    A dataset with 50k training images and 10k testing images, with the
    following classes:
    * Airplane
    * Automobile
    * Bird
    * Cat
    * Deer
    * Dog
    * Frog
    * Horse
    * Ship
    * Truck
    .. [Kri09] Krizhevsky, A (2009). Learning Multiple Layers of Features
        from Tiny Images. Technical Report.
    """
    def __init__(self, data_path='/tmp/', **kwargs):
        ds_kwargs = {
            'num_classes': 10,
            'mean': ch.tensor([0.4914, 0.4822, 0.4465]),
            'std': ch.tensor([0.2023, 0.1994, 0.2010]),
            'custom_class': datasets.CIFAR10,
            'label_mapping': None,
            'transform_train': da.TRAIN_TRANSFORMS_DEFAULT(32),
            'transform_test': da.TEST_TRANSFORMS_DEFAULT(32)
            }
        super(Cifar10, self).__init__('cifar', data_path, **ds_kwargs)

    def get_model(self, arch, pretrained):
        if pretrained:
            raise ValueError('CIFAR does not support pytorch_pretrained=True')
        return cifar_models.__dict__[arch](num_classes=self.num_classes)


class Cifar100(object):
    """Documentation for Cifar100

    """
    def __init__(self, args):
        super(Cifar100, self).__init__()
        self.args = args


class Svhn(object):
    """Documentation for Cifar100

    """
    def __init__(self, args):
        super(Cifar100, self).__init__()
        self.args = args


class FashionMnist(object):
    """Documentation for Cifar100

    """
    def __init__(self, args):
        super(Cifar100, self).__init__()
        self.args = args


class Mnist(object):
    """Documentation for Cifar100

    """
    def __init__(self, args):
        super(Cifar100, self).__init__()
        self.args = args


class Datasets(object):
    """Documentation for Datasets

    """
    def __init__(self, dataset_name, config, valid_size=0.1,
                 batch_size=128, num_workers=4, pin_memory=True,
                 augment=True, sampler=torch.utils.data.SubsetRandomSampler):
        super(Datasets, self).__init__()

        self.data_path = config.data_path
        self.dataset_name = dataset_name
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sampler = sampler
        self.augment = augment
        self.augments = [
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.RandomCrop(32, padding=4)
        ]

        self.transforms = {
            'imagenet': tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225))
            ]),
            'restricted_imagenet': tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=(0.4717, 0.4499, 0.3837),
                    std=(0.2600, 0.2516, 0.2575))
            ]),
            'CIFAR10': tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010))
            ]),
            'CIFAR100': tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=(0.5071, 0.4867, 0.4408),
                    std=(0.2675, 0.2565, 0.2761))
            ]),
            'cinic': tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(
                    mean=(0.47889522, 0.47227842, 0.43047404),
                    std=(0.24205776, 0.23828046, 0.25874835))
            ]),
            'SVHN': tv.transforms.Compose([
                tv.transforms.ToTensor(),
                # tv.transforms.Lambda(lambda x: (x - x.mean())/x.std()),
                # these seem to work better
                # mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
                tv.transforms.Normalize(
                    mean=(0.4376, 0.4437, 0.4728),
                    std=(0.1980, 0.2010, 0.1970))
            ]),
        }

        self.dataset = getattr(tv.datasets, self.dataset_name)
        logging.info("Loading dataset...{}".format(
            str(self.dataset).split('.')[-1].split("'")[0]
        ))

        self.dsetsplit = {
            'CIFAR10': {'train': True, 'test': False,
                        'data': 'data', 'targets': 'targets'},
            'SVHN': {'train': 'train', 'test': 'test',
                     'data': 'data', 'targets': 'labels'}
        }
        if augment:
            import copy
            self.test_transforms = copy.deepcopy(self.transforms)
            self.augments.reverse()
            for k, v in self.transforms.items():
                for aug in self.augments:
                    v.transforms.insert(0, aug)

        self.train = self.dataset(
            self.data_path, self.dsetsplit[self.dataset_name]['train'],
            download=False if os.path.exists(os.getcwd() + self.data_path)
            else True,
            transform=self.transforms[dataset_name])

        self.test = self.dataset(
            self.data_path, self.dsetsplit[self.dataset_name]['test'],
            download=False if os.path.exists(os.getcwd() + self.data_path)
            else True,
            transform=self.test_transforms[dataset_name] if augment else
            self.transforms[dataset_name])

    def split_image_data(self, train_data, test_data=None, dist=False):

        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        if dist:
            import copy
            valid_data = copy.deepcopy(train_data)
            for idx, split in zip([train_idx, valid_idx], [train_data, valid_data]):
                data = getattr(
                    split, self.dsetsplit[self.dataset_name]['data']
                )[idx]
                targets = getattr(
                    split, self.dsetsplit[self.dataset_name]['targets']
                )
                targets = np.array(targets)[idx].tolist()
                setattr(split, self.dsetsplit[self.dataset_name]['data'], data)
                setattr(split, self.dsetsplit[self.dataset_name]['targets'], targets)

            train_sampler = self.sampler(train_data)
            valid_sampler = self.sampler(valid_data)
        else:
            train_sampler = self.sampler(train_idx)
            valid_sampler = self.sampler(valid_idx)

        if test_data is not None:
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.batch_size,
                num_workers=self.num_workers)
        else:
            train_idx, test_idx = train_idx[split:], train_idx[:split]
            train_sampler = self.sampler(train_idx)
            test_sampler = self.sampler(test_idx)

            test_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.batch_size,
                sampler=test_sampler,
                num_workers=self.num_workers)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers)

        valid_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=self.num_workers)

        if self.augment:
            import copy
            valid_loader = copy.deepcopy(valid_loader)
            valid_loader.dataset.transforms = self.test_transforms[self.dataset_name]

        return train_loader, valid_loader, test_loader
