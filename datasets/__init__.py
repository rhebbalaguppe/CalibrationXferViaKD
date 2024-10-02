from .cifar10 import get_train_valid_test_loader as cifar10loader

from .cifar100 import get_train_valid_test_loader as cifar100loader

from .svhn import get_train_valid_test_loader as svhnloader

from .tiny_imagenet import get_train_valid_test_loader as tiny_imagenetloader

dataloader_dict = {
    "cifar10" : cifar10loader,
    "cifar100" : cifar100loader,
    "svhn" : svhnloader,
    "tiny_imagenet" : tiny_imagenetloader
}

dataset_nclasses_dict = {
    "cifar10" : 10,
    "cifar100" : 100,
    "svhn" : 10,
    "tiny_imagenet" : 200
}
