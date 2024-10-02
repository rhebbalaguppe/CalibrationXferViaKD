from .resnet_cifar import _get_cifar_resnet
from .convnet_cifar import _get_cifar_convnet
from .resnet_tinyimagenet import _get_tinyimagenet_resnet
from .mobilenetv2 import _get_mobilenetv2
from .shufflenetv2 import _get_shufflenetv2
from .shufflenetv1 import _get_shufflenetv1
from .densenet import _get_densenet
from .wrn import _get_wrn


def get_model(model, dataset, args):
    if "cifar" in dataset:
        if "resnet" in model:
            return _get_cifar_resnet(model, dataset)
        elif "convnet" in model:
            return _get_cifar_convnet(model, dataset)
        elif "mobilenetv2" in model:
            return _get_mobilenetv2(model, dataset)
        elif "shufflenetv1" in model:
            return _get_shufflenetv1(model, dataset)
        elif "shufflenetv2" in model:
            return _get_shufflenetv2(model, dataset)
        elif "wrn" in model:
            return _get_wrn(model, dataset)
    elif "tiny_imagenet" in dataset:
        if "resnet" in model:
            return _get_tinyimagenet_resnet(model, dataset)
        elif "mobilenetv2" in model:
            return _get_mobilenetv2(model, dataset)
        elif "shufflenetv1" in model:
            return _get_shufflenetv1(model, dataset)
        elif "shufflenetv2" in model:
            return _get_shufflenetv2(model, dataset)
        elif "densenet" in model:
            return _get_densenet(model, dataset)