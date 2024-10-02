# '''ShuffleNetV2 in PyTorch.
# See the paper "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design" for more details.
# '''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ShuffleBlock(nn.Module):
#     def __init__(self, groups=2):
#         super(ShuffleBlock, self).__init__()
#         self.groups = groups

#     def forward(self, x):
#         '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
#         N, C, H, W = x.size()
#         g = self.groups
#         return x.view(N, g, C//g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


# class SplitBlock(nn.Module):
#     def __init__(self, ratio):
#         super(SplitBlock, self).__init__()
#         self.ratio = ratio

#     def forward(self, x):
#         c = int(x.size(1) * self.ratio)
#         return x[:, :c, :, :], x[:, c:, :, :]


# class BasicBlock(nn.Module):
#     def __init__(self, in_channels, split_ratio=0.5, is_last=False):
#         super(BasicBlock, self).__init__()
#         self.is_last = is_last
#         self.split = SplitBlock(split_ratio)
#         in_channels = int(in_channels * split_ratio)
#         self.conv1 = nn.Conv2d(in_channels, in_channels,
#                                kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.conv2 = nn.Conv2d(in_channels, in_channels,
#                                kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
#         self.bn2 = nn.BatchNorm2d(in_channels)
#         self.conv3 = nn.Conv2d(in_channels, in_channels,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(in_channels)
#         self.shuffle = ShuffleBlock()

#     def forward(self, x):
#         x1, x2 = self.split(x)
#         out = F.relu(self.bn1(self.conv1(x2)))
#         out = self.bn2(self.conv2(out))
#         preact = self.bn3(self.conv3(out))
#         out = F.relu(preact)
#         # out = F.relu(self.bn3(self.conv3(out)))
#         preact = torch.cat([x1, preact], 1)
#         out = torch.cat([x1, out], 1)
#         out = self.shuffle(out)
#         if self.is_last:
#             return out, preact
#         else:
#             return out


# class DownBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DownBlock, self).__init__()
#         mid_channels = out_channels // 2
#         # left
#         self.conv1 = nn.Conv2d(in_channels, in_channels,
#                                kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
#         self.bn1 = nn.BatchNorm2d(in_channels)
#         self.conv2 = nn.Conv2d(in_channels, mid_channels,
#                                kernel_size=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(mid_channels)
#         # right
#         self.conv3 = nn.Conv2d(in_channels, mid_channels,
#                                kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(mid_channels)
#         self.conv4 = nn.Conv2d(mid_channels, mid_channels,
#                                kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
#         self.bn4 = nn.BatchNorm2d(mid_channels)
#         self.conv5 = nn.Conv2d(mid_channels, mid_channels,
#                                kernel_size=1, bias=False)
#         self.bn5 = nn.BatchNorm2d(mid_channels)

#         self.shuffle = ShuffleBlock()

#     def forward(self, x):
#         # left
#         out1 = self.bn1(self.conv1(x))
#         out1 = F.relu(self.bn2(self.conv2(out1)))
#         # right
#         out2 = F.relu(self.bn3(self.conv3(x)))
#         out2 = self.bn4(self.conv4(out2))
#         out2 = F.relu(self.bn5(self.conv5(out2)))
#         # concat
#         out = torch.cat([out1, out2], 1)
#         out = self.shuffle(out)
#         return out


# class ShuffleNetV2(nn.Module):
#     def __init__(self, net_size, num_classes=10):
#         super(ShuffleNetV2, self).__init__()
#         out_channels = configs[net_size]['out_channels']
#         num_blocks = configs[net_size]['num_blocks']

#         # self.conv1 = nn.Conv2d(3, 24, kernel_size=3,
#         #                        stride=1, padding=1, bias=False)
#         self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(24)
#         self.in_channels = 24
#         self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
#         self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
#         self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
#         self.conv2 = nn.Conv2d(out_channels[2], out_channels[3],
#                                kernel_size=1, stride=1, padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels[3])
#         self.linear = nn.Linear(out_channels[3], num_classes)

#     def _make_layer(self, out_channels, num_blocks):
#         layers = [DownBlock(self.in_channels, out_channels)]
#         for i in range(num_blocks):
#             layers.append(BasicBlock(out_channels, is_last=(i == num_blocks - 1)))
#             self.in_channels = out_channels
#         return nn.Sequential(*layers)

#     def get_feat_modules(self):
#         feat_m = nn.ModuleList([])
#         feat_m.append(self.conv1)
#         feat_m.append(self.bn1)
#         feat_m.append(self.layer1)
#         feat_m.append(self.layer2)
#         feat_m.append(self.layer3)
#         return feat_m

#     def get_bn_before_relu(self):
#         raise NotImplementedError('ShuffleNetV2 currently is not supported for "Overhaul" teacher')

#     def forward(self, x, is_feat=False, preact=False):
#         out = F.relu(self.bn1(self.conv1(x)))
#         # out = F.max_pool2d(out, 3, stride=2, padding=1)
#         f0 = out
#         out, f1_pre = self.layer1(out)
#         f1 = out
#         out, f2_pre = self.layer2(out)
#         f2 = out
#         out, f3_pre = self.layer3(out)
#         f3 = out
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         f4 = out
#         out = self.linear(out)
#         if is_feat:
#             if preact:
#                 return [f0, f1_pre, f2_pre, f3_pre, f4], out
#             else:
#                 return [f0, f1, f2, f3, f4], out
#         else:
#             return out


# configs = {
#     0.2: {
#         'out_channels': (40, 80, 160, 512),
#         'num_blocks': (3, 3, 3)
#     },

#     0.3: {
#         'out_channels': (40, 80, 160, 512),
#         'num_blocks': (3, 7, 3)
#     },

#     0.5: {
#         'out_channels': (48, 96, 192, 1024),
#         'num_blocks': (3, 7, 3)
#     },

#     1: {
#         'out_channels': (116, 232, 464, 1024),
#         'num_blocks': (3, 7, 3)
#     },
#     1.5: {
#         'out_channels': (176, 352, 704, 1024),
#         'num_blocks': (3, 7, 3)
#     },
#     2: {
#         'out_channels': (224, 488, 976, 2048),
#         'num_blocks': (3, 7, 3)
#     }
# }


# def shufflenetv2(**kwargs):
#     model = ShuffleNetV2(net_size=1, **kwargs)
#     return model

# model_dict = {
#     "shufflenetv2" : shufflenetv2
# }

# def _get_shufflenetv2(model, dataset):
#     if dataset == "cifar10":
#         model = model_dict[model](num_classes=10)
#     elif dataset == "cifar100":
#         model = model_dict[model](num_classes=100)
#     elif dataset == "tiny_imagenet":
#         model = model_dict[model](num_classes=200)
#     else:
#         assert False, "not a compatible dataset for ShuffleNet v2..."
#     return model

# if __name__ == '__main__':
#     model = shufflenetv2(num_classes=100)

#     x = torch.randn(1, 3, 32, 32)
#     y = model(x)
#     print(y.shape)

#     from torchsummary import summary
#     summary(model, (3, 32, 32), batch_size=-1, device="cpu")

"""shufflenetv2 in pytorch
[1] Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, Jian Sun
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)


def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x


class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, int(out_channels / 2), 1),
                nn.BatchNorm2d(int(out_channels / 2)),
                nn.ReLU(inplace=True)
            )
        else:
            self.shortcut = nn.Sequential()

            in_channels = int(in_channels / 2)
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):

        if self.stride == 1 and self.out_channels == self.in_channels:
            shortcut, residual = channel_split(x, int(self.in_channels / 2))
        else:
            shortcut = x
            residual = x

        shortcut = self.shortcut(shortcut)
        residual = self.residual(residual)
        x = torch.cat([shortcut, residual], dim=1)
        x = channel_shuffle(x, 2)

        return x


class ShuffleNetV2(nn.Module):

    def __init__(self, ratio=1, class_num=100):
        super().__init__()
        if ratio == 0.5:
            out_channels = [48, 96, 192, 1024]
        elif ratio == 1:
            out_channels = [116, 232, 464, 1024]
        elif ratio == 1.5:
            out_channels = [176, 352, 704, 1024]
        elif ratio == 2:
            out_channels = [244, 488, 976, 2048]
        else:
            ValueError('unsupported ratio number')

        self.pre = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24)
        )

        self.stage2 = self._make_stage(24, out_channels[0], 3)
        self.stage3 = self._make_stage(out_channels[0], out_channels[1], 7)
        self.stage4 = self._make_stage(out_channels[1], out_channels[2], 3)
        self.conv5 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 1),
            nn.BatchNorm2d(out_channels[3]),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(out_channels[3], class_num)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_stage(self, in_channels, out_channels, repeat):
        layers = []
        layers.append(ShuffleUnit(in_channels, out_channels, 2))

        while repeat:
            layers.append(ShuffleUnit(out_channels, out_channels, 1))
            repeat -= 1

        return nn.Sequential(*layers)


def shufflenetv2(num_classes=10):
    return ShuffleNetV2(class_num=num_classes)

model_dict = {
    "shufflenetv2" : shufflenetv2
}

def _get_shufflenetv2(model, dataset):
    if dataset == "cifar10":
        model = model_dict[model](num_classes=10)
    elif dataset == "cifar100":
        model = model_dict[model](num_classes=100)
    elif dataset == "tiny_imagenet":
        model = model_dict[model](num_classes=200)
    else:
        assert False, "not a compatible dataset for ShuffleNetv2..."
    return model


if __name__ == '__main__':
    import torch
    model = shufflenetv2(num_classes=200)
    x = torch.randn(2,3,64,64)
    y = model(x)
    print(y.shape)

    x = torch.randn(2,3,32,32)
    y = model(x)
    print(y.shape)