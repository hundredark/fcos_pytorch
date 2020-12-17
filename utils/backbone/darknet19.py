import torch
import torch.nn as nn


cfg1 = [32, 'M', 64, 'M', 128, 64, 128, 'M', 256, 128, 256]
cfg2 = ['M', 512, 256, 512, 256, 512]
cfg3 = ['M', 1024, 512, 1024, 512, 1024]


def make_layers(cfg, in_channels=3, batch_norm=True, flag=True):
    layers = []
    # TODO: delete this line after test
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=v,
                                    kernel_size=(1, 3)[flag],
                                    stride=1,
                                    padding=(0, 1)[flag],
                                    bias=False))
            if batch_norm:
                bn = nn.BatchNorm2d(v)
                layers.append(bn)
            in_channels = v

            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        flag = not flag

    return nn.Sequential(*layers)


class Darknet19(nn.Module):
    def __init__(self, in_channels=3, batch_norm=True, pretrained=False, bef_neck=True):
        super(Darknet19, self).__init__()
        self.bef_neck = bef_neck
        self.block1 = make_layers(cfg1, in_channels=in_channels, batch_norm=batch_norm, flag=True)
        self.block2 = make_layers(cfg2, in_channels=cfg1[-1], batch_norm=batch_norm, flag=False)
        self.block3 = make_layers(cfg3, in_channels=cfg2[-1], batch_norm=batch_norm, flag=False)

        if pretrained is None or not pretrained:
            self._initialize_weights()
        else:
            self.load_weight(pretrained)

    def forward(self, x):
        feature1 = self.block1(x)
        feature2 = self.block2(feature1)
        feature3 = self.block3(feature2)

        if self.bef_neck:
            return [feature1, feature2, feature3]
        else:
            return feature3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_weight(self, weight_file):
        dic = {}
        for now_keys, values in zip(self.state_dict().keys(), torch.load(weight_file).values()):
            dic[now_keys] = values
        self.load_state_dict(dic)
