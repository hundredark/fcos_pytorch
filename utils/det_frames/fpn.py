import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, features=256, backbone="resnet50"):
        super(FPN, self).__init__()

        if backbone == "resnet50":
            print("INFO: using resnet50 backbone")
            self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)
            self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
            self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        elif backbone == "darknet19":
            print("INFO: using darnet19 backbone")
            self.prj_5 = nn.Conv2d(1024, features, kernel_size=1)
            self.prj_4 = nn.Conv2d(512, features, kernel_size=1)
            self.prj_3 = nn.Conv2d(256, features, kernel_size=1)
        # elif backbone == "slimdarknet19":
        #     print("INFO: using slimdarnet19 backbone")
        #     self.prj_5 = nn.Conv2d(717, features, kernel_size=1)
        #     self.prj_4 = nn.Conv2d(358, features, kernel_size=1)
        #     self.prj_3 = nn.Conv2d(179, features, kernel_size=1)
        else:
            raise ValueError("arg 'backbone' only support 'resnet50' or 'darknet19'")

        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)

        self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)

        self.apply(self.init_conv_kaiming)

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x

        # lateral connection
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        # feature fusion
        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])

        # smooth after fusion
        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)

        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3, P4, P5, P6, P7]

