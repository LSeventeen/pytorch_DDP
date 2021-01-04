import torch
import torch.nn as nn


class unetConv2(nn.Module):
    def  __init__(self, in_size, out_size, n=1, ks=3, stride=1, padding=1, dropout=0):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding

        for i in range(1, n + 1):
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                 nn.BatchNorm2d(out_size),
                                 nn.Dropout2d(dropout),
                                 nn.ReLU(inplace=True), )
            setattr(self, 'conv%d' % i, conv)
            in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x


class encode(nn.Module):
    def __init__(self,in_channel,out_channel,):
        super(encode, self).__init__()
        self



class unetUp(nn.Module):
    def __init__(self, c, dropout=0):
        super(unetUp, self).__init__()
        self.conv = unetConv2(int(c / 2 + c * 2 + c), c, dropout=dropout)
        self.down = nn.Conv2d(int(c / 2), int(c / 2), stride=2, kernel_size=2)
        self.up = nn.ConvTranspose2d(c * 2, c * 2, kernel_size=2, stride=2, padding=0)

    def forward(self, down_feature, up_feature, same_feature):
        up = self.up(up_feature)
        down = self.down(down_feature)

        outputs0 = torch.cat([up, same_feature, down], 1)
        return self.conv(outputs0)


class UNet_Test(nn.Module):

    def __init__(self, n_classes=1, in_channels=1, feature_scale=2, is_ds=True, dropout=0, **_):
        super(UNet_Test, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_ds = is_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        # small vessel
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv01 = unetConv2(self.in_channels, filters[0], dropout=dropout)
        self.conv02 = unetConv2(filters[0], filters[0], dropout=dropout)
        self.conv03 = unetConv2(filters[0], filters[0], dropout=dropout)

        # downsampling
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(self.in_channels, filters[1], dropout=dropout)
        self.down1 = nn.Conv2d(filters[1], filters[1], stride=2, kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], dropout=dropout)
        self.down2 = nn.Conv2d(filters[2], filters[2], stride=2, kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], dropout=dropout)
        self.down3 = nn.Conv2d(filters[3], filters[3], stride=2, kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], dropout=dropout)

        # upsampling
        self.up_concat11 = unetUp(filters[1], dropout=dropout)
        self.up_concat21 = unetUp(filters[2], dropout=dropout)
        self.up_concat31 = unetUp(filters[3], dropout=dropout)

        self.up_concat12 = unetUp(filters[1], dropout=dropout)

        self.up_concat13 = unetUp(filters[1], dropout=dropout)

        # final conv (without any concat)
        self.final_small = nn.Conv2d(filters[0], 1, stride=2, kernel_size=2)
        self.final = nn.Conv2d(filters[1], n_classes,1)

    def forward(self, inputs):
        # column : 0
        X_01 = self.conv01(self.up(inputs)) # 32*96*96
        X_02 = self.conv02(X_01)            # 32*96*96
        X_03 = self.conv03(X_02)            # 32*96*96
        X_small = self.final_small(X_03)    # 1*96*96

        X_10 = self.conv10(inputs)          # 64*48*48
        X_20 = self.conv20(self.down1(X_10))# 128*24*24
        X_30 = self.conv30(self.down2(X_20))# 256*12*12
        X_40 = self.conv40(self.down3(X_30))# 512*6*6

        X_11 = self.up_concat11(X_01, X_20, X_10)# 64*48*48
        X_21 = self.up_concat21(X_11, X_30, X_20)# 128*24*24
        X_31 = self.up_concat31(X_21, X_40, X_30)# 256*12*12

        X_12 = self.up_concat12(X_02, X_21, X_11)# 64*48*48
        X_22 = self.up_concat22(X_12, X_31, X_21)# 128*24*24

        X_13 = self.up_concat13(X_03, X_22, X_12)

        X = self.final(X_13)

        return X_small, X
