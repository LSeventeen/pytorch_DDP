import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(encoder, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Conv2d(out_channels, out_channels, stride=2, kernel_size=2)

    def forward(self, x):
        x = self.down_conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(self, x_copy, x):
        x = self.up(x)
        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNet_Oetd(nn.Module):
    def __init__(self, in_channels=1, feature_scale=2, dropout=0.1):
        super(UNet_Oetd, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.down1_f = encoder(in_channels, filters[1], dropout)
        self.down2_f = encoder(filters[1], filters[2], dropout)

        self.middle_conv_f = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

        )

        self.up1_l = decoder(filters[3], filters[2], dropout)
        self.up2_l = decoder(filters[2], filters[1], dropout)
        self.final_conv_l = nn.Sequential(nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
                                          nn.BatchNorm2d(filters[1]),
                                          # nn.Dropout2d(dropout),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(filters[1], 1, kernel_size=1),
                                          )

        self.up1_s = decoder(filters[3], filters[2], dropout)
        self.up2_s = decoder(filters[2], filters[1], dropout)
        self.final_conv_s = nn.Sequential(nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
                                          nn.BatchNorm2d(filters[1]),
                                          # nn.Dropout2d(dropout),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(filters[1], 1, kernel_size=1),
                                          )

        self.down1 = encoder(3, filters[1], dropout)
        self.down2 = encoder(filters[1], filters[2], dropout)

        self.middle_conv = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(filters[3], filters[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

        )

        self.up1 = decoder(filters[3], filters[2], dropout)
        self.up2 = decoder(filters[2], filters[1], dropout)
        self.final_conv = nn.Sequential(nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
                                        nn.BatchNorm2d(filters[1]),
                                        # nn.Dropout2d(dropout),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(filters[1], 1, kernel_size=1),
                                        )

    def forward(self, x):
        f1, f = self.down1_f(x)
        f2, f = self.down2_f(f)

        f = self.middle_conv_f(f)

        l = self.up1_l(f2, f)
        l1 = self.up2_l(f1, l)
        l2 = self.final_conv_l(l1)
        

        s = self.up1_s(f2, f)
        s1 = self.up2_s(f1, s)

        s2 = self.final_conv_s(s1)

        y = torch.cat([l2, s2, x], dim=1)
        y1, y = self.down1(y)
        y2, y = self.down2(y)
        y = self.middle_conv(y)
        y = self.up1(y2, y)
        y = self.up2(y1, y)

        y = self.final_conv(y)

        return l2, s2, y
