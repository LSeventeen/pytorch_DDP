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
        self.pool = nn.Conv2d(out_channels, out_channels, stride=3, kernel_size=2)

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


class UNet_Multi(nn.Module):
    def __init__(self, in_channels=1, dropout=0.1):
        super(UNet_Multi, self).__init__()
        self.down1_l = encoder(in_channels, 32, dropout)
        self.down2_l = encoder(32, 64, dropout)

        self.middle_conv_l = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

        )
        self.up1_l = decoder(128, 64, dropout)
        self.up2_l = decoder(64, 32, dropout)
        self.final_conv_l = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(32),
                                          nn.Dropout2d(dropout),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, 1, kernel_size=3, padding=1),

                                          )
        self.down1_s = encoder(in_channels, 32, dropout)
        self.down2_s = encoder(32, 64, dropout)
        self.middle_conv_s = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

        )
        self.up1_s = decoder(128, 64, dropout)
        self.up2_s = decoder(64, 32, dropout)

        self.final_conv_s = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(32),
                                          nn.Dropout2d(dropout),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, 1, kernel_size=3, padding=1),
                                          )

        self.down1 = encoder(64, 64, dropout)
        self.down2 = encoder(64, 128, dropout)

        self.middle_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

        )
        self.up1 = decoder(256, 128, dropout)
        self.up2 = decoder(128, 64, dropout)

        self.final_conv = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(32),
                                        # nn.Dropout2d(dropout),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(32, 1, kernel_size=3, padding=1)
                                        )

    def forward(self, x):
        l1, l = self.down1_l(x)
        l2, l = self.down2_l(l)
        l = self.middle_conv_l(l)
        l = self.up1_l(l2, l)
        l1 = self.up2_l(l1, l)

        l2 = self.final_conv_l(l1)

        s1, s = self.down1_s(x)
        s2, s = self.down2_s(s)
        s = self.middle_conv_s(s)
        s = self.up1_s(s2, s)
        s1 = self.up2_s(s1, s)

        s2 = self.final_conv_s(s1)

        y = torch.cat([l1, s1], dim=1)
        y1, y = self.down1(y)
        y2, y = self.down2(y)
        y = self.middle_conv(y)
        y = self.up1(y2, y)
        y = self.up2(y1, y)

        y = self.final_conv(y)

        return l2, s2, y
