import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0):
        super(encoder, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=False),
        )
        self.pool = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.down_conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0):
        super(decoder, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout2d(dropout),
            # nn.ReLU(inplace=True),
        )
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

        )

    def forward(self, x_copy, x):
        x = self.up(x)
        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNet_Up(nn.Module):
    def __init__(self, feature_scale=1, in_channels=1, dropout=0):
        super(UNet_Up, self).__init__()
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down1 = encoder(in_channels, filters[1], dropout)
        self.down2 = encoder(filters[1], filters[2], dropout)
        self.down3 = encoder(filters[2], filters[3], dropout)

        self.middle_conv = nn.Sequential(
            nn.Conv2d(filters[3], filters[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[4]),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(filters[4], filters[4], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[4]),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

        )

        self.up1 = decoder(filters[4], filters[3], dropout)
        self.up2 = decoder(filters[3], filters[2], dropout)
        self.up3 = decoder(filters[2], filters[1], dropout)
        self.final_conv = nn.Sequential(nn.Conv2d(filters[1], filters[1], kernel_size=3, padding=1),
                                        nn.BatchNorm2d(filters[1]),
                                        # nn.Dropout2d(dropout),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(filters[1], 1, kernel_size=2, stride=2),
                                        )

    def forward(self, x):
        x = self.up(x)
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        # x4, x = self.down4(x)
        x = self.middle_conv(x)
        x = self.up1(x3, x)
        x = self.up2(x2, x)
        x = self.up3(x1, x)
        # x = self.up4(x1, x)
        x = self.final_conv(x)

        return x
