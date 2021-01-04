import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0):
        super(encoder, self).__init__()

        self.res_dowm = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
        )
        self.relu = nn.ReLU()

        self.pool = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout2d(dropout),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = self.res_dowm(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + res
        x = self.relu(x)
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

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
        )
        self.res_dowm = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=1, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x_copy, x):
        x = self.up(x)
        x = torch.cat([x_copy, x], dim=1)
        res = self.res_dowm(x)
        # Concatenate
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = x + res
        x = self.relu(x)
        return x


class UNet_Res(nn.Module):
    def __init__(self, num_classes=1, in_channels=1, dropout=0):
        super(UNet_Res, self).__init__()
        self.down1 = encoder(in_channels, 64, dropout)
        self.down2 = encoder(64, 128, dropout)
        self.down3 = encoder(128, 256, dropout)
        self.down4 = encoder(256, 512, dropout)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout2d(dropout),
            nn.ReLU(inplace=True),

        )
        self.up1 = decoder(1024, 512, dropout)
        self.up2 = decoder(512, 256, dropout)
        self.up3 = decoder(256, 128, dropout)
        self.up4 = decoder(128, 64, dropout)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x = self.middle_conv(x)
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.final_conv(x)

        return x
