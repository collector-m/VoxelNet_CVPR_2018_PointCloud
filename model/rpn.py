import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg

import pdb


class ConvMD(nn.Module):
    def __init__(self, M, cin, cout, k, s, p, bn = True, activation = True):
        super(ConvMD, self).__init__()

        self.M = M  # Dimension of input
        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s
        self.p = p
        self.bn = bn
        self.activation = activation

        if self.M == 2:     # 2D input
            self.conv = nn.Conv2d(self.cin, self.cout, self.k, self.s, self.p)
            if self.bn:
                self.batch_norm = nn.BatchNorm2d(self.cout)
        elif self.M == 3:   # 3D input
            self.conv = nn.Conv3d(self.cin, self.cout, self.k, self.s, self.p)
            if self.bn:
                self.batch_norm = nn.BatchNorm3d(self.cout)
        else:
            raise Exception('No such mode!')


    def forward(self, inputs):

        out = self.conv(inputs)

        if self.bn:
            out = self.batch_norm(out)

        if self.activation:
            return F.relu(out)
        else:
            return out


class Deconv2D(nn.Module):
    def __init__(self, cin, cout, k, s, p, bn = True):
        super(Deconv2D, self).__init__()

        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s
        self.p = p
        self.bn = bn

        self.deconv = nn.ConvTranspose2d(self.cin, self.cout, self.k, self.s, self.p)

        if self.bn:
            self.batch_norm = nn.BatchNorm2d(self.cout)


    def forward(self, inputs):
        out = self.deconv(inputs)

        if self.bn == True:
            out = self.batch_norm(out)

        return F.relu(out)


class MiddleAndRPN(nn.Module):
    def __init__(self, alpha = 1.5, beta = 1, sigma = 3, training = True, name = ''):
        super(MiddleAndRPN, self).__init__()

        self.middle_layer = nn.Sequential(ConvMD(3, 128, 64, 3, (2, 1, 1,), (1, 1, 1)),
                                          ConvMD(3, 64, 64, 3, (1, 1, 1), (0, 1, 1)),
                                          ConvMD(3, 64, 64, 3, (2, 1, 1), (1, 1, 1)))


        if cfg.DETECT_OBJ == 'Car':
            self.block1 = nn.Sequential(ConvMD(2, 128, 128, 3, (2, 2), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)))
        else:   # Pedestrian/Cyclist
            self.block1 = nn.Sequential(ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)))

        self.deconv1 = Deconv2D(128, 256, 3, (1, 1), (1, 1))

        self.block2 = nn.Sequential(ConvMD(2, 128, 128, 3, (2, 2), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)))

        self.deconv2 = Deconv2D(128, 256, 2, (2, 2), (0, 0))

        self.block3 = nn.Sequential(ConvMD(2, 128, 256, 3, (2, 2), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)))

        self.deconv3 = Deconv2D(256, 256, 4, (4, 4), (0, 0))

        self.prob_conv = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), bn = False, activation = False)

        self.reg_conv = ConvMD(2, 768, 14, 1, (1, 1), (0, 0), bn = False, activation = False)

        self.output_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]


    def forward(self, inputs):

        batch_size, DEPTH, HEIGHT, WIDTH, C = inputs.shape  # [batch_size, 10, 400/200, 352/240, 128]

        inputs = inputs.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)

        temp_conv = self.middle_layer(inputs)   # [batch, 64, 2, 400, 352]
        temp_conv = temp_conv.view(batch_size, -1, HEIGHT, WIDTH)   # [batch, 128, 400, 352]

        temp_conv = self.block1(temp_conv)      # [batch, 128, 200, 176]
        temp_deconv1 = self.deconv1(temp_conv)

        temp_conv = self.block2(temp_conv)      # [batch, 128, 100, 88]
        temp_deconv2 = self.deconv2(temp_conv)

        temp_conv = self.block3(temp_conv)      # [batch, 256, 50, 44]
        temp_deconv3 = self.deconv3(temp_conv)

        temp_conv = torch.cat([temp_deconv3, temp_deconv2, temp_deconv1], dim = 1)

        # Probability score map, [batch, 2, 200/100, 176/120]
        p_map = self.prob_conv(temp_conv)

        # Regression map, [batch, 14, 200/100, 176/120]
        r_map = self.reg_conv(temp_conv)

        return torch.sigmoid(p_map), r_map
