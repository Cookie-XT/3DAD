import torch
import torch.nn as nn
from torch.nn import functional as F

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class TRA(nn.Module):

    def __init__(self, inplanes, num):

        super(TRA, self).__init__()

        self.inplanes = inplanes
        self.num = num
        self.relu = nn.ReLU(True)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        print('Build ' + self.num + ' layer TRA!')

        self.gamma_temporal = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes / 8),
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(inplanes / 8)),
            self.relu
        )
        self.gamma_temporal.apply(weights_init_kaiming)

        self.beta_temporal = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=int(inplanes / 8),
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int(inplanes / 8)),
            self.relu
        )
        self.beta_temporal.apply(weights_init_kaiming)

        self.gg_temporal = nn.Sequential(
            nn.Conv2d(in_channels=98, out_channels=128,#自己
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
        )
        self.gg_temporal.apply(weights_init_kaiming)

        self.tte_para = nn.Sequential(
            nn.Conv2d(in_channels=2 * 128, out_channels=128,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu,
        )
        self.tte_para.apply(weights_init_kaiming)

        self.te_para = nn.Sequential(
            nn.Conv2d(in_channels=2 * 128, out_channels=1,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.te_para.apply(weights_init_kaiming)

        self.theta_channel = nn.Sequential(
            nn.Conv1d(in_channels=inplanes, out_channels=int(inplanes / 8),
                      kernel_size=1, stride=1, padding=0, bias=False),
            self.relu,
        )
        self.theta_channel.apply(weights_init_kaiming)

        self.channel_para = nn.Sequential(
            nn.Linear(in_features=int(inplanes / 4), out_features=int(inplanes / 8)),
            self.relu,
            nn.Linear(in_features=int(inplanes / 8), out_features=inplanes),
            nn.Sigmoid()
        )
        self.channel_para.apply(weights_init_kaiming)

    def forward(self, featmap, re_featmap, vect_featmap, embed_feat):

        b, t, c, h, w = featmap.size()
        #print(featmap.size(),'featmap.size()1')#torch.Size([8, 8, 1024, 7, 7]) featmap.size()1

        gamma_feat = self.gamma_temporal(re_featmap).view(b, t, -1, h * w)
        #print('gamma_feat',gamma_feat.shape)(8,8,128,49)

        beta_feat = self.beta_temporal(re_featmap).view(b, t, -1, h * w)
        # print('beta_feat',beta_feat.shape)torch.Size([8, 8, 128, 49])

        channel_para = self.theta_channel(vect_featmap.permute(0, 2, 1))
        # print('channel_para',channel_para.shape)torch.Size([8, 128, 8])
        gap_feat_map0 = []

        for idx in range(0, t, 2):
            # print(idx)
            #print(channel_para[:, :, idx].shape,channel_para[:, :, idx + 1].shape)
            # torch.Size([8, 128]) torch.Size([8, 128])
            para0 = torch.cat((channel_para[:, :, idx], channel_para[:, :, idx + 1]), 1)
            # torch.Size([8, 256])
            para_00 = self.channel_para(para0).view(b, -1, 1, 1)
            # print('para_00',para_00.shape) torch.Size([8, 1024, 1, 1])
            para1 = torch.cat((channel_para[:, :, idx + 1], channel_para[:, :, idx]), 1)
            para_01 = self.channel_para(para1).view(b, -1, 1, 1)

            embed_feat0 = embed_feat[:, idx, :, :, :]
            # print('embed_feat0.shape',embed_feat0.shape) torch.Size([8, 128, 7, 7])
            embed_feat1 = embed_feat[:, idx + 1, :, :, :]

            gamma_feat0 = gamma_feat[:, idx, :, :].permute(0, 2, 1)
            # print('gamma_feat0',gamma_feat0.shape)  torch.Size([8, 49, 128])
            beta_feat0 = beta_feat[:, idx + 1, :, :]
            # print('beta_feat0',beta_feat0.shape) torch.Size([8, 128, 49])
            Gs0 = torch.matmul(gamma_feat0, beta_feat0)

            Gs_in0 = Gs0.permute(0, 2, 1).view(b, h * w, h, w)
            #print(Gs_in0.shape,'Gs_in0.shape')#torch.Size([8, 49, 7, 7]) Gs_in0.shape
            Gs_out0 = Gs0.view(b, h * w, h, w)
 
            gamma_feat1 = gamma_feat[:, idx + 1, :, :].permute(0, 2, 1)
            beta_feat1 = beta_feat[:, idx, :, :]
            Gs1 = torch.matmul(gamma_feat1, beta_feat1)
            Gs_in1 = Gs1.permute(0, 2, 1).view(b, h * w, h, w)
            Gs_out1 = Gs1.view(b, h * w, h, w)
            #print(Gs_out1.shape,'Gs_out1.shape' )#torch.Size([8, 49, 7, 7]) Gs_out1.shape

            Gs_joint0 = torch.cat((Gs_in0, Gs_out1), 1)
            #print(Gs_joint0.shape,'Gs_joint0.shape')#torch.Size([4, 98, 7, 7]) Gs_joint0.shape

            Gs_joint0 = self.gg_temporal(Gs_joint0)
            para_alpha = self.tte_para(torch.cat((embed_feat0, embed_feat1), 1))
            para_alpha = self.te_para(torch.cat((para_alpha, Gs_joint0), 1))

            Gs_joint1 = torch.cat((Gs_in1, Gs_out0), 1)
            Gs_joint1 = self.gg_temporal(Gs_joint1)
            para_beta = self.tte_para(torch.cat((embed_feat1, embed_feat0), 1))
            para_beta = self.te_para(torch.cat((para_beta, Gs_joint1), 1))

            para_00 = para_00 * para_alpha
            para_01 = para_01 * para_beta

            gap_map0 = para_00 * featmap[:, idx, :, :, :] + para_01 * featmap[:, idx + 1, :, :, :]
            gap_map0 = self.relu(gap_map0)
            gap_map0 = gap_map0 ** 2
            gap_feat_map0.append(gap_map0)

        gap_feat_map0 = torch.stack(gap_feat_map0, 1)
        torch.cuda.empty_cache()

        return gap_feat_map0
