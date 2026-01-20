import numbers
import os
import random
from datetime import datetime
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

from Src.utils.Dataloader import get_loader, test_dataset
from utils.utils import clip_gradient
from lib.pvtv2 import pvt_v2_b2
class LMSFNet(nn.Module):
    """
    - backbone: pvt_v2_b2，输出 x1,x2,x3,x4
    - Translayer*：统一到 fuse_channels 维度
    - MSSEP：x1->x2, x1->x3 的小目标增强
    - 融合：x1, x2', x3', x4' 上采样到同一尺寸 concat，再 conv 融合 -> seg_head 输出
    - 最后再上采样到输入大小
    """

    def __init__(self,
                 pretrained_pvt_path=None,
                 out_channels=1,
                 fuse_channels=64,
                 use_mssep=True):
        super().__init__()
        self.backbone = pvt_v2_b2()
        self.use_mssep = use_mssep
        self.fuse_channels = fuse_channels

        # --- 加载 PVT 预训练 ---
        if pretrained_pvt_path is not None:
            try:
                state_dict = torch.load(pretrained_pvt_path, map_location="cpu")
                model_dict = self.backbone.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                model_dict.update(state_dict)
                self.backbone.load_state_dict(model_dict)
                print(f"[Backbone] Loaded PVTv2-B2 weights from {pretrained_pvt_path}")
            except Exception as e:
                print(f"[Backbone] WARNING: failed to load PVTv2-B2 weights: {e}")

        # pvt_v2_b2 输出通道: [64, 128, 320, 512]
        self.trans1 = nn.Conv2d(64, fuse_channels, kernel_size=1)
        self.trans2 = nn.Conv2d(128, fuse_channels, kernel_size=1)
        self.trans3 = nn.Conv2d(320, fuse_channels, kernel_size=1)
        self.trans4 = nn.Conv2d(512, fuse_channels, kernel_size=1)

        # MSSEP（只在 use_mssep=True 的时候用）
        if self.use_mssep:
            self.ssep_2 = MS_SSEP(fuse_channels, fuse_channels)
            self.ssep_3 = MS_SSEP(fuse_channels, fuse_channels)

        # 多尺度融合：4 个尺度 concat 后卷积
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fuse_channels * 5, fuse_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fuse_channels, fuse_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.intra = Interaction(64 * 4, 4, False)
        self.glcf = GLCF(64)
        # segmentation head
        self.seg_head = nn.Conv2d(fuse_channels, out_channels, kernel_size=1)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]

        # 1) backbone 提取多尺度特征
        x1, x2, x3, x4 = self.backbone(x)  # [B,64,H/4,W/4], [B,128,H/8,W/8]...

        # 2) 通道统一
        x1_t = self.trans1(x1)  # [B,C,H/4,W/4]
        x2_t = self.trans2(x2)  # [B,C,H/8,W/8]
        x3_t = self.trans3(x3)  # [B,C,H/16,W/16]
        x4_t = self.trans4(x4)  # [B,C,H/32,W/32]

        # 3) MSSEP 小目标增强
        if self.use_mssep:
            # x1 -> x2
            x1_down2 = F.interpolate(x1_t, size=x2_t.shape[2:], mode='bilinear', align_corners=True)
            x2_t = x2_t + self.ssep_2(x1_down2, x2_t)

            # x1 -> x3
            x1_down3 = F.interpolate(x1_t, size=x3_t.shape[2:], mode='bilinear', align_corners=True)
            x3_t = x3_t + self.ssep_3(x1_down3, x3_t)

        # 4) 多尺度交互：全部上采样到 x1_t 的分辨率，再 concat
        x_qkv = self.intra(torch.cat((x1, F.interpolate(x2_t, size=x1.size()[2:], mode='bilinear'),
                                      F.interpolate(x3_t, size=x1.size()[2:], mode='bilinear'),
                                      F.interpolate(x4_t, size=x1.size()[2:], mode='bilinear')), 1))
	#5)上下文融合
        x_share = self.glcf(x4_t, x3_t, x2_t, x1)
        x_share = x_qkv + x_share
        target_size = x1_t.shape[2:]  # H/4, W/4
        f1 = x1_t
        f2 = F.interpolate(x2_t, size=target_size, mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3_t, size=target_size, mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4_t, size=target_size, mode='bilinear', align_corners=True)

        feats_cat = torch.cat([f1, f2, f3, f4, x_share], dim=1)  # [B, 4C, H/4, W/4]
        fused = self.fuse_conv(feats_cat)  # [B, C, H/4, W/4]

        # 6) 输出预测 + 上采样到输入大小
        logits = self.seg_head(fused)  # [B, 1, H/4, W/4]
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)  # [B,1,H,W]

        return logits
