import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DICTS = {}
MODEL_DICTS.update(timm.models.__dict__)


# ==========================================
#  辅助模块
# ==========================================

class EncoderHead(nn.Module):
    """
    单个编码器分支:从共享的backbone特征中提取一个潜空间的分布参数。

    设计说明：
      - 用 1*1 conv 做通道降维，保留 8*8 的空间结构
      - 用 3*3 conv 进一步提取局部特征（对伪造痕迹更敏感）
      - AdaptiveAvgPool2d(1) 做全局平均池化，汇聚空间信息
      - 两个独立的线性头分别输出 mu 和 logvar

    Args:
        in_channels (int): backbone输出的通道数,ResNet50为2048
        hidden_dim (int): 中间特征维度
        z_dim (int): 潜变量维度
    """
    def __init__(self, in_channels=2048, hidden_dim=512, z_dim=256):
        super().__init__()

        self.conv = nn.Sequential(
            # 1×1 conv: 通道降维 2048→hidden_dim，不改变空间尺寸
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            # 3×3 conv: 提取局部特征（伪造痕迹往往是局部的高频信号）
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 全局平均池化：[B, hidden_dim, 8, 8] → [B, hidden_dim, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 输出 mu 和 logvar
        self.fc_mu     = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        """
        Args:
            x: backbone特征 [B, in_channels, H, W]
        Returns:
            mu:     [B, z_dim]
            logvar: [B, z_dim]
        """
        x = self.conv(x)            # [B, hidden_dim, H, W]
        x = self.pool(x)            # [B, hidden_dim, 1, 1]
        x = x.flatten(1)            # [B, hidden_dim]
        mu     = self.fc_mu(x)      # [B, z_dim]
        logvar = self.fc_logvar(x)  # [B, z_dim]
        return mu, logvar


class ResidualBlock(nn.Module):
    """
    解码器中使用的残差块，用于在上采样过程中保持特征质量。

    设计说明：
      残差连接让梯度更顺畅地流动，缓解深层解码器的梯度消失问题，
      同时让网络有机会学习"改进量"而非完整映射，训练更稳定。
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


# ==========================================
#  主模型：双潜空间 VAE
# ==========================================

class DualLatentVAE(nn.Module):
    """
    双潜空间VAE,将图像特征解耦为语义潜变量(z_sem)和伪造潜变量(z_forg)。

    设计目标：
      - z_sem: 编码人脸身份/语义信息，与伪造方法无关
      - z_forg: 编码伪造方法特有的痕迹，与身份无关
      - z_sem + z_forg 合起来可以重构出原图（信息完整性约束）

    Args:
        encoder (str): timm backbone名称,默认 'resnet50'
        num_classes (int): 分类头的类别数
        pretrained (bool): 是否加载预训练权重
        z_dim (int): 每个潜空间的维度(sem和forg各z_dim维)
    """
    def __init__(self,
                 encoder='resnet50',
                 num_classes=41,
                 pretrained=False,
                 **kwargs):
        super().__init__()

        self.z_sem_dim       = 256
        self.z_forg_dim      = 128
        self.num_classes = num_classes

        # ------------------------------------------------------------------
        # Backbone：ResNet50 去掉最后的 avgpool 和 fc
        # 对于 256×256 输入，输出形状为 [B, 2048, 8, 8]
        # ------------------------------------------------------------------
        _backbone = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs)
        self.backbone = nn.Sequential(*list(_backbone.children())[:-2])

        # 冻结backbone前几层（可选，提升训练稳定性）
        # 如果显存充裕且需要微调，可注释掉这段
        # for name, param in self.backbone.named_parameters():
        #     if 'layer4' not in name:   # 只训练layer4及之后
        #         param.requires_grad = False

        # ------------------------------------------------------------------
        # 编码器：两个独立分支，在backbone之后立刻分叉
        # ------------------------------------------------------------------
        self.sem_head  = EncoderHead(in_channels=2048, hidden_dim=512, z_dim=self.z_sem_dim)
        self.forg_head = EncoderHead(in_channels=2048, hidden_dim=512, z_dim=self.z_forg_dim)

        # ------------------------------------------------------------------
        # 分类器（共享，作用于 z_forg）
        # ------------------------------------------------------------------
        self.shared_classifier = nn.Sequential(
            nn.Linear(self.z_forg_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),          # 防止分类器过拟合已知类
            nn.Linear(128, num_classes),
        )

        self.decoder_input = nn.Sequential(
            nn.Linear(self.z_sem_dim + self.z_forg_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------
        # 解码器：逐级上采样 4×4 → 256×256（共6次）
        # ------------------------------------------------------------------
        def up_block(in_c, out_c, last=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            if not last:
                layers.append(ResidualBlock(out_c))
            return nn.Sequential(*layers)

        self.decoder = nn.Sequential(
            up_block(256, 256),       # 4×4   → 8×8
            up_block(256, 128),       # 8×8   → 16×16
            up_block(128, 64),        # 16×16 → 32×32
            up_block(64,  32),        # 32×32 → 64×64
            up_block(32,  16),        # 64×64 → 128×128
            up_block(16,   3, last=True),  # 128×128 → 256×256（最后一层不加残差）
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """对非backbone部分统一用 kaiming_normal_ 初始化"""
        modules_to_init = [
            self.sem_head, self.forg_head,
            self.decoder_input, self.decoder,
            self.shared_classifier,
        ]
        for module in modules_to_init:
            for layer in module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(
                        layer.weight, mode='fan_out', nonlinearity='leaky_relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)


    def encode(self, x):
        """
        提取语义和伪造的分布参数。
        Args:
            x: [B, 3, 256, 256]
        Returns:
            mu_sem, logvar_sem, mu_forg, logvar_forg
        """
        features = self.backbone(x)                          # [B, 2048, 8, 8]
        mu_sem,  logvar_sem  = self.sem_head(features)       # 各 [B, z_sem_dim]
        mu_forg, logvar_forg = self.forg_head(features)      # 各 [B, z_forg_dim]
        return mu_sem, logvar_sem, mu_forg, logvar_forg

    def reparameterize(self, mu, logvar):
        """
        重参数化采样。
        训练时:z = mu + eps * std(带随机性,支持梯度传播）
        推理时:z = mu(确定性,减少方差,与VAE标准做法一致)

        Args:
            mu:     均值 [B, z_dim]
            logvar: 对数方差 [B, z_dim]
        Returns:
            z: 采样的潜变量 [B, z_dim]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu   # 推理时用均值，更稳定

    def decode(self, z_sem, z_forg):
        """
        从两个潜变量重构图像。

        Args:
            z_sem:  语义潜变量 [B, z_sem_dim]
            z_forg: 伪造潜变量 [B, z_forg_dim]
        Returns:
            x_recon: 重构图像 [B, 3, 256, 256]
        """
        z = torch.cat([z_sem, z_forg], dim=1)   # [B, z_dim*2]
        x = self.decoder_input(z)               # [B, 256*4*4]
        x = x.view(-1, 256, 4, 4)              # [B, 256, 4, 4]
        return self.decoder(x)                  # [B, 3, 256, 256]

    def forward(self, x):
        # 完整前向传播，主要供验证/测试使用

        mu_sem, logvar_sem, mu_forg, logvar_forg = self.encode(x)

        z_sem  = self.reparameterize(mu_sem,  logvar_sem)
        z_forg = self.reparameterize(mu_forg, logvar_forg)

        x_recon   = self.decode(z_sem, z_forg)
        forg_pred = self.shared_classifier(z_forg)

        return x_recon, mu_sem, logvar_sem, mu_forg, logvar_forg, z_sem, z_forg, forg_pred

