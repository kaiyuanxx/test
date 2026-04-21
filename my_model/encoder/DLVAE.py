import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

MODEL_DICTS = {}
MODEL_DICTS.update(timm.models.__dict__)

# ==========================================
#    双潜空间 VAE 模型
# ==========================================
class DualLatentVAE(nn.Module):
    def __init__(self, 
                 encoder,
                 num_classes=41,
                 pretrained=False,
                 in_channels=2048,
                 z_dim=256,
                 **kwargs):
        super().__init__()

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.backbone = MODEL_DICTS[encoder](pretrained=pretrained, **kwargs) 

        # 去掉 ResNet50 最后的全连接层和池化层，输出维度为 [B, 2048, 8, 8]
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        modules = []
        hidden_dims = [1024, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 4, stride= 2, padding  = 1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder_conv = nn.Sequential(*modules)  #[B, 2048, 8, 8] --> [B, 512, 2, 2]

        self.fc = nn.Linear(hidden_dims[-1]*4, 512)
        # 语义潜空间 (Semantic)
        self.fc_mu_sem = nn.Linear(512, z_dim)
        self.fc_var_sem = nn.Linear(512, z_dim)
        
        # 伪造潜空间 (Forgery)
        self.fc_mu_forg = nn.Linear(512, z_dim)
        self.fc_var_forg = nn.Linear(512, z_dim)

        # --- 参数共享的分类器 ---
        self.shared_classifier = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
        # --- 解码器 ---
        # 接收 z_sem 和 z_forg 拼接后的向量，重构 3x256x256 的图像
        self.decoder_input = nn.Linear(z_dim * 2, 512 * 8 * 8) 
        
        self.decoder = nn.Sequential(
            # 输入: 512 x 8 x 8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(256), nn.LeakyReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(128), nn.LeakyReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(64), nn.LeakyReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 128x128
            nn.BatchNorm2d(32), nn.LeakyReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 【最终输出】: 256x256
            nn.Tanh() 
        )
    
    def encode(self, x):
        features = self.backbone(x)   # [Batch, 2048, 8, 8]
        features = self.encoder_conv(features) # [Batch, 512, 2, 2]
        features = torch.flatten(features, start_dim=1) # [Batch, 512*4]
        features = self.fc(features) # [Batch, 512]

        mu_sem = self.fc_mu_sem(features)
        logvar_sem = self.fc_var_sem(features)
        mu_forg = self.fc_mu_forg(features)
        logvar_forg = self.fc_var_forg(features)

        return mu_sem, logvar_sem, mu_forg, logvar_forg

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z_sem, z_forg):
        z_combined = torch.cat([z_sem, z_forg], dim=1)
        dec_in = self.decoder_input(z_combined)
        dec_in = dec_in.view(-1, 512, 8, 8)
        return self.decoder(dec_in)


    def forward(self, x):
        # 提供一个完整的前向传播基础接口，主要给验证/测试使用
        mu_sem, logvar_sem, mu_forg, logvar_forg = self.encode(x)
        
        z_sem = self.reparameterize(mu_sem, logvar_sem)
        z_forg = self.reparameterize(mu_forg, logvar_forg)
        
        x_recon = self.decode(z_sem, z_forg)
        
        adv_preds = self.shared_classifier(z_sem)
        forg_preds = self.shared_classifier(z_forg)
    
        return x_recon, mu_sem, logvar_sem, mu_forg, logvar_forg, adv_preds, forg_preds, z_sem, z_forg
    
    
    
    
