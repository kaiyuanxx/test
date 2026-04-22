import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

from loguru import logger
from my_model.algorithm import SLModel
from my_model.utils import AverageMeter, update_meter



# 计算信息熵
def entropy_loss(preds):
    p = F.softmax(preds, dim=1)
    entropy = -torch.mean(torch.sum(p * torch.log(p + 1e-8), dim=1))
    return entropy

# 冻结与解冻参数
def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


class TestNet(SLModel):
    def __init__(self, args):
        super().__init__()
    
        # 定义损失函数
        self.criterion_recon = nn.L1Loss()
        self.criterion_ce = nn.CrossEntropyLoss()  

        self.w_recon = args.loss.recon_loss
        self.w_kl = args.loss.kl_loss
        self.w_adv = args.loss.adv_loss
        self.w_cls = args.loss.cls_loss
        # 半监督一致性损失权重 & 伪标签阈值
        self.w_consist_high = 1.0  
        self.w_consist_low = 1.0   
        self.pseudo_threshold = 0.95
    

    def get_loss_names(self):
        loss_name = {
            'total_loss',
            'recon_loss',
            'kl_loss',
            'adv_loss',
            'cls_loss',
            'consist_high_loss',
            'consist_low_loss'
        }
        return loss_name

    def on_train_epoch_start(self):

        self.train_losses = {loss: AverageMeter() for loss in self.get_loss_names()}

    def training_step(self, batch, batch_idx):

        # Log current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=True, on_epoch=False, prog_bar=True)

        loss_map = {}

        images = batch['image']
        targets = batch['target']
        # MergedDataset 会附加一个 tag (1为有标签, 2为无标签)
        tags = batch['tag'].squeeze() 

        # 区分有标签和无标签掩码
        labeled_mask = (tags == 1)
        unlabeled_mask = (tags == 2)

        # =================================================================
        # 第一阶段：全局前向传播与全局损失 (适用于 Batch 中所有样本)
        # =================================================================
        # 1. 编码 & 采样 (所有样本)
        mu_sem, logvar_sem, mu_forg, logvar_forg = self.encoder.encode(images)
        z_sem = self.encoder.reparameterize(mu_sem, logvar_sem)
        z_forg = self.encoder.reparameterize(mu_forg, logvar_forg)

        # 2. 重构图像
        x_recon = self.encoder.decode(z_sem, z_forg)
        
        # 3. 计算全局重构损失
        recon_loss = self.criterion_recon(x_recon, images)
        
        # 4. 计算全局 KL 散度 
        # (使用 dim=1 求和特征维，然后对 batch 维求均值，这样更稳定)
        kl_sem = -0.5 * torch.mean(torch.sum(1 + logvar_sem - mu_sem.pow(2) - logvar_sem.exp(), dim=1))
        kl_forg = -0.5 * torch.mean(torch.sum(1 + logvar_forg - mu_forg.pow(2) - logvar_forg.exp(), dim=1))
        kl_loss = kl_sem + kl_forg

        # 5. 计算全局对抗损失 (语义分支冻结分类器，最大化所有样本的分类熵)
        set_requires_grad(self.encoder.shared_classifier, False)
        pred_sem = self.encoder.shared_classifier(z_sem)
        adv_loss = -entropy_loss(pred_sem)
        set_requires_grad(self.encoder.shared_classifier, True) # 恢复解冻


        # =================================================================
        # 第二阶段：分离有标签数据 (全监督分类)
        # =================================================================
        if labeled_mask.sum() > 0:
            # 提取有标签样本的伪造潜变量和对应标签
            z_forg_l = z_forg[labeled_mask]
            y_l = targets[labeled_mask]

            # 正常交叉熵优化共享分类器
            pred_forg_l = self.encoder.shared_classifier(z_forg_l)
            cls_loss = self.criterion_ce(pred_forg_l, y_l)

        # =================================================================
        # 第三阶段：分离无标签数据 (伪标签打靶 + 跨样本换脸一致性)
        # =================================================================
        if unlabeled_mask.sum() > 0:
            # 提取无标签样本的分布参数和当前采样的伪造潜变量
            mu_sem_u = mu_sem[unlabeled_mask]
            logvar_sem_u = logvar_sem[unlabeled_mask]
            mu_forg_u = mu_forg[unlabeled_mask]
            logvar_forg_u = logvar_forg[unlabeled_mask]
            z_forg_u = z_forg[unlabeled_mask]
            
            # 预测类别并计算置信度（不回传分类器梯度以防崩坏）
            with torch.no_grad():
                pred_forg_u = self.encoder.shared_classifier(z_forg_u)
                probs_u = F.softmax(pred_forg_u, dim=1)
                max_probs, pseudo_labels = torch.max(probs_u, dim=1)
                
            high_conf_mask = max_probs >= self.pseudo_threshold
            low_conf_mask = max_probs < self.pseudo_threshold

            # (A) 高置信度样本：基于伪标签重新采样对齐分布
            if high_conf_mask.sum() > 0:
                # 在高置信度样本的分布中*重新采样*一个潜变量
                z_forg_u_high = self.encoder.reparameterize(
                    mu_forg_u[high_conf_mask], logvar_forg_u[high_conf_mask])
                
                pred_high = self.encoder.shared_classifier(z_forg_u_high)
                loss_consist_high = self.criterion_ce(pred_high, pseudo_labels[high_conf_mask])

            # (B) 低置信度样本：基于语义扰动的伪造特征一致性
            if low_conf_mask.sum() > 0:
                # 1. 获取低置信样本自身的语义和伪造分布并重新采样
                z_sem_self = self.encoder.reparameterize(mu_sem_u[low_conf_mask], logvar_sem_u[low_conf_mask])
                z_forg_self = z_forg_u[low_conf_mask]  

                # 2. 随机打乱语义特征，模拟“换脸”
                idx_shuffle = torch.randperm(z_sem_self.size(0), device=images.device)
                z_sem_other = z_sem_self[idx_shuffle]
                
                # 3. 解码得到 原图重构 和 换脸重构
                x_orig = self.encoder.decode(z_sem_self, z_forg_self)
                x_swap = self.encoder.decode(z_sem_other, z_forg_self)
                
                # 4. 重新送入编码器提取新的伪造分布
                _, _, mu_f_orig, logvar_f_orig = self.encoder.encode(x_orig)
                _, _, mu_f_swap, logvar_f_swap = self.encoder.encode(x_swap)
                
                z_f_orig = self.encoder.reparameterize(mu_f_orig, logvar_f_orig)
                z_f_swap = self.encoder.reparameterize(mu_f_swap, logvar_f_swap)
                
                # 5. 约束预测输出概率一致
                pred_orig = self.encoder.shared_classifier(z_f_orig)
                pred_swap = self.encoder.shared_classifier(z_f_swap)
                
                loss_consist_low = F.mse_loss(F.softmax(pred_orig, dim=1), F.softmax(pred_swap, dim=1))
                
                total_loss = self.w_recon * recon_loss + self.w_kl * kl_loss + \
                             self.w_adv * adv_loss + self.w_cls * cls_loss + \
                             self.w_consist_high * loss_consist_high + self.w_consist_low * loss_consist_low
                
        loss_map['total_loss'] = total_loss
        loss_map['recon_loss'] = recon_loss
        loss_map['kl_loss'] = kl_loss
        loss_map['adv_loss'] = adv_loss
        loss_map['cls_loss'] = cls_loss
        if unlabeled_mask.sum() > 0:
            if high_conf_mask.sum() > 0:
                loss_map['consist_high_loss'] = loss_consist_high
            else:
                loss_map['consist_high_loss'] = torch.tensor(0.0, device=images.device)
            
            if low_conf_mask.sum() > 0:
                loss_map['consist_low_loss'] = loss_consist_low
            else:
                loss_map['consist_low_loss'] = torch.tensor(0.0, device=images.device)
        else:
            loss_map['consist_high_loss'] = torch.tensor(0.0, device=images.device)
            loss_map['consist_low_loss'] = torch.tensor(0.0, device=images.device)

        
        for key, value in loss_map.items():
            update_meter(
                self.train_losses[key], value, self.args.train.batch_size)
        for ls in self.train_losses.values():
            self.log(ls.name, ls.avg, on_step=True, prog_bar=True)
            
        return total_loss
    
    def on_train_epoch_end(self):
        """Log averaged training losses for the current epoch."""
        results = {key: meter.avg for key, meter in self.train_losses.items()}
        logger.info(results, step=self.current_epoch)

