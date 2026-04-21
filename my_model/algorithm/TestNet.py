import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from loguru import logger
from my_model.algorithm import SLModel



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
    

    def get_loss_names(self):
        loss_name = {
            'total_loss',
            'recon_loss',
            'kl_loss',
            'adv_loss',
            'cls_loss',
        }
        return loss_name

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        images = batch['image']
        targets = batch['target']
        # MergedDataset 会附加一个 tag (1为有标签, 2为无标签)
        tags = batch['tag'].squeeze() 

        # 区分有标签和无标签掩码
        labeled_mask = (tags == 1)
        unlabeled_mask = (tags == 2)
        
        total_loss = 0.0

        # ==========================================
        # 1. 有标签数据 (监督学习 + 对抗学习)
        # ==========================================
        if labeled_mask.sum() > 0:
            x_l = images[labeled_mask]
            y_l = targets[labeled_mask]

            mu_sem_l, logvar_sem_l, mu_forg_l, logvar_forg_l = self.encoder.encode(x_l)
            z_sem_l = self.encoder.reparameterize(mu_sem_l, logvar_sem_l)
            z_forg_l = self.encoder.reparameterize(mu_forg_l, logvar_forg_l)

            # 重构与 KL Loss
            x_recon_l = self.encoder.decode(z_sem_l, z_forg_l)
            recon_loss = self.criterion_recon(x_recon_l, x_l)
            
            kl_sem = -0.5 * torch.sum(1 + logvar_sem_l - mu_sem_l.pow(2) - logvar_sem_l.exp()) / x_l.size(0)
            kl_forg = -0.5 * torch.sum(1 + logvar_forg_l - mu_forg_l.pow(2) - logvar_forg_l.exp()) / x_l.size(0)
            kl_loss = kl_sem + kl_forg

            # 伪造潜空间分类 (正常优化分类器)
            set_requires_grad(self.encoder.shared_classifier, True)
            pred_forg_l = self.encoder.shared_classifier(z_forg_l)
            cls_loss = self.criterion_ce(pred_forg_l, y_l)

            # 语义潜空间分类 (冻结分类器，最大化分类熵)
            set_requires_grad(self.encoder.shared_classifier, False)
            pred_sem_l = self.encoder.shared_classifier(z_sem_l)
            # 最小化负熵 = 最大化熵
            adv_loss = -entropy_loss(pred_sem_l)
            
            # 恢复分类器梯度
            set_requires_grad(self.encoder.shared_classifier, True)

            labeled_loss = (self.w_recon * recon_loss + self.w_kl * kl_loss + 
                            self.w_cls * cls_loss + self.w_adv * adv_loss)
            total_loss += labeled_loss

            self.log('t_recon_loss', recon_loss)
            self.log('t_cls_loss', cls_loss)

        # ==========================================
        # 2. 无标签数据 (半监督：伪标签 & 换脸一致性)
        # ==========================================
        if unlabeled_mask.sum() > 0:
            x_u = images[unlabeled_mask]
            
            # 编码获取无标签数据的分布
            mu_sem_u, logvar_sem_u, mu_forg_u, logvar_forg_u = self.encoder.encode(x_u)
            z_forg_u = self.encoder.reparameterize(mu_forg_u, logvar_forg_u)
            
            # 预测类别并计算置信度
            with torch.no_grad():
                pred_forg_u = self.encoder.shared_classifier(z_forg_u)
                probs_u = F.softmax(pred_forg_u, dim=1)
                max_probs, pseudo_labels = torch.max(probs_u, dim=1)
                
            high_conf_mask = max_probs >= self.pseudo_threshold
            low_conf_mask = max_probs < self.pseudo_threshold

            # (A) 高置信度样本: 多次采样一致性
            if high_conf_mask.sum() > 0:
                # 再次在分布中采样潜变量
                z_forg_u_high = self.encoder.reparameterize(
                    mu_forg_u[high_conf_mask], logvar_forg_u[high_conf_mask])
                pred_high = self.encoder.shared_classifier(z_forg_u_high)
                
                # 使新的预测逼近伪标签 (交叉熵)
                loss_consist_high = self.criterion_ce(pred_high, pseudo_labels[high_conf_mask])
                total_loss += self.w_consist_high * loss_consist_high
                self.log('t_consist_high', loss_consist_high)

            # (B) 低置信度样本: 换脸 (Swap) 一致性
            if low_conf_mask.sum() > 0:
                # 1. 获取自身的语义与伪造潜变量
                mu_s_low = mu_sem_u[low_conf_mask]
                logvar_s_low = logvar_sem_u[low_conf_mask]
                z_sem_self = self.encoder.reparameterize(mu_s_low, logvar_s_low)
                
                mu_f_low = mu_forg_u[low_conf_mask]
                logvar_f_low = logvar_forg_u[low_conf_mask]
                z_forg_self = self.encoder.reparameterize(mu_f_low, logvar_f_low)
                
                # 2. 从当前 batch 随机采样另一组语义潜变量 (模拟换脸)
                idx_shuffle = torch.randperm(z_sem_self.size(0))
                z_sem_other = z_sem_self[idx_shuffle]
                
                # 3. 拼合并解码得到 "原表示" 和 "换脸表示"
                x_orig = self.encoder.decode(z_sem_self, z_forg_self)
                x_swap = self.encoder.decode(z_sem_other, z_forg_self)
                
                # 4. 将两组图像重新送入网络
                _, _, mu_f_orig, logvar_f_orig = self.encoder.encode(x_orig)
                _, _, mu_f_swap, logvar_f_swap = self.encoder.encode(x_swap)
                
                z_f_orig = self.encoder.reparameterize(mu_f_orig, logvar_f_orig)
                z_f_swap = self.encoder.reparameterize(mu_f_swap, logvar_f_swap)
                
                pred_orig = self.encoder.shared_classifier(z_f_orig)
                pred_swap = self.encoder.shared_classifier(z_f_swap)
                
                # 5. 约束两次分类结果一致 (均方误差约束概率分布)
                loss_consist_low = F.mse_loss(F.softmax(pred_orig, dim=1), F.softmax(pred_swap, dim=1))
                total_loss += self.w_consist_low * loss_consist_low
                self.log('t_consist_low', loss_consist_low)

        self.log('t_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss


    def on_validation_epoch_end(self):
        metrics = self.trainer.callback_metrics
        
        epoch = self.current_epoch
        val_recon = metrics.get('val_recon', 0.0)
        val_adv_acc = metrics.get('val_adv_acc', 0.0)
        val_cls_acc = metrics.get('val_cls_acc', 0.0)
        #score = metrics.get('score', 0.0)

        log_msg = (f"Epoch {epoch:03d} Validation End| "
                    f"Recon_loss: {val_recon:.6f} | "
                    f"Adv_Acc: {val_adv_acc:.4f} | "
                    f"Cls_Acc: {val_cls_acc:.4f}")
        
        logger.success(log_msg) 


    def on_test_start(self):
        self.vis_dir = os.path.join(self.args.exam_dir, "visualizations")
        os.makedirs(self.vis_dir, exist_ok=True)
        logger.info(f"测试可视化结果将保存在: {self.vis_dir}")
        self.test_outputs = []
 
    def test_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target']
        
        # 1. 解包 7 个返回值
        x_recon, _s, _, _, _, adv_preds, forg_preds, z_sem, z_forg = self.encoder(image)
        
        # 2. 计算指标
        recon_loss = self.criterion_recon(x_recon, image)
        preds = forg_preds.argmax(dim=1)
        cls_acc = (preds == target).float().mean()
        adv_acc = (adv_preds.argmax(dim=1) == target).float().mean()

        self.test_outputs.append({
            'targets': target.cpu(),
            'preds': preds.cpu()
        })

        self.log('test_recon', recon_loss, sync_dist=True)
        self.log('test_cls_acc', cls_acc, sync_dist=True)
        self.log('test_adv_acc', adv_acc, sync_dist=True)

        if batch_idx <= 10:
            with torch.no_grad():
                # 消融 1: 抹除伪造信息 (Z_forg 置零)，得到纯净语义图
                x_no_forg = self.encoder.decode(z_sem, torch.zeros_like(z_forg))
                
                # 消融 2: 抹除语义信息 (Z_sem 置零)，观察仅剩伪造特征时的重构
                x_no_sem = self.encoder.decode(torch.zeros_like(z_sem), z_forg)
            self._save_visual_results(image, x_recon, x_no_forg, x_no_sem, batch_idx)

        return {'test_recon': recon_loss, 'test_cls_acc': cls_acc, 'test_adv_acc': adv_acc}
        


    def on_test_epoch_end(self):
        all_targets = torch.cat([x['targets'] for x in self.test_outputs]).numpy()
        all_preds = torch.cat([x['preds'] for x in self.test_outputs]).numpy()

        df = pd.DataFrame({
            'True_Label': all_targets,
            'Predicted_Label': all_preds
        })
        csv_path = os.path.join(self.args.exam_dir, "test_predictions.csv")
        df.to_csv(csv_path, index=False)
        logger.success(f"预测结果已保存至: {csv_path}")
        self._plot_confusion_matrix(all_targets, all_preds)


    def _plot_confusion_matrix(self, y_true, y_pred):
        """生成并保存混淆矩阵热力图"""
        cm_path = os.path.join(self.args.exam_dir, "test_confusion_matrix.png")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 设置画板大小
        plt.figure(figsize=(16, 14))
        
        # 使用 seaborn 画热力图
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='d', cbar=True)
        
        plt.title('Disentanglement Analysis: Confusion Matrix', fontsize=18)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
        plt.close()
        logger.success(f"混淆矩阵保存至: {cm_path}")


    def _save_visual_results(self, orig, recon, x_no_forg, x_no_sem, batch_idx):
        n_imgs = min(8, orig.size(0))
        
        # 动态命名
        filename = f"batch_{batch_idx:02d}.png"
        save_path = os.path.join(self.vis_dir, filename)
        
        sem_attn_maps = []
        forg_attn_maps = []
        
        for i in range(n_imgs):
            # 计算语义关注点：全重构 vs 无语义重构 的差值
            sem_diff = torch.mean(torch.abs(recon[i] - x_no_sem[i]), dim=0, keepdim=True)
            sem_attn_maps.append(self._generate_heatmap_overlay(sem_diff, orig[i]))
            
            # 计算伪造关注点：全重构 vs 无伪造重构 的差值
            forg_diff = torch.mean(torch.abs(recon[i] - x_no_forg[i]), dim=0, keepdim=True)
            forg_attn_maps.append(self._generate_heatmap_overlay(forg_diff, orig[i]))

        # 拼接 5 行数据
        # 1. 原始图像
        # 2. 完整重构 (Recon)
        # 3. 语义关注热力图 (Z_sem Attention) -> 应该亮在五官
        # 4. 伪造关注热力图 (Z_forg Attention) -> 应该亮在伪造瑕疵
        # 5. 纯净语义重构 (Recon without Forgery) -> 理想状态下的“去伪存真”脸
        comparison = torch.cat([
            orig[:n_imgs].cpu(),
            recon[:n_imgs].cpu(),
            torch.stack(sem_attn_maps).cpu(),
            torch.stack(forg_attn_maps).cpu(),
            x_no_forg[:n_imgs].cpu()
        ])

        grid = torchvision.utils.make_grid(comparison, nrow=n_imgs, normalize=True, padding=4, pad_value=1.0)
        torchvision.utils.save_image(grid, save_path)

    def _generate_heatmap_overlay(self, activation_map, img_single):
        """通用热力图生成函数 (自动处理不同的空间尺寸)"""
        # 确保 activation_map 是 2D 的 [H, W]
        if activation_map.dim() == 3:
            activation_map = activation_map.squeeze(0)

        # Min-Max 归一化
        act_min, act_max = activation_map.min(), activation_map.max()
        if act_max - act_min > 1e-8:
            activation_map = (activation_map - act_min) / (act_max - act_min)
        else:
            activation_map = torch.zeros_like(activation_map)

        # 双线性插值放大到与原图相同的尺寸 (例如 256x256)
        activation_map = activation_map.unsqueeze(0).unsqueeze(0)
        activation_map = F.interpolate(
            activation_map, 
            size=(img_single.shape[1], img_single.shape[2]), 
            mode='bilinear', 
            align_corners=False
        ).squeeze()

        # 转为 numpy 并使用 OpenCV 生成彩色热力图
        am_np = activation_map.cpu().numpy()
        am_cv = np.uint8(255 * am_np)
        heatmap_cv = cv2.applyColorMap(am_cv, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_cv, cv2.COLOR_BGR2RGB) / 255.0
        
        # 叠加热力图与原图
        heatmap_tensor = torch.from_numpy(heatmap_rgb).permute(2, 0, 1).to(img_single.device).float()
        
        # 采用 0.7的原图 + 0.5的热力图 可以让底层图像保留得更清晰
        overlay = img_single + 0.6 * heatmap_tensor 
        return torch.clamp(overlay, 0, 1)
    
def entropy_loss(preds):
    # preds: [Batch, Num_Classes] -> 经过 Softmax 后的概率
    p = F.softmax(preds, dim=1)
    # 计算熵: -sum(p * log(p))
    # 加上 1e-8 防止 log(0)
    entropy = -torch.mean(torch.sum(p * torch.log(p + 1e-8), dim=1))
    return entropy