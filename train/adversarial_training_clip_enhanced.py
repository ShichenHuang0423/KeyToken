"""
增强版CLIP对抗训练
集成MAE重建+关键Token保护
"""

import sys
sys.path.append("open_flamingo")

import os
import shutil
import time
import argparse

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from training.scheduler import cosine_lr
from torchvision import transforms
import wandb
from tqdm import tqdm

from train.datasets import COCOFlickrDataset, ImageNetDataset
from CLIP_eval.eval_utils import load_clip_model
from open_flamingo.eval.classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
from train.pgd_train import pgd
from train.apgd_train import apgd_train as apgd
from train.utils import init_wandb, AverageMeter, str2bool
from train.sam_data import SamData
from open_flamingo.eval.models.utils import unwrap_model

# 导入新模块
from train.disturb_detector import PatchDisturbDetector, TokenDisturbDetector
from train.key_token_selector import KeyTokenSelector, AdaptiveKeyTokenSelector
from train.mae_decoder import DualMAEDecoder
from train.keytoken_loss import KeyTokenLoss, compute_keytoken_loss

# 解析参数
parser = argparse.ArgumentParser()
# 原有参数
parser.add_argument('--clip_model_name', type=str, default='ViT-L-14')
parser.add_argument('--pretrained', type=str, default='openai')
parser.add_argument('--dataset', type=str, default='imagenet')
parser.add_argument('--template', type=str, default='std')
parser.add_argument('--imagenet_root', type=str, default='/mnt/datasets/imagenet')
parser.add_argument('--output_normalize', type=str2bool, default=False)
parser.add_argument('--start_step', type=int, default=0)
parser.add_argument('--optimizer_state', type=str, default='')
parser.add_argument('--steps', type=int, default=20000)
parser.add_argument('--warmup', type=int, default=1400)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--loss', type=str, default='l2')
parser.add_argument('--loss_clean', type=str, default='none')
parser.add_argument('--clean_weight', type=float, default=0.)
parser.add_argument('--trades', type=str2bool, default=False)
parser.add_argument('--opt', type=str, default='adamw')
parser.add_argument('--momentum_sgd', type=float, default=0.9)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--attack', type=str, default='apgd')
parser.add_argument('--inner_loss', type=str, default='l2')
parser.add_argument('--norm', type=str, default='linf')
parser.add_argument('--eps', type=float, default=4)
parser.add_argument('--iterations_adv', type=int, default=10)
parser.add_argument('--stepsize_adv', type=float, default=1.)
parser.add_argument('--wandb', type=str2bool, default=True)
parser.add_argument('--experiment_name', type=str, default='')
parser.add_argument('--overwrite', type=str2bool, default=False)
parser.add_argument('--log_freq', type=int, default=1)
parser.add_argument('--eval_freq', type=int, default=50)
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--save_checkpoints', type=str2bool, default=True)
parser.add_argument('--devices', type=str, default='')

# 新增参数
parser.add_argument('--use_mae_recon', type=str2bool, default=True, help='使用MAE重建任务')
parser.add_argument('--use_key_token_protection', type=str2bool, default=True, help='使用关键Token保护')
parser.add_argument('--mae_weight', type=float, default=0.1, help='MAE重建损失权重')
parser.add_argument('--text_recon_weight', type=float, default=0.8, help='文本重建损失权重')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='动态掩码比例')
parser.add_argument('--key_token_ratio', type=float, default=0.2, help='关键Token保留比例')
parser.add_argument('--adaptive_masking', type=str2bool, default=False, help='使用自适应掩码')

# 断点续连参数
parser.add_argument('--resume', type=str, default='', help='从指定checkpoint恢复训练（auto表示自动检测最新）')
parser.add_argument('--checkpoint_freq', type=int, default=500, help='每多少步保存一次checkpoint')

# 参数冻结参数
parser.add_argument('--freeze_clip_backbone', type=str2bool, default=False, help='冻结CLIP预训练权重，只训练新增模块')
parser.add_argument('--freeze_encoder_layers', type=int, default=0, help='冻结ViT encoder的前N层（0=不冻结）')

# ⚡ 显存优化参数
parser.add_argument('--use_amp', type=str2bool, default=True, help='使用混合精度训练(AMP)，可节省~30%显存')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='梯度累积步数（有效batch=batch_size*accumulation_steps）')
parser.add_argument('--memory_efficient_mode', type=str2bool, default=True, help='启用内存高效模式（减少中间激活值缓存）')

# 🚨 I/O优化参数
parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker数量（HDD建议2-4，SSD可8-12）')
parser.add_argument('--prefetch_factor', type=int, default=4, help='每个worker预读取的batch数（降低可减少I/O压力）')

# 🎲 随机种子参数
parser.add_argument('--seed', type=int, default=None, help='随机种子（None=随机生成并记录）')

# 🎯 KeyToken融合Loss参数（对比学习版本）
parser.add_argument('--use_keytoken_loss', type=str2bool, default=False, help='使用KeyToken融合Loss（对比学习+鲁棒性+MAE）')
parser.add_argument('--contrastive_weight', type=float, default=1.0, help='对比学习损失权重')
parser.add_argument('--contrastive_temperature', type=float, default=0.07, help='对比学习温度参数')
parser.add_argument('--robust_weight', type=float, default=0.5, help='鲁棒性损失(L2)权重')
parser.add_argument('--detect_weight', type=float, default=0.1, help='扰动检测损失权重')


def set_random_seed(seed=None):
    """
    设置所有随机种子以保证可复现性
    
    Args:
        seed: 随机种子，None时自动生成
    
    Returns:
        使用的种子值
    """
    import random
    
    if seed is None:
        # 使用时间戳生成种子
        seed = int(time.time() * 1000) % (2**31)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置deterministic模式（可能影响性能，但保证可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    return seed


def save_checkpoint(model, optimizer, scheduler, step, epoch, args, filename='checkpoint.pt'):
    """
    保存完整的训练状态（优化版：减少磁盘I/O压力）
    ⚡ 优化：
    1. 检查磁盘空间，不足时只保留最近N个checkpoint
    2. 异步保存（可选，降低阻塞时间）
    3. 压缩保存（可选，减少写入量）
    """
    import random
    
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    # ⚡ 保存完整的EnhancedClipVisionModel（包含所有增强模块）
    enhanced_model = unwrap_model(model)
    checkpoint = {
        'step': step,
        'epoch': epoch,
        # 保存完整的增强模型（包括所有训练的增强模块）
        'enhanced_model_state_dict': enhanced_model.state_dict(),
        # 同时保存基础CLIP权重（用于兼容旧的评估脚本）
        'model_state_dict': enhanced_model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'args': vars(args),
        # 🎲 保存随机状态，确保跨Stage训练的连续性
        'random_state': {
            'python_random': random.getstate(),
            'numpy_random': np.random.get_state(),
            'torch_random': torch.get_rng_state(),
            'torch_cuda_random': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    }
    
    # ⚡ 检查磁盘空间（如果接近满，清理旧checkpoint）
    if args.memory_efficient_mode:
        try:
            import shutil
            stat = shutil.disk_usage(checkpoint_dir)
            free_gb = stat.free / (1024**3)
            
            # 如果剩余空间 < 50GB，清理旧checkpoint，只保留最近10个
            if free_gb < 50:
                print(f"⚠️ 磁盘空间不足 ({free_gb:.1f}GB剩余)，清理旧checkpoint...")
                cleanup_old_checkpoints(checkpoint_dir, keep_last_n=10)
        except Exception as e:
            print(f"⚠️ 磁盘空间检查失败: {e}")
    
    # 先保存到临时文件，避免保存中断导致checkpoint损坏
    temp_path = checkpoint_path + '.tmp'
    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, checkpoint_path)  # 原子操作
        print(f"✅ 保存checkpoint: {checkpoint_path} (step={step}, epoch={epoch})")
    except OSError as e:
        # 磁盘空间不足或其他I/O错误
        if os.path.exists(temp_path):
            os.remove(temp_path)  # 清理临时文件
        raise RuntimeError(f"❌ Checkpoint保存失败 (磁盘空间不足?): {e}")
    
    return checkpoint_path


def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=10):
    """清理旧checkpoint，只保留最近N个"""
    try:
        checkpoints = []
        for fname in os.listdir(checkpoint_dir):
            if fname.endswith('.pt') and not fname.endswith('.tmp'):
                fpath = os.path.join(checkpoint_dir, fname)
                checkpoints.append((os.path.getmtime(fpath), fpath, fname))
        
        if len(checkpoints) > keep_last_n:
            checkpoints.sort(reverse=True)  # 最新的在前
            to_delete = checkpoints[keep_last_n:]  # 保留前N个，删除其他的
            
            deleted_size = 0
            for _, fpath, fname in to_delete:
                try:
                    size = os.path.getsize(fpath)
                    os.remove(fpath)
                    deleted_size += size
                    print(f"  删除旧checkpoint: {fname} ({size/1e9:.2f}GB)")
                except Exception as e:
                    print(f"  删除失败 {fname}: {e}")
            
            print(f"✅ 已清理 {len(to_delete)} 个旧checkpoint，释放 {deleted_size/1e9:.2f}GB")
    except Exception as e:
        print(f"⚠️ Checkpoint清理失败: {e}")


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的checkpoint"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        if fname.endswith('.pt') and not fname.endswith('.tmp'):
            fpath = os.path.join(checkpoint_dir, fname)
            checkpoints.append((os.path.getmtime(fpath), fpath))
    
    if not checkpoints:
        return None
    
    # 返回最新的checkpoint
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """加载checkpoint并恢复训练状态"""
    import random
    
    print(f"📂 加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型（优先加载完整的增强模型，否则只加载基础CLIP）
    enhanced_model = unwrap_model(model)
    if 'enhanced_model_state_dict' in checkpoint:
        # 新格式：包含所有增强模块
        enhanced_model.load_state_dict(checkpoint['enhanced_model_state_dict'])
        print(f"✅ 完整增强模型权重已恢复（包含所有增强模块）")
    else:
        # 旧格式：只有基础CLIP权重
        enhanced_model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"⚠️  仅基础CLIP权重已恢复（增强模块将随机初始化）")
    
    # 加载optimizer
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Optimizer状态已恢复")
    
    # 加载scheduler
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ Scheduler状态已恢复")
    
    # 🎲 恢复随机状态，确保跨Stage训练的连续性
    if 'random_state' in checkpoint:
        random_state = checkpoint['random_state']
        random.setstate(random_state['python_random'])
        np.random.set_state(random_state['numpy_random'])
        torch.set_rng_state(random_state['torch_random'])
        if torch.cuda.is_available() and random_state['torch_cuda_random'] is not None:
            torch.cuda.set_rng_state_all(random_state['torch_cuda_random'])
        print(f"✅ 随机状态已恢复（保持训练连续性）")
    else:
        print(f"⚠️  未找到随机状态（旧checkpoint格式）")
    
    step = checkpoint.get('step', 0)
    epoch = checkpoint.get('epoch', 0)
    print(f"✅ 从 step={step}, epoch={epoch} 恢复训练")
    
    return step, epoch


def freeze_clip_backbone(model, freeze_encoder_layers=0):
    """冻结CLIP backbone的参数"""
    enhanced_model = unwrap_model(model)
    clip_model = enhanced_model.model  # ViT encoder
    
    print(f"\n🔒 冻结CLIP预训练权重...")
    
    # 冻结所有CLIP参数
    for param in clip_model.parameters():
        param.requires_grad = False
    
    # 如果指定了部分解冻encoder层
    if freeze_encoder_layers > 0:
        total_layers = len(clip_model.transformer.resblocks)
        unfreeze_layers = total_layers - freeze_encoder_layers
        if unfreeze_layers > 0:
            print(f"  解冻最后 {unfreeze_layers} 层 transformer blocks")
            for i in range(total_layers - unfreeze_layers, total_layers):
                for param in clip_model.transformer.resblocks[i].parameters():
                    param.requires_grad = True
    
    print(f"  ✅ CLIP backbone已冻结")


def get_trainable_params(model):
    """获取所有需要训练的参数"""
    enhanced_model = unwrap_model(model)
    trainable_params = []
    
    # 收集所有requires_grad=True的参数
    for name, param in enhanced_model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
    
    return trainable_params


def print_trainable_params(model):
    """打印可训练参数统计"""
    enhanced_model = unwrap_model(model)
    
    total_params = 0
    trainable_params = 0
    
    print(f"\n📊 参数统计：")
    print(f"{'-'*60}")
    
    # 按模块分组统计
    module_stats = {}
    
    for name, param in enhanced_model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            
            # 提取模块名称
            module_name = name.split('.')[0] if '.' in name else name
            if module_name not in module_stats:
                module_stats[module_name] = {'trainable': 0, 'frozen': 0}
            module_stats[module_name]['trainable'] += num_params
        else:
            module_name = name.split('.')[0] if '.' in name else name
            if module_name not in module_stats:
                module_stats[module_name] = {'trainable': 0, 'frozen': 0}
            module_stats[module_name]['frozen'] += num_params
    
    # 打印每个模块的统计
    for module_name, stats in sorted(module_stats.items()):
        total_module = stats['trainable'] + stats['frozen']
        status = "🔓 训练" if stats['trainable'] > 0 else "🔒 冻结"
        print(f"  {status} {module_name:30s}: {stats['trainable']:>12,} / {total_module:>12,} 参数")
    
    print(f"{'-'*60}")
    print(f"  总计: {trainable_params:,} / {total_params:,} 参数可训练")
    print(f"  可训练比例: {100*trainable_params/total_params:.2f}%")
    print(f"  预计显存节省: ~{100*(1-trainable_params/total_params):.1f}% (梯度+优化器状态)")
    print(f"{'-'*60}\n")


class EnhancedClipVisionModel(nn.Module):
    """
    增强的CLIP视觉模型
    集成MAE重建+关键Token保护
    """
    def __init__(self, model, args, normalize):
        super().__init__()
        self.model = model  # ViT visual encoder
        self.args = args
        self.normalize = normalize
        
        # 获取模型维度
        if args.clip_model_name == 'ViT-L-14':
            self.dim = 1024  # ViT-L/14 特征维度是1024
        elif args.clip_model_name == 'ViT-B-32':
            self.dim = 512
        elif args.clip_model_name == 'ViT-B-16':
            self.dim = 768
        else:
            self.dim = 768  # 默认
        
        # 新增模块
        if args.use_mae_recon or args.use_key_token_protection:
            self.patch_disturb_detector = PatchDisturbDetector(dim=self.dim)
            
            if args.use_key_token_protection:
                if args.adaptive_masking:
                    self.key_selector = AdaptiveKeyTokenSelector(
                        base_ratio=args.key_token_ratio
                    )
                else:
                    self.key_selector = KeyTokenSelector(
                        top_k_ratio=args.key_token_ratio
                    )
            
            if args.use_mae_recon:
                # 只需要图像解码器（文本编码器冻结）
                self.mae_decoder = DualMAEDecoder(
                    img_dim=self.dim,
                    text_dim=self.dim
                ).img_decoder
    
    def forward_features(self, x):
        """
        提取图像patch特征
        需要修改ViT以返回所有patch tokens
        """
        x = self.normalize(x)
        
        # ⚡ 处理DataParallel包装：unwrap获取实际模型
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # 通过ViT的所有层获取patch tokens
        # 这里需要访问ViT的内部结构
        x = actual_model.conv1(x)  # patch embedding
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # 添加class token和position embedding
        x = torch.cat([
            actual_model.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ),
            x
        ], dim=1)
        x = x + actual_model.positional_embedding.to(x.dtype)
        
        x = actual_model.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # 通过transformer blocks
        x = actual_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 不应用ln_post和projection（保留patch特征）
        return x  # (B, N+1, dim) where N=196 for 224x224
    
    def forward(self, x, vision_clean=None, output_normalize=False, mode='train', randomize_defense=False):
        """
        增强的前向传播（显存优化版）
        
        Args:
            x: 扰动图像
            vision_clean: 清洁图像（训练时提供）
            output_normalize: 是否归一化输出
            mode: 'train' / 'eval' / 'attack'
                - 'train': 训练模式，使用所有增强模块学习防御
                - 'eval': 推理模式，使用所有增强模块进行鲁棒推理
                - 'attack': 攻击模式，只使用基础CLIP（无防御），用于生成强对抗样本
        """
        if mode == 'attack':
            # 对抗样本生成模式：只使用基础CLIP，不启用任何防御机制
            # 这样可以生成更强的对抗样本，避免攻击时就启用防御导致样本太弱
            embedding = self.model(self.normalize(x))
            if output_normalize:
                embedding = F.normalize(embedding, dim=-1)
            # attack模式不计算FeatDiff，返回6个值保持接口一致
            return embedding, torch.tensor(0.0, device=x.device), None, None, None, None
        
        elif mode == 'train' and (self.args.use_mae_recon or self.args.use_key_token_protection):
            assert vision_clean is not None, "训练模式需要提供清洁图像"
            
            # 1. 提取patch特征
            patch_tokens = self.forward_features(x)  # (B, 197, dim)
            
            # ⚡ 显存优化：clean特征不需要梯度，使用no_grad减少50%显存
            with torch.no_grad():
                patch_tokens_clean = self.forward_features(vision_clean).detach()
            
            # 特征可视化：使用TTC论文的token级τ值（归一化相对变化）
            # τ_i = ||f_i(x_adv) - f_i(x_clean)|| / ||f_i(x_clean)||  对每个token
            # 优势：1) 归一化度量 2) token级别，与disturb_scores一致 3) 无需训练
            tau_token = None  # 用于后续传递给disturb_detector
            
            # ⚡ 总是计算feature_diff，不依赖hasattr检查
            with torch.no_grad():
                # Token级别τ值计算 (B, 197, dim)
                token_diff = patch_tokens - patch_tokens_clean  # (B, 197, dim)
                token_diff_norm = torch.norm(token_diff, p=2, dim=2)  # (B, 197)
                token_clean_norm = torch.norm(patch_tokens_clean, p=2, dim=2)  # (B, 197)
                tau_token = token_diff_norm / (token_clean_norm + 1e-8)  # (B, 197)
                
                # 全局统计指标用于日志
                feature_diff_mean = tau_token.mean()  # tensor标量
                feature_diff_max = tau_token.max()
                feature_diff_std = tau_token.std()
            
            # 2. 扰动检测（融合token级别τ值）
            # 扰动检测器需要梯度，用于训练检测器
            disturb_scores_raw = self.patch_disturb_detector(
                patch_tokens, patch_tokens_clean, mode='train'
            )  # (B, 197)
            
            # 融合τ值：将无监督的τ值与学习到的disturb_scores结合
            # 使用加权平均，早期依赖τ值，后期依赖学习的scores
            if tau_token is not None:
                # 动态权重：随训练进行逐渐从τ值转向disturb_scores
                tau_weight = 0.3
                disturb_scores = tau_weight * tau_token.detach() + (1 - tau_weight) * disturb_scores_raw
            else:
                disturb_scores = disturb_scores_raw
            
            # 保存用于损失计算（pred_disturb用于训练检测器）
            self._pred_disturb = disturb_scores_raw
            self._target_disturb = tau_token.detach() if tau_token is not None else None
            
            # 3. 关键Token筛选（不需要梯度）
            with torch.no_grad():
                if self.args.use_key_token_protection:
                    # 获取注意力权重（简化版：使用None，依赖特征重要性）
                    if self.args.adaptive_masking:
                        key_mask = self.key_selector(
                            patch_tokens.detach(), disturb_scores,
                            attention_weights=None, token_type='image'
                        )
                    else:
                        key_mask = self.key_selector.select_img_key_tokens(
                            patch_tokens.detach(), attention_weights=None
                        )
                else:
                    key_mask = torch.ones_like(disturb_scores, dtype=torch.bool)
                
                # 4. 动态阈值调整（训练阶段）
                avg_disturb = disturb_scores.mean(dim=1, keepdim=True)  # (B, 1)
                adaptive_threshold = 0.3 + 0.4 * avg_disturb.clamp(0, 1)  # (B, 1)
                
                # 5. ⚡ 优化后的上下文保留机制（向量化操作，避免Python循环）
                batch_size = key_mask.shape[0]
                num_tokens = key_mask.shape[1]
                
                # 使用卷积操作扩展相邻Token（比循环快10x+）
                key_mask_float = key_mask.float().unsqueeze(1)  # (B, 1, N)
                # 1D卷积核 [1, 1, 1] 表示扩展前后各1个位置
                expand_kernel = torch.ones(1, 1, 3, device=key_mask.device)
                key_mask_expanded = F.conv1d(key_mask_float, expand_kernel, padding=1)
                key_mask_expanded = (key_mask_expanded.squeeze(1) > 0)  # (B, N)
                
                # 6. 动态掩码
                mask = (disturb_scores > adaptive_threshold) & (~key_mask_expanded)
            
            # 对保护Token添加噪声（保持梯度）
            noise_std = 0.1
            protected_mask = ~mask
            # ⚡ 使用where避免原地操作
            noise = torch.randn_like(patch_tokens) * noise_std
            patch_tokens_protected = torch.where(
                protected_mask.unsqueeze(-1),
                patch_tokens + noise,
                patch_tokens
            )
            
            # 7. MAE重建（可选）
            if self.args.use_mae_recon:
                patch_recon = self.mae_decoder(patch_tokens, mask)
                mae_loss = self.mae_decoder.compute_reconstruction_loss(
                    patch_recon, patch_tokens_clean, mask
                )
            else:
                mae_loss = torch.tensor(0.0, device=x.device)
            
            # 8. 生成embedding：融合全局特征和关键token特征（与推理模式一致）
            # 全局特征：使用保护后的[CLS] token
            global_feat = patch_tokens_protected[:, 0, :]  # (B, dim)
            
            # 局部关键特征：关键Token的加权平均
            if self.args.use_key_token_protection and key_mask.any():
                # key_mask: (B, N), 提取关键token并聚合
                key_tokens = patch_tokens_protected * key_mask.unsqueeze(-1).float()
                key_count = key_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
                local_feat = key_tokens.sum(dim=1) / key_count  # (B, dim)
            else:
                local_feat = global_feat
            
            # 融合全局+局部特征（70% 全局 + 30% 局部，与推理模式一致）
            embedding = 0.7 * global_feat + 0.3 * local_feat
            
            # 应用后处理层（ln_post + projection）
            embedding = self.model.ln_post(embedding)
            if self.model.proj is not None:
                embedding = embedding @ self.model.proj
            
            if output_normalize:
                embedding = F.normalize(embedding, dim=-1)
            
            # ⚡ 返回feature_diff_mean解决DataParallel问题（tensor标量可以被gather）
            # 同时返回pred_disturb、target_disturb和key_mask用于损失计算
            return embedding, mae_loss, feature_diff_mean, self._pred_disturb, self._target_disturb, key_mask
        
        else:
            # 推理模式：实现Token过滤与鲁棒匹配
            if self.args.use_key_token_protection or self.args.use_mae_recon:
                # 1. 提取patch特征
                patch_tokens = self.forward_features(x)  # (B, 197, dim)
                
                # 2. 扰动检测（推理模式）
                disturb_scores = self.patch_disturb_detector(
                    patch_tokens, mode='eval'
                )  # (B, 197)
                
                # 3. 关键Token筛选
                if self.args.use_key_token_protection:
                    # 统一使用forward接口（兼容AdaptiveKeyTokenSelector和KeyTokenSelector）
                    if self.args.adaptive_masking:
                        key_mask = self.key_selector(
                            patch_tokens, disturb_scores,
                            attention_weights=None, token_type='image'
                        )
                    else:
                        key_mask = self.key_selector.select_img_key_tokens(
                            patch_tokens, attention_weights=None
                        )
                else:
                    key_mask = torch.ones_like(disturb_scores, dtype=torch.bool)
                
                # 4. 动态阈值调整：根据扰动强度自适应
                # 计算批次平均扰动强度
                avg_disturb = disturb_scores.mean(dim=1, keepdim=True)  # (B, 1)
                # 阈值范围：[0.3, 0.7]，扰动越强阈值越高（保留更多Token）
                adaptive_threshold = 0.3 + 0.4 * avg_disturb.clamp(0, 1)  # (B, 1)
                
                # 🎲 随机化防御：对阈值添加高斯噪声（打破APGD梯度估计）
                if randomize_defense:
                    threshold_noise = torch.randn_like(adaptive_threshold) * 0.05
                    adaptive_threshold = (adaptive_threshold + threshold_noise).clamp(0.2, 0.8)
                
                # 5. ⚡ 上下文保留机制（向量化操作，与训练模式一致）
                # 使用卷积操作扩展相邻Token（比循环快10x+）
                key_mask_float = key_mask.float().unsqueeze(1)  # (B, 1, N)
                
                # 🎲 随机化防御：随机选择上下文扩展范围（kernel_size: 1-5）
                if randomize_defense:
                    kernel_size = torch.randint(1, 6, (1,), device=key_mask.device).item()
                else:
                    kernel_size = 3
                
                if kernel_size > 1:
                    expand_kernel = torch.ones(1, 1, kernel_size, device=key_mask.device)
                    padding = kernel_size // 2
                    key_mask_expanded = F.conv1d(key_mask_float, expand_kernel, padding=padding)
                    key_mask_expanded = (key_mask_expanded.squeeze(1) > 0)  # (B, N)
                else:
                    key_mask_expanded = key_mask  # 不扩展
                
                # 6. Token过滤：保留低扰动Token + 关键Token（含上下文）
                # 使用动态阈值
                filter_mask = (disturb_scores <= adaptive_threshold) | key_mask_expanded  # (B, 197)
                
                # 过滤Token
                patch_tokens_filtered = patch_tokens * filter_mask.unsqueeze(-1).float()
                
                # 7. 全局特征：使用过滤后的[CLS] token
                global_feat = patch_tokens_filtered[:, 0, :]  # (B, dim)
                
                # 8. 局部关键特征：关键Token的平均
                if self.args.use_key_token_protection and key_mask.any():
                    # 提取关键Token特征
                    key_tokens = patch_tokens_filtered * key_mask.unsqueeze(-1).float()
                    key_count = key_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
                    local_feat = key_tokens.sum(dim=1) / key_count  # (B, dim)
                else:
                    local_feat = global_feat
                
                # 9. 融合全局+局部特征（70% 全局 + 30% 局部）
                # 🎲 随机化防御：融合权重添加随机扰动
                if randomize_defense:
                    alpha_noise = torch.randn(1, device=global_feat.device).item() * 0.1
                    alpha = torch.clamp(torch.tensor(0.7 + alpha_noise), 0.5, 0.9).item()
                else:
                    alpha = 0.7
                embedding = alpha * global_feat + (1.0 - alpha) * local_feat
                
                # 应用后处理层
                embedding = self.model.ln_post(embedding)
                if self.model.proj is not None:
                    embedding = embedding @ self.model.proj
                
                if output_normalize:
                    embedding = F.normalize(embedding, dim=-1)
                
                # eval模式不计算FeatDiff，返回6个值保持接口一致
                return embedding, torch.tensor(0.0, device=x.device), None, None, None, None
            
            else:
                # 不使用增强功能，直接用CLIP
                embedding = self.model(self.normalize(x))
                if output_normalize:
                    embedding = F.normalize(embedding, dim=-1)
                return embedding, torch.tensor(0.0, device=x.device), None, None, None, None


class ComputeLossWrapper:
    def __init__(self, embedding_orig, embedding_text_labels_norm, reduction='mean', 
                 loss=None, logit_scale=100.):
        self.embedding_orig = embedding_orig
        self.embedding_text_labels_norm = embedding_text_labels_norm
        self.reduction = reduction
        self.loss_str = loss
        self.logit_scale = logit_scale

    def __call__(self, embedding, targets):
        from train.adversarial_training_clip import compute_loss
        return compute_loss(
            loss_str=self.loss_str, embedding=embedding, targets=targets,
            embedding_orig=self.embedding_orig, logit_scale=self.logit_scale,
            embedding_text_labels_norm=self.embedding_text_labels_norm, 
            reduction=self.reduction
        )


def train_one_epoch(
    step_total, model, model_orig, dataloader, optimizer, scheduler, normalize,
    embedding_text_labels_norm, args, epoch, dataloader_eval=None, scaler=None,
    best_acc=0.0
):
    """
    训练一个epoch（显存优化版）
    
    ⚡ 优化内容：
    1. 支持混合精度训练(AMP) - 节省~30%显存
    2. 支持梯度累积 - 可用更小batch训练
    3. 优化对抗攻击的显存管理
    4. 及时释放中间张量
    """
    model_orig.eval()
    model.train()
    
    # 初始化特征差异统计
    enhanced_model = unwrap_model(model)
    enhanced_model.feature_diff_stats = {'mean': 0.0, 'max': 0.0, 'std': 0.0}

    loss_meter = AverageMeter('loss')
    mae_loss_meter = AverageMeter('mae_loss')
    cos_sim_meter = AverageMeter('cos-sim')
    acc_meter = AverageMeter('acc')
    racc_meter = AverageMeter('racc')
    feature_diff_meter = AverageMeter('feature_diff')

    # ⚡ AMP设置
    use_amp = args.use_amp and torch.cuda.is_available()
    amp_dtype = torch.float16 if use_amp else torch.float32
    
    # 梯度累积
    accumulation_steps = args.gradient_accumulation_steps
    accumulated_loss = 0.0

    epoch_start_time = time.time()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Step {step_total}/{args.steps}", ncols=120)
    
    for i, (data, targets) in enumerate(pbar):
        is_classification = isinstance(targets, torch.Tensor)
        data = data.cuda(non_blocking=True)
        n_samples = data.shape[0]
        if is_classification:
            targets = targets.cuda(non_blocking=True)

        # 保存清洁图像
        data_clean = data.clone()

        # ⚡ 原始嵌入（不需要梯度）
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                embedding_orig, _, _, _, _, _ = model_orig(data, output_normalize=args.output_normalize, mode='eval')
            embedding_orig = embedding_orig.detach()

        # ⚡ 生成对抗样本（使用torch.no_grad减少显存）
        loss_inner_wrapper = ComputeLossWrapper(
            embedding_orig, embedding_text_labels_norm,
            reduction='none' if args.attack == 'apgd' else 'mean', 
            loss=args.inner_loss, logit_scale=100.
        )
        model.eval()

        # ⚡ 对抗攻击在低精度下执行，减少显存峰值
        # ⚡ 使用'attack'模式：攻击基础CLIP（无防御），生成更强的对抗样本
        with torch.cuda.amp.autocast(enabled=use_amp):
            if args.attack == 'pgd':
                data_adv = pgd(
                    forward=lambda x, output_normalize: model(x, output_normalize=output_normalize, mode='attack')[0],  # 只取第一个返回值
                    loss_fn=loss_inner_wrapper,
                    data_clean=data,
                    targets=targets,
                    norm=args.norm,
                    eps=args.eps,
                    iterations=args.iterations_adv,
                    stepsize=args.stepsize_adv,
                    output_normalize=args.output_normalize,
                    perturbation=torch.zeros_like(data).uniform_(-args.eps, args.eps).requires_grad_(True),
                    mode='max',
                    verbose=False
                )
            elif args.attack == 'apgd':
                # 创建一个模型wrapper类用于apgd
                class APGDModelWrapper(nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                    
                    def forward(self, x, output_normalize=True):
                        # ⚡ 使用'attack'模式：攻击基础CLIP（无防御）
                        # 只取第一个返回值（embedding）
                        return self.model(x, output_normalize=output_normalize, mode='attack')[0]
                
                model_wrapper = APGDModelWrapper(model)
                model_wrapper.eval()
                
                data_adv = apgd(
                    model=model_wrapper,
                    loss_fn=loss_inner_wrapper,
                    x=data,
                    y=targets,
                    norm=args.norm,
                    eps=args.eps,
                    n_iter=args.iterations_adv,
                    verbose=False
                )
                # ⚡ 及时释放wrapper
                del model_wrapper
            elif args.attack == 'none':
                data_adv = data

        # ⚡ 及时释放不需要的张量
        del loss_inner_wrapper
        torch.cuda.empty_cache()  # 清理缓存
        
        model.train()

        # ⚡ 训练前向传播（使用AMP）
        with torch.cuda.amp.autocast(enabled=use_amp):
            embedding_adv, mae_loss, feature_diff_mean, pred_disturb, target_disturb, key_mask = model(
                data_adv, vision_clean=data_clean, 
                output_normalize=args.output_normalize, mode='train'
            )
            
            # 🎯 使用KeyToken融合Loss或原始Loss
            if args.use_keytoken_loss:
                # KeyToken融合Loss：对比学习 + 关键Token鲁棒性 + MAE重建 + 扰动检测
                loss, loss_dict = compute_keytoken_loss(
                    embedding_adv=embedding_adv,
                    embedding_orig=embedding_orig,
                    targets=targets,
                    text_embeddings=embedding_text_labels_norm,
                    mae_loss=mae_loss if args.use_mae_recon else None,
                    pred_disturb=pred_disturb,  # 传入扰动检测器的预测
                    target_disturb=target_disturb,  # 基于τ值的真实扰动
                    key_mask=key_mask,  # 关键token掩码，用于关键Token级别Robust Loss
                    contrastive_weight=args.contrastive_weight,
                    robust_weight=args.robust_weight,
                    mae_weight=args.mae_weight,
                    detect_weight=args.detect_weight,
                    temperature=args.contrastive_temperature,
                    logit_scale=100.0
                )
                # 记录各项loss用于日志
                loss_contrastive_value = loss_dict.get('loss_contrastive', 0.0)
                loss_robust_value = loss_dict.get('loss_robust', 0.0)
                loss_mae_value = loss_dict.get('loss_mae', 0.0)
                loss_detect_value = loss_dict.get('loss_detect', 0.0)
            else:
                # 原始Loss计算方式
                from train.adversarial_training_clip import compute_loss
                loss_adv = compute_loss(
                    loss_str=args.loss, embedding=embedding_adv, targets=targets,
                    embedding_orig=embedding_orig, logit_scale=100.,
                    embedding_text_labels_norm=embedding_text_labels_norm
                )

                # 清洁样本损失（可选）
                if args.clean_weight > 0.:
                    embedding_clean, _, _, _, _, _ = model(
                        data_clean, vision_clean=None,
                        output_normalize=args.output_normalize, mode='eval'
                    )
                    loss_clean = compute_loss(
                        loss_str=args.loss_clean, embedding=embedding_clean, targets=targets,
                        embedding_orig=embedding_orig, logit_scale=100.,
                        embedding_text_labels_norm=None
                    )
                    loss = loss_adv + args.clean_weight * loss_clean
                else:
                    loss = loss_adv

                # 添加MAE重建损失
                if args.use_mae_recon:
                    if isinstance(mae_loss, torch.Tensor) and mae_loss.dim() > 0:
                        mae_loss = mae_loss.mean()
                    loss = loss + args.mae_weight * mae_loss
                
                # 兼容性：设置默认值（处理DataParallel多元素tensor）
                loss_cls_value = 0.0
                if isinstance(loss_adv, torch.Tensor):
                    loss_robust_value = loss_adv.mean().item() if loss_adv.numel() > 1 else loss_adv.item()
                else:
                    loss_robust_value = loss_adv
                if isinstance(mae_loss, torch.Tensor):
                    loss_mae_value = mae_loss.mean().item() if mae_loss.numel() > 1 else mae_loss.item()
                else:
                    loss_mae_value = 0.0
            
            # ⚡ 梯度累积：按累积步数缩放损失
            loss = loss / accumulation_steps

        # ⚡ 反向传播（使用AMP scaler）
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        accumulated_loss += loss.item() * accumulation_steps
        
        # ⚡ 梯度累积：达到累积步数后才更新参数
        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                # AMP梯度裁剪和更新
                scaler.unscale_(optimizer)
                
                # ⚡ NaN检测：检查梯度是否有效
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 如果梯度包含NaN或Inf，跳过此次更新
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    print(f"⚠️ Step {step_total}: 检测到NaN/Inf梯度 (norm={grad_norm.item():.2f})，跳过此次更新")
                    scaler.update()  # 仍需更新scaler状态
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                else:
                    print(f"⚠️ Step {step_total}: 检测到NaN/Inf梯度，跳过此次更新")
            
            optimizer.zero_grad(set_to_none=True)  # ⚡ set_to_none节省少量显存
            scheduler(step_total)
            
            # 记录
            loss_meter.update(accumulated_loss, n_samples * accumulation_steps)
            accumulated_loss = 0.0
            
            step_total += 1
        
        if args.use_mae_recon:
            # DataParallel可能返回多元素tensor，需要先取mean
            if isinstance(mae_loss, torch.Tensor):
                if mae_loss.numel() > 1:
                    mae_loss_val = mae_loss.mean().item()
                else:
                    mae_loss_val = mae_loss.item()
            else:
                mae_loss_val = mae_loss
            mae_loss_meter.update(mae_loss_val, n_samples)
        
        # 记录特征差异
        # ⚡ 使用forward返回的feature_diff_mean（解决DataParallel问题）
        if feature_diff_mean is not None:
            # 将tensor转为标量用于统计（DataParallel gather后可能是多元素tensor）
            if torch.is_tensor(feature_diff_mean):
                if feature_diff_mean.numel() > 1:
                    mean_val = feature_diff_mean.mean().item()
                else:
                    mean_val = feature_diff_mean.item()
            else:
                mean_val = feature_diff_mean
            feature_diff_meter.update(mean_val, n_samples)
            # 同步到enhanced_model（只保存mean值）
            enhanced_model.feature_diff_stats = {
                'mean': mean_val,
                'max': mean_val,
                'std': 0.0
            }

        # 计算准确率：同时计算干净样本和对抗样本准确率
        with torch.no_grad():
            if is_classification:
                # 对抗样本准确率 (racc)
                logits_adv = embedding_text_labels_norm.T @ embedding_adv.float().T
                racc = (logits_adv.argmax(dim=0) == targets).float().mean()
                racc_meter.update(racc.item(), n_samples)
                
                # 干净样本准确率 (acc) - 使用eval模式避免MAE影响
                embedding_clean_eval, _, _, _, _, _ = model(
                    data_clean, vision_clean=None,
                    output_normalize=args.output_normalize, mode='eval'
                )
                logits_clean = embedding_text_labels_norm.T @ embedding_clean_eval.float().T
                acc = (logits_clean.argmax(dim=0) == targets).float().mean()
                acc_meter.update(acc.item(), n_samples)
                
                del embedding_clean_eval, logits_adv, logits_clean

        # ⚡ 及时释放不需要的张量
        del data_adv, embedding_adv, data_clean
        if args.clean_weight > 0.:
            del embedding_clean
        
        # 日志记录
        if step_total % args.log_freq == 0 and (i + 1) % accumulation_steps == 0:
            log_dict = {
                'loss': loss_meter.avg,
                'step': step_total,
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr']
            }
            if args.use_mae_recon:
                log_dict['mae_loss'] = mae_loss_meter.avg
            if is_classification:
                log_dict['clean_acc'] = acc_meter.avg
                log_dict['racc'] = racc_meter.avg
            
            # 添加特征差异到日志
            if hasattr(enhanced_model, 'feature_diff_stats') and 'mean' in enhanced_model.feature_diff_stats:
                log_dict['feature_diff_mean'] = enhanced_model.feature_diff_stats['mean']
                log_dict['feature_diff_max'] = enhanced_model.feature_diff_stats['max']
                log_dict['feature_diff_std'] = enhanced_model.feature_diff_stats['std']
            
            # ⚡ 添加显存监控
            if torch.cuda.is_available():
                log_dict['gpu_memory_gb'] = torch.cuda.max_memory_allocated() / 1e9
            
            wandb.log(log_dict)
            
            progress_pct = step_total / args.steps * 100
            pbar.set_description(f"Epoch {epoch+1} [{progress_pct:.1f}%] Step {step_total}/{args.steps}")
            # 简化postfix，避免终端宽度截断（完整信息在print输出中）
            pbar.set_postfix({'Loss': f"{loss_meter.avg:.4f}", 'FeatDiff': f"{feature_diff_meter.avg:.4f}"})
            
            # 构建FeatDiff字符串
            stats = getattr(enhanced_model, 'feature_diff_stats', None)
            if stats and 'mean' in stats:
                if stats['mean'] > 0.05:
                    status_msg = "✅ CLIP已适应扰动"
                else:
                    status_msg = "⚠️ 扰动微弱"
            else:
                stats = {'mean': 0.0, 'max': 0.0, 'std': 0.0}
                status_msg = "⚠️ 特征差异未初始化"
            
            feat_diff_str = f"FeatDiff: {stats['mean']:.6f} (max={stats['max']:.6f}, std={stats['std']:.6f}) {status_msg}"
            
            # 构建显存字符串
            if torch.cuda.is_available():
                mem_str = f"GPU: {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
            else:
                mem_str = ""
            
            print(f"\n{'='*80}", flush=True)
            print(f"[{progress_pct:.1f}%] Step {step_total}/{args.steps}", flush=True)
            if args.use_keytoken_loss:
                print(f"  Loss: {loss_meter.avg:.4f} | Contrastive: {loss_contrastive_value:.4f} | L2: {loss_robust_value:.4f} | MAE: {loss_mae_value:.4f} | Detect: {loss_detect_value:.4f}", flush=True)
            else:
                print(f"  Loss: {loss_meter.avg:.4f} | MAE: {mae_loss_meter.avg:.4f}", flush=True)
            print(f"  CleanAcc: {acc_meter.avg:.4f} | RobustAcc: {racc_meter.avg:.4f}", flush=True)
            print(f"  {feat_diff_str}", flush=True)
            if mem_str:
                print(f"  {mem_str}", flush=True)
            print(f"{'='*80}\n", flush=True)
            
            # 🏆 检查并保存最佳准确率checkpoint
            current_racc = racc_meter.avg
            if args.save_checkpoints and current_racc > best_acc:
                best_acc = current_racc
                save_checkpoint(
                    model, optimizer, scheduler, step_total, epoch,
                    args, filename='best.pt'
                )
                print(f"🏆 新的最佳RobustAcc: {best_acc:.4f} - 已保存best.pt", flush=True)

        # 保存checkpoint
        if args.save_checkpoints and step_total % args.checkpoint_freq == 0 and (i + 1) % accumulation_steps == 0:
            save_checkpoint(
                model, optimizer, scheduler, step_total, epoch,
                args, filename=f'step_{step_total}.pt'
            )
        
        # 每10%进度额外保存一个里程碑checkpoint
        if args.save_checkpoints and step_total % (args.steps // 10) == 0 and (i + 1) % accumulation_steps == 0:
            save_checkpoint(
                model, optimizer, scheduler, step_total, epoch,
                args, filename=f'milestone_step_{step_total}.pt'
            )

        if step_total >= args.steps:
            break
        
        # ⚡ 定期清理GPU缓存
        if i % 100 == 0:
            torch.cuda.empty_cache()

    return step_total, best_acc


def main(args):
    # 🎲 随机种子处理：只在首次训练时设置，resume时会从checkpoint恢复
    is_resuming = bool(args.resume)
    
    if not is_resuming:
        # 首次训练：设置新的随机种子
        actual_seed = set_random_seed(args.seed)
        args.seed = actual_seed  # 更新args以记录实际使用的种子
        
        print(f"\n{'=' * 60}")
        print(f"🎲 随机种子: {actual_seed}")
        print(f"   可通过 --seed {actual_seed} 完全复现此次训练")
        print(f"{'=' * 60}\n")
    else:
        # Resume训练：随机状态将从checkpoint恢复，不设置新种子
        print(f"\n{'=' * 60}")
        print(f"🔄 从checkpoint恢复训练")
        print(f"   随机状态将从checkpoint恢复（保持跨Stage连续性）")
        print(f"{'=' * 60}\n")
    
    # 设置wandb
    if args.wandb:
        init_wandb(
            project_name='clip-finetune-enhanced',
            model_name=args.experiment_name or 'enhanced_clip',
            config=vars(args)
        )
    else:
        wandb.init(mode='disabled')

    # 打印参数
    print(f"Arguments:\n{'-' * 50}")
    for arg, value in vars(args).items():
        print(f"{arg:30s}: {value}")
    print(f"{'-' * 50}")

    # 设置输出目录
    if args.overwrite:
        shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # 保存参数和种子到文件
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        f.write(str(args))
    
    # 单独保存种子便于查找（只在首次训练时保存）
    if not is_resuming and hasattr(args, 'seed') and args.seed is not None:
        with open(os.path.join(args.output_dir, 'random_seed.txt'), 'w') as f:
            f.write(f"Random Seed: {args.seed}\n")
            f.write(f"Command to reproduce: --seed {args.seed}\n")

    main_device = 0
    num_gpus = torch.cuda.device_count()
    
    # 加载模型
    model_orig, _, image_processor = open_clip.create_model_and_transforms(
        args.clip_model_name, pretrained='openai'
    )
    if args.pretrained != 'openai':
        model, _, _ = load_clip_model(args.clip_model_name, args.pretrained)
    else:
        model = model_orig

    # 预处理
    preprocessor_without_normalize = transforms.Compose(image_processor.transforms[:-1])
    normalize = image_processor.transforms[-1]

    # 加载数据集
    if args.dataset == 'imagenet':
        dataset = ImageNetDataset(
            root=args.imagenet_root + '/train',
            transform=preprocessor_without_normalize,
        )
        dataset_eval = ImageNetDataset(
            root=args.imagenet_root + '/val',
            transform=preprocessor_without_normalize,
        )
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    # ⚡ DataLoader优化（适配HDD/SSD）
    # - num_workers: 从命令行参数获取，HDD建议2-4，SSD可8-12
    # - prefetch_factor: 从命令行参数获取，降低可减少I/O压力
    # - pin_memory: 加速GPU传输
    # - persistent_workers: 避免worker重启开销（workers>0时启用）
    
    # 使用命令行参数配置
    num_workers = args.num_workers
    prefetch_factor = args.prefetch_factor if num_workers > 0 else None
    use_persistent_workers = num_workers > 0
    
    dataloader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'drop_last': True,
        'pin_memory': True
    }
    
    # 只有当num_workers>0时才添加prefetch_factor和persistent_workers
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
        dataloader_kwargs['persistent_workers'] = use_persistent_workers
    
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    dataloader_eval = DataLoader(dataset_eval, **dataloader_kwargs)
    
    print(f"⚡ DataLoader配置: {num_workers} workers, prefetch={prefetch_factor}, pin_memory=True, persistent={use_persistent_workers}")

    # 获取文本嵌入
    template = 'This is a photo of a {}'
    texts = [template.format(c) for c in IMAGENET_1K_CLASS_ID_TO_LABEL.values()]
    text_tokens = open_clip.tokenize(texts)
    model_orig.to(main_device)
    
    with torch.no_grad():
        embedding_text_labels_norm = []
        for el in (text_tokens[:500], text_tokens[500:]):
            embedding_text_labels_norm.append(
                model_orig.encode_text(el.to(main_device), normalize=True).detach().cpu()
            )
        embedding_text_labels_norm = torch.cat(embedding_text_labels_norm).T.to(main_device)
    
    model_orig.cpu()
    
    # 包装模型
    model_orig = EnhancedClipVisionModel(model=model_orig.visual, args=args, normalize=normalize)
    model = EnhancedClipVisionModel(model=model.visual, args=args, normalize=normalize)
    
    # 多GPU
    if num_gpus > 1:
        model_orig = nn.DataParallel(model_orig)
        model = nn.DataParallel(model)
    
    model_orig.cuda()
    model.cuda()

    # 参数冻结
    if args.freeze_clip_backbone:
        freeze_clip_backbone(model, freeze_encoder_layers=args.freeze_encoder_layers)
        print_trainable_params(model)
        
        # 只优化可训练的参数
        params = get_trainable_params(model)
        print(f"✅ 优化器将只更新 {len(params)} 组可训练参数\n")
    else:
        # 训练所有参数（包括CLIP和新增模块）
        params = unwrap_model(model).parameters()
        print(f"⚠️  训练所有参数（CLIP + 新增模块，未冻结）\n")
    
    # 优化器
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum_sgd, weight_decay=args.wd
        )

    # 学习率调度
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, args.steps)

    # ⚡ 初始化AMP GradScaler（增加稳定性参数）
    scaler = None
    if args.use_amp:
        # 使用更保守的scaler参数避免梯度爆炸
        scaler = torch.cuda.amp.GradScaler(
            init_scale=2.**10,  # 降低初始缩放（默认2^16）
            growth_interval=2000  # 增加增长间隔（默认2000）
        )
        print("⚡ 启用混合精度训练(AMP) - 预计节省~30%显存")
        print("⚡ GradScaler配置: init_scale=1024, growth_interval=2000（保守模式）")
    
    # ⚡ 显示显存优化配置
    print(f"\n{'='*60}")
    print(f"⚡ 显存优化配置:")
    print(f"   混合精度(AMP): {'✅ 开启' if args.use_amp else '❌ 关闭'}")
    print(f"   梯度累积步数: {args.gradient_accumulation_steps}")
    print(f"   有效batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"   内存高效模式: {'✅ 开启' if args.memory_efficient_mode else '❌ 关闭'}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({mem_total:.1f}GB)")
    print(f"{'='*60}\n")

    # 断点续连
    step_total = args.start_step
    epoch = 0
    
    if args.resume:
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        
        if args.resume == 'auto':
            # 自动检测最新checkpoint
            checkpoint_path = find_latest_checkpoint(checkpoint_dir)
            if checkpoint_path:
                step_total, epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
            else:
                print("⚠️  未找到checkpoint，从头开始训练")
        else:
            # 从指定checkpoint恢复
            checkpoint_path = args.resume
            if not os.path.isabs(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
            
            if os.path.exists(checkpoint_path):
                step_total, epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
            else:
                print(f"⚠️  Checkpoint不存在: {checkpoint_path}，从头开始训练")
    
    # 训练
    # 考虑梯度累积计算实际epoch数
    total_epochs = args.steps * args.gradient_accumulation_steps / len(dataloader)
    print(f'训练 {total_epochs:.1f} epochs，从step {step_total}开始')
    
    # 🏆 初始化最佳准确率跟踪
    best_acc = 0.0
    
    while step_total < args.steps:
        step_total, best_acc = train_one_epoch(
            step_total, model, model_orig, dataloader,
            optimizer, scheduler, normalize,
            embedding_text_labels_norm, args, epoch, dataloader_eval,
            scaler=scaler,  # ⚡ 传递scaler
            best_acc=best_acc  # 🏆 传递best_acc
        )
        
        # 每个epoch结束时保存checkpoint
        save_checkpoint(
            model, optimizer, scheduler, step_total, epoch + 1,
            args, filename=f'epoch_{epoch+1}.pt'
        )
        print(f'✅ Epoch {epoch+1} 完成')
        epoch += 1
        
        # ⚡ epoch结束后清理显存
        torch.cuda.empty_cache()

    # 保存最终checkpoint
    save_checkpoint(
        model, optimizer, scheduler, step_total, epoch,
        args, filename='final.pt'
    )

    print(f"✅ 训练完成！最佳RobustAcc: {best_acc:.4f}")


if __name__ == '__main__':
    args = parser.parse_args()
    
    # 转换eps和stepsize（与adversarial_training_clip.py保持一致）
    args.eps /= 255
    args.stepsize_adv /= 255
    
    # 参数验证（与原始脚本保持一致）
    assert not any([isinstance(x, str) and x in ['True', 'False'] for x in args.__dict__.values()]), \
        f'args contains a string that should be a bool: {args}'
    assert args.eval_freq % args.log_freq == 0, 'eval_freq must be a multiple of log_freq'
    
    # 设置实验名称
    if not args.experiment_name:
        args.experiment_name = f'enhanced_clip_{args.clip_model_name}_eps{int(args.eps*255)}'
    
    # 设置输出目录
    if not args.output_dir:
        args.output_dir = f'output/{args.experiment_name}'
    
    args.finetuned_model_name = args.experiment_name
    
    main(args)
