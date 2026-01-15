#!/usr/bin/env python
"""
KeyToken 快速测试脚本
用于验证环境配置和模型加载是否正常
"""

import sys
import torch
from PIL import Image
import requests
from io import BytesIO

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_imports():
    """测试关键库导入"""
    print_section("测试 1: 检查依赖库导入")
    
    required_libs = [
        'torch', 'torchvision', 'open_clip', 'transformers', 
        'accelerate', 'einops', 'huggingface_hub', 'PIL',
        'numpy', 'wandb', 'timm'
    ]
    
    failed = []
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"  ✓ {lib}")
        except ImportError as e:
            print(f"  ✗ {lib}: {e}")
            failed.append(lib)
    
    if failed:
        print(f"\n警告: {len(failed)} 个库导入失败")
        return False
    else:
        print("\n所有依赖库导入成功！")
        return True

def test_cuda():
    """测试CUDA可用性"""
    print_section("测试 2: CUDA环境检查")
    
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            # 显示显存
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"    显存: {mem_total:.2f} GB")
        return True
    else:
        print("  警告: 未检测到CUDA支持，将使用CPU运行（速度较慢）")
        return False

def test_clip_model_loading():
    """测试CLIP模型加载"""
    print_section("测试 3: CLIP模型加载")
    
    try:
        import open_clip
        
        print("  正在加载OpenAI ViT-L/14 CLIP模型...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', 
            pretrained='openai',
            device='cpu'  # 先在CPU测试
        )
        print("  ✓ 模型加载成功")
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  模型参数量: {total_params/1e6:.2f}M")
        
        # 测试前向传播
        print("  正在测试前向传播...")
        dummy_image = torch.randn(1, 3, 224, 224)
        dummy_text = open_clip.tokenize(["a photo of a cat"])
        
        with torch.no_grad():
            image_features = model.encode_image(dummy_image)
            text_features = model.encode_text(dummy_text)
        
        print(f"  ✓ 图像特征维度: {image_features.shape}")
        print(f"  ✓ 文本特征维度: {text_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        return False

def test_image_inference():
    """测试图像推理"""
    print_section("测试 4: 图像推理测试")
    
    try:
        import open_clip
        from PIL import Image
        
        print("  正在下载测试图片...")
        # 使用一个公开的测试图片
        url = "https://raw.githubusercontent.com/openai/CLIP/main/CLIP.png"
        try:
            response = requests.get(url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            print("  ✓ 图片下载成功")
        except:
            print("  ! 无法下载测试图片，创建随机图片")
            image = Image.new('RGB', (224, 224), color='red')
        
        print("  正在加载模型...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', 
            pretrained='openai',
            device='cpu'
        )
        
        # 预处理图片
        image_tensor = preprocess(image).unsqueeze(0)
        
        # 定义类别
        text_labels = ["a dog", "a cat", "a bird", "a car", "a building"]
        text_tokens = open_clip.tokenize(text_labels)
        
        print("  正在进行零样本分类...")
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        print("\n  预测结果:")
        for label, prob in zip(text_labels, similarity[0]):
            print(f"    {label:15s}: {prob.item()*100:5.2f}%")
        
        print("\n  ✓ 推理测试成功")
        return True
        
    except Exception as e:
        print(f"  ✗ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_project_structure():
    """测试项目结构"""
    print_section("测试 5: 项目结构检查")
    
    import os
    
    required_dirs = [
        'train', 'CLIP_eval', 'CLIP_benchmark', 'vlm_eval',
        'pope_eval', 'scienceqa_eval', 'llava', 'open_flamingo',
        'bash', 'autoattack'
    ]
    
    required_files = [
        'requirements.txt', 'README.md', 'SETUP_GUIDE.md'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (缺失)")
            missing_dirs.append(dir_name)
    
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"  ✓ {file_name}")
        else:
            print(f"  ! {file_name} (缺失)")
            missing_files.append(file_name)
    
    if missing_dirs:
        print(f"\n  警告: {len(missing_dirs)} 个必需目录缺失")
        return False
    
    print("\n  项目结构完整")
    return True

def test_huggingface_connection():
    """测试HuggingFace连接"""
    print_section("测试 6: HuggingFace连接")
    
    try:
        from huggingface_hub import hf_hub_download
        print("  正在测试HuggingFace Hub连接...")
        
        # 尝试访问一个公开的配置文件（很小，不会真正下载大文件）
        try:
            # 只是检查能否访问，不实际下载
            from huggingface_hub import model_info
            info = model_info("openai/clip-vit-large-patch14")
            print(f"  ✓ 成功连接到HuggingFace Hub")
            print(f"  ✓ 测试模型: {info.modelId}")
            return True
        except Exception as e:
            print(f"  ! 连接HuggingFace可能较慢或需要代理: {e}")
            print("  提示: 如在国内，可设置镜像:")
            print("    export HF_ENDPOINT=https://hf-mirror.com")
            return False
            
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "+"*60)
    print("  KeyToken (RobustVLM) 环境验证测试")
    print("+"*60)
    
    results = []
    
    # 运行所有测试
    results.append(("依赖库导入", test_imports()))
    results.append(("CUDA环境", test_cuda()))
    results.append(("CLIP模型加载", test_clip_model_loading()))
    results.append(("图像推理", test_image_inference()))
    results.append(("项目结构", test_project_structure()))
    results.append(("HuggingFace连接", test_huggingface_connection()))
    
    # 总结
    print_section("测试总结")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name:20s}: {status}")
    
    print(f"\n  总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n  🎉 恭喜！所有测试通过，环境配置完成！")
        print("\n  下一步:")
        print("    1. 查看 SETUP_GUIDE.md 了解详细使用说明")
        print("    2. 下载数据集（参考SETUP_GUIDE.md中的数据集准备部分）")
        print("    3. 运行实验（参考SETUP_GUIDE.md中的实验复现步骤）")
        return 0
    else:
        print("\n  ⚠️  部分测试未通过，请检查环境配置")
        print("     参考 SETUP_GUIDE.md 中的故障排除部分")
        return 1

if __name__ == "__main__":
    sys.exit(main())
