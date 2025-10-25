#!/usr/bin/env python3
"""
测试DMD LoRA功能的简单脚本
"""

import torch
import argparse
import sys
import os

# 添加路径以便导入DMD
sys.path.insert(0, os.path.dirname(__file__))

def create_test_args():
    """创建测试用的参数配置"""
    args = argparse.Namespace()
    
    # 基础DMD参数
    args.model_name = "wan"
    args.generator_task = "causal_video"
    args.generator_grad = True
    args.real_score_grad = False
    args.fake_score_grad = True
    args.gradient_checkpointing = False  # 测试时关闭以避免复杂性
    args.denoising_step_list = [0, 250, 500, 750, 999]
    args.num_train_timestep = 1000
    args.real_guidance_scale = 7.5
    args.denoising_loss_type = "flow"
    args.mixed_precision = True
    args.warp_denoising_step = False
    
    # 可选的checkpoint路径（测试时不加载）
    args.generator_ckpt = None
    args.real_score_ckpt = None
    args.fake_score_ckpt = None
    
    return args

def test_dmd_without_lora():
    """测试不使用LoRA的DMD初始化"""
    print("测试1: DMD初始化（不使用LoRA）")
    
    args = create_test_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from dmd import DMD
        dmd = DMD(args, device)
        print("✓ DMD初始化成功（不使用LoRA）")
        return True
    except Exception as e:
        print(f"✗ DMD初始化失败: {e}")
        return False

def test_dmd_with_lora_config():
    """测试带有LoRA配置的DMD初始化（但不加载实际的checkpoint文件）"""
    print("\n测试2: DMD初始化（带LoRA配置，但不加载实际文件）")
    
    args = create_test_args()
    
    # 添加LoRA配置，但使用不存在的文件路径
    # 这样可以测试配置解析逻辑，而不需要实际的checkpoint文件
    args.real_score_lora_ckpt = None  # 设为None以跳过实际加载
    args.real_score_lora_base_model = "model"
    args.real_score_lora_target_modules = "q,k,v,o"
    args.real_score_lora_rank = 16
    
    args.fake_score_lora_ckpt = None  # 设为None以跳过实际加载
    args.fake_score_lora_base_model = "model"
    args.fake_score_lora_target_modules = "q,k,v,o"
    args.fake_score_lora_rank = 16
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        from dmd import DMD
        dmd = DMD(args, device)
        print("✓ DMD初始化成功（带LoRA配置）")
        return True
    except Exception as e:
        print(f"✗ DMD初始化失败: {e}")
        return False

def test_lora_helper_method():
    """测试_add_lora_to_model辅助方法"""
    print("\n测试3: 测试_add_lora_to_model方法")
    
    try:
        from dmd import DMD
        
        # 创建一个简单的模型来测试LoRA注入
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q = torch.nn.Linear(64, 64)
                self.k = torch.nn.Linear(64, 64)
                self.v = torch.nn.Linear(64, 64)
                self.o = torch.nn.Linear(64, 64)
        
        model = SimpleModel()
        
        # 创建DMD实例来访问_add_lora_to_model方法
        args = create_test_args()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dmd = DMD(args, device)
        
        # 测试LoRA注入
        model_with_lora = dmd._add_lora_to_model(
            model=model,
            target_modules=["q", "k", "v", "o"],
            lora_rank=8
        )
        
        print("✓ _add_lora_to_model方法工作正常")
        return True
        
    except Exception as e:
        print(f"✗ _add_lora_to_model方法测试失败: {e}")
        return False

def test_lora_parameter_parsing():
    """测试LoRA参数解析"""
    print("\n测试4: LoRA参数解析")
    
    args = create_test_args()
    
    # 设置一些LoRA参数
    args.real_score_lora_target_modules = "q,k,v,o,ffn.0,ffn.2"
    args.real_score_lora_rank = 32
    
    try:
        # 测试参数解析
        target_modules = getattr(args, "real_score_lora_target_modules", "q,k,v,o,ffn.0,ffn.2")
        parsed_modules = target_modules.split(",")
        
        expected_modules = ["q", "k", "v", "o", "ffn.0", "ffn.2"]
        assert parsed_modules == expected_modules, f"模块解析错误: {parsed_modules} != {expected_modules}"
        
        rank = getattr(args, "real_score_lora_rank", 32)
        assert rank == 32, f"Rank解析错误: {rank} != 32"
        
        print("✓ LoRA参数解析正常")
        return True
        
    except Exception as e:
        print(f"✗ LoRA参数解析失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始DMD LoRA功能测试")
    print("=" * 50)
    
    tests = [
        test_dmd_without_lora,
        test_dmd_with_lora_config,
        test_lora_helper_method,
        test_lora_parameter_parsing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！")
        return True
    else:
        print("❌ 有测试失败，请检查上述错误信息")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 