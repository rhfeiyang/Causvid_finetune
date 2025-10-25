#!/usr/bin/env python3
"""
CausVid增强轨迹可视化功能使用示例
展示如何使用新的擦除/新增操作区分功能
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_autoregressive_inference import generate_trajectory_visualization

def example_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 输入视频路径（请修改为您的视频路径）
    video_path = "/home/coder/code/video_sketch/data/custom_sketch0618/trunc_compress81_sample/bird.mp4"
    
    if not os.path.exists(video_path):
        print(f"示例视频不存在: {video_path}")
        print("请修改 video_path 为有效的视频文件路径")
        return
    
    # 生成默认设置的轨迹图
    output_path = "trajectory_basic.png"
    
    print(f"正在生成轨迹可视化: {output_path}")
    result = generate_trajectory_visualization(
        video_path=video_path,
        output_path=output_path
    )
    
    if result:
        print(f"✓ 成功生成: {result}")
        print("默认设置: 新增100%透明度，擦除30%透明度，heat色彩方案")
    else:
        print("✗ 生成失败")

def example_custom_settings():
    """自定义设置示例"""
    print("\n=== 自定义设置示例 ===")
    
    video_path = "/home/coder/code/video_sketch/data/custom_sketch0618/trunc_compress81_sample/bird.mp4"
    
    if not os.path.exists(video_path):
        print(f"示例视频不存在: {video_path}")
        return
    
    # 示例1: 提高擦除可见度
    print("1. 提高擦除操作可见度（擦除80%，新增100%）")
    result1 = generate_trajectory_visualization(
        video_path=video_path,
        output_path="trajectory_high_erase.png",
        color_scheme="heat",
        erase_opacity=0.8,
        add_opacity=1.0
    )
    if result1:
        print(f"   ✓ 生成: {result1}")
    
    # 示例2: 平衡显示
    print("2. 平衡显示（擦除60%，新增80%）")
    result2 = generate_trajectory_visualization(
        video_path=video_path,
        output_path="trajectory_balanced.png",
        color_scheme="cool",
        erase_opacity=0.6,
        add_opacity=0.8
    )
    if result2:
        print(f"   ✓ 生成: {result2}")
    
    # 示例3: 彩虹色方案
    print("3. 彩虹色方案（擦除40%，新增100%）")
    result3 = generate_trajectory_visualization(
        video_path=video_path,
        output_path="trajectory_rainbow.png",
        color_scheme="rainbow",
        erase_opacity=0.4,
        add_opacity=1.0
    )
    if result3:
        print(f"   ✓ 生成: {result3}")

def example_analysis_scenarios():
    """不同分析场景的设置建议"""
    print("\n=== 分析场景设置建议 ===")
    
    scenarios = [
        {
            "name": "强调主要绘制过程",
            "description": "突出显示绘画的主要创作内容，淡化修正过程",
            "settings": {"erase_opacity": 0.2, "add_opacity": 1.0, "color_scheme": "heat"}
        },
        {
            "name": "分析修正过程",
            "description": "重点观察画家的修正和调整行为",
            "settings": {"erase_opacity": 0.9, "add_opacity": 0.7, "color_scheme": "cool"}
        },
        {
            "name": "全面过程分析",
            "description": "平衡显示所有操作，用于综合分析",
            "settings": {"erase_opacity": 0.6, "add_opacity": 0.8, "color_scheme": "rainbow"}
        },
        {
            "name": "教学演示",
            "description": "清晰区分不同操作类型，便于教学说明",
            "settings": {"erase_opacity": 0.5, "add_opacity": 1.0, "color_scheme": "heat"}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}")
        print(f"   用途: {scenario['description']}")
        settings = scenario['settings']
        print(f"   设置: 擦除{settings['erase_opacity']*100:.0f}%, 新增{settings['add_opacity']*100:.0f}%, {settings['color_scheme']}色彩")
        print(f"   命令: --erase_opacity {settings['erase_opacity']} --add_opacity {settings['add_opacity']} --trajectory_color_scheme {settings['color_scheme']}")

def main():
    """主函数"""
    print("CausVid增强轨迹可视化功能使用示例")
    print("=" * 50)
    
    # 基础使用
    example_basic_usage()
    
    # 自定义设置
    example_custom_settings()
    
    # 分析场景建议
    example_analysis_scenarios()
    
    print("\n" + "=" * 50)
    print("示例完成！")
    print("生成的图片文件:")
    for file in ["trajectory_basic.png", "trajectory_high_erase.png", 
                 "trajectory_balanced.png", "trajectory_rainbow.png"]:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024
            print(f"  ✓ {file} ({size:.1f} KB)")
        else:
            print(f"  - {file} (未生成)")
    
    print("\n💡 提示:")
    print("- 调整erase_opacity来控制擦除操作的可见度")
    print("- 调整add_opacity来控制新增操作的可见度") 
    print("- 尝试不同的color_scheme获得最佳视觉效果")
    print("- 查看README_trajectory_enhancement.md了解更多详情")

if __name__ == "__main__":
    main() 