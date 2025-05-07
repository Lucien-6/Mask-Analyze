#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打包脚本：将应用程序打包成独立的可执行文件
"""
import os
import sys
import shutil
import subprocess
import platform

def main():
    """主函数：执行打包操作"""
    print("开始打包应用程序...")
    
    # 清理之前的构建目录
    for dir_name in ['build', 'dist']:
        if os.path.exists(dir_name):
            print(f"清理 {dir_name} 目录...")
            shutil.rmtree(dir_name)
    
    # 检测操作系统
    system = platform.system()
    
    # 设置图标路径
    if system == 'Windows':
        icon_option = '--icon=resources/app_icon.ico'
    elif system == 'Darwin':  # macOS
        icon_option = '--icon=resources/app_icon.icns'
    else:  # Linux
        icon_option = '--icon=resources/app_icon.png'
    
    # 构建 PyInstaller 命令
    cmd = [
        'pyinstaller',
        '--name=掩膜数据分析工具',
        '--onefile',  # 打包成单个文件
        '--windowed',  # 不显示控制台窗口
        icon_option,
        '--add-data=resources;resources',  # 包含资源文件夹
        '--noconfirm',  # 不询问覆盖确认
        '--clean',  # 在构建之前清理PyInstaller缓存
        '--upx-dir=E:\\VSCode\\UPX',  # 指定UPX可执行文件目录
        '--add-binary=resources/app_icon.ico;.',  # 确保图标文件也作为二进制文件添加
        'main.py'  # 主程序文件
    ]
    
    # Windows系统使用分号，Unix系统使用冒号作为路径分隔符
    if system != 'Windows':
        cmd[5] = '--add-data=resources:resources'
        # 对于非Windows系统，需要调整UPX路径
        cmd[8] = '--upx-dir=/usr/local/bin'  # Unix系统中UPX的常见位置
        cmd[9] = '--add-binary=resources/app_icon.png:.'  # 在非Windows系统中使用PNG图标
    
    # 执行打包命令
    print("正在执行PyInstaller打包...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    
    print("打包完成!")
    print(f"可执行文件位于: {os.path.join('dist', '掩膜数据分析工具')}")
    
    # 复制示例数据（如果有）
    try:
        example_src = os.path.join('resources', 'example_data')
        example_dst = os.path.join('dist', 'example_data')
        if os.path.exists(example_src):
            print("复制示例数据...")
            shutil.copytree(example_src, example_dst)
    except Exception as e:
        print(f"复制示例数据时出错: {str(e)}")

if __name__ == "__main__":
    main() 