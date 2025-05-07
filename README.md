[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/Lucien-6/mask-analyze)](https://github.com/Lucien-6/mask-analyze/releases)

# Mask Analyzer | 掩膜数据分析工具

一个用于分析处理掩膜图像序列的工具，特别适用于生物和微生物图像分析。

## 目录

- [功能特点](#功能特点)
- [系统要求](#系统要求)
- [使用方法](#使用方法)
  - [使用可执行文件（推荐）](#使用可执行文件推荐)
  - [从源码运行](#从源码运行)
  - [使用指南](#使用指南)
- [打包为可执行文件](#打包为可执行文件)
- [常见问题](#常见问题)
- [详细文档](#详细文档)
- [许可证](#许可证)

## 功能特点

- 加载并处理原始图片序列与掩膜图片序列
- 支持多种掩膜处理操作（膨胀、腐蚀、开闭运算等）
- 实时预览原始图片与掩膜叠加效果
- 分析计算每个对象的面积、长短轴、纵横比等数据
- 生成多种统计图表（时间序列图、分布直方图等）
- 支持导出处理后的掩膜和分析数据
- 支持手动排除特定对象的功能

## 系统要求

- Windows 10/11, macOS 10.12+, 或 Linux
- 如使用 Python 源码运行，需要安装以下 Python 库:
  - Python 3.10+
  - PyQt5
  - NumPy
  - OpenCV (opencv-python)
  - Matplotlib
  - Pandas

## 使用方法

### 使用可执行文件（推荐）

1. 从[发布页面](https://github.com/Lucien-6/mask-analyze/releases)下载最新版本的可执行文件
2. 直接运行可执行文件，无需安装任何依赖

### 从源码运行

1. 克隆或下载此代码库
2. 安装所需依赖:

```bash
pip install numpy opencv-python matplotlib PyQt5 pandas
```

3. 运行主程序:

```bash
python main.py
```

### 使用指南

1. 程序启动后:

   - 浏览选择原始图片目录和掩膜图片目录
   - 设置参数（μm/pixel 换算系数和帧率）
   - 点击"加载图像"按钮

2. 数据加载后:
   - 使用左侧面板设置掩膜处理操作
   - 使用中央面板预览和播放图像序列
   - 使用右侧面板查看统计图表
   - 通过菜单栏的"导出"功能导出结果

## 打包为可执行文件

如果需要将程序打包成独立的可执行文件，可以使用提供的打包脚本:

1. 安装 PyInstaller:

```bash
pip install pyinstaller
```

2. 运行打包脚本:

```bash
python build.py
```

3. 打包完成后，可执行文件将位于`dist`目录中

## 常见问题

- **问题**: 程序无法启动或加载图像
  **解决方案**: 确保安装了所有必要的依赖，并且提供的图像路径正确

- **问题**: 导出功能失败
  **解决方案**: 确保导出目录有写入权限

## 详细文档

详细的使用说明可通过程序的"帮助"菜单查看。

## 许可证

本仓库遵循 [MIT 许可证](./LICENSE)。

Copyright (c) 2025 Lucien
