#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
掩膜数据分析工具 - 主程序
"""
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from ui.main_window import MainWindow

# 添加matplotlib字体配置
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 优先使用微软雅黑，备选黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用图标
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources', 'app_icon.ico')
    # 如果是打包后的可执行文件，可能需要调整路径
    if not os.path.exists(icon_path):
        # 尝试在打包环境中找到图标
        base_path = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        possible_paths = [
            os.path.join(base_path, 'resources', 'app_icon.ico'),  # Windows图标
            os.path.join(base_path, 'resources', 'app_icon.png'),  # 通用图标
        ]
        for path in possible_paths:
            if os.path.exists(path):
                icon_path = path
                break
    
    # 如果找到图标，则设置应用图标
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 