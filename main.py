#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
掩膜数据分析工具 - 主程序

Author: Lucien
Email: lucien-6@qq.com
Date: 2025-12-05
"""
import sys
import os
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QIcon

from core.logger import get_logger

# 获取主模块日志记录器
logger = get_logger("main")

# 添加matplotlib字体配置
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 优先使用微软雅黑，备选黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.family'] = 'sans-serif'  # 使用无衬线字体


def global_exception_handler(exc_type, exc_value, exc_tb):
    """
    全局异常处理函数，捕获未处理的异常并记录日志
    
    Args:
        exc_type: 异常类型
        exc_value: 异常值
        exc_tb: 异常追踪信息
    """
    # 忽略键盘中断异常
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    
    # 格式化异常追踪信息
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_text = ''.join(tb_lines)
    
    # 记录到日志
    logger.error(f"未捕获的异常:\n{tb_text}")
    
    # 显示错误对话框给用户
    try:
        error_msg = f"程序发生未预期的错误:\n\n{exc_type.__name__}: {exc_value}\n\n详细信息已记录到日志文件。"
        QMessageBox.critical(None, "程序错误", error_msg)
    except Exception:
        # 如果无法显示对话框，至少打印到控制台
        print(f"严重错误: {exc_type.__name__}: {exc_value}")


def main():
    """主函数"""
    # 注册全局异常处理
    sys.excepthook = global_exception_handler
    
    try:
        # 导入主窗口（延迟导入，以便异常处理器先注册）
        from ui.main_window import MainWindow
        
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
        
        logger.info("应用程序启动")
        
        window = MainWindow()
        window.show()
        
        exit_code = app.exec_()
        
        logger.info(f"应用程序退出，退出码: {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        # 捕获主函数中的异常
        logger.error(f"主函数执行出错: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
