#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志模块：提供统一的日志记录功能

Author: Lucien
Email: lucien-6@qq.com
Date: 2025-12-05
"""
import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
    name: str = "mask_analyzer",
    log_file: Optional[str] = None,
    level: int = logging.DEBUG,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3
) -> logging.Logger:
    """
    配置并返回应用程序日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，None则使用默认路径
        level: 日志级别
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份日志文件数量
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 日志格式
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file is None:
        # 默认日志文件路径
        log_dir = os.path.join(os.path.expanduser("~"), ".mask_analyzer", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "mask_analyzer.log")
    
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except Exception as e:
        # 如果无法创建文件处理器，仅使用控制台
        logger.warning(f"无法创建日志文件处理器: {e}")
    
    return logger


# 创建全局日志记录器
logger = setup_logger()


def get_logger(name: str = None) -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 子日志记录器名称，None则返回主日志记录器
        
    Returns:
        日志记录器实例
    """
    if name:
        return logging.getLogger(f"mask_analyzer.{name}")
    return logger

