#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预览控制部件：用于控制图像序列播放
"""
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider, QLabel
)
from PyQt5.QtCore import Qt


class PreviewControlWidget(QWidget):
    """预览控制部件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建播放控制按钮
        self.prev_button = QPushButton("上一帧")
        self.prev_button.setFixedWidth(80)
        layout.addWidget(self.prev_button)
        
        self.play_button = QPushButton("播放")
        self.play_button.setFixedWidth(80)
        layout.addWidget(self.play_button)
        
        self.next_button = QPushButton("下一帧")
        self.next_button.setFixedWidth(80)
        layout.addWidget(self.next_button)
        
        # 创建帧计数标签
        self.frame_label = QLabel("帧: 0/0")
        self.frame_label.setMinimumWidth(100)
        self.frame_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.frame_label)
        
        # 创建滑块
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(0)
        self.slider.setTracking(True)
        layout.addWidget(self.slider)
        
        # 设置布局比例
        layout.setStretchFactor(self.slider, 10)
        layout.setStretchFactor(self.frame_label, 1)
        layout.setStretchFactor(self.prev_button, 1)
        layout.setStretchFactor(self.play_button, 1)
        layout.setStretchFactor(self.next_button, 1) 