#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载部件：用于选择数据源和设置参数
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QLineEdit, QFileDialog, QGroupBox,
    QFormLayout, QSpinBox, QDoubleSpinBox, QSizePolicy
)
from PyQt5.QtCore import Qt
import os


class DataLoaderWidget(QWidget):
    """数据加载部件"""
    
    def __init__(self, parent=None):
        """初始化"""
        super().__init__(parent)
        # 存储完整路径
        self.original_dir_full_path = ""
        self.mask_dir_full_path = ""
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(15)  # 增加垂直间距
        
        # 表单布局
        form_layout = QFormLayout()
        form_layout.setSpacing(15)  # 增加表单项之间的间距
        form_layout.setLabelAlignment(Qt.AlignRight)  # 标签右对齐
        form_layout.setFormAlignment(Qt.AlignLeft)   # 控件左对齐
        
        # 原始图片目录
        self.original_dir_edit = QLineEdit()
        self.original_dir_edit.setMinimumWidth(150)  # 设置最小宽度
        browse_original_btn = QPushButton("浏览...")
        browse_original_btn.setMinimumWidth(80)  # 设置最小宽度
        browse_original_btn.clicked.connect(self.browse_original_dir)
        
        original_layout = QHBoxLayout()
        original_layout.setSpacing(10)  # 增加水平间距
        original_layout.addWidget(self.original_dir_edit)
        original_layout.addWidget(browse_original_btn)
        form_layout.addRow("原始图片目录:", original_layout)
        
        # 掩膜图片目录
        self.mask_dir_edit = QLineEdit()
        self.mask_dir_edit.setMinimumWidth(150)  # 设置最小宽度
        browse_mask_btn = QPushButton("浏览...")
        browse_mask_btn.setMinimumWidth(80)  # 设置最小宽度
        browse_mask_btn.clicked.connect(self.browse_mask_dir)
        
        mask_layout = QHBoxLayout()
        mask_layout.setSpacing(10)  # 增加水平间距
        mask_layout.addWidget(self.mask_dir_edit)
        mask_layout.addWidget(browse_mask_btn)
        form_layout.addRow("掩膜图片目录:", mask_layout)
        
        # 起始帧和结束帧
        frame_range_layout = QHBoxLayout()
        frame_range_layout.setSpacing(20)  # 增加水平间距
        
        # 添加左侧弹性空间
        frame_range_layout.addStretch()
        
        # 起始帧
        start_layout = QHBoxLayout()
        start_layout.setSpacing(5)
        start_label = QLabel("起始帧:")
        self.start_idx_spin = QSpinBox()
        self.start_idx_spin.setRange(0, 9999)
        self.start_idx_spin.setValue(0)
        self.start_idx_spin.setMinimumWidth(80)  # 设置最小宽度
        start_layout.addWidget(start_label)
        start_layout.addWidget(self.start_idx_spin)
        frame_range_layout.addLayout(start_layout)
        
        # 结束帧
        end_layout = QHBoxLayout()
        end_layout.setSpacing(5)
        end_label = QLabel("结束帧:")
        self.end_idx_spin = QSpinBox()
        self.end_idx_spin.setRange(0, 9999)
        self.end_idx_spin.setValue(0)
        self.end_idx_spin.setMinimumWidth(80)  # 设置最小宽度
        end_layout.addWidget(end_label)
        end_layout.addWidget(self.end_idx_spin)
        frame_range_layout.addLayout(end_layout)
        
        # 添加右侧弹性空间
        frame_range_layout.addStretch()
        
        # 使用空标签添加到表单布局，并设置对齐方式
        form_layout.addRow("", frame_range_layout)  # 空标签，让控件对齐
        
        layout.addLayout(form_layout)
        
        # 创建帧率和像素换算系数控件（但不显示在界面上）
        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.1, 1000)
        self.frame_rate_spin.setValue(30)
        self.frame_rate_spin.setSingleStep(1)
        self.frame_rate_spin.setDecimals(1)

        self.scale_factor_spin = QDoubleSpinBox()
        self.scale_factor_spin.setRange(0.001, 1000)
        self.scale_factor_spin.setValue(1.0)
        self.scale_factor_spin.setSingleStep(0.1)
        self.scale_factor_spin.setDecimals(3)
        
        # 添加空白
        layout.addSpacing(10)
        
        # 加载按钮
        self.load_button = QPushButton("加载图像")
        self.load_button.setMinimumHeight(45)  # 增加按钮高度
        self.load_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 10px 25px;
                font-size: 24px;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        layout.addWidget(self.load_button, alignment=Qt.AlignCenter)
        
        # 连接信号
        self.original_dir_edit.textChanged.connect(self.update_frame_range)
        self.mask_dir_edit.textChanged.connect(self.update_frame_range)

    def browse_original_dir(self):
        """浏览原始图片目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择原始图片目录")
        if directory:
            # 存储完整路径
            self.original_dir_full_path = directory
            # 只显示文件夹名称
            folder_name = os.path.basename(directory)
            self.original_dir_edit.setText(folder_name)
            self.update_frame_range()
            
    def browse_mask_dir(self):
        """浏览掩膜图片目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择掩膜图片目录")
        if directory:
            # 存储完整路径
            self.mask_dir_full_path = directory
            # 只显示文件夹名称
            folder_name = os.path.basename(directory)
            self.mask_dir_edit.setText(folder_name)
            self.update_frame_range()

    def update_frame_range(self):
        """更新起始帧和结束帧范围"""
        if not self.original_dir_full_path or not self.mask_dir_full_path:
            return
            
        try:
            import glob
            
            # 使用完整路径获取原始图像文件列表
            original_files = sorted(glob.glob(os.path.join(self.original_dir_full_path, '*.*')))
            mask_files = sorted(glob.glob(os.path.join(self.mask_dir_full_path, '*.*')))
            
            # 检查是否有图像文件
            if not original_files or not mask_files:
                return
                
            # 设置结束帧为图像数量-1
            image_count = min(len(original_files), len(mask_files))
            if image_count > 0:
                self.start_idx_spin.setValue(0)
                self.end_idx_spin.setValue(image_count - 1)
                
        except Exception as e:
            print(f"更新帧范围时出错: {str(e)}")
            
    # 获取原始图片目录的完整路径
    def get_original_dir(self):
        return self.original_dir_full_path
    
    # 获取掩膜图片目录的完整路径
    def get_mask_dir(self):
        return self.mask_dir_full_path 