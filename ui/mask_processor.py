#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
掩膜处理部件：用于设置和控制掩膜处理参数
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QCheckBox, QGroupBox, QGridLayout,
    QFormLayout, QSpinBox, QDoubleSpinBox, QScrollArea,
    QListWidget
)
from PyQt5.QtCore import Qt


class MaskProcessorWidget(QWidget):
    """掩膜处理部件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建滚动区域包含所有控件
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        main_layout.addWidget(scroll)
        
        # 滚动区域的内容部件
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(2)  # 减小整体间距
        
        # 创建处理操作区域容器
        operations_widget = QWidget()
        operations_layout = QVBoxLayout(operations_widget)
        operations_layout.setSpacing(2)
        operations_layout.setContentsMargins(0, 0, 0, 0)
        
        # 膨胀操作
        self.dilation_check = QCheckBox("膨胀")
        operations_layout.addWidget(self.dilation_check)
        
        dilation_form = QFormLayout()
        dilation_form.setContentsMargins(20, 0, 0, 0)  # 缩进布局
        
        self.dilation_kernel_spin = QSpinBox()
        self.dilation_kernel_spin.setMinimum(1)
        self.dilation_kernel_spin.setMaximum(20)
        self.dilation_kernel_spin.setValue(3)
        dilation_form.addRow("核大小 (像素):", self.dilation_kernel_spin)
        
        self.dilation_iter_spin = QSpinBox()
        self.dilation_iter_spin.setMinimum(1)
        self.dilation_iter_spin.setMaximum(10)
        self.dilation_iter_spin.setValue(1)
        dilation_form.addRow("迭代次数:", self.dilation_iter_spin)
        
        operations_layout.addLayout(dilation_form)
        
        # 腐蚀操作
        self.erosion_check = QCheckBox("腐蚀")
        operations_layout.addWidget(self.erosion_check)
        
        erosion_form = QFormLayout()
        erosion_form.setContentsMargins(20, 0, 0, 0)
        
        self.erosion_kernel_spin = QSpinBox()
        self.erosion_kernel_spin.setMinimum(1)
        self.erosion_kernel_spin.setMaximum(20)
        self.erosion_kernel_spin.setValue(3)
        erosion_form.addRow("核大小 (像素):", self.erosion_kernel_spin)
        
        self.erosion_iter_spin = QSpinBox()
        self.erosion_iter_spin.setMinimum(1)
        self.erosion_iter_spin.setMaximum(10)
        self.erosion_iter_spin.setValue(1)
        erosion_form.addRow("迭代次数:", self.erosion_iter_spin)
        
        operations_layout.addLayout(erosion_form)
        
        # 开运算
        self.opening_check = QCheckBox("开运算 (先腐蚀后膨胀)")
        operations_layout.addWidget(self.opening_check)
        
        opening_form = QFormLayout()
        opening_form.setContentsMargins(20, 0, 0, 0)
        
        self.opening_kernel_spin = QSpinBox()
        self.opening_kernel_spin.setMinimum(1)
        self.opening_kernel_spin.setMaximum(20)
        self.opening_kernel_spin.setValue(3)
        opening_form.addRow("核大小 (像素):", self.opening_kernel_spin)
        
        self.opening_iter_spin = QSpinBox()
        self.opening_iter_spin.setMinimum(1)
        self.opening_iter_spin.setMaximum(10)
        self.opening_iter_spin.setValue(1)
        opening_form.addRow("迭代次数:", self.opening_iter_spin)
        
        operations_layout.addLayout(opening_form)
        
        # 闭运算
        self.closing_check = QCheckBox("闭运算 (先膨胀后腐蚀)")
        operations_layout.addWidget(self.closing_check)
        
        closing_form = QFormLayout()
        closing_form.setContentsMargins(20, 0, 0, 0)
        
        self.closing_kernel_spin = QSpinBox()
        self.closing_kernel_spin.setMinimum(1)
        self.closing_kernel_spin.setMaximum(20)
        self.closing_kernel_spin.setValue(3)
        closing_form.addRow("核大小 (像素):", self.closing_kernel_spin)
        
        self.closing_iter_spin = QSpinBox()
        self.closing_iter_spin.setMinimum(1)
        self.closing_iter_spin.setMaximum(10)
        self.closing_iter_spin.setValue(1)
        closing_form.addRow("迭代次数:", self.closing_iter_spin)
        
        operations_layout.addLayout(closing_form)
        
        # 填补孔洞
        self.fill_holes_check = QCheckBox("填补孔洞")
        operations_layout.addWidget(self.fill_holes_check)
        
        # 添加额外间距
        operations_layout.addSpacing(20)
        
        # 面积阈值过滤
        self.area_filter_check = QCheckBox("面积阈值过滤")
        operations_layout.addWidget(self.area_filter_check)
        
        area_filter_form = QFormLayout()
        area_filter_form.setContentsMargins(20, 0, 0, 0)
        
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setMinimum(0)
        self.min_area_spin.setMaximum(100000)
        self.min_area_spin.setValue(10)
        area_filter_form.addRow("最小面积 (μm²):", self.min_area_spin)
        
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setMinimum(0)
        self.max_area_spin.setMaximum(100000)
        self.max_area_spin.setValue(0)
        self.max_area_spin.setSpecialValueText("无限制")
        area_filter_form.addRow("最大面积 (μm²):", self.max_area_spin)
        
        operations_layout.addLayout(area_filter_form)
        
        # 添加应用于所有帧的选项
        self.apply_all_check = QCheckBox("应用到所有帧")
        operations_layout.addWidget(self.apply_all_check)
        
        # 设置处理操作区域的固定高度
        operations_widget.setFixedHeight(580)  # 设置固定高度
        scroll_layout.addWidget(operations_widget)
        
        # 添加分隔线
        separator = QWidget()
        separator.setFixedHeight(20)
        separator.setStyleSheet("background-color: transparent;")
        scroll_layout.addWidget(separator)
        
        # 全局应用设置组
        global_group = QGroupBox("边缘对象处理")
        global_group.setStyleSheet("font-size: 24px; font-weight: bold; color: #005500;")
        global_layout = QVBoxLayout(global_group)
        global_layout.setContentsMargins(10, 15, 10, 5)  # 减小内边距
        global_layout.setSpacing(5)  # 减小控件间距
        
        # 剔除边缘对象
        self.exclude_edge_check = QCheckBox("剔除被边缘截断对象")
        self.exclude_edge_check.setStyleSheet("font-size: 20px; color: black;") #颜色设为黑色
        global_layout.addWidget(self.exclude_edge_check)
        
        # 添加说明标签
        note_label = QLabel("注意：此设置默认应用于所有帧！")
        note_label.setStyleSheet("color: #555555; font-style: italic;font-size: 18px;")
        note_label.setWordWrap(True)
        global_layout.addWidget(note_label)
        
        # 设置边缘对象处理区域的固定高度
        global_group.setFixedHeight(120)  # 设置固定高度
        scroll_layout.addWidget(global_group)
        
        # 添加分隔线
        separator2 = QWidget()
        separator2.setFixedHeight(20)
        separator2.setStyleSheet("background-color: transparent;")
        scroll_layout.addWidget(separator2)
        
        # 创建手动剔除对象GroupBox
        manual_exclude_group = QGroupBox("手动剔除对象")
        manual_exclude_group.setStyleSheet("font-size: 24px; font-weight: bold; color: #005500;")
        manual_exclude_layout = QVBoxLayout(manual_exclude_group)
        manual_exclude_layout.setContentsMargins(10, 15, 10, 10)
        manual_exclude_layout.setSpacing(5)  # 减小控件间距
        
        # 添加说明标签
        manual_exclude_label = QLabel("右键点击对象可添加/移除标记")
        manual_exclude_label.setStyleSheet("color: #555555;font-size: 18px;")
        manual_exclude_label.setWordWrap(True)
        manual_exclude_label.setFixedHeight(25)  # 设置固定高度
        manual_exclude_layout.addWidget(manual_exclude_label)
        
        # 添加手动剔除对象列表
        self.excluded_objects_list = QListWidget()
        self.excluded_objects_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.excluded_objects_list.setToolTip("显示当前帧中被手动剔除的对象列表")
        manual_exclude_layout.addWidget(self.excluded_objects_list, 1)  # 分配更多空间比例
        
        # 添加清空按钮
        self.clear_excluded_btn = QPushButton("清空当前帧剔除标记")
        self.clear_excluded_btn.setFixedHeight(35)  # 设置固定高度
        self.clear_excluded_btn.setStyleSheet("font-size: 18px;")
        manual_exclude_layout.addWidget(self.clear_excluded_btn)
        
        # 不设置固定高度，让其自动占用剩余空间
        scroll_layout.addWidget(manual_exclude_group, 1)  # 分配更多空间比例
        
        # 添加按钮容器
        buttons_container = QWidget()
        buttons_container_layout = QVBoxLayout(buttons_container)
        buttons_container_layout.setContentsMargins(0, 0, 0, 0)
        buttons_container_layout.setSpacing(5)
        
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(5)
        
        # 添加应用按钮
        self.apply_button = QPushButton("应用处理")
        self.apply_button.setFixedHeight(50)  # 减小高度
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 24px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #367c39;
            }
        """)
        buttons_layout.addWidget(self.apply_button)
        
        # 添加撤销和还原按钮的容器
        undo_restore_layout = QHBoxLayout()
        undo_restore_layout.setSpacing(5)
        
        # 添加撤销最近处理按钮
        self.undo_button = QPushButton("撤销最近处理")
        self.undo_button.setFixedHeight(40)  # 减小高度
        self.undo_button.setStyleSheet("""
            QPushButton {
                background-color: #FFB74D;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 18px;
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #FFA726;
            }
            QPushButton:pressed {
                background-color: #FB8C00;
            }
        """)
        undo_restore_layout.addWidget(self.undo_button)
        
        # 添加还原初始掩膜按钮
        self.restore_button = QPushButton("还原初始掩膜")
        self.restore_button.setFixedHeight(40)  # 减小高度
        self.restore_button.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 18px;
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #E53935;
            }
            QPushButton:pressed {
                background-color: #D32F2F;
            }
        """)
        undo_restore_layout.addWidget(self.restore_button)
        
        buttons_layout.addLayout(undo_restore_layout)
        buttons_container_layout.addLayout(buttons_layout)
        
        # 设置按钮区域的固定高度
        buttons_container.setFixedHeight(200)  # 设置固定高度
        scroll_layout.addWidget(buttons_container)
        
        # 设置滚动区域
        scroll.setWidget(scroll_content) 