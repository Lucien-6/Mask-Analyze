#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图表显示部件：用于显示各种统计图表
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget,
    QScrollArea, QGridLayout, QGroupBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import Dict, Any

from core.data_analyzer import ChartGenerator


class ChartViewWidget(QWidget):
    """图表显示部件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.init_ui()
        
        # 初始化存储图表的变量
        self.object_count_chart = None
        self.area_fraction_chart = None
        self.global_area_histogram = None
        self.global_aspect_ratio_histogram = None
        self.area_histogram = None
        self.major_axis_histogram = None
        self.minor_axis_histogram = None
        self.aspect_ratio_histogram = None
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建图表选项卡
        self.tab_widget = QTabWidget()
        
        # 创建全局结果选项卡 (原时间序列选项卡)
        global_tab = QWidget()
        global_layout = QVBoxLayout(global_tab)
        
        # 对象数量-时间曲线
        self.object_count_canvas_container = QWidget()
        self.object_count_canvas_layout = QVBoxLayout(self.object_count_canvas_container)
        self.object_count_canvas_layout.setContentsMargins(0, 0, 0, 0)
        global_layout.addWidget(self.object_count_canvas_container)
        
        # 面积分数-时间曲线
        self.area_fraction_canvas_container = QWidget()
        self.area_fraction_canvas_layout = QVBoxLayout(self.area_fraction_canvas_container)
        self.area_fraction_canvas_layout.setContentsMargins(0, 0, 0, 0)
        global_layout.addWidget(self.area_fraction_canvas_container)
        
        # 添加全局面积分布直方图
        self.global_area_histogram_container = QWidget()
        self.global_area_histogram_layout = QVBoxLayout(self.global_area_histogram_container)
        self.global_area_histogram_layout.setContentsMargins(0, 0, 0, 0)
        global_layout.addWidget(self.global_area_histogram_container)
        
        # 添加全局纵横比分布直方图
        self.global_aspect_ratio_histogram_container = QWidget()
        self.global_aspect_ratio_histogram_layout = QVBoxLayout(self.global_aspect_ratio_histogram_container)
        self.global_aspect_ratio_histogram_layout.setContentsMargins(0, 0, 0, 0)
        global_layout.addWidget(self.global_aspect_ratio_histogram_container)
        
        # 添加滚动区域来包含所有全局图表
        global_scroll = QScrollArea()
        global_scroll.setWidget(global_tab)
        global_scroll.setWidgetResizable(True)
        global_scroll.setFrameShape(QScrollArea.NoFrame)
        
        self.tab_widget.addTab(global_scroll, "全局结果")
        
        # 创建当前帧结果选项卡 (原分布直方图选项卡)
        frame_tab = QWidget()
        frame_layout = QVBoxLayout(frame_tab)
        
        # 面积分布直方图
        self.area_histogram_container = QWidget()
        self.area_histogram_layout = QVBoxLayout(self.area_histogram_container)
        self.area_histogram_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.area_histogram_container)
        
        # 长轴分布直方图
        self.major_axis_histogram_container = QWidget()
        self.major_axis_histogram_layout = QVBoxLayout(self.major_axis_histogram_container)
        self.major_axis_histogram_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.major_axis_histogram_container)
        
        # 短轴分布直方图
        self.minor_axis_histogram_container = QWidget()
        self.minor_axis_histogram_layout = QVBoxLayout(self.minor_axis_histogram_container)
        self.minor_axis_histogram_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.minor_axis_histogram_container)
        
        # 纵横比分布直方图
        self.aspect_ratio_histogram_container = QWidget()
        self.aspect_ratio_histogram_layout = QVBoxLayout(self.aspect_ratio_histogram_container)
        self.aspect_ratio_histogram_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.aspect_ratio_histogram_container)
        
        # 添加滚动区域来包含所有直方图
        frame_scroll = QScrollArea()
        frame_scroll.setWidget(frame_tab)
        frame_scroll.setWidgetResizable(True)
        frame_scroll.setFrameShape(QScrollArea.NoFrame)
        
        self.tab_widget.addTab(frame_scroll, "当前帧结果")
        
        main_layout.addWidget(self.tab_widget)
    
    def update_charts(self, chart_generator: ChartGenerator, global_stats: Dict[str, Any]):
        """
        更新所有图表
        
        Args:
            chart_generator: 图表生成器
            global_stats: 全局统计数据
        """
        # 更新对象数量-时间曲线
        fig, canvas = chart_generator.create_object_count_chart(global_stats)
        self._update_canvas(self.object_count_canvas_layout, canvas)
        self.object_count_chart = (fig, canvas)
        
        # 更新面积分数-时间曲线
        fig, canvas = chart_generator.create_area_fraction_chart(global_stats)
        self._update_canvas(self.area_fraction_canvas_layout, canvas)
        self.area_fraction_chart = (fig, canvas)
        
        # 更新全局面积分布直方图
        fig, canvas = chart_generator.create_area_histogram(global_stats)
        self._update_canvas(self.global_area_histogram_layout, canvas)
        self.global_area_histogram = (fig, canvas)
        
        # 更新全局纵横比分布直方图
        fig, canvas = chart_generator.create_aspect_ratio_histogram(global_stats)
        self._update_canvas(self.global_aspect_ratio_histogram_layout, canvas)
        self.global_aspect_ratio_histogram = (fig, canvas)
    
    def update_histograms(self, chart_generator: ChartGenerator, global_stats: Dict[str, Any]):
        """
        更新所有直方图
        
        Args:
            chart_generator: 图表生成器
            global_stats: 全局统计数据
        """
        # 更新面积分布直方图
        fig, canvas = chart_generator.create_area_histogram(global_stats)
        self._update_canvas(self.area_histogram_layout, canvas)
        self.area_histogram = (fig, canvas)
        
        # 更新长轴分布直方图
        fig, canvas = chart_generator.create_major_axis_histogram(global_stats)
        self._update_canvas(self.major_axis_histogram_layout, canvas)
        self.major_axis_histogram = (fig, canvas)
        
        # 更新短轴分布直方图
        fig, canvas = chart_generator.create_minor_axis_histogram(global_stats)
        self._update_canvas(self.minor_axis_histogram_layout, canvas)
        self.minor_axis_histogram = (fig, canvas)
        
        # 更新纵横比分布直方图
        fig, canvas = chart_generator.create_aspect_ratio_histogram(global_stats)
        self._update_canvas(self.aspect_ratio_histogram_layout, canvas)
        self.aspect_ratio_histogram = (fig, canvas)
    
    def update_frame_histograms(self, chart_generator: ChartGenerator, 
                               frame_stats: Dict[str, Any], global_stats: Dict[str, Any]):
        """
        更新当前帧的直方图
        
        Args:
            chart_generator: 图表生成器
            frame_stats: 当前帧的统计数据
            global_stats: 全局统计数据
        """
        # 使用当前帧的数据创建直方图
        # 更新面积分布直方图
        fig, canvas = chart_generator.create_frame_area_histogram(frame_stats)
        self._update_canvas(self.area_histogram_layout, canvas)
        self.area_histogram = (fig, canvas)
        
        # 更新长轴分布直方图
        fig, canvas = chart_generator.create_frame_major_axis_histogram(frame_stats)
        self._update_canvas(self.major_axis_histogram_layout, canvas)
        self.major_axis_histogram = (fig, canvas)
        
        # 更新短轴分布直方图
        fig, canvas = chart_generator.create_frame_minor_axis_histogram(frame_stats)
        self._update_canvas(self.minor_axis_histogram_layout, canvas)
        self.minor_axis_histogram = (fig, canvas)
        
        # 更新纵横比分布直方图
        fig, canvas = chart_generator.create_frame_aspect_ratio_histogram(frame_stats)
        self._update_canvas(self.aspect_ratio_histogram_layout, canvas)
        self.aspect_ratio_histogram = (fig, canvas)
    
    def _update_canvas(self, layout, canvas):
        """
        更新布局中的画布
        
        Args:
            layout: 布局
            canvas: 画布
        """
        # 清空当前布局
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # 添加新画布
        layout.addWidget(canvas) 