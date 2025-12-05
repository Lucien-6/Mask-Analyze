#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出对话框：用于设置和控制数据导出
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QCheckBox, QGroupBox, QFileDialog, QProgressBar,
    QMessageBox, QDialogButtonBox
)
from PyQt5.QtCore import Qt, QCoreApplication
from typing import Dict, Any, List

from core.image_processor import ImageProcessor
from core.data_analyzer import ChartGenerator
from core.logger import get_logger

# 获取模块日志记录器
logger = get_logger("export_dialog")


class ExportDialog(QDialog):
    """导出对话框，提供数据和图表导出功能"""
    
    def __init__(self, parent, image_processor: ImageProcessor, 
                 chart_generator: ChartGenerator, export_data: Dict[str, Any]):
        """
        初始化导出对话框
        
        Args:
            parent: 父窗口
            image_processor: 图像处理器实例
            chart_generator: 图表生成器实例
            export_data: 导出数据字典
        """
        super().__init__(parent)
        
        self.image_processor = image_processor
        self.chart_generator = chart_generator
        self.export_data = export_data
        self.export_dir = None  # 导出目录，初始为None
        
        self.init_ui()
    
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("导出分析结果")
        self.setMinimumWidth(450)
        self.setMinimumHeight(550)
        
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        
        # 创建导出选项组
        export_group = QGroupBox("选择导出项目")
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(10)
        
        # 修改后的掩膜图片序列
        self.export_masks_check = QCheckBox("修改后的掩膜图片序列")
        self.export_masks_check.setChecked(True)
        export_layout.addWidget(self.export_masks_check)
        
        # 分析结果图表
        self.export_charts_check = QCheckBox("分析结果图表（4个全局图表和每帧4张分析图表）")
        self.export_charts_check.setChecked(True)
        export_layout.addWidget(self.export_charts_check)
        
        # 详细数据表格
        self.export_objects_data_check = QCheckBox("详细数据表格（每帧中所有对象的信息）")
        self.export_objects_data_check.setChecked(True)
        export_layout.addWidget(self.export_objects_data_check)
        
        main_layout.addWidget(export_group)
        
        # 创建导出路径组
        path_group = QGroupBox("导出路径")
        path_layout = QHBoxLayout(path_group)
        
        self.path_label = QLabel("请选择导出目录")
        self.path_label.setStyleSheet("font-weight: normal;")
        path_layout.addWidget(self.path_label, 1)
        
        self.browse_button = QPushButton("浏览...")
        self.browse_button.setFixedWidth(100)
        self.browse_button.clicked.connect(self.browse_export_dir)
        path_layout.addWidget(self.browse_button)
        
        main_layout.addWidget(path_group)
        
        # 创建进度信息显示
        self.progress_group = QGroupBox("导出进度")
        progress_layout = QVBoxLayout(self.progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("准备导出...")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)
        
        main_layout.addWidget(self.progress_group)
        
        # 创建按钮组
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        # 设置对话框属性
        self.setModal(True)
        self.resize(500, 350)
    
    def browse_export_dir(self):
        """浏览选择导出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "选择导出目录", os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        
        if dir_path:
            self.export_dir = dir_path
            # 如果路径太长，截断显示
            display_path = dir_path
            if len(display_path) > 40:
                display_path = "..." + display_path[-37:]
            self.path_label.setText(display_path)
            self.path_label.setToolTip(dir_path)  # 设置完整路径为工具提示
    
    def accept(self):
        """确认导出，执行导出操作"""
        # 检查是否选择了导出目录
        if not self.export_dir:
            QMessageBox.warning(self, "错误", "请先选择导出目录")
            return
        
        # 检查是否至少选择了一个导出项目
        if not any([
            self.export_masks_check.isChecked(),
            self.export_charts_check.isChecked(),
            self.export_objects_data_check.isChecked()
        ]):
            QMessageBox.warning(self, "错误", "请至少选择一个导出项目")
            return
        
        # 开始导出操作
        try:
            self.progress_bar.setValue(0)
            export_count = sum([
                self.export_masks_check.isChecked(),
                self.export_charts_check.isChecked(),
                self.export_objects_data_check.isChecked()
            ])
            progress_step = 100 / export_count
            current_progress = 0
            
            # 1. 导出修改后的掩膜图片序列
            if self.export_masks_check.isChecked():
                self.status_label.setText("导出掩膜图片序列...")
                QCoreApplication.processEvents()  # 刷新UI
                
                self.export_masks()
                
                current_progress += progress_step
                self.progress_bar.setValue(int(current_progress))
                QCoreApplication.processEvents()  # 刷新UI
            
            # 2. 导出分析结果图表
            if self.export_charts_check.isChecked():
                self.status_label.setText("导出分析结果图表...")
                QCoreApplication.processEvents()  # 刷新UI
                
                self.export_all_charts()
                
                current_progress += progress_step
                self.progress_bar.setValue(int(current_progress))
                QCoreApplication.processEvents()  # 刷新UI
            
            # 3. 导出详细数据表格
            if self.export_objects_data_check.isChecked():
                self.status_label.setText("导出详细数据表格...")
                QCoreApplication.processEvents()  # 刷新UI
                
                self.export_objects_data_excel()
                
                current_progress += progress_step
                self.progress_bar.setValue(int(current_progress))
                QCoreApplication.processEvents()  # 刷新UI
            
            # 导出完成
            self.status_label.setText("导出完成")
            self.progress_bar.setValue(100)
            
            QMessageBox.information(self, "成功", "数据导出完成")
            super().accept()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出过程中发生错误:\n{str(e)}")
            self.status_label.setText(f"导出失败: {str(e)}")
    
    def export_masks(self):
        """导出修改后的掩膜图片序列"""
        # 创建掩膜导出目录
        masks_dir = os.path.join(self.export_dir, "processed_masks")
        os.makedirs(masks_dir, exist_ok=True)
        
        # 获取掩膜图像总数
        total_masks = len(self.image_processor.processed_masks)
        if total_masks == 0:
            raise ValueError("没有可用的掩膜图像")
        
        # 导出每一帧的掩膜
        for i, mask in enumerate(self.image_processor.processed_masks):
            # 更新进度和状态
            if i % 5 == 0 or i == total_masks - 1:  # 每5帧更新一次UI，减少UI更新开销
                progress = int(100 * (i + 1) / total_masks)
                self.progress_bar.setValue(progress)
                self.status_label.setText(f"导出掩膜图像 {i+1}/{total_masks}...")
                QCoreApplication.processEvents()  # 刷新UI
            
            # 保存掩膜图像
            filename = os.path.join(masks_dir, f"mask_{i:04d}.png")
            cv2.imwrite(filename, mask)
    
    def export_all_charts(self):
        """导出所有分析图表，包括4个全局图表和每帧的4个分析图表"""
        # 创建图表导出目录
        charts_dir = os.path.join(self.export_dir, "analysis_charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # 设置计数器，用于进度更新
        total_charts = 4  # 4个全局图表
        completed_charts = 0
        
        # 1. 导出对象数量-时间曲线图表
        self.status_label.setText("导出对象数量-时间曲线图...")
        QCoreApplication.processEvents()  # 刷新UI
        
        if hasattr(self.parent(), "chart_view_widget") and hasattr(self.parent().chart_view_widget, "object_count_chart"):
            try:
                fig, _ = self.parent().chart_view_widget.object_count_chart
                chart_path = os.path.join(charts_dir, "object_count_chart.png")
                self.chart_generator.save_chart(fig, chart_path)
                completed_charts += 1
                self.progress_bar.setValue(int(100 * completed_charts / total_charts))
                QCoreApplication.processEvents()  # 刷新UI
            except Exception as e:
                logger.error(f"导出对象数量-时间曲线图失败: {e}")
        
        # 2. 导出面积分数-时间曲线图表
        self.status_label.setText("导出面积分数-时间曲线图...")
        QCoreApplication.processEvents()  # 刷新UI
        
        if hasattr(self.parent(), "chart_view_widget") and hasattr(self.parent().chart_view_widget, "area_fraction_chart"):
            try:
                fig, _ = self.parent().chart_view_widget.area_fraction_chart
                chart_path = os.path.join(charts_dir, "area_fraction_chart.png")
                self.chart_generator.save_chart(fig, chart_path)
                completed_charts += 1
                self.progress_bar.setValue(int(100 * completed_charts / total_charts))
                QCoreApplication.processEvents()  # 刷新UI
            except Exception as e:
                logger.error(f"导出面积分数-时间曲线图失败: {e}")
        
        # 3. 导出全局对象面积分布图
        self.status_label.setText("导出全局面积分布图...")
        QCoreApplication.processEvents()  # 刷新UI
        
        if hasattr(self.parent(), "chart_view_widget") and hasattr(self.parent().chart_view_widget, "global_area_histogram"):
            try:
                fig, _ = self.parent().chart_view_widget.global_area_histogram
                chart_path = os.path.join(charts_dir, "area_distribution_chart.png")
                self.chart_generator.save_chart(fig, chart_path)
                completed_charts += 1
                self.progress_bar.setValue(int(100 * completed_charts / total_charts))
                QCoreApplication.processEvents()  # 刷新UI
            except Exception as e:
                logger.error(f"导出全局面积分布图失败: {e}")
        
        # 4. 导出全局对象纵横比图
        self.status_label.setText("导出全局纵横比分布图...")
        QCoreApplication.processEvents()  # 刷新UI
        
        if hasattr(self.parent(), "chart_view_widget") and hasattr(self.parent().chart_view_widget, "global_aspect_ratio_histogram"):
            try:
                fig, _ = self.parent().chart_view_widget.global_aspect_ratio_histogram
                chart_path = os.path.join(charts_dir, "aspect_ratio_distribution_chart.png")
                self.chart_generator.save_chart(fig, chart_path)
                completed_charts += 1
                self.progress_bar.setValue(int(100 * completed_charts / total_charts))
                QCoreApplication.processEvents()  # 刷新UI
            except Exception as e:
                logger.error(f"导出全局纵横比分布图失败: {e}")
        
        # 创建帧图表目录
        frame_charts_dir = os.path.join(charts_dir, "frame_charts")
        os.makedirs(frame_charts_dir, exist_ok=True)
        
        # 导出所有帧的分析图表
        total_frames = len(self.image_processor.processed_masks)
        if total_frames == 0:
            self.status_label.setText("没有帧数据，跳过帧图表导出")
            return
        
        # 更新进度条初始值
        self.progress_bar.setValue(10)
        QCoreApplication.processEvents()  # 刷新UI
        
        # 导出成功的帧计数
        successful_frames = 0
        
        # 遍历所有帧
        for frame_idx in range(total_frames):
            # 更新状态
            if frame_idx % 5 == 0 or frame_idx == total_frames - 1:  # 每5帧更新一次UI
                self.status_label.setText(f"导出帧 {frame_idx+1}/{total_frames} 的图表...")
                progress = 10 + int(90 * (frame_idx + 1) / total_frames)  # 帧图表从10%进度开始，占总进度的90%
                self.progress_bar.setValue(progress)
                QCoreApplication.processEvents()  # 刷新UI
            
            # 获取当前帧的统计数据
            try:
                frame_stats = self.image_processor.get_frame_stats(frame_idx)
                if not frame_stats or not frame_stats.get('objects'):
                    logger.info(f"跳过帧 {frame_idx}: 没有对象数据")
                    continue  # 跳过无数据的帧
                
                # 为当前帧生成各种分析图表
                try:
                    # 1. 面积分布直方图 - 使用专门为帧设计的方法
                    self.status_label.setText(f"生成帧 {frame_idx+1} 的面积分布图...")
                    QCoreApplication.processEvents()  # 刷新UI
                    area_hist_fig, area_hist_canvas = self.chart_generator.create_frame_area_histogram(frame_stats)
                    chart_path = os.path.join(frame_charts_dir, f"frame_{frame_idx:04d}_area_chart.png")
                    success = self.chart_generator.save_chart(area_hist_fig, chart_path)
                    if not success:
                        logger.error(f"保存帧 {frame_idx} 的面积分布图失败")
                    
                    # 2. 长轴分布直方图 - 使用专门为帧设计的方法
                    self.status_label.setText(f"生成帧 {frame_idx+1} 的长轴分布图...")
                    QCoreApplication.processEvents()  # 刷新UI
                    major_axis_hist_fig, major_axis_hist_canvas = self.chart_generator.create_frame_major_axis_histogram(frame_stats)
                    chart_path = os.path.join(frame_charts_dir, f"frame_{frame_idx:04d}_major_axis_chart.png")
                    success = self.chart_generator.save_chart(major_axis_hist_fig, chart_path)
                    if not success:
                        logger.error(f"保存帧 {frame_idx} 的长轴分布图失败")
                    
                    # 3. 短轴分布直方图 - 使用专门为帧设计的方法
                    self.status_label.setText(f"生成帧 {frame_idx+1} 的短轴分布图...")
                    QCoreApplication.processEvents()  # 刷新UI
                    minor_axis_hist_fig, minor_axis_hist_canvas = self.chart_generator.create_frame_minor_axis_histogram(frame_stats)
                    chart_path = os.path.join(frame_charts_dir, f"frame_{frame_idx:04d}_minor_axis_chart.png")
                    success = self.chart_generator.save_chart(minor_axis_hist_fig, chart_path)
                    if not success:
                        logger.error(f"保存帧 {frame_idx} 的短轴分布图失败")
                    
                    # 4. 纵横比分布直方图 - 使用专门为帧设计的方法
                    self.status_label.setText(f"生成帧 {frame_idx+1} 的纵横比分布图...")
                    QCoreApplication.processEvents()  # 刷新UI
                    aspect_ratio_hist_fig, aspect_ratio_hist_canvas = self.chart_generator.create_frame_aspect_ratio_histogram(frame_stats)
                    chart_path = os.path.join(frame_charts_dir, f"frame_{frame_idx:04d}_aspect_ratio_chart.png")
                    success = self.chart_generator.save_chart(aspect_ratio_hist_fig, chart_path)
                    if not success:
                        logger.error(f"保存帧 {frame_idx} 的纵横比分布图失败")
                    
                    successful_frames += 1
                except Exception as e:
                    logger.error(f"处理帧 {frame_idx} 的图表时出错: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"获取帧 {frame_idx} 的统计数据时出错: {e}")
                continue
        
        if successful_frames > 0:
            self.status_label.setText(f"已成功导出 {successful_frames}/{total_frames} 帧的分析图表")
        else:
            self.status_label.setText("警告：未能成功导出任何帧的分析图表")
    
    def export_objects_data_excel(self):
        """导出每帧中对象详细数据为Excel文件"""
        try:
            # 导出数据
            excel_path = os.path.join(self.export_dir, "objects_detailed_data.xlsx")
            
            # 获取所有对象数据
            all_objects = []
            total_frames = len(self.image_processor.processed_masks)
            
            # 更新UI
            self.status_label.setText("收集对象数据...")
            self.progress_bar.setValue(0)
            QCoreApplication.processEvents()  # 刷新UI
            
            for frame_idx in range(total_frames):
                # 更新进度和状态
                if frame_idx % 5 == 0 or frame_idx == total_frames - 1:  # 每5帧更新一次UI
                    progress = int(50 * (frame_idx + 1) / total_frames)  # 数据收集占50%进度
                    self.progress_bar.setValue(progress)
                    self.status_label.setText(f"收集帧 {frame_idx+1}/{total_frames} 的对象数据...")
                    QCoreApplication.processEvents()  # 刷新UI
                
                try:
                    # 检查是否已分析且帧索引在范围内
                    if (self.image_processor.is_analyzed and 
                        frame_idx < len(self.image_processor.analyzed_data) and
                        self.image_processor.analyzed_data[frame_idx]):
                        
                        objects_data = self.image_processor.analyzed_data[frame_idx]
                        
                        # 添加每个对象的数据
                        for obj in objects_data:
                            if isinstance(obj, dict):  # 确保对象是字典类型
                                try:
                                    obj_copy = obj.copy()
                                    # 添加帧索引（从1开始，便于用户理解）
                                    obj_copy['frame'] = frame_idx + 1
                                    all_objects.append(obj_copy)
                                except Exception as obj_err:
                                    logger.error(f"处理帧 {frame_idx} 的对象数据时出错: {obj_err}")
                except Exception as frame_err:
                    logger.error(f"获取帧 {frame_idx} 的对象数据时出错: {frame_err}")
            
            # 更新UI
            self.status_label.setText("准备导出Excel数据...")
            self.progress_bar.setValue(50)
            QCoreApplication.processEvents()  # 刷新UI
            
            # 导出Excel
            if all_objects:
                try:
                    # 转换为DataFrame
                    df = pd.DataFrame(all_objects)
                    
                    # 保存所有列，但排除不必要的内部列
                    columns_to_exclude = ['contour', '_scale_factor']
                    columns_to_save = [col for col in df.columns if col not in columns_to_exclude]
                    
                    # 重新排序列，使关键信息排在前面
                    key_columns = ['frame', 'id', 'label_value', 
                                  'center_x', 'center_y', 'center_x_um', 'center_y_um',
                                  'area_pixels', 'area_um2', 
                                  'major_axis_pixels', 'major_axis_um',
                                  'minor_axis_pixels', 'minor_axis_um',
                                  'aspect_ratio', 'circularity']
                    
                    # 确保key_columns中的列都存在于columns_to_save中
                    ordered_columns = [col for col in key_columns if col in columns_to_save]
                    
                    # 添加其他列
                    for col in columns_to_save:
                        if col not in ordered_columns:
                            ordered_columns.append(col)
                    
                    # 更新UI
                    self.status_label.setText("写入Excel文件...")
                    self.progress_bar.setValue(80)
                    QCoreApplication.processEvents()  # 刷新UI
                    
                    # 保存Excel
                    df[ordered_columns].to_excel(excel_path, index=False)
                    
                    # 完成
                    self.status_label.setText(f"已导出 {len(all_objects)} 个对象的数据")
                    self.progress_bar.setValue(100)
                    QCoreApplication.processEvents()  # 刷新UI
                    
                except Exception as e:
                    # 备用处理：如果排序或筛选列失败，尝试直接保存
                    error_msg = f"格式化Excel时出错: {str(e)}，尝试以原始格式保存..."
                    self.status_label.setText(error_msg)
                    QCoreApplication.processEvents()  # 刷新UI
                    
                    # 尝试以原始格式保存，但仍然排除不必要的列
                    df_simplified = pd.DataFrame(all_objects)
                    # 确保即使在备用处理中也能剔除不必要的列
                    for col in columns_to_exclude:
                        if col in df_simplified.columns:
                            df_simplified = df_simplified.drop(columns=[col])
                    
                    df_simplified.to_excel(excel_path, index=False)
                    
                    self.status_label.setText("已以简化格式导出数据")
                    self.progress_bar.setValue(100)
                    QCoreApplication.processEvents()  # 刷新UI
            else:
                # 如果没有数据，创建一个空的Excel文件
                pd.DataFrame().to_excel(excel_path, index=False)
                
                self.status_label.setText("未找到对象数据，已创建空Excel文件")
                self.progress_bar.setValue(100)
                QCoreApplication.processEvents()  # 刷新UI
                
        except Exception as e:
            error_msg = f"导出Excel时出错: {str(e)}"
            self.status_label.setText(error_msg)
            raise Exception(error_msg) 