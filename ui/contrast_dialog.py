#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像对比度调整对话框：用于调整图像的对比度和亮度参数

Author: Lucien
Email: lucien-6@qq.com
Date: 2025-12-05
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QSplitter, QSizePolicy, QApplication, QWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from core.logger import get_logger

# 获取模块日志记录器
logger = get_logger("contrast_dialog")


class ContrastDialog(QDialog):
    """图像对比度调整对话框"""
    
    def __init__(self, parent, image_processor, current_frame_idx):
        """
        初始化对话框
        
        Args:
            parent: 父窗口
            image_processor: 图像处理器实例
            current_frame_idx: 当前帧索引
        """
        super().__init__(parent)
        self.image_processor = image_processor
        self.current_frame_idx = current_frame_idx
        
        # 获取原始图像副本
        self.orig_image = self.image_processor.get_original_image(current_frame_idx)
        if self.orig_image is not None:
            self.orig_image = self.orig_image.copy()
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("调整图像对比度")
        self.setMinimumSize(1000, 600)
        
        # 创建布局
        main_layout = QVBoxLayout(self)
        
        # 创建左右分栏
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧：预览区域
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_label = QLabel("预览效果")
        preview_label.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(preview_label)
        
        # 创建预览图像标签
        self.preview_image_label = QLabel()
        self.preview_image_label.setMinimumSize(400, 400)
        self.preview_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_image_label.setAlignment(Qt.AlignCenter)
        self.preview_image_label.setStyleSheet("border: 1px solid #ccc; background-color: black;")
        preview_layout.addWidget(self.preview_image_label)
        
        # 右侧：控制区域
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # 直方图显示区域
        histogram_label = QLabel("灰度直方图")
        histogram_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(histogram_label)
        
        # 直方图图像标签
        self.histogram_image_label = QLabel()
        self.histogram_image_label.setMinimumSize(300, 200)
        self.histogram_image_label.setAlignment(Qt.AlignCenter)
        self.histogram_image_label.setStyleSheet("border: 1px solid #ccc;")
        control_layout.addWidget(self.histogram_image_label)
        
        # 直方图阈值控制
        threshold_group = QGroupBox("直方图阈值")
        threshold_layout = QVBoxLayout(threshold_group)
        
        # 下限百分比滑块
        lower_layout = QHBoxLayout()
        lower_label = QLabel("下限百分比:")
        self.lower_slider = QSlider(Qt.Horizontal)
        self.lower_slider.setRange(0, 200)  # 0-20%
        self.lower_slider.setValue(int(self.image_processor.lower_percent * 10))
        self.lower_value_label = QLabel(f"{self.image_processor.lower_percent:.1f}%")
        lower_layout.addWidget(lower_label)
        lower_layout.addWidget(self.lower_slider)
        lower_layout.addWidget(self.lower_value_label)
        threshold_layout.addLayout(lower_layout)
        
        # 上限百分比滑块
        upper_layout = QHBoxLayout()
        upper_label = QLabel("上限百分比:")
        self.upper_slider = QSlider(Qt.Horizontal)
        self.upper_slider.setRange(800, 1000)  # 80-100%
        self.upper_slider.setValue(int(self.image_processor.upper_percent * 10))
        self.upper_value_label = QLabel(f"{self.image_processor.upper_percent:.1f}%")
        upper_layout.addWidget(upper_label)
        upper_layout.addWidget(self.upper_slider)
        upper_layout.addWidget(self.upper_value_label)
        threshold_layout.addLayout(upper_layout)
        
        control_layout.addWidget(threshold_group)
        
        # 亮度对比度控制
        adjust_group = QGroupBox("亮度与对比度")
        adjust_layout = QVBoxLayout(adjust_group)
        
        # 亮度滑块
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("亮度:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(50, 200)  # 0.5-2.0倍
        self.brightness_slider.setValue(int(self.image_processor.brightness_factor * 100))
        self.brightness_value_label = QLabel(f"{self.image_processor.brightness_factor:.2f}")
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(self.brightness_slider)
        brightness_layout.addWidget(self.brightness_value_label)
        adjust_layout.addLayout(brightness_layout)
        
        # 对比度滑块
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("对比度:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 200)  # 0.5-2.0倍
        self.contrast_slider.setValue(int(self.image_processor.contrast_factor * 100))
        self.contrast_value_label = QLabel(f"{self.image_processor.contrast_factor:.2f}")
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(self.contrast_slider)
        contrast_layout.addWidget(self.contrast_value_label)
        adjust_layout.addLayout(contrast_layout)
        
        control_layout.addWidget(adjust_group)
        
        # 重置按钮
        reset_button = QPushButton("重置参数")
        reset_button.clicked.connect(self.on_reset)
        control_layout.addWidget(reset_button)
        
        # 添加左右分栏
        splitter.addWidget(preview_widget)
        splitter.addWidget(control_widget)
        splitter.setSizes([600, 400])
        
        # 按钮区域
        button_layout = QHBoxLayout()
        apply_button = QPushButton("应用")
        apply_button.clicked.connect(self.on_apply)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)
        
        # 状态标签
        self.status_label = QLabel("等待操作...")
        main_layout.addWidget(self.status_label)
        
        # 连接信号
        self.lower_slider.valueChanged.connect(self.on_lower_changed)
        self.upper_slider.valueChanged.connect(self.on_upper_changed)
        self.brightness_slider.valueChanged.connect(self.on_brightness_changed)
        self.contrast_slider.valueChanged.connect(self.on_contrast_changed)
        
        # 初始显示 - 延迟执行确保界面已完成布局
        QTimer.singleShot(100, self.update_preview)
    
    def resizeEvent(self, event):
        """窗口大小变化事件"""
        super().resizeEvent(event)
        self.update_preview()
    
    def enhance_image(self, image, lower_percent, upper_percent, 
                      brightness_factor, contrast_factor):
        """
        增强图像
        
        Args:
            image: 输入图像
            lower_percent: 直方图下限百分比
            upper_percent: 直方图上限百分比
            brightness_factor: 亮度因子
            contrast_factor: 对比度因子
            
        Returns:
            增强后的图像
        """
        if image is None:
            return None
        
        dtype = image.dtype
        
        # 针对高位深图像进行增强
        if dtype == np.uint16 or dtype == np.float32 or dtype == np.float64:
            if dtype == np.float32 or dtype == np.float64:
                min_val = np.min(image)
                max_val = np.max(image)
                
                if min_val < max_val:
                    image = ((image - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
                else:
                    return image
            
            if len(image.shape) == 3:  # 彩色图像
                if image.shape[2] == 3:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    l_enhanced = self.apply_histogram_stretch(l, lower_percent, upper_percent)
                    l_enhanced = self.adjust_brightness_contrast(l_enhanced, brightness_factor, contrast_factor)
                    enhanced_lab = cv2.merge([l_enhanced, a, b])
                    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                else:
                    return cv2.convertScaleAbs(image, alpha=255/65535)
            else:  # 灰度图像
                enhanced = self.apply_histogram_stretch(image, lower_percent, upper_percent)
                return self.adjust_brightness_contrast(enhanced, brightness_factor, contrast_factor)
        
        elif dtype == np.uint8:
            if len(image.shape) == 3:  # 彩色图像
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_enhanced = clahe.apply(l)
                l_enhanced = self.adjust_brightness_contrast(l_enhanced, brightness_factor, contrast_factor)
                enhanced_lab = cv2.merge([l_enhanced, a, b])
                return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            else:  # 灰度图像
                enhanced = cv2.equalizeHist(image)
                return self.adjust_brightness_contrast(enhanced, brightness_factor, contrast_factor)
        
        return image
    
    def apply_histogram_stretch(self, image, lower_percent, upper_percent):
        """应用直方图拉伸"""
        if image.dtype == np.uint16:
            min_val = np.percentile(image, lower_percent)
            max_val = np.percentile(image, upper_percent)
            
            if min_val < max_val:
                alpha = 255.0 / (max_val - min_val)
                beta = -min_val * alpha
                return np.clip(image * alpha + beta, 0, 255).astype(np.uint8)
            else:
                return cv2.convertScaleAbs(image, alpha=255/65535)
        else:
            return cv2.equalizeHist(image) if image.dtype == np.uint8 else image
    
    def adjust_brightness_contrast(self, image, brightness, contrast):
        """调整亮度和对比度"""
        if brightness == 1.0 and contrast == 1.0:
            return image
        
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image)
        
        alpha = contrast
        beta = (brightness - 1.0) * 128
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def draw_histogram(self, image):
        """绘制直方图"""
        if image is None:
            return None
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if gray.dtype != np.uint8:
            gray = cv2.convertScaleAbs(gray, alpha=255/65535)
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        hist_w = 300
        hist_h = 200
        bin_w = int(hist_w / 256)
        hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
        
        cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
        
        for i in range(256):
            cv2.rectangle(
                hist_img,
                (i * bin_w, hist_h),
                ((i + 1) * bin_w, hist_h - int(hist[i])),
                (255, 255, 255),
                -1
            )
        
        # 绘制阈值线
        lower_thresh = int(self.lower_slider.value() / 10 * 256 / 100)
        upper_thresh = int(self.upper_slider.value() / 10 * 256 / 100)
        
        cv2.line(hist_img, (lower_thresh * bin_w, 0), (lower_thresh * bin_w, hist_h), (0, 0, 255), 2)
        cv2.line(hist_img, (upper_thresh * bin_w, 0), (upper_thresh * bin_w, hist_h), (0, 255, 0), 2)
        
        return hist_img
    
    def get_preview_size(self):
        """获取预览区域尺寸"""
        w = self.preview_image_label.width()
        h = self.preview_image_label.height()
        return max(200, w - 10), max(200, h - 10)
    
    def update_preview(self):
        """更新预览"""
        try:
            lower_percent = self.lower_slider.value() / 10.0
            upper_percent = self.upper_slider.value() / 10.0
            brightness = self.brightness_slider.value() / 100.0
            contrast = self.contrast_slider.value() / 100.0
            
            self.status_label.setText(
                f"处理图像: L={lower_percent:.1f}%, U={upper_percent:.1f}%, "
                f"B={brightness:.2f}, C={contrast:.2f}"
            )
            QApplication.processEvents()
            
            if self.orig_image is None:
                return
            
            img_copy = self.orig_image.copy()
            enhanced = self.enhance_image(img_copy, lower_percent, upper_percent, brightness, contrast)
            
            # 绘制直方图
            hist_img = self.draw_histogram(enhanced)
            if hist_img is not None:
                histogram_qimg = QImage(
                    hist_img.data,
                    hist_img.shape[1],
                    hist_img.shape[0],
                    hist_img.strides[0],
                    QImage.Format_RGB888
                )
                self.histogram_image_label.setPixmap(QPixmap.fromImage(histogram_qimg))
            
            # 缩放并显示预览
            preview_w, preview_h = self.get_preview_size()
            h, w = enhanced.shape[:2]
            ratio = min(preview_w / w, preview_h / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            
            resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if len(resized.shape) == 3:
                qimg = QImage(
                    resized.data,
                    resized.shape[1],
                    resized.shape[0],
                    resized.strides[0],
                    QImage.Format_BGR888
                )
            else:
                qimg = QImage(
                    resized.data,
                    resized.shape[1],
                    resized.shape[0],
                    resized.strides[0],
                    QImage.Format_Grayscale8
                )
            
            self.preview_image_label.setPixmap(QPixmap.fromImage(qimg))
            
            self.status_label.setText(
                f"预览更新完成: L={lower_percent:.1f}%, U={upper_percent:.1f}%, "
                f"B={brightness:.2f}, C={contrast:.2f}"
            )
            QApplication.processEvents()
            
        except Exception as e:
            self.status_label.setText(f"预览更新出错: {e}")
            logger.error(f"预览更新出错: {e}")
    
    def on_lower_changed(self, value):
        """下限滑块变化处理"""
        val = value / 10.0
        self.lower_value_label.setText(f"{val:.1f}%")
        
        upper_val = self.upper_slider.value() / 10.0
        if val >= upper_val - 1.0:
            self.lower_slider.blockSignals(True)
            self.lower_slider.setValue(int((upper_val - 1.0) * 10))
            self.lower_slider.blockSignals(False)
            self.lower_value_label.setText(f"{(upper_val - 1.0):.1f}%")
            return
        
        self.update_preview()
    
    def on_upper_changed(self, value):
        """上限滑块变化处理"""
        val = value / 10.0
        self.upper_value_label.setText(f"{val:.1f}%")
        
        lower_val = self.lower_slider.value() / 10.0
        if val <= lower_val + 1.0:
            self.upper_slider.blockSignals(True)
            self.upper_slider.setValue(int((lower_val + 1.0) * 10))
            self.upper_slider.blockSignals(False)
            self.upper_value_label.setText(f"{(lower_val + 1.0):.1f}%")
            return
        
        self.update_preview()
    
    def on_brightness_changed(self, value):
        """亮度滑块变化处理"""
        val = value / 100.0
        self.brightness_value_label.setText(f"{val:.2f}")
        self.update_preview()
    
    def on_contrast_changed(self, value):
        """对比度滑块变化处理"""
        val = value / 100.0
        self.contrast_value_label.setText(f"{val:.2f}")
        self.update_preview()
    
    def on_reset(self):
        """重置参数"""
        self.lower_slider.blockSignals(True)
        self.upper_slider.blockSignals(True)
        self.brightness_slider.blockSignals(True)
        self.contrast_slider.blockSignals(True)
        
        self.lower_slider.setValue(10)  # 1.0%
        self.upper_slider.setValue(990)  # 99.0%
        self.brightness_slider.setValue(100)  # 1.0
        self.contrast_slider.setValue(100)  # 1.0
        
        self.lower_value_label.setText("1.0%")
        self.upper_value_label.setText("99.0%")
        self.brightness_value_label.setText("1.00")
        self.contrast_value_label.setText("1.00")
        
        self.lower_slider.blockSignals(False)
        self.upper_slider.blockSignals(False)
        self.brightness_slider.blockSignals(False)
        self.contrast_slider.blockSignals(False)
        
        self.update_preview()
    
    def on_apply(self):
        """应用参数"""
        try:
            self.image_processor.lower_percent = self.lower_slider.value() / 10.0
            self.image_processor.upper_percent = self.upper_slider.value() / 10.0
            self.image_processor.brightness_factor = self.brightness_slider.value() / 100.0
            self.image_processor.contrast_factor = self.contrast_slider.value() / 100.0
            
            self.accept()
            
        except Exception as e:
            self.status_label.setText(f"应用参数时出错: {e}")
            logger.error(f"应用参数时出错: {e}")

