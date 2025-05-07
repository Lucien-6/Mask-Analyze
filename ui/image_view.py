#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像预览部件：用于显示图像并处理交互事件
"""
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QEvent, QRect


class ImageViewWidget(QWidget):
    """图像预览部件"""
    
    # 自定义信号
    left_clicked = pyqtSignal(int, int)  # x, y
    right_clicked = pyqtSignal(int, int)  # x, y
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.init_ui()
        
        # 初始化成员变量
        self.image = None
        self.image_ratio = 1.0  # 图像缩放比例
        self.original_size = (0, 0)  # 原始图像尺寸
        self.mouse_position = QPoint()  # 鼠标位置
        self.show_crosshair = True  # 是否显示十字线
        
    def init_ui(self):
        """初始化UI"""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建图像标签
        self.image_label = CustomLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.image_label.setMinimumSize(400, 300)
        
        self.layout.addWidget(self.image_label)
        
        # 启用鼠标追踪
        self.setMouseTracking(True)
        self.image_label.setMouseTracking(True)
        
        # 允许接收鼠标事件
        self.image_label.installEventFilter(self)
        
        # 连接鼠标移动信号
        self.image_label.mouseMoveEvent = self.on_mouse_move
    
    def set_image(self, image: np.ndarray):
        """
        设置要显示的图像
        
        Args:
            image: OpenCV图像(BGR格式)
        """
        if image is None:
            return
            
        self.image = image.copy()
        self.original_size = (image.shape[1], image.shape[0])  # 宽度, 高度
        
        self.update_display()
    
    def update_display(self):
        """更新图像显示"""
        if self.image is None:
            return
            
        # 获取窗口尺寸
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        
        # 计算缩放比例
        image_width, image_height = self.original_size
        width_ratio = label_width / image_width
        height_ratio = label_height / image_height
        
        # 选择较小的比例，保持图像的纵横比
        self.image_ratio = min(width_ratio, height_ratio)
        
        # 计算缩放后的图像尺寸
        display_width = int(image_width * self.image_ratio)
        display_height = int(image_height * self.image_ratio)
        
        # 如果缩放尺寸太小，使用原始尺寸
        if display_width < 10 or display_height < 10:
            display_width, display_height = image_width, image_height
            self.image_ratio = 1.0
        
        # 转换OpenCV图像格式为QImage
        if len(self.image.shape) == 3:  # 彩色图像
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_image.data, image_width, image_height, 
                         rgb_image.strides[0], QImage.Format_RGB888)
        else:  # 灰度图像
            qimg = QImage(self.image.data, image_width, image_height, 
                         self.image.strides[0], QImage.Format_Grayscale8)
        
        # 创建缩放后的QPixmap并设置到标签
        pixmap = QPixmap.fromImage(qimg).scaled(
            display_width, display_height, 
            Qt.KeepAspectRatio, Qt.FastTransformation  # 使用快速变换保持清晰度
        )
        
        self.image_label.setPixmap(pixmap)
        self.image_label.setImageSize(display_width, display_height)
    
    def toggle_crosshair(self, enable=True):
        """切换是否显示十字交叉线"""
        self.show_crosshair = enable
        if hasattr(self, 'image_label'):
            self.image_label.setCrosshairEnabled(enable)
    
    def on_mouse_move(self, event):
        """处理鼠标移动事件"""
        if self.image is None:
            return
            
        # 更新鼠标位置
        self.mouse_position = event.pos()
        self.image_label.setMousePosition(self.mouse_position)
        self.image_label.update()  # 触发重绘
    
    def leaveEvent(self, event):
        """处理鼠标离开窗口事件"""
        # 当鼠标离开窗口时，清除十字线
        self.image_label.clearCrosshair()
        super().leaveEvent(event)
    
    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        super().resizeEvent(event)
        self.update_display()
    
    def eventFilter(self, source, event):
        """事件过滤器，处理图像标签的鼠标事件"""
        if source is self.image_label:
            if event.type() == QEvent.MouseButtonPress:
                if self.image is not None:
                    # 获取点击位置
                    pos = event.pos()
                    
                    # 计算标签中心点
                    label_center = QPoint(self.image_label.width() // 2, self.image_label.height() // 2)
                    
                    # 计算缩放后图像的尺寸
                    scaled_width = int(self.original_size[0] * self.image_ratio)
                    scaled_height = int(self.original_size[1] * self.image_ratio)
                    
                    # 计算图像左上角在标签中的位置
                    image_top_left = QPoint(
                        label_center.x() - scaled_width // 2,
                        label_center.y() - scaled_height // 2
                    )
                    
                    # 计算点击位置在原始图像中的坐标
                    if pos.x() >= image_top_left.x() and pos.y() >= image_top_left.y() and \
                       pos.x() < image_top_left.x() + scaled_width and pos.y() < image_top_left.y() + scaled_height:
                        image_x = int((pos.x() - image_top_left.x()) / self.image_ratio)
                        image_y = int((pos.y() - image_top_left.y()) / self.image_ratio)
                        
                        # 确保坐标在原始图像范围内
                        image_x = max(0, min(image_x, self.original_size[0] - 1))
                        image_y = max(0, min(image_y, self.original_size[1] - 1))
                        
                        # 根据鼠标按钮发送相应信号
                        if event.button() == Qt.LeftButton:
                            self.left_clicked.emit(image_x, image_y)
                        elif event.button() == Qt.RightButton:
                            self.right_clicked.emit(image_x, image_y)
                            
        return super().eventFilter(source, event)


class CustomLabel(QLabel):
    """自定义标签类，支持绘制十字交叉线"""
    
    def __init__(self):
        super().__init__()
        self.mouse_pos = QPoint()
        self.image_width = 0
        self.image_height = 0
        self.enable_crosshair = True
        self.show_current_crosshair = True  # 控制当前是否显示十字线
    
    def setMousePosition(self, pos):
        """设置鼠标位置"""
        self.mouse_pos = pos
        self.show_current_crosshair = True  # 重新显示十字线
    
    def clearCrosshair(self):
        """清除十字线"""
        self.show_current_crosshair = False
        self.update()  # 触发重绘
    
    def setImageSize(self, width, height):
        """设置图像尺寸"""
        self.image_width = width
        self.image_height = height
    
    def setCrosshairEnabled(self, enable):
        """设置是否启用十字交叉线"""
        self.enable_crosshair = enable
        self.update()
    
    def paintEvent(self, event):
        """绘制事件"""
        super().paintEvent(event)
        
        if not self.enable_crosshair or not self.pixmap() or not self.show_current_crosshair:
            return
            
        # 计算图像在标签中的位置
        pixmap_rect = self.getImageRect()
        
        # 检查鼠标是否在图像区域内
        if not pixmap_rect.contains(self.mouse_pos):
            return
            
        # 创建绘图对象
        painter = QPainter(self)
        
        # 设置画笔 - 使用黄色，2像素宽度，虚线样式
        pen = QPen(QColor(255, 255, 0, 180), 1, Qt.DashLine)
        painter.setPen(pen)
        
        # 绘制水平线
        painter.drawLine(pixmap_rect.left(), self.mouse_pos.y(), 
                         pixmap_rect.right(), self.mouse_pos.y())
        
        # 绘制垂直线
        painter.drawLine(self.mouse_pos.x(), pixmap_rect.top(),
                         self.mouse_pos.x(), pixmap_rect.bottom())
        
        painter.end()
    
    def getImageRect(self):
        """获取图像在标签中的位置矩形"""
        # 标签的尺寸
        label_width = self.width()
        label_height = self.height()
        
        # 计算图像的位置
        if self.pixmap():
            pixmap_width = self.pixmap().width()
            pixmap_height = self.pixmap().height()
            
            # 图像居中显示的左上角坐标
            x = (label_width - pixmap_width) / 2
            y = (label_height - pixmap_height) / 2
            
            return QRect(int(x), int(y), pixmap_width, pixmap_height)
        
        return QRect(0, 0, 0, 0) 