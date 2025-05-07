#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主窗口模块：实现用户界面的主窗口
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QSlider, QFileDialog, 
    QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QProgressBar, QStatusBar, 
    QMessageBox, QAction, QMenu, QToolBar, QSplitter,
    QSpacerItem, QSizePolicy, QScrollArea, QFrame,
    QTabWidget, QDialog, QFormLayout, QLineEdit, QApplication,
    QListWidget
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, pyqtSlot, QThread, QSize,
    QRunnable, QThreadPool, QObject
)
from PyQt5.QtGui import (
    QPixmap, QImage, QIcon, QPalette, QColor
)

from ui.image_view import ImageViewWidget
from ui.preview_control import PreviewControlWidget
from ui.mask_processor import MaskProcessorWidget
from ui.data_loader import DataLoaderWidget
from ui.chart_view import ChartViewWidget
from ui.export_dialog import ExportDialog
from ui.help_dialog import HelpDialog

from core.image_processor import ImageProcessor
from core.data_analyzer import ChartGenerator


# 添加工作线程类
class AnalysisSignals(QObject):
    """分析工作线程的信号类"""
    finished = pyqtSignal()  # 分析完成信号
    progress = pyqtSignal(int)  # 进度信号
    error = pyqtSignal(str)  # 错误信号
    frame_analyzed = pyqtSignal(int, object)  # 单帧分析完成信号，传递帧索引和分析结果

class AnalysisWorker(QRunnable):
    """分析工作线程"""
    
    def __init__(self, image_processor, frame_indices):
        """
        初始化分析工作线程
        
        Args:
            image_processor: 图像处理器对象
            frame_indices: 需要分析的帧索引列表
        """
        super().__init__()
        self.image_processor = image_processor
        self.frame_indices = frame_indices
        self.signals = AnalysisSignals()
        
    def run(self):
        """运行分析任务"""
        total_frames = len(self.frame_indices)
        
        try:
            # 获取缩放因子，避免在循环中重复访问
            scale_factor = self.image_processor.scale_factor
            
            # 按批次处理帧，而不是一次处理所有，减少内存占用
            batch_size = 5  # 每批处理的帧数
            
            for batch_start in range(0, total_frames, batch_size):
                batch_end = min(batch_start + batch_size, total_frames)
                batch_indices = self.frame_indices[batch_start:batch_end]
                
                # 处理当前批次的帧
                for i, frame_idx in enumerate(batch_indices):
                    # 分析单个帧
                    frame_objects = self.image_processor.analyze_objects(frame_idx)
                    
                    # 确保每个对象都有必要的字段，使用高效的批处理
                    if frame_objects:
                        # 直接向分析成功的帧传递已经计算好的结果
                        # 避免额外的字段检查和计算
                        
                        # 发送单帧分析完成信号
                        self.signals.frame_analyzed.emit(frame_idx, frame_objects)
                    
                    # 发送进度信号 - 计算总体进度
                    overall_idx = batch_start + i
                    progress = int((overall_idx + 1) / total_frames * 100)
                    self.signals.progress.emit(progress)
            
            # 发送完成信号
            self.signals.finished.emit()
            
        except Exception as e:
            self.signals.error.emit(str(e))

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        
        # 初始化成员变量
        self.image_processor = ImageProcessor()
        self.chart_generator = ChartGenerator()  # 这里已经设置了中文字体
        self.current_frame_idx = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_next_frame)
        self.play_speed_fps = 30
        self.hidden_objects = set()  # 记录被用户隐藏的对象
        self.threadpool = QThreadPool()  # 添加线程池
        self.analyzed_frames_count = 0  # 已分析的帧数
        self.total_frames_to_analyze = 0  # 总共需要分析的帧数
        
        # 初始化UI
        self.init_ui()
        
        # 连接信号和槽
        self.connect_signals()
        
    def init_ui(self):
        """初始化用户界面"""
        # 设置窗口属性
        self.setWindowTitle("掩膜数据分析工具")
        self.setMinimumSize(1600, 900)  # 增加最小尺寸以适应更宽的面板
        self.showMaximized()  # 默认最大化显示
        
        # 设置窗口图标
        try:
            import os
            import sys
            # 尝试寻找图标文件
            icon_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'app_icon.ico'),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'resources', 'app_icon.png')
            ]
            # 如果是打包后的环境
            if getattr(sys, 'frozen', False):
                base_path = os.path.dirname(sys.executable)
                icon_paths.extend([
                    os.path.join(base_path, 'resources', 'app_icon.ico'),
                    os.path.join(base_path, 'resources', 'app_icon.png'),
                    os.path.join(base_path, 'app_icon.ico'),
                    os.path.join(base_path, 'app_icon.png')
                ])
            
            # 尝试每个可能的路径
            for icon_path in icon_paths:
                if os.path.exists(icon_path):
                    from PyQt5.QtGui import QIcon
                    self.setWindowIcon(QIcon(icon_path))
                    break
        except Exception as e:
            print(f"设置窗口图标时出错: {str(e)}")
        
        # 设置QGroupBox标题样式 - 增大字号并加粗
        self.setStyleSheet("""
            QGroupBox {
                font-size: 28px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)  # 设置布局间距
        main_layout.setContentsMargins(10, 10, 10, 10)  # 设置边距
        
        # 创建左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)  # 增加控件之间的间距
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_panel.setFixedWidth(600)  # 设置固定宽度为600px
        
        # 创建所有部件
        self.create_widgets()
        
        # 创建数据加载模块（使用GroupBox包装）
        data_loader_group = QGroupBox("数据加载")
        data_loader_layout = QVBoxLayout(data_loader_group)
        data_loader_layout.setSpacing(15)  # 增加内部控件间距
        data_loader_layout.setContentsMargins(10, 15, 10, 15)  # 增加内边距
        data_loader_layout.addWidget(self.data_loader_widget)
        data_loader_group.setFixedHeight(280)  # 增加高度到280px
        left_layout.addWidget(data_loader_group)
        
        # 创建参数设置模块（使用GroupBox包装）
        params_group = QGroupBox("参数设置")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(15)  # 增加内部控件间距
        params_layout.setContentsMargins(10, 15, 10, 15)  # 增加内边距
        
        # 创建参数设置表单布局
        form_layout = QFormLayout()
        form_layout.setSpacing(15)  # 增加表单项间距
        form_layout.setLabelAlignment(Qt.AlignRight)  # 标签右对齐
        form_layout.setFormAlignment(Qt.AlignLeft)  # 控件左对齐
        
        # 使用数据加载模块中的帧率和像素换算系数控件
        # 设置控件的最小宽度，使其更容易操作
        self.data_loader_widget.frame_rate_spin.setMinimumWidth(120)
        self.data_loader_widget.scale_factor_spin.setMinimumWidth(120)
        
        form_layout.addRow("帧率 (fps):", self.data_loader_widget.frame_rate_spin)
        form_layout.addRow("μm/pixel:", self.data_loader_widget.scale_factor_spin)
        
        params_layout.addLayout(form_layout)
        
        # 添加一些空间再放置分析按钮
        params_layout.addSpacing(10)
        
        # 添加分析按钮
        self.analyze_button = QPushButton("分析计算")
        self.analyze_button.setMinimumHeight(45)  # 增加按钮高度
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 25px;
                font-size: 24px;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.analyze_button.clicked.connect(self.analyze_data)
        params_layout.addWidget(self.analyze_button, alignment=Qt.AlignCenter)
        
        params_group.setFixedHeight(200)  # 增加高度到200px
        left_layout.addWidget(params_group)
        
        # 创建掩膜处理模块（使用GroupBox包装）
        mask_processor_group = QGroupBox("掩膜处理")
        mask_processor_layout = QVBoxLayout(mask_processor_group)
        mask_processor_layout.addWidget(self.mask_processor_widget)
        left_layout.addWidget(mask_processor_group)
        
        # 创建中间面板
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.setSpacing(10)
        middle_layout.setContentsMargins(5, 5, 5, 5)
        
        # 创建显示控制组（使用GroupBox包装）
        display_control_group = QGroupBox("显示控制")
        display_control_layout = QHBoxLayout(display_control_group)
        display_control_layout.setSpacing(8)  # 按钮之间的间距
        
        # 统一按钮样式
        button_style = """
            QPushButton {
                padding: 5px 10px;
                min-width: 120px;
                min-height: 30px;
            }
            QPushButton:checked {
                background-color: #e0e0e0;
            }
        """
        
        self.show_edges_btn = QPushButton("显示边缘轮廓")
        self.show_edges_btn.setCheckable(True)
        self.show_edges_btn.setStyleSheet(button_style)
        display_control_layout.addWidget(self.show_edges_btn)
        
        self.show_centers_btn = QPushButton("显示对象中心")
        self.show_centers_btn.setCheckable(True)
        self.show_centers_btn.setStyleSheet(button_style)
        display_control_layout.addWidget(self.show_centers_btn)
        
        self.show_axes_btn = QPushButton("显示对象轴线")
        self.show_axes_btn.setCheckable(True)
        self.show_axes_btn.setStyleSheet(button_style)
        display_control_layout.addWidget(self.show_axes_btn)
        
        middle_layout.addWidget(display_control_group)
        
        # 创建图像预览组（使用GroupBox包装）
        preview_group = QGroupBox("图像预览")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(5)
        
        # 创建图像预览窗口
        preview_layout.addWidget(self.image_view_widget)
        
        # 创建预览控制面板
        preview_layout.addWidget(self.preview_control_widget)
        
        middle_layout.addWidget(preview_group)
        
        # 创建右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_panel.setFixedWidth(1000)  # 设置固定宽度为1000px
        
        # 创建图表显示模块（使用GroupBox包装）
        chart_group = QGroupBox("数据分析图表")
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.addWidget(self.chart_view_widget)
        right_layout.addWidget(chart_group)
        
        # 设置布局比例
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(middle_panel)
        splitter.addWidget(right_panel)
        
        # 禁用拖动调整大小
        splitter.setHandleWidth(1)
        for i in range(splitter.count()):
            splitter.handle(i).setEnabled(False)
        
        main_layout.addWidget(splitter)
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        self.statusBar.addPermanentWidget(self.progress_bar)
        
        # 创建状态标签
        self.status_label = QLabel("就绪")
        self.statusBar.addWidget(self.status_label)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 初始状态更新
        self.update_ui_state(False)
        
    def create_widgets(self):
        """创建所有部件"""
        # 创建数据加载部件
        self.data_loader_widget = DataLoaderWidget()
        
        # 创建掩膜处理部件
        self.mask_processor_widget = MaskProcessorWidget()
        
        # 创建图像预览部件
        self.image_view_widget = ImageViewWidget()
        
        # 创建预览控制部件
        self.preview_control_widget = PreviewControlWidget()
        
        # 创建图表显示部件
        self.chart_view_widget = ChartViewWidget()
        
        # 为帧率SpinBox添加变化事件处理
        self.data_loader_widget.frame_rate_spin.valueChanged.connect(self.on_frame_rate_changed)
        
        # 为像素换算系数SpinBox添加变化事件处理
        self.data_loader_widget.scale_factor_spin.valueChanged.connect(self.on_scale_factor_changed)
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        # 导出菜单项
        export_action = QAction("导出分析结果", self)
        export_action.triggered.connect(self.open_export_dialog)
        file_menu.addAction(export_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        # 图像增强子菜单
        enhance_menu = view_menu.addMenu("图像增强")
        
        # 启用/禁用自动增强选项
        self.auto_enhance_action = QAction("自动增强高位深图像", self)
        self.auto_enhance_action.setCheckable(True)
        self.auto_enhance_action.setChecked(self.image_processor.auto_enhance)
        self.auto_enhance_action.triggered.connect(self.toggle_auto_enhance)
        enhance_menu.addAction(self.auto_enhance_action)
        
        # 调整图像对比度选项
        adjust_contrast_action = QAction("调整图像对比度", self)
        adjust_contrast_action.triggered.connect(self.open_clahe_dialog)
        enhance_menu.addAction(adjust_contrast_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        # 关于菜单项
        about_action = QAction("使用说明", self)
        about_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(about_action)
        
    def connect_signals(self):
        """连接信号和槽"""
        # 数据加载模块信号
        self.data_loader_widget.load_button.clicked.connect(self.load_data)
        
        # 掩膜处理模块信号
        self.mask_processor_widget.apply_button.clicked.connect(self.apply_mask_processing)
        self.mask_processor_widget.undo_button.clicked.connect(self.undo_last_mask_processing)
        self.mask_processor_widget.restore_button.clicked.connect(self.restore_original_masks)
        self.mask_processor_widget.clear_excluded_btn.clicked.connect(self.clear_excluded_objects)
        
        # 预览控制模块信号
        self.preview_control_widget.play_button.clicked.connect(self.toggle_play)
        self.preview_control_widget.prev_button.clicked.connect(self.show_prev_frame)
        self.preview_control_widget.next_button.clicked.connect(self.show_next_frame)
        self.preview_control_widget.slider.valueChanged.connect(self.jump_to_frame)
        
        # 显示/隐藏按钮信号
        self.show_edges_btn.clicked.connect(self.update_display)
        self.show_centers_btn.clicked.connect(self.update_display)
        self.show_axes_btn.clicked.connect(self.update_display)
        
        # 图像视图信号
        self.image_view_widget.left_clicked.connect(self.handle_left_click)
        self.image_view_widget.right_clicked.connect(self.toggle_object_visibility)
    
    def update_ui_state(self, data_loaded: bool, data_analyzed: bool = False):
        """
        更新UI状态
        
        Args:
            data_loaded: 是否已加载数据
            data_analyzed: 是否已分析数据
        """
        # 更新按钮状态
        self.analyze_button.setEnabled(data_loaded)
        self.mask_processor_widget.setEnabled(data_loaded)
        self.preview_control_widget.setEnabled(data_loaded)
        
        # 显示控制按钮只有在数据被分析后才启用
        self.show_edges_btn.setEnabled(data_analyzed)
        self.show_centers_btn.setEnabled(data_analyzed)
        self.show_axes_btn.setEnabled(data_analyzed)
        
        # 如果数据未加载，禁用播放
        if not data_loaded and self.timer.isActive():
            self.timer.stop()
            self.preview_control_widget.play_button.setText("播放")
    
    def load_data(self):
        """加载数据"""
        original_dir = self.data_loader_widget.get_original_dir()
        mask_dir = self.data_loader_widget.get_mask_dir()
        
        if not original_dir or not mask_dir:
            QMessageBox.warning(self, "错误", "请先选择原始图片目录和掩膜图片目录")
            return
        
        start_idx = self.data_loader_widget.start_idx_spin.value()
        end_idx = self.data_loader_widget.end_idx_spin.value()
        
        # 设置状态
        self.status_label.setText("正在加载图像...")
        self.progress_bar.setValue(0)
        
        # 加载图像
        count, error_msg = self.image_processor.load_images(
            original_dir=original_dir,
            mask_dir=mask_dir,
            start_idx=start_idx,
            end_idx=end_idx
        )
        
        if count == 0:
            QMessageBox.warning(self, "错误", f"加载图像失败: {error_msg}")
            self.status_label.setText("加载失败")
            return
        
        # 设置参数
        scale_factor = self.data_loader_widget.scale_factor_spin.value()
        frame_rate = self.data_loader_widget.frame_rate_spin.value()
        self.image_processor.set_parameters(scale_factor, frame_rate)
        
        # 不再需要单独设置play_speed_fps，因为toggle_play会从image_processor获取最新值
        
        # 清空隐藏对象集合
        self.hidden_objects = set()
        
        # 重置分析状态
        self.image_processor.is_analyzed = False
        
        # 更新UI状态 - 数据已加载但未分析
        self.update_ui_state(True, False)
        self.status_label.setText(f"已加载 {count} 张图像")
        self.progress_bar.setValue(100)
        
        # 设置帧范围
        self.current_frame_idx = 0
        self.preview_control_widget.slider.setRange(0, count - 1)
        self.preview_control_widget.slider.setValue(0)
        
        # 显示第一帧
        self.display_current_frame()
    
    def analyze_data(self):
        """分析数据"""
        if not self.image_processor.original_images:
            QMessageBox.warning(self, "错误", "请先加载图像")
            return
        
        # 设置状态
        self.status_label.setText("正在分析数据...")
        self.progress_bar.setValue(0)
        
        # 禁用界面按钮，防止分析过程中操作
        self.analyze_button.setEnabled(False)
        self.mask_processor_widget.setEnabled(False)
        self.preview_control_widget.setEnabled(False)
        QApplication.processEvents()  # 刷新UI
        
        # 清空之前的分析数据
        self.image_processor.analyzed_data = []
        self.analyzed_frames_count = 0
        
        # 获取需要分析的总帧数
        total_frames = len(self.image_processor.processed_masks)
        self.total_frames_to_analyze = total_frames
        
        # 调整analyzed_data列表大小以适应所有帧
        self.image_processor.analyzed_data = [None] * total_frames
        
        # 计算可用的线程数（根据CPU核心数、帧数和系统负载调整）
        max_threads = QThreadPool.globalInstance().maxThreadCount()
        # 使用80%的可用线程，避免系统过载
        available_threads = max(1, int(max_threads * 0.8))
        # 确保线程数不超过总帧数
        thread_count = min(available_threads, total_frames)
        
        # 优化帧分配策略 - 使用交错分配而不是连续分配
        # 这样可以更均匀地分配复杂度，避免某些线程因为处理复杂帧而落后
        frame_indices_groups = [[] for _ in range(thread_count)]
        
        # 交错分配帧，使每个线程处理分散的帧
        for i in range(total_frames):
            group_idx = i % thread_count
            frame_indices_groups[group_idx].append(i)
        
        # 创建和启动工作线程
        for i in range(thread_count):
            frame_indices = frame_indices_groups[i]
            
            # 如果这个线程没有分配到帧，跳过
            if not frame_indices:
                continue
                
            # 创建工作线程
            worker = AnalysisWorker(self.image_processor, frame_indices)
            
            # 连接信号
            worker.signals.frame_analyzed.connect(self.on_frame_analyzed)
            worker.signals.finished.connect(self.on_thread_finished)
            worker.signals.error.connect(self.on_analysis_error)
            
            # 启动线程
            self.threadpool.start(worker)
        
        # 记录分析开始时间（用于估计剩余时间）
        self.analysis_start_time = QApplication.instance().startTimer(0)
    
    def on_frame_analyzed(self, frame_idx, frame_objects):
        """
        处理单帧分析完成信号
        
        Args:
            frame_idx: 帧索引
            frame_objects: 分析结果
        """
        # 保存分析结果
        self.image_processor.analyzed_data[frame_idx] = frame_objects
        
        # 增加已分析的帧数
        self.analyzed_frames_count += 1
        
        # 更新进度
        progress = int(self.analyzed_frames_count / self.total_frames_to_analyze * 90)  # 预留10%给图表生成
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"正在分析数据... {self.analyzed_frames_count}/{self.total_frames_to_analyze}帧")
        QApplication.processEvents()  # 刷新UI
        
        # 如果当前帧已分析完成，更新显示
        if frame_idx == self.current_frame_idx:
            self.display_current_frame()
    
    def on_thread_finished(self):
        """处理线程完成信号"""
        # 检查是否所有帧都已分析完成
        if self.analyzed_frames_count >= self.total_frames_to_analyze:
            # 设置分析完成标志
            self.image_processor.is_analyzed = True
            
            # 第二步：生成统计数据和图表
            global_stats = self.image_processor.get_global_stats()
            self.chart_view_widget.update_charts(self.chart_generator, global_stats)
            
            # 更新当前帧的直方图
            frame_stats = self.image_processor.get_frame_stats(self.current_frame_idx)
            self.chart_view_widget.update_frame_histograms(
                self.chart_generator, frame_stats, global_stats
            )
            
            # 更新UI状态 - 数据已加载且已分析
            self.update_ui_state(True, True)
            
            # 更新显示
            self.display_current_frame()
            
            # 完成
            self.status_label.setText("数据分析完成")
            self.progress_bar.setValue(100)
    
    def on_analysis_error(self, error_msg):
        """处理分析错误信号"""
        # 启用界面按钮
        self.update_ui_state(True, False)
        
        # 显示错误消息
        self.status_label.setText("分析出错")
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "错误", f"分析数据时出错：{error_msg}")
    
    def apply_mask_processing(self):
        """应用掩膜处理"""
        if not self.image_processor.processed_masks:
            QMessageBox.warning(self, "错误", "请先加载图像")
            return
        
        # 获取剔除边缘对象的原始状态，以检测是否发生改变
        original_edge_exclusion = self.image_processor.edge_exclusion
        
        # 记录处理前的分析状态
        original_analysis_state = self.image_processor.is_analyzed
        
        # 记录处理前掩膜的哈希值，用于检测掩膜是否真正改变
        current_mask = self.image_processor.get_processed_mask(self.current_frame_idx)
        if current_mask is not None:
            original_mask_hash = hash(current_mask.tobytes())
        else:
            original_mask_hash = None
            
        # 记录手动剔除对象状态，转换为字符串后计算哈希值，更可靠地检测变化
        original_excluded_objects = self.image_processor.manually_excluded_objects.copy()
        original_excluded_hash = hash(str(sorted(original_excluded_objects)))
        
        # 检查是否有手动剔除的对象
        has_manually_excluded = len(self.image_processor.manually_excluded_objects) > 0
        
        # 设置状态
        self.status_label.setText("正在处理掩膜...")
        self.progress_bar.setValue(0)
        
        # 首先确定针对哪些帧进行处理
        # 除"剔除被边缘截断对象"外的操作根据apply_all决定处理范围
        apply_all = self.mask_processor_widget.apply_all_check.isChecked()
        frame_idx = -1 if apply_all else self.current_frame_idx
        
        # 特殊标记：是否执行了填充孔洞操作
        did_fill_holes = False
        
        # 首先重置掩膜
        self.image_processor.reset_masks(frame_idx)
        
        # 应用各种操作
        # 膨胀
        if self.mask_processor_widget.dilation_check.isChecked():
            kernel_size = self.mask_processor_widget.dilation_kernel_spin.value()
            iterations = self.mask_processor_widget.dilation_iter_spin.value()
            self.image_processor.apply_dilation(kernel_size, iterations, frame_idx)
        
        # 腐蚀
        if self.mask_processor_widget.erosion_check.isChecked():
            kernel_size = self.mask_processor_widget.erosion_kernel_spin.value()
            iterations = self.mask_processor_widget.erosion_iter_spin.value()
            self.image_processor.apply_erosion(kernel_size, iterations, frame_idx)
        
        # 开运算
        if self.mask_processor_widget.opening_check.isChecked():
            kernel_size = self.mask_processor_widget.opening_kernel_spin.value()
            iterations = self.mask_processor_widget.opening_iter_spin.value()
            self.image_processor.apply_opening(kernel_size, iterations, frame_idx)
        
        # 闭运算
        if self.mask_processor_widget.closing_check.isChecked():
            kernel_size = self.mask_processor_widget.closing_kernel_spin.value()
            iterations = self.mask_processor_widget.closing_iter_spin.value()
            self.image_processor.apply_closing(kernel_size, iterations, frame_idx)
        
        # 填充孔洞
        if self.mask_processor_widget.fill_holes_check.isChecked():
            self.image_processor.fill_holes(frame_idx)
            did_fill_holes = True
        
        # 面积阈值过滤
        if self.mask_processor_widget.area_filter_check.isChecked():
            min_area = self.mask_processor_widget.min_area_spin.value()
            max_area = -1 if self.mask_processor_widget.max_area_spin.value() == 0 else self.mask_processor_widget.max_area_spin.value()
            self.image_processor.filter_by_area(min_area, max_area, frame_idx)
        
        # 边缘剔除 - 始终应用于所有帧
        new_edge_exclusion = self.mask_processor_widget.exclude_edge_check.isChecked()
        edge_exclusion_changed = original_edge_exclusion != new_edge_exclusion
        self.image_processor.exclude_edge_objects(new_edge_exclusion)
        
        # 如果有手动剔除的对象，即使掩膜未改变也强制执行分析
        if has_manually_excluded:
            # 强制分析
            self.image_processor.is_analyzed = False
            
            # 更新显示
            self.display_current_frame(skip_analysis_update=True)
            
            # 设置状态提示
            self.status_label.setText("检测到手动剔除的对象，正在分析...")
            self.progress_bar.setValue(50)
            QApplication.processEvents()  # 刷新UI
            
            # 自动执行分析
            if apply_all:
                # 使用多线程分析所有帧
                self.start_analysis()
            else:
                # 使用优化的单帧分析方法
                self.analyze_single_frame(self.current_frame_idx)
                
            return
            
        # 检查掩膜是否真正改变
        if not apply_all:
            current_mask = self.image_processor.get_processed_mask(self.current_frame_idx)
            if current_mask is not None:
                new_mask_hash = hash(current_mask.tobytes())
                mask_changed = original_mask_hash != new_mask_hash
            else:
                mask_changed = True
        else:
            mask_changed = True
            
        # 检查手动剔除对象列表是否发生变化（使用更可靠的哈希值比较）
        current_excluded_hash = hash(str(sorted(self.image_processor.manually_excluded_objects)))
        excluded_changed = original_excluded_hash != current_excluded_hash
        
        # 如果掩膜未改变且剔除边缘状态未改变且手动剔除对象列表未变化，直接返回
        if not mask_changed and not edge_exclusion_changed and not excluded_changed:
            self.status_label.setText("掩膜未发生变化")
            self.progress_bar.setValue(100)
            return
        
        # 重置分析状态 - 掩膜改变后需要重新分析
        self.image_processor.is_analyzed = False
        
        # 立即更新显示 - 确保边缘剔除的效果能立即看到，但跳过分析更新
        self.display_current_frame(skip_analysis_update=True)
        
        # 设置状态提示
        self.status_label.setText("掩膜处理完成，正在分析数据...")
        self.progress_bar.setValue(50)
        QApplication.processEvents()  # 刷新UI
        
        # 自动执行分析 - 如果应用到所有帧或边缘剔除状态改变或执行了填充孔洞，则始终分析所有帧
        if apply_all or edge_exclusion_changed or did_fill_holes:
            # 应用到所有帧或状态改变，使用多线程分析所有帧
            self.start_analysis()
        else:
            # 如果仅处理当前帧，使用优化的单帧分析方法
            self.analyze_single_frame(self.current_frame_idx)
    
    def analyze_single_frame(self, frame_idx):
        """
        优化的单帧分析方法，提高当前帧的处理和图表更新效率
        
        Args:
            frame_idx: 要分析的帧索引
        """
        try:
            # 使用异步方式进行分析，避免UI阻塞
            QApplication.setOverrideCursor(Qt.WaitCursor)
            
            # 分析当前帧
            frame_objects = self.image_processor.analyze_objects(frame_idx)
            
            # 确保分析结果存储到analyzed_data中
            if len(self.image_processor.analyzed_data) != len(self.image_processor.processed_masks):
                self.image_processor.analyzed_data = [None] * len(self.image_processor.processed_masks)
            
            # 添加帧索引
            for obj in frame_objects:
                obj["frame"] = frame_idx
            
            # 保存分析结果到对应位置
            self.image_processor.analyzed_data[frame_idx] = frame_objects
            
            # 设置分析已完成标志 - 部分完成
            self.image_processor.is_analyzed = True
            
            # 高效获取当前帧统计信息
            frame_stats = self.image_processor.get_frame_stats(frame_idx)
            
            # 只有当前帧改变，其他帧保持不变，可以增量更新全局统计
            global_stats = self.update_global_stats_incrementally(frame_idx, frame_stats)
            
            # 延迟图表更新，使用低优先级的定时器，避免UI阻塞
            def update_charts():
                # 先更新当前帧直方图，因为更快且对用户更重要
                self.chart_view_widget.update_frame_histograms(
                    self.chart_generator, frame_stats, global_stats
                )
                
                # 再更新全局图表
                self.chart_view_widget.update_charts(self.chart_generator, global_stats)
                
                # 设置完成状态
                self.status_label.setText("当前帧掩膜处理和分析已完成")
                self.progress_bar.setValue(100)
                QApplication.restoreOverrideCursor()
            
            # 使用单发定时器在主线程中更新UI
            QTimer.singleShot(10, update_charts)
            
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "错误", f"分析当前帧数据时出错：{str(e)}")
            self.status_label.setText("分析出错")
            self.progress_bar.setValue(0)
    
    def update_global_stats_incrementally(self, frame_idx, frame_stats):
        """
        增量更新全局统计数据，避免重新计算所有帧数据
        
        Args:
            frame_idx: 更新的帧索引
            frame_stats: 该帧的统计数据
            
        Returns:
            更新后的全局统计数据
        """
        # 获取现有的全局统计数据
        global_stats = self.image_processor.get_global_stats()
        
        # 更新时间点数据数组
        time_point = frame_idx / self.image_processor.frame_rate
        
        # 如果全局统计中存在时间点数组和对象计数数组
        if 'time_points' in global_stats and 'object_counts' in global_stats:
            # 确保数组长度足够
            if len(global_stats['time_points']) > frame_idx:
                # 更新对象计数和面积比例
                global_stats['object_counts'][frame_idx] = frame_stats.get('object_count', 0)
                global_stats['area_fractions'][frame_idx] = frame_stats.get('area_fraction', 0)
                
                # 更新对象数据
                all_objects = global_stats.get('all_objects', [])
                
                # 移除同一帧的旧对象
                all_objects = [obj for obj in all_objects if obj.get('frame') != frame_idx]
                
                # 添加当前帧的新对象
                frame_objects = frame_stats.get('objects', [])
                all_objects.extend(frame_objects)
                
                # 更新全局对象列表
                global_stats['all_objects'] = all_objects
                
                # 重新计算全局最大值、最小值等统计数据
                self.update_global_statistics(global_stats)
        
        return global_stats
    
    def update_global_statistics(self, global_stats):
        """
        更新全局统计数据中的最大值、最小值等
        
        Args:
            global_stats: 要更新的全局统计数据
        """
        if 'all_objects' not in global_stats or not global_stats['all_objects']:
            return
            
        # 收集所有属性的值
        areas = []
        circularities = []
        aspect_ratios = []
        
        for obj in global_stats['all_objects']:
            if 'area_um2' in obj:
                areas.append(obj['area_um2'])
            if 'circularity' in obj:
                circularities.append(obj['circularity'])
            if 'aspect_ratio' in obj:
                aspect_ratios.append(obj['aspect_ratio'])
        
        # 更新统计数据
        if areas:
            global_stats['area_stats'] = {
                'min': min(areas),
                'max': max(areas),
                'mean': sum(areas) / len(areas)
            }
        
        if circularities:
            global_stats['circularity_stats'] = {
                'min': min(circularities),
                'max': max(circularities),
                'mean': sum(circularities) / len(circularities)
            }
        
        if aspect_ratios:
            global_stats['aspect_ratio_stats'] = {
                'min': min(aspect_ratios),
                'max': max(aspect_ratios),
                'mean': sum(aspect_ratios) / len(aspect_ratios)
            }
    
    def display_current_frame(self, skip_analysis_update=False, force_reload=False):
        """
        显示当前帧
        
        Args:
            skip_analysis_update: 是否跳过分析更新（直方图等），用于避免重复更新
            force_reload: 是否强制重新加载图像，用于图像增强参数改变时
        """
        if not self.image_processor.original_images:
            return
        
        # 如果强制重新加载，清除当前帧的缓存
        if force_reload:
            with self.image_processor.cache_lock:
                if self.current_frame_idx in self.image_processor.original_cache:
                    del self.image_processor.original_cache[self.current_frame_idx]
        
        # 检查是否显示轴线，且数据是否已经分析或需要重新分析
        show_axes = self.show_axes_btn.isChecked()
        needs_analysis = show_axes and (
            not self.image_processor.is_analyzed or 
            self.current_frame_idx >= len(self.image_processor.analyzed_data) or
            self.image_processor.analyzed_data[self.current_frame_idx] is None
        )
        
        # 如果需要显示轴线但尚未分析数据，先进行分析
        if needs_analysis:
            try:
                # 临时分析当前帧
                frame_objects = self.image_processor.analyze_objects(self.current_frame_idx)
                
                # 确保分析结果存储到analyzed_data中
                if len(self.image_processor.analyzed_data) <= self.current_frame_idx:
                    self.image_processor.analyzed_data = [None] * max(len(self.image_processor.processed_masks), self.current_frame_idx + 1)
                
                # 添加帧索引
                for obj in frame_objects:
                    obj["frame"] = self.current_frame_idx
                
                # 更新分析数据
                self.image_processor.analyzed_data[self.current_frame_idx] = frame_objects
            except Exception as e:
                print(f"分析当前帧时出错: {str(e)}")
        
        # 获取叠加图像
        overlay = self.image_processor.get_overlay_image(
            frame_idx=self.current_frame_idx,
            show_edges=self.show_edges_btn.isChecked(),
            show_centers=self.show_centers_btn.isChecked(),
            show_axes=show_axes
        )
        
        if overlay is None:
            return
        
        # 显示图像
        self.image_view_widget.set_image(overlay)
        
        # 更新帧编号显示
        total_frames = len(self.image_processor.original_images)
        self.preview_control_widget.frame_label.setText(f"帧: {self.current_frame_idx + 1}/{total_frames}")
        
        # 更新滑块位置
        self.preview_control_widget.slider.blockSignals(True)
        self.preview_control_widget.slider.setValue(self.current_frame_idx)
        self.preview_control_widget.slider.blockSignals(False)
        
        # 更新手动剔除对象列表
        self.update_excluded_objects_list()
        
        # 更新当前帧的直方图，如果不跳过分析更新
        if hasattr(self, "chart_view_widget") and not skip_analysis_update:
            frame_stats = self.image_processor.get_frame_stats(self.current_frame_idx)
            global_stats = self.image_processor.get_global_stats()
            self.chart_view_widget.update_frame_histograms(
                self.chart_generator, frame_stats, global_stats
            )
    
    def toggle_play(self):
        """切换播放/暂停状态"""
        if self.timer.isActive():
            self.timer.stop()
            self.preview_control_widget.play_button.setText("播放")
        else:
            # 如果已经到最后一帧，从头开始
            if self.current_frame_idx >= len(self.image_processor.original_images) - 1:
                self.current_frame_idx = 0
            
            # 获取当前设置的帧率，而不是使用初始值
            current_fps = self.image_processor.frame_rate
            self.play_speed_fps = current_fps  # 更新播放速度
            
            # 使用当前帧率设置定时器间隔
            self.timer.start(int(1000 / current_fps))  # 以毫秒计
            self.preview_control_widget.play_button.setText("暂停")
    
    def play_next_frame(self):
        """播放下一帧"""
        if not self.image_processor.original_images:
            return
        
        self.current_frame_idx += 1
        if self.current_frame_idx >= len(self.image_processor.original_images):
            self.current_frame_idx = 0
        
        self.display_current_frame()
    
    def show_prev_frame(self):
        """显示上一帧"""
        if not self.image_processor.original_images:
            return
        
        self.current_frame_idx -= 1
        if self.current_frame_idx < 0:
            self.current_frame_idx = len(self.image_processor.original_images) - 1
        
        self.display_current_frame()
    
    def show_next_frame(self):
        """显示下一帧"""
        if not self.image_processor.original_images:
            return
        
        self.current_frame_idx += 1
        if self.current_frame_idx >= len(self.image_processor.original_images):
            self.current_frame_idx = 0
        
        self.display_current_frame()
    
    def jump_to_frame(self, frame_idx):
        """跳转到指定帧"""
        if not self.image_processor.original_images:
            return
        
        if 0 <= frame_idx < len(self.image_processor.original_images):
            self.current_frame_idx = frame_idx
            self.display_current_frame()
    
    def update_display(self):
        """更新显示模式"""
        self.display_current_frame()
    
    def handle_left_click(self, x, y):
        """处理鼠标左键点击事件"""
        if not self.image_processor.processed_masks:
            return
        
        # 获取当前掩膜
        mask = self.image_processor.processed_masks[self.current_frame_idx]
        
        # 获取点击位置的对象ID
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            obj_value = mask[y, x]
            if obj_value > 0:  # 如果不是背景
                # 查找对象数据
                if self.image_processor.is_analyzed and self.current_frame_idx < len(self.image_processor.analyzed_data):
                    # 使用已分析的数据
                    objects_data = self.image_processor.analyzed_data[self.current_frame_idx]
                else:
                    # 如果没有分析过，则临时分析
                    objects_data = self.image_processor.analyze_objects(self.current_frame_idx)
                
                # 确保所有对象都有最新的物理坐标
                current_scale = self.image_processor.scale_factor
                for obj in objects_data:
                    # 更新中心点物理坐标
                    obj["center_x_um"] = obj["center_x"] * current_scale
                    obj["center_y_um"] = obj["center_y"] * current_scale
                    
                    # 检查对象是否有_scale_factor标记（表示之前已计算过物理值）
                    if "_scale_factor" in obj and obj["_scale_factor"] != current_scale:
                        # 如果缩放系数发生变化，重新计算物理量
                        # 面积：从像素值重新计算
                        obj["area_um2"] = obj["area_pixels"] * (current_scale ** 2)
                        # 长短轴：从之前的物理值调整到新的缩放系数
                        obj["major_axis_um"] = obj["major_axis_um"] / obj["_scale_factor"] * current_scale
                        obj["minor_axis_um"] = obj["minor_axis_um"] / obj["_scale_factor"] * current_scale
                    elif "_scale_factor" not in obj:
                        # 如果是首次计算，确保area_um2已经计算（analyze_objects中已乘以scale_factor）
                        # 不要重复计算长短轴，因为analyze_objects中已经乘以了scale_factor
                        pass
                    
                    # 记录当前使用的缩放系数，用于后续更新
                    obj["_scale_factor"] = current_scale
                
                selected_object = None
                for obj in objects_data:
                    if obj["label_value"] == obj_value:
                        selected_object = obj
                        break
                
                if selected_object:
                    self.show_object_info(selected_object, x, y)
    
    def show_object_info(self, obj_data, x, y):
        """显示对象信息弹窗"""
        # 使用直接计算好的物理坐标
        info_text = (
            f"对象ID: {obj_data['id']}\n"
            f"中心坐标: ({obj_data['center_x_um']:.1f}, {obj_data['center_y_um']:.1f}) μm\n"
            f"面积: {obj_data['area_um2']:.2f} μm²\n"
            f"长轴: {obj_data['major_axis_um']:.2f} μm\n"
            f"短轴: {obj_data['minor_axis_um']:.2f} μm\n"
            f"纵横比: {obj_data['aspect_ratio']:.2f}"
        )
        
        QMessageBox.information(self, f"对象信息", info_text)
    
    def toggle_object_visibility(self, x, y):
        """切换对象可见性与手动剔除状态"""
        if not self.image_processor.processed_masks:
            return
        
        # 获取当前掩膜
        mask = self.image_processor.get_processed_mask(self.current_frame_idx)
        if mask is None:
            return
        
        # 获取点击位置的对象ID
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            obj_value = mask[y, x]
            if obj_value > 0:  # 如果不是背景
                # 找到对象的实际中心点
                binary_mask = (mask == obj_value).astype(np.uint8)
                moments = cv2.moments(binary_mask)
                
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx, cy = x, y  # 如果无法计算中心点，则使用点击位置
                
                # 切换对象的手动剔除状态
                is_excluded = self.image_processor.toggle_manually_exclude_object(
                    self.current_frame_idx, obj_value, cx, cy)
                
                # 更新显示
                self.update_excluded_objects_list()
                self.display_current_frame()
                
                # 显示状态提示
                action = "添加到" if is_excluded else "从"
                self.status_label.setText(f"已{action}手动剔除目录中{action}对象ID: {obj_value}")
    
    def update_excluded_objects_list(self):
        """更新手动剔除对象列表显示"""
        # 清空当前列表
        self.mask_processor_widget.excluded_objects_list.clear()
        
        # 获取当前帧的手动剔除对象
        excluded_objects = self.image_processor.get_manually_excluded_objects(self.current_frame_idx)
        
        # 按ID排序
        excluded_objects.sort()
        
        # 获取所有已剔除对象的完整IDs，用于获取更多信息
        all_excluded_ids = self.image_processor.manually_excluded_objects
        
        # 添加到列表
        for obj_value in excluded_objects:
            # 构建完整ID格式
            complete_id = f"{self.current_frame_idx}-{obj_value}"
            self.mask_processor_widget.excluded_objects_list.addItem(f"{complete_id}")
    
    def clear_excluded_objects(self):
        """清空当前帧的所有手动剔除标记"""
        if not self.image_processor.processed_masks:
            return
        
        # 获取当前帧的手动剔除对象
        excluded_objects = self.image_processor.get_manually_excluded_objects(self.current_frame_idx)
        
        # 如果没有剔除对象，直接返回
        if not excluded_objects:
            return
        
        # 清除所有手动剔除标记
        for obj_value in excluded_objects.copy():  # 使用副本进行遍历
            # 查找对象中心点
            mask = self.image_processor.get_processed_mask(self.current_frame_idx)
            if mask is not None:
                # 查找该对象中心点
                binary_mask = (mask == obj_value).astype(np.uint8)
                moments = cv2.moments(binary_mask)
                if moments["m00"] > 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    
                    # 切换状态（从已剔除变为未剔除）
                    self.image_processor.toggle_manually_exclude_object(
                        self.current_frame_idx, obj_value, cx, cy)
        
        # 更新显示
        self.update_excluded_objects_list()
        self.display_current_frame()
        self.status_label.setText(f"已清空当前帧的所有手动剔除标记")
    
    def open_export_dialog(self):
        """打开导出对话框"""
        if not self.image_processor.original_images:
            QMessageBox.warning(self, "错误", "请先加载并分析数据")
            return
        
        # 获取当前统计数据
        global_stats = self.image_processor.get_global_stats()
        export_data = self.chart_generator.prepare_export_data(global_stats)
        
        # 创建并显示导出对话框
        dialog = ExportDialog(self, self.image_processor, self.chart_generator, export_data)
        dialog.exec_()
    
    def show_help_dialog(self):
        """显示帮助对话框"""
        dialog = HelpDialog(self)
        dialog.exec_()
    
    def on_frame_rate_changed(self, new_value):
        """处理帧率变化事件"""
        # 更新图像处理器的帧率
        self.image_processor.frame_rate = new_value
        
        # 如果正在播放，需要重新设置定时器
        if self.timer.isActive():
            self.timer.stop()
            self.timer.start(int(1000 / new_value))
    
    def on_scale_factor_changed(self, new_value):
        """处理像素换算系数变化事件"""
        # 更新图像处理器的像素换算系数
        self.image_processor.scale_factor = new_value
        
        # 如果已经分析过数据，需要清除或更新已分析的结果
        if self.image_processor.is_analyzed:
            # 清除已分析的数据
            self.image_processor.is_analyzed = False
            self.image_processor.analyzed_data = []
            
            # 更新UI状态 - 数据已加载但未分析
            self.update_ui_state(True, False)
            
            # 显示提醒
            self.status_label.setText("像素换算系数已更改，需要重新分析计算")
            QMessageBox.information(self, "提示", "像素换算系数已更改，请点击「分析计算」按钮重新分析数据，以确保物理坐标正确。")

    def start_analysis(self):
        """启动多线程分析"""
        # 设置状态（如果不是从apply_mask_processing调用的，才设置状态）
        if self.status_label.text() != "掩膜处理完成，正在分析数据...":
            self.status_label.setText("正在分析数据...")
            self.progress_bar.setValue(0)
        
        # 禁用界面按钮，防止分析过程中操作
        self.analyze_button.setEnabled(False)
        self.mask_processor_widget.setEnabled(False)
        self.preview_control_widget.setEnabled(False)
        QApplication.processEvents()  # 刷新UI
        
        # 清空之前的分析数据
        self.image_processor.analyzed_data = []
        self.analyzed_frames_count = 0
        
        # 获取需要分析的总帧数
        total_frames = len(self.image_processor.processed_masks)
        self.total_frames_to_analyze = total_frames
        
        # 调整analyzed_data列表大小以适应所有帧
        self.image_processor.analyzed_data = [None] * total_frames
        
        # 计算可用的线程数（根据CPU核心数、帧数和系统负载调整）
        max_threads = QThreadPool.globalInstance().maxThreadCount()
        # 使用80%的可用线程，避免系统过载
        available_threads = max(1, int(max_threads * 0.8))
        # 确保线程数不超过总帧数
        thread_count = min(available_threads, total_frames)
        
        # 优化帧分配策略 - 使用交错分配而不是连续分配
        # 这样可以更均匀地分配复杂度，避免某些线程因为处理复杂帧而落后
        frame_indices_groups = [[] for _ in range(thread_count)]
        
        # 交错分配帧，使每个线程处理分散的帧
        for i in range(total_frames):
            group_idx = i % thread_count
            frame_indices_groups[group_idx].append(i)
        
        # 创建和启动工作线程
        for i in range(thread_count):
            frame_indices = frame_indices_groups[i]
            
            # 如果这个线程没有分配到帧，跳过
            if not frame_indices:
                continue
                
            # 创建工作线程
            worker = AnalysisWorker(self.image_processor, frame_indices)
            
            # 连接信号
            worker.signals.frame_analyzed.connect(self.on_frame_analyzed)
            worker.signals.finished.connect(self.on_thread_finished)
            worker.signals.error.connect(self.on_analysis_error)
            
            # 启动线程
            self.threadpool.start(worker)

    def undo_last_mask_processing(self):
        """撤销最近一次掩膜处理操作"""
        if not hasattr(self, 'image_processor') or self.image_processor is None:
            return
            
        # 获取是否应用到所有帧的选项
        apply_all = self.mask_processor_widget.apply_all_check.isChecked()
        frame_idx = -1 if apply_all else self.current_frame_idx
        
        # 设置状态
        self.status_label.setText("正在撤销掩膜处理...")
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        
        # 执行撤销操作
        success = self.image_processor.undo_last_mask_processing(frame_idx)
        
        if not success:
            self.status_label.setText("没有可撤销的处理操作")
            self.progress_bar.setValue(100)
            return
            
        # 更新显示
        self.display_current_frame(skip_analysis_update=True)
        
        # 设置状态提示
        self.status_label.setText("已撤销最近一次掩膜处理，正在更新分析...")
        self.progress_bar.setValue(50)
        QApplication.processEvents()
        
        # 执行分析
        if apply_all:
            # 如果应用到所有帧，使用多线程分析所有帧
            self.start_analysis()
        else:
            # 如果仅处理当前帧，使用优化的单帧分析方法
            self.analyze_single_frame(self.current_frame_idx)
    
    def restore_original_masks(self):
        """还原掩膜至初始状态"""
        if not hasattr(self, 'image_processor') or self.image_processor is None:
            return
            
        # 获取是否应用到所有帧的选项
        apply_all = self.mask_processor_widget.apply_all_check.isChecked()
        frame_idx = -1 if apply_all else self.current_frame_idx
        
        # 设置状态
        self.status_label.setText("正在还原掩膜至初始状态...")
        self.progress_bar.setValue(0)
        QApplication.processEvents()
        
        # 执行还原操作
        success = self.image_processor.restore_original_masks(frame_idx)
        
        if not success:
            self.status_label.setText("还原掩膜失败")
            self.progress_bar.setValue(100)
            return
            
        # 更新显示
        self.display_current_frame(skip_analysis_update=True)
        
        # 设置状态提示
        self.status_label.setText("已还原掩膜至初始状态，正在更新分析...")
        self.progress_bar.setValue(50)
        QApplication.processEvents()
        
        # 执行分析
        if apply_all:
            # 如果应用到所有帧，使用多线程分析所有帧
            self.start_analysis()
        else:
            # 如果仅处理当前帧，使用优化的单帧分析方法
            self.analyze_single_frame(self.current_frame_idx)

    def toggle_auto_enhance(self):
        """切换自动增强选项"""
        self.image_processor.auto_enhance = self.auto_enhance_action.isChecked()
        
        # 清除缓存并强制重新加载当前图像
        self.image_processor.clear_all_caches()
        self.display_current_frame(force_reload=True)
        
        self.status_label.setText("自动增强选项已" + ("启用" if self.image_processor.auto_enhance else "禁用"))
        self.progress_bar.setValue(100)

    def open_clahe_dialog(self):
        """打开图像对比度调整对话框"""
        # 获取当前帧图像作为预览
        if not self.image_processor.original_images or self.current_frame_idx >= len(self.image_processor.original_images):
            QMessageBox.warning(self, "警告", "请先加载图像")
            return
            
        # 获取原始图像的副本
        orig_image = self.image_processor.get_original_image(self.current_frame_idx)
        if orig_image is None:
            QMessageBox.warning(self, "警告", "获取图像失败")
            return
            
        # 创建一个完全独立的图像副本
        orig_image = orig_image.copy()
        
        # 创建对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("调整图像对比度")
        dialog.setMinimumSize(1000, 600)
        
        # 创建布局
        main_layout = QVBoxLayout(dialog)
        
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
        preview_image_label = QLabel()
        preview_image_label.setMinimumSize(400, 400)
        preview_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_image_label.setAlignment(Qt.AlignCenter)
        preview_image_label.setStyleSheet("border: 1px solid #ccc; background-color: black;")
        preview_layout.addWidget(preview_image_label)
        
        # 右侧：控制区域
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # 直方图显示区域
        histogram_label = QLabel("灰度直方图")
        histogram_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(histogram_label)
        
        # 直方图图像标签
        histogram_image_label = QLabel()
        histogram_image_label.setMinimumSize(300, 200)
        histogram_image_label.setAlignment(Qt.AlignCenter)
        histogram_image_label.setStyleSheet("border: 1px solid #ccc;")
        control_layout.addWidget(histogram_image_label)
        
        # 直方图阈值控制
        threshold_group = QGroupBox("直方图阈值")
        threshold_layout = QVBoxLayout(threshold_group)
        
        # 下限百分比滑块
        lower_layout = QHBoxLayout()
        lower_label = QLabel("下限百分比:")
        lower_slider = QSlider(Qt.Horizontal)
        lower_slider.setRange(0, 200)  # 0-20%
        lower_slider.setValue(int(self.image_processor.lower_percent * 10))  # 转为整数值
        lower_value_label = QLabel(f"{self.image_processor.lower_percent:.1f}%")
        lower_layout.addWidget(lower_label)
        lower_layout.addWidget(lower_slider)
        lower_layout.addWidget(lower_value_label)
        threshold_layout.addLayout(lower_layout)
        
        # 上限百分比滑块
        upper_layout = QHBoxLayout()
        upper_label = QLabel("上限百分比:")
        upper_slider = QSlider(Qt.Horizontal)
        upper_slider.setRange(800, 1000)  # 80-100%
        upper_slider.setValue(int(self.image_processor.upper_percent * 10))  # 转为整数值
        upper_value_label = QLabel(f"{self.image_processor.upper_percent:.1f}%")
        upper_layout.addWidget(upper_label)
        upper_layout.addWidget(upper_slider)
        upper_layout.addWidget(upper_value_label)
        threshold_layout.addLayout(upper_layout)
        
        control_layout.addWidget(threshold_group)
        
        # 亮度对比度控制
        adjust_group = QGroupBox("亮度与对比度")
        adjust_layout = QVBoxLayout(adjust_group)
        
        # 亮度滑块
        brightness_layout = QHBoxLayout()
        brightness_label = QLabel("亮度:")
        brightness_slider = QSlider(Qt.Horizontal)
        brightness_slider.setRange(50, 200)  # 0.5-2.0倍
        brightness_slider.setValue(int(self.image_processor.brightness_factor * 100))
        brightness_value_label = QLabel(f"{self.image_processor.brightness_factor:.2f}")
        brightness_layout.addWidget(brightness_label)
        brightness_layout.addWidget(brightness_slider)
        brightness_layout.addWidget(brightness_value_label)
        adjust_layout.addLayout(brightness_layout)
        
        # 对比度滑块
        contrast_layout = QHBoxLayout()
        contrast_label = QLabel("对比度:")
        contrast_slider = QSlider(Qt.Horizontal)
        contrast_slider.setRange(50, 200)  # 0.5-2.0倍
        contrast_slider.setValue(int(self.image_processor.contrast_factor * 100))
        contrast_value_label = QLabel(f"{self.image_processor.contrast_factor:.2f}")
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(contrast_slider)
        contrast_layout.addWidget(contrast_value_label)
        adjust_layout.addLayout(contrast_layout)
        
        control_layout.addWidget(adjust_group)
        
        # 重置按钮
        reset_button = QPushButton("重置参数")
        control_layout.addWidget(reset_button)
        
        # 添加左右分栏
        splitter.addWidget(preview_widget)
        splitter.addWidget(control_widget)
        splitter.setSizes([600, 400])  # 设置初始宽度比例
        
        # 按钮区域
        button_layout = QHBoxLayout()
        apply_button = QPushButton("应用")
        cancel_button = QPushButton("取消")
        button_layout.addStretch()
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        main_layout.addLayout(button_layout)
        
        # 添加一个状态标签，用于调试
        status_label = QLabel("等待操作...")
        main_layout.addWidget(status_label)
        
        # 直接从core/image_processor.py导入所需的函数，而不是依赖实例方法
        def enhance_image(image, lower_percent, upper_percent, brightness_factor, contrast_factor):
            """直接增强图像的函数，不依赖ImageProcessor类实例"""
            if image is None:
                return None
                
            # 获取图像数据类型和位深
            dtype = image.dtype
            
            # 针对高位深图像（如16位深度）或浮点型图像进行增强
            if dtype == np.uint16 or dtype == np.float32 or dtype == np.float64:
                # 如果是浮点型图像，先转换到适合处理的范围
                if dtype == np.float32 or dtype == np.float64:
                    # 计算最小值和最大值
                    min_val = np.min(image)
                    max_val = np.max(image)
                    
                    if min_val < max_val:
                        # 归一化到0-65535范围，以便用16位整数处理
                        image = ((image - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
                    else:
                        # 如果图像没有动态范围，直接返回原图
                        return image
                
                # 对彩色和灰度图像分别处理
                if len(image.shape) == 3:  # 彩色图像
                    # 如果是彩色图像，分离通道，只对亮度通道应用直方图调整
                    if image.shape[2] == 3:  # 确保是BGR格式
                        # 转换到LAB色彩空间
                        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        
                        # 应用直方图拉伸到亮度通道
                        l_enhanced = apply_histogram_stretch(l, lower_percent, upper_percent)
                        
                        # 应用亮度和对比度调整
                        l_enhanced = adjust_brightness_contrast(
                            l_enhanced, 
                            brightness_factor, 
                            contrast_factor
                        )
                        
                        # 合并通道并转回BGR
                        enhanced_lab = cv2.merge([l_enhanced, a, b])
                        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                        return enhanced_image
                    else:
                        # 其他通道数的彩色图像，直接转为8位
                        return cv2.convertScaleAbs(image, alpha=255/65535)
                else:  # 灰度图像
                    # 应用直方图拉伸
                    enhanced = apply_histogram_stretch(image, lower_percent, upper_percent)
                    
                    # 应用亮度和对比度调整
                    enhanced = adjust_brightness_contrast(
                        enhanced, 
                        brightness_factor, 
                        contrast_factor
                    )
                    
                    return enhanced
            elif dtype == np.uint8:
                # 8位图像处理
                if len(image.shape) == 3:  # 彩色图像
                    # 转换到LAB色彩空间
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # 对亮度通道应用直方图均衡
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    l_enhanced = clahe.apply(l)
                    
                    # 应用亮度和对比度调整
                    l_enhanced = adjust_brightness_contrast(
                        l_enhanced, 
                        brightness_factor, 
                        contrast_factor
                    )
                    
                    # 合并通道并转回BGR
                    enhanced_lab = cv2.merge([l_enhanced, a, b])
                    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                    return enhanced_image
                else:  # 灰度图像
                    # 应用直方图均衡
                    enhanced = cv2.equalizeHist(image)
                    
                    # 应用亮度和对比度调整
                    enhanced = adjust_brightness_contrast(
                        enhanced, 
                        brightness_factor, 
                        contrast_factor
                    )
                    
                    return enhanced
                    
            return image
            
        def apply_histogram_stretch(image, lower_percent, upper_percent):
            """应用直方图拉伸"""
            # 计算指定百分位的像素值
            if image.dtype == np.uint16:
                min_val = np.percentile(image, lower_percent)
                max_val = np.percentile(image, upper_percent)
                
                # 确保有效范围
                if min_val < max_val:
                    # 线性拉伸：将范围映射到0-255
                    alpha = 255.0 / (max_val - min_val)
                    beta = -min_val * alpha
                    
                    # 执行线性变换，并限制范围
                    enhanced = np.clip(image * alpha + beta, 0, 255).astype(np.uint8)
                    return enhanced
                else:
                    # 如果范围无效，直接转换为8位
                    return cv2.convertScaleAbs(image, alpha=255/65535)
            else:
                # 如果已经是8位图像，应用普通的直方图均衡化
                return cv2.equalizeHist(image) if image.dtype == np.uint8 else image
        
        def adjust_brightness_contrast(image, brightness, contrast):
            """调整图像的亮度和对比度"""
            if brightness == 1.0 and contrast == 1.0:
                return image
                
            # 确保图像是uint8类型
            if image.dtype != np.uint8:
                image = cv2.convertScaleAbs(image)
            
            # 亮度调整：添加或减去一个常数
            # 对比度调整：乘以一个因子，以128为中心
            alpha = contrast  # 对比度因子
            beta = (brightness - 1.0) * 128  # 亮度偏移
            
            # 应用变换: pixel = pixel * alpha + beta
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
            return adjusted
        
        # 绘制直方图的函数
        def draw_histogram(image):
            if image is None:
                return None
                
            # 如果是彩色图像，转为灰度
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # 如果是高位深图像，压缩到8位
            if gray.dtype != np.uint8:
                gray = cv2.convertScaleAbs(gray, alpha=255/65535)
                
            # 计算直方图
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # 创建直方图图像
            hist_w = 300
            hist_h = 200
            bin_w = int(hist_w / 256)
            hist_img = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)
            
            # 归一化直方图
            cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)
            
            # 绘制直方图
            for i in range(256):
                cv2.rectangle(
                    hist_img, 
                    (i * bin_w, hist_h), 
                    ((i + 1) * bin_w, hist_h - int(hist[i])), 
                    (255, 255, 255), 
                    -1
                )
                
            # 标记当前阈值
            lower_thresh = int(lower_slider.value() / 10 * 256 / 100)
            upper_thresh = int(upper_slider.value() / 10 * 256 / 100)
            
            # 绘制阈值线
            cv2.line(hist_img, (lower_thresh * bin_w, 0), (lower_thresh * bin_w, hist_h), (0, 0, 255), 2)
            cv2.line(hist_img, (upper_thresh * bin_w, 0), (upper_thresh * bin_w, hist_h), (0, 255, 0), 2)
            
            return hist_img
        
        # 获取图像控件尺寸的函数
        def get_preview_size():
            # 获取预览区域的实际可用尺寸
            w = preview_image_label.width()
            h = preview_image_label.height()
            
            # 确保尺寸合理，至少为200x200
            return max(200, w - 10), max(200, h - 10)
            
        # 更新预览的函数
        def update_preview():
            try:
                # 获取当前参数值
                lower_percent = lower_slider.value() / 10.0
                upper_percent = upper_slider.value() / 10.0
                brightness = brightness_slider.value() / 100.0
                contrast = contrast_slider.value() / 100.0
                
                # 更新状态
                status_label.setText(f"处理图像: L={lower_percent:.1f}%, U={upper_percent:.1f}%, B={brightness:.2f}, C={contrast:.2f}")
                QApplication.processEvents()  # 更新UI
                
                # 创建图像副本进行处理
                img_copy = orig_image.copy()
                
                # 使用直接函数处理图像，不依赖ImageProcessor类
                enhanced = enhance_image(
                    img_copy, 
                    lower_percent,
                    upper_percent,
                    brightness,
                    contrast
                )
                
                # 绘制直方图
                hist_img = draw_histogram(enhanced)
                if hist_img is not None:
                    histogram_qimg = QImage(
                        hist_img.data, 
                        hist_img.shape[1], 
                        hist_img.shape[0], 
                        hist_img.strides[0], 
                        QImage.Format_RGB888
                    )
                    histogram_image_label.setPixmap(QPixmap.fromImage(histogram_qimg))
                
                # 获取预览区域尺寸
                preview_w, preview_h = get_preview_size()
                
                # 保持原始图像高宽比缩放
                h, w = enhanced.shape[:2]
                ratio = min(preview_w / w, preview_h / h)
                
                # 计算缩放后尺寸
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                
                # 缩放图像
                resized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 转换为QImage并显示
                if len(resized.shape) == 3:  # 彩色图像
                    qimg = QImage(
                        resized.data, 
                        resized.shape[1], 
                        resized.shape[0], 
                        resized.strides[0], 
                        QImage.Format_BGR888
                    )
                else:  # 灰度图像
                    qimg = QImage(
                        resized.data, 
                        resized.shape[1], 
                        resized.shape[0], 
                        resized.strides[0], 
                        QImage.Format_Grayscale8
                    )
                    
                # 显示图像
                pixmap = QPixmap.fromImage(qimg)
                preview_image_label.setPixmap(pixmap)
                
                # 更新状态
                status_label.setText(f"预览更新完成: L={lower_percent:.1f}%, U={upper_percent:.1f}%, B={brightness:.2f}, C={contrast:.2f}")
                
                # 强制刷新UI
                QApplication.processEvents()
                
            except Exception as e:
                status_label.setText(f"预览更新出错: {str(e)}")
                print(f"预览更新出错: {str(e)}")
        
        # 连接信号
        def on_lower_changed(value):
            val = value / 10.0
            lower_value_label.setText(f"{val:.1f}%")
            
            # 确保下限不超过上限
            upper_val = upper_slider.value() / 10.0
            if val >= upper_val - 1.0:
                lower_slider.blockSignals(True)
                lower_slider.setValue(int((upper_val - 1.0) * 10))
                lower_slider.blockSignals(False)
                lower_value_label.setText(f"{(upper_val - 1.0):.1f}%")
                return
                
            # 触发更新
            update_preview()
            
        def on_upper_changed(value):
            val = value / 10.0
            upper_value_label.setText(f"{val:.1f}%")
            
            # 确保上限不低于下限
            lower_val = lower_slider.value() / 10.0
            if val <= lower_val + 1.0:
                upper_slider.blockSignals(True)
                upper_slider.setValue(int((lower_val + 1.0) * 10))
                upper_slider.blockSignals(False)
                upper_value_label.setText(f"{(lower_val + 1.0):.1f}%")
                return
                
            # 触发更新
            update_preview()
            
        def on_brightness_changed(value):
            val = value / 100.0
            brightness_value_label.setText(f"{val:.2f}")
            # 触发更新
            update_preview()
            
        def on_contrast_changed(value):
            val = value / 100.0
            contrast_value_label.setText(f"{val:.2f}")
            # 触发更新
            update_preview()
            
        def on_reset():
            # 重置参数到默认值
            lower_slider.blockSignals(True)
            upper_slider.blockSignals(True)
            brightness_slider.blockSignals(True)
            contrast_slider.blockSignals(True)
            
            lower_slider.setValue(10)  # 1.0%
            upper_slider.setValue(990)  # 99.0%
            brightness_slider.setValue(100)  # 1.0
            contrast_slider.setValue(100)  # 1.0
            
            lower_value_label.setText("1.0%")
            upper_value_label.setText("99.0%")
            brightness_value_label.setText("1.00")
            contrast_value_label.setText("1.00")
            
            lower_slider.blockSignals(False)
            upper_slider.blockSignals(False)
            brightness_slider.blockSignals(False)
            contrast_slider.blockSignals(False)
            
            # 触发更新
            update_preview()
            
        def on_apply():
            try:
                # 保存参数
                self.image_processor.lower_percent = lower_slider.value() / 10.0
                self.image_processor.upper_percent = upper_slider.value() / 10.0
                self.image_processor.brightness_factor = brightness_slider.value() / 100.0
                self.image_processor.contrast_factor = contrast_slider.value() / 100.0
                
                # 清除缓存并重新加载当前帧
                self.image_processor.clear_all_caches()
                self.display_current_frame(force_reload=True)
                
                # 关闭对话框
                dialog.accept()
                
                # 显示状态
                self.status_label.setText("已应用图像对比度调整")
                self.progress_bar.setValue(100)
            except Exception as e:
                status_label.setText(f"应用参数时出错: {str(e)}")
            
        def on_cancel():
            dialog.reject()
            
        # 窗口大小变化事件处理
        def on_resize():
            # 窗口大小变化时更新预览
            update_preview()
            
        # 每次大小变化都更新预览
        dialog.resizeEvent = lambda event: on_resize()
            
        # 连接信号和槽
        lower_slider.valueChanged.connect(on_lower_changed)
        upper_slider.valueChanged.connect(on_upper_changed)
        brightness_slider.valueChanged.connect(on_brightness_changed)
        contrast_slider.valueChanged.connect(on_contrast_changed)
        reset_button.clicked.connect(on_reset)
        apply_button.clicked.connect(on_apply)
        cancel_button.clicked.connect(on_cancel)
        
        # 初始显示
        # 延迟一点执行初始预览，确保界面已完成布局
        QTimer.singleShot(100, update_preview)
        
        # 显示对话框
        dialog.exec_()