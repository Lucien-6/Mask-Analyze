#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图像处理模块：提供掩膜图像的各种处理功能
"""
import cv2
import numpy as np
import os
import glob
import threading
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Optional
from scipy import ndimage

from core.logger import get_logger

# 获取模块日志记录器
logger = get_logger("image_processor")


class ImageProcessor:
    def __init__(self):
        # 基本参数
        self.scale_factor = 1.0    # 像素与实际物理单位的换算系数 (μm/pixel)
        self.frame_rate = 30       # 默认帧率
        self.edge_exclusion = False  # 是否剔除边缘对象
        self.analyzed_data = []    # 存储分析结果数据
        self.is_analyzed = False   # 标记是否已分析
        
        # 图像增强参数
        self.auto_enhance = True        # 是否自动增强高位深图像
        self.lower_percent = 1.0        # 直方图截断下限百分比
        self.upper_percent = 99.0       # 直方图截断上限百分比
        self.brightness_factor = 1.0    # 亮度调整因子
        self.contrast_factor = 1.0      # 对比度调整因子
        
        # 图像路径信息
        self.original_files = []   # 存储原始图像文件路径
        self.mask_files = []       # 存储掩膜图像文件路径
        self.total_frames = 0      # 总帧数
        
        # 缓存系统
        self.original_cache = OrderedDict()  # 原始图像缓存 {frame_idx: image}
        self.mask_cache = OrderedDict()      # 掩膜图像缓存 {frame_idx: mask}
        self.processed_masks_cache = OrderedDict()  # 处理后的掩膜缓存 {frame_idx: processed_mask}
        self.thumbnail_cache = OrderedDict()  # 缩略图缓存 {frame_idx: thumbnail}
        
        # 缓存设置
        self.max_cache_size = 50   # 最大缓存帧数
        self.preload_range = 5     # 预加载范围
        
        # 图像元数据
        self.image_metadata = {}   # 存储图像元数据 {frame_idx: {"width": w, "height": h}}
        
        # 锁，防止多线程同时修改缓存
        self.cache_lock = threading.Lock()
        
        # 预生成颜色映射
        self._color_cache = {}     # 缓存生成的颜色
        self._cached_base_colors = self._generate_colors(20)  # 预生成20种不同颜色
        
        # 掩膜历史记录
        self.mask_history = {}     # 帧索引 -> 掩膜历史记录列表
        self.max_history_size = 50  # 每帧最大历史记录数量
        
        # 手动剔除对象跟踪
        self.manually_excluded_objects = set()  # 存储手动剔除的对象ID: "帧索引-对象标签值"
        self.excluded_objects_marked = {}  # 存储标记为剔除的对象中心点: {帧索引: [(对象标签值, x, y), ...]}
    
    def _generate_colors(self, n):
        """
        生成HSV颜色空间的颜色，然后转换为BGR
        
        Args:
            n: 需要生成的颜色数量
            
        Returns:
            BGR颜色列表
        """
        colors = []
        for i in range(n):
            # 均匀分布的色相值
            h = i * 179 // n
            s = 255
            v = 255
            # 将HSV转换为BGR
            bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        return colors
        
    def load_images(self, original_dir: str, mask_dir: str, start_idx: int = 0, end_idx: int = -1) -> Tuple[int, str]:
        """
        加载图像路径信息，而不是立即加载全部图像
        
        Args:
            original_dir: 原始图像目录
            mask_dir: 掩膜图像目录
            start_idx: 起始图像索引
            end_idx: 结束图像索引
        
        Returns:
            加载的图像数量和错误消息(如果有)
        """
        # 获取原始图像和掩膜图像的文件列表并按自然顺序排序
        original_files = sorted(glob.glob(os.path.join(original_dir, '*.*')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.*')))
        
        # 检查文件数量
        if not original_files or not mask_files:
            return 0, "未找到图像文件"
        
        # 调整范围
        if end_idx < 0 or end_idx >= len(original_files):
            end_idx = len(original_files) - 1
        
        if start_idx < 0:
            start_idx = 0
        
        if start_idx > end_idx:
            return 0, "起始索引大于结束索引"
        
        # 检查两个目录中的图像数量是否匹配
        if len(original_files) != len(mask_files):
            return 0, f"原始图像和掩膜图像数量不匹配 ({len(original_files)} vs {len(mask_files)})"
        
        # 清空之前的图像路径和缓存
        self.original_files = []
        self.mask_files = []
        self.clear_all_caches()
        
        # 存储指定范围内的图像路径
        self.original_files = original_files[start_idx:end_idx+1]
        self.mask_files = mask_files[start_idx:end_idx+1]
        self.total_frames = len(self.original_files)
        
        # 初始化分析数据
        self.analyzed_data = [None] * self.total_frames
        self.is_analyzed = False
        
        # 加载第一帧用于初始化
        try:
            first_original = self.get_original_image(0)
            first_mask = self.get_mask_image(0)
            
            # 检查第一帧的掩膜是否为灰度图像
            if first_mask is not None and (len(first_mask.shape) != 2 and first_mask.shape[2] != 1):
                return 0, f"掩膜图像必须为灰度图像: {self.mask_files[0]} 不满足要求！"
                
                # 检查图像尺寸是否匹配
            if first_original is not None and first_mask is not None and first_original.shape[:2] != first_mask.shape[:2]:
                return 0, f"图像尺寸不匹配: {self.original_files[0]} ({first_original.shape[:2]}) vs {self.mask_files[0]} ({first_mask.shape[:2]})"
            
            # 启动异步预加载线程
            threading.Thread(target=self.preload_adjacent_frames, args=(0,), daemon=True).start()
            
        except Exception as e:
            return 0, f"加载图像出错: {str(e)}"
        
        return self.total_frames, ""
    
    def clear_all_caches(self):
        """清空所有缓存"""
        with self.cache_lock:
            self.original_cache.clear()
            self.mask_cache.clear()
            self.processed_masks_cache.clear()
            self.thumbnail_cache.clear()
            self.image_metadata.clear()
    
    def get_original_image(self, frame_idx: int) -> np.ndarray:
        """
        获取指定帧的原始图像，如果不在缓存中则从文件加载
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            原始图像，如果不存在则返回None
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        
        with self.cache_lock:
            # 检查缓存
            if frame_idx in self.original_cache:
                # 将访问的图像移到缓存队列末尾（最近使用）
                image = self.original_cache.pop(frame_idx)
                self.original_cache[frame_idx] = image
                return image
        
        # 缓存未命中，从文件加载
        try:
            image_path = self.original_files[frame_idx]
            # 先尝试读取高位深图像
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if image is not None:
                # 检测并增强高位深图像
                image = self.enhance_high_bit_depth_image(image)
                
                # 记录图像元数据
                self.image_metadata[frame_idx] = {
                    "width": image.shape[1],
                    "height": image.shape[0]
                }
                
                with self.cache_lock:
                    # 检查缓存是否已满
                    if len(self.original_cache) >= self.max_cache_size:
                        # 移除最早使用的图像（LRU策略）
                        self.original_cache.popitem(last=False)
                    
                    # 添加到缓存
                    self.original_cache[frame_idx] = image
            
            return image
        except Exception as e:
            logger.error(f"加载原始图像 {frame_idx} 失败: {e}")
            return None
            
    def enhance_high_bit_depth_image(self, image: np.ndarray) -> np.ndarray:
        """
        增强高位深图像的显示效果，使用直方图调整方式
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        # 检查图像是否存在或是否开启自动增强
        if image is None or not self.auto_enhance:
            return image
            
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
                    l_enhanced = self._apply_histogram_stretch(l)
                    
                    # 应用亮度和对比度调整
                    l_enhanced = self._adjust_brightness_contrast(
                        l_enhanced, 
                        self.brightness_factor, 
                        self.contrast_factor
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
                enhanced = self._apply_histogram_stretch(image)
                
                # 应用亮度和对比度调整
                enhanced = self._adjust_brightness_contrast(
                    enhanced, 
                    self.brightness_factor, 
                    self.contrast_factor
                )
                
                return enhanced
        
        return image
    
    def _apply_histogram_stretch(self, image: np.ndarray) -> np.ndarray:
        """
        应用直方图拉伸，将图像的亮度分布重新映射到指定的百分位范围
        
        Args:
            image: 输入图像（灰度图或单通道）
            
        Returns:
            增强后的图像
        """
        # 计算指定百分位的像素值
        if image.dtype == np.uint16:
            min_val = np.percentile(image, self.lower_percent)
            max_val = np.percentile(image, self.upper_percent)
            
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
    
    def _adjust_brightness_contrast(self, image: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
        """
        调整图像的亮度和对比度
        
        Args:
            image: 输入图像
            brightness: 亮度因子（1.0表示原始亮度）
            contrast: 对比度因子（1.0表示原始对比度）
            
        Returns:
            调整后的图像
        """
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
    
    def get_mask_image(self, frame_idx: int) -> np.ndarray:
        """
        获取指定帧的掩膜图像，如果不在缓存中则从文件加载
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            掩膜图像，如果不存在则返回None
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        
        with self.cache_lock:
            # 检查缓存
            if frame_idx in self.mask_cache:
                # 将访问的图像移到缓存队列末尾（最近使用）
                mask = self.mask_cache.pop(frame_idx)
                self.mask_cache[frame_idx] = mask
                return mask
        
        # 缓存未命中，从文件加载
        try:
            mask_path = self.mask_files[frame_idx]
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            
            if mask is not None:
                with self.cache_lock:
                    # 检查缓存是否已满
                    if len(self.mask_cache) >= self.max_cache_size:
                        # 移除最早使用的图像（LRU策略）
                        self.mask_cache.popitem(last=False)
                    
                    # 添加到缓存
                    self.mask_cache[frame_idx] = mask
                    
                    # 初始化处理后的掩膜为原始掩膜的副本
                    if frame_idx not in self.processed_masks_cache:
                        self.processed_masks_cache[frame_idx] = mask.copy()
            
            return mask
        except Exception as e:
            logger.error(f"加载掩膜图像 {frame_idx} 失败: {e}")
            return None
    
    def get_processed_mask(self, frame_idx: int) -> np.ndarray:
        """
        获取指定帧的处理后掩膜，如果不在缓存中则从原始掩膜初始化
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            处理后的掩膜，如果不存在则返回None
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        
        with self.cache_lock:
            # 检查缓存
            if frame_idx in self.processed_masks_cache:
                # 将访问的掩膜移到缓存队列末尾（最近使用）
                mask = self.processed_masks_cache.pop(frame_idx)
                self.processed_masks_cache[frame_idx] = mask
                return mask
        
        # 缓存未命中，获取原始掩膜并进行初始化
        original_mask = self.get_mask_image(frame_idx)
        if original_mask is not None:
            with self.cache_lock:
                # 检查缓存是否已满
                if len(self.processed_masks_cache) >= self.max_cache_size:
                    # 移除最早使用的掩膜（LRU策略）
                    self.processed_masks_cache.popitem(last=False)
                
                # 添加到缓存
                processed_mask = original_mask.copy()
                self.processed_masks_cache[frame_idx] = processed_mask
                return processed_mask
        
        return None
    
    def set_processed_mask(self, frame_idx: int, mask: np.ndarray) -> None:
        """
        设置指定帧的处理后掩膜，并保存历史记录
        
        Args:
            frame_idx: 帧索引
            mask: 处理后的掩膜
        """
        if frame_idx < 0 or frame_idx >= self.total_frames or mask is None:
            return
        
        with self.cache_lock:
            # 保存当前掩膜到历史记录
            if frame_idx in self.processed_masks_cache:
                current_mask = self.processed_masks_cache[frame_idx]
                self._add_to_history(frame_idx, current_mask.copy())
            
            # 检查缓存是否已满
            if frame_idx not in self.processed_masks_cache and len(self.processed_masks_cache) >= self.max_cache_size:
                # 移除最早使用的掩膜（LRU策略）
                self.processed_masks_cache.popitem(last=False)
            
            # 添加或更新缓存
            self.processed_masks_cache[frame_idx] = mask
    
    def _add_to_history(self, frame_idx: int, mask: np.ndarray) -> None:
        """
        将掩膜添加到历史记录
        
        Args:
            frame_idx: 帧索引
            mask: 掩膜图像
        """
        # 初始化该帧的历史记录列表
        if frame_idx not in self.mask_history:
            self.mask_history[frame_idx] = []
        
        # 限制历史记录数量
        if len(self.mask_history[frame_idx]) >= self.max_history_size:
            self.mask_history[frame_idx].pop(0)  # 移除最旧的记录
        
        # 添加新记录
        self.mask_history[frame_idx].append(mask)
    
    def undo_last_mask_processing(self, frame_idx: int = -1) -> bool:
        """
        撤销最近一次掩膜处理操作
        
        Args:
            frame_idx: 帧索引，-1表示应用到所有帧
            
        Returns:
            是否成功撤销
        """
        if frame_idx >= 0 and frame_idx < self.total_frames:
            # 处理单一帧
            return self._undo_single_frame(frame_idx)
        else:
            # 处理所有帧
            success = False
            for i in range(self.total_frames):
                if self._undo_single_frame(i):
                    success = True
            return success
    
    def _undo_single_frame(self, frame_idx: int) -> bool:
        """
        撤销单一帧的最近一次掩膜处理操作
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            是否成功撤销
        """
        with self.cache_lock:
            # 检查是否有历史记录可撤销
            if frame_idx not in self.mask_history or not self.mask_history[frame_idx]:
                return False
            
            # 获取最新的历史记录
            last_mask = self.mask_history[frame_idx].pop()
            
            # 恢复到上一状态
            self.processed_masks_cache[frame_idx] = last_mask
            
            # 重置分析状态
            self.is_analyzed = False
            
            return True
    
    def restore_original_masks(self, frame_idx: int = -1) -> bool:
        """
        将掩膜还原至最初状态
        
        Args:
            frame_idx: 帧索引，-1表示应用到所有帧
            
        Returns:
            是否成功还原
        """
        if frame_idx >= 0 and frame_idx < self.total_frames:
            # 处理单一帧
            return self._restore_original_single_frame(frame_idx)
        else:
            # 处理所有帧
            success = False
            for i in range(self.total_frames):
                if self._restore_original_single_frame(i):
                    success = True
            return success
    
    def _restore_original_single_frame(self, frame_idx: int) -> bool:
        """
        将单一帧的掩膜还原至最初状态
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            是否成功还原
        """
        original_mask = self.get_mask_image(frame_idx)
        if original_mask is None:
            return False
        
        with self.cache_lock:
            # 清空该帧的历史记录
            if frame_idx in self.mask_history:
                self.mask_history[frame_idx] = []
            
            # 还原到原始状态
            self.processed_masks_cache[frame_idx] = original_mask.copy()
            
            # 重置分析状态
            self.is_analyzed = False
            
            return True
    
    def preload_adjacent_frames(self, current_frame_idx: int, range_size: int = None) -> None:
        """
        预加载当前帧附近的图像
        
        Args:
            current_frame_idx: 当前帧索引
            range_size: 预加载范围，默认使用self.preload_range
        """
        if range_size is None:
            range_size = self.preload_range
        
        # 计算预加载范围
        start_idx = max(0, current_frame_idx - range_size)
        end_idx = min(self.total_frames, current_frame_idx + range_size + 1)
        
        # 优先加载当前帧及其后面的几帧，再加载前面的帧
        prioritized_indices = list(range(current_frame_idx, end_idx)) + list(range(start_idx, current_frame_idx))
        
        # 预加载原始图像和掩膜
        for idx in prioritized_indices:
            if idx != current_frame_idx:  # 当前帧已加载，跳过
                # 异步加载原始图像和掩膜
                try:
                    # 检查是否已经在缓存中
                    with self.cache_lock:
                        original_in_cache = idx in self.original_cache
                        mask_in_cache = idx in self.mask_cache
                    
                    # 低优先级加载未缓存的图像
                    if not original_in_cache:
                        self.get_original_image(idx)
                    
                    if not mask_in_cache:
                        self.get_mask_image(idx)
                except Exception as e:
                    logger.warning(f"预加载图像 {idx} 失败: {e}")
    
    def generate_thumbnail(self, image: np.ndarray, max_size: int = 256) -> np.ndarray:
        """
        生成缩略图
        
        Args:
            image: 原始图像
            max_size: 缩略图最大尺寸
            
        Returns:
            缩略图
        """
        if image is None:
            return None
        
        h, w = image.shape[:2]
        # 计算缩放比例
        scale = min(max_size / w, max_size / h)
        if scale >= 1:  # 图像小于缩略图尺寸
            return image.copy()
        
        # 缩放图像
        new_w, new_h = int(w * scale), int(h * scale)
        thumbnail = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return thumbnail
    
    def get_thumbnail(self, frame_idx: int, max_size: int = 256) -> np.ndarray:
        """
        获取缩略图
        
        Args:
            frame_idx: 帧索引
            max_size: 缩略图最大尺寸
            
        Returns:
            缩略图
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        
        with self.cache_lock:
            # 检查缓存
            if frame_idx in self.thumbnail_cache:
                thumbnail = self.thumbnail_cache.pop(frame_idx)
                self.thumbnail_cache[frame_idx] = thumbnail
                return thumbnail
        
        # 获取原图（优先从缓存获取）
        image = self.get_original_image(frame_idx)
        if image is None:
            return None
        
        # 生成缩略图
        thumbnail = self.generate_thumbnail(image, max_size)
        
        with self.cache_lock:
            # 缓存缩略图
            if len(self.thumbnail_cache) >= self.max_cache_size * 2:  # 缩略图可以缓存更多
                self.thumbnail_cache.popitem(last=False)
            self.thumbnail_cache[frame_idx] = thumbnail
        
        return thumbnail
    
    def set_parameters(self, scale_factor: float, frame_rate: float):
        """
        设置参数
        
        Args:
            scale_factor: 像素与实际物理单位的换算系数 (μm/pixel)
            frame_rate: 帧率
        """
        self.scale_factor = scale_factor
        self.frame_rate = frame_rate
    
    def apply_dilation(self, kernel_size: int, iterations: int, frame_idx: int = -1) -> np.ndarray:
        """
        应用膨胀操作
        
        Args:
            kernel_size: 核大小
            iterations: 迭代次数
            frame_idx: 帧索引，-1表示应用到所有帧
            
        Returns:
            处理后的掩膜
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if frame_idx >= 0 and frame_idx < self.total_frames:
            # 处理单一帧
            mask = self.get_processed_mask(frame_idx)
            if mask is None:
                return None
                
            dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
            self.set_processed_mask(frame_idx, dilated_mask)
            return dilated_mask
        else:
            # 处理所有帧
            for i in range(self.total_frames):
                self.apply_dilation(kernel_size, iterations, i)
            return self.get_processed_mask(0)
            
    def apply_erosion(self, kernel_size: int, iterations: int, frame_idx: int = -1) -> np.ndarray:
        """
        应用腐蚀操作
        
        Args:
            kernel_size: 核大小
            iterations: 迭代次数
            frame_idx: 帧索引，-1表示应用到所有帧
            
        Returns:
            处理后的掩膜
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if frame_idx >= 0 and frame_idx < self.total_frames:
            # 处理单一帧
            mask = self.get_processed_mask(frame_idx)
            if mask is None:
                return None
                
            eroded_mask = cv2.erode(mask, kernel, iterations=iterations)
            self.set_processed_mask(frame_idx, eroded_mask)
            return eroded_mask
        else:
            # 处理所有帧
            for i in range(self.total_frames):
                self.apply_erosion(kernel_size, iterations, i)
            return self.get_processed_mask(0)
    
    def apply_opening(self, kernel_size: int, iterations: int, frame_idx: int = -1) -> np.ndarray:
        """
        应用开运算操作 (先腐蚀后膨胀)
        
        Args:
            kernel_size: 核大小
            iterations: 迭代次数
            frame_idx: 帧索引，-1表示应用到所有帧
            
        Returns:
            处理后的掩膜
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if frame_idx >= 0 and frame_idx < self.total_frames:
            # 处理单一帧
            mask = self.get_processed_mask(frame_idx)
            if mask is None:
                return None
                
            opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
            self.set_processed_mask(frame_idx, opened_mask)
            return opened_mask
        else:
            # 处理所有帧
            for i in range(self.total_frames):
                self.apply_opening(kernel_size, iterations, i)
            return self.get_processed_mask(0)
    
    def apply_closing(self, kernel_size: int, iterations: int, frame_idx: int = -1) -> np.ndarray:
        """
        应用闭运算操作 (先膨胀后腐蚀)
        
        Args:
            kernel_size: 核大小
            iterations: 迭代次数
            frame_idx: 帧索引，-1表示应用到所有帧
            
        Returns:
            处理后的掩膜
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if frame_idx >= 0 and frame_idx < self.total_frames:
            # 处理单一帧
            mask = self.get_processed_mask(frame_idx)
            if mask is None:
                return None
                
            closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            self.set_processed_mask(frame_idx, closed_mask)
            return closed_mask
        else:
            # 处理所有帧
            for i in range(self.total_frames):
                self.apply_closing(kernel_size, iterations, i)
            return self.get_processed_mask(0)
    
    # 为兼容现有代码，添加以下属性
    @property
    def original_images(self):
        """返回当前已缓存的原始图像列表，用于兼容旧代码"""
        images = []
        for i in range(self.total_frames):
            image = self.get_original_image(i)
            if image is not None:
                images.append(image)
        return images
    
    @property
    def mask_images(self):
        """返回当前已缓存的掩膜图像列表，用于兼容旧代码"""
        masks = []
        for i in range(self.total_frames):
            mask = self.get_mask_image(i)
            if mask is not None:
                masks.append(mask)
        return masks
    
    @property
    def processed_masks(self):
        """返回当前已缓存的处理后掩膜列表，用于兼容旧代码"""
        masks = []
        for i in range(self.total_frames):
            mask = self.get_processed_mask(i)
            if mask is not None:
                masks.append(mask)
        return masks
    
    def fill_holes(self, frame_idx: int = -1) -> np.ndarray:
        """
        填充掩膜中的孔洞，特别处理边缘对象以确保它们不会丢失
        
        Args:
            frame_idx: 帧索引，-1表示应用到所有帧
            
        Returns:
            处理后的掩膜
        """
        if not self.processed_masks:
            return None
            
        def fill_holes_single(mask):
            # 保留原始掩膜数据类型，不强制转换为8位
            # 创建相同数据类型的副本
            mask_copy = mask.copy()
                
            # 特殊处理 - 如果掩膜为空，直接返回原始掩膜
            if np.max(mask_copy) == 0:
                return mask_copy
            
            # 创建结果掩膜 - 复制原始掩膜
            result_mask = mask_copy.copy()
            
            # 找到所有不同的标签值
            unique_values = np.unique(mask_copy)
            unique_values = unique_values[unique_values > 0]  # 排除背景值
            
            # 统计处理前的对象数
            pre_fill_object_count = 0
            for val in unique_values:
                binary_mask = (mask_copy == val).astype(np.uint8)  # 二值掩膜可以是uint8
                num_labels, _ = cv2.connectedComponents(binary_mask, connectivity=8)
                pre_fill_object_count += (num_labels - 1)  # 减去背景
            
            # 获取掩膜尺寸
            h, w = mask_copy.shape
            
            # 创建边缘掩膜用于检测边缘对象
            edge_mask = np.zeros((h, w), dtype=np.uint8)
            edge_mask[0, :] = edge_mask[h-1, :] = edge_mask[:, 0] = edge_mask[:, w-1] = 1
            # 扩展边缘区域，确保边缘对象完全覆盖
            edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
            
            # 依次处理每个标签值
            for val in unique_values:
                # 创建当前标签的二值掩膜 - 二值掩膜可以安全地转为uint8，因为只有0和1两个值
                binary_mask = (mask_copy == val).astype(np.uint8)
                
                # 检查是否与边缘相交
                is_edge_object = np.any(binary_mask & edge_mask)
                
                # 获取所有连通区域
                num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
                
                # 对每个连通区域单独处理
                for i in range(1, num_labels):
                    # 提取当前连通区域
                    component = (labels == i).astype(np.uint8)
                    
                    # 检查当前组件是否与边缘相交
                    component_is_edge = np.any(component & edge_mask)
                    
                    # 为边缘对象使用更安全的填充方法
                    if component_is_edge:
                        # 创建一个安全副本避免修改原始数据
                        component_filled = component.copy()
                        
                        # 使用形态学闭操作填充小孔洞 (安全方法，不会扩大边界)
                        kernel_size = max(3, int(min(h, w) * 0.01))  # 自适应核大小
                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        component_filled = cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel)
                        
                        # 确保不扩大对象 - 与原始组件取交集
                        component_filled = np.logical_and(component_filled > 0, component > 0).astype(np.uint8)
                        component_filled = component_filled + component  # 确保原始区域保留
                        component_filled = np.clip(component_filled, 0, 1).astype(np.uint8)
                    else:
                        # 非边缘对象使用 scipy 的 binary_fill_holes（更高效）
                        # 将组件转换为布尔数组进行孔洞填充
                        component_bool = component.astype(bool)
                        # 使用 scipy.ndimage.binary_fill_holes 填充孔洞
                        filled_bool = ndimage.binary_fill_holes(component_bool)
                        # 转换回 uint8 格式
                        component_filled = filled_bool.astype(np.uint8)
                    
                    # 更新结果掩膜 - 精确更新，只修改这个组件
                    # 先清除这个区域的原始值
                    result_mask[component > 0] = 0
                    # 然后设置新值，确保使用原始标签值
                    result_mask[component_filled > 0] = val
            
            # 安全检查 - 确保填充后没有减少对象数量且没有丢失标签值
            post_fill_object_count = 0
            result_labels = np.unique(result_mask)
            result_labels = result_labels[result_labels > 0]
            
            # 检查是否所有原始标签值都在填充后的掩膜中
            missing_labels = set(unique_values) - set(result_labels)
            
            for val in result_labels:
                # 创建二值掩膜进行连通性分析
                binary_mask = (result_mask == val).astype(np.uint8)
                num_labels, _ = cv2.connectedComponents(binary_mask, connectivity=8)
                post_fill_object_count += (num_labels - 1)  # 减去背景
            
            # 如果对象数量减少或有标签丢失，警告并回退到原始掩膜
            if post_fill_object_count < pre_fill_object_count or len(missing_labels) > 0:
                if len(missing_labels) > 0:
                    logger.warning(f"填充孔洞后丢失标签值 {missing_labels}，使用原始掩膜")
                else:
                    logger.warning(f"填充孔洞导致对象数量从{pre_fill_object_count}减少到{post_fill_object_count}，使用原始掩膜")
                return mask_copy
            
            return result_mask
        
        # 开始处理掩膜
        if frame_idx >= 0 and frame_idx < len(self.processed_masks):
            # 处理单一帧
            original_mask = self.processed_masks[frame_idx].copy()  # 保存原始掩膜用于恢复
            try:
                # 尝试填充孔洞
                filled_mask = fill_holes_single(self.processed_masks[frame_idx])
                # 检查填充后的掩膜是否有问题
                unique_before = np.unique(self.processed_masks[frame_idx])
                unique_after = np.unique(filled_mask)
                
                # 确保填充后不会丢失任何标签值
                before_positive = unique_before[unique_before > 0]
                after_positive = unique_after[unique_after > 0]
                if len(before_positive) > len(after_positive):
                    logger.warning(f"填充孔洞后丢失标签值，使用原始掩膜。原标签值数量：{len(before_positive)}，填充后：{len(after_positive)}")
                    self.processed_masks[frame_idx] = original_mask
                else:
                    self.processed_masks[frame_idx] = filled_mask
            except Exception as e:
                logger.error(f"填充孔洞时出错: {e}，使用原始掩膜")
                # 出错时恢复原始掩膜
                self.processed_masks[frame_idx] = original_mask
            
            # 处理后重置分析状态，强制重新分析
            self.is_analyzed = False
            return self.processed_masks[frame_idx]
        else:
            # 处理所有帧
            original_masks = [mask.copy() for mask in self.processed_masks]  # 保存原始掩膜
            for i in range(len(self.processed_masks)):
                try:
                    # 尝试填充孔洞
                    filled_mask = fill_holes_single(self.processed_masks[i])
                    # 检查填充后的掩膜是否有问题
                    unique_before = np.unique(self.processed_masks[i])
                    unique_after = np.unique(filled_mask)
                    
                    # 确保填充后不会丢失任何标签值
                    before_positive = unique_before[unique_before > 0]
                    after_positive = unique_after[unique_after > 0]
                    if len(before_positive) > len(after_positive):
                        logger.warning(f"第{i}帧填充孔洞后丢失标签值，使用原始掩膜。原标签值数量：{len(before_positive)}，填充后：{len(after_positive)}")
                        self.processed_masks[i] = original_masks[i]
                    else:
                        self.processed_masks[i] = filled_mask
                except Exception as e:
                    logger.error(f"处理第{i}帧时出错: {e}，使用原始掩膜")
                    # 出错时恢复原始掩膜
                    self.processed_masks[i] = original_masks[i]
            
            # 处理后重置分析状态，强制重新分析
            self.is_analyzed = False
            return self.processed_masks[0] if self.processed_masks else None
    
    def filter_by_area(self, min_area: float, max_area: float, frame_idx: int = -1) -> np.ndarray:
        """
        根据面积过滤掩膜对象
        
        Args:
            min_area: 最小面积（μm²）
            max_area: 最大面积（μm²），-1表示无上限
            frame_idx: 帧索引，-1表示应用到所有帧
            
        Returns:
            处理后的掩膜
        """
        if not self.processed_masks:
            return None
        
        # 转换物理面积到像素面积
        min_area_pixels = min_area / (self.scale_factor ** 2) if min_area > 0 else 0
        max_area_pixels = max_area / (self.scale_factor ** 2) if max_area > 0 else -1
            
        def filter_by_area_single(mask):
            # 找到所有不同的对象值
            unique_values = np.unique(mask)
            unique_values = unique_values[unique_values > 0]  # 排除背景值
            
            filtered_mask = np.zeros_like(mask)
            
            for val in unique_values:
                # 为每个不同的对象值创建单独的二值掩膜
                binary_mask = (mask == val).astype(np.uint8)
                
                # 找到所有连通区域 - 使用8连通性
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
                
                for i in range(1, num_labels):  # 跳过背景
                    area = stats[i, cv2.CC_STAT_AREA]
                    if (area >= min_area_pixels) and (max_area_pixels == -1 or area <= max_area_pixels):
                        # 保留符合面积条件的对象
                        filtered_mask[labels == i] = val
                        
            return filtered_mask
        
        if frame_idx >= 0 and frame_idx < len(self.processed_masks):
            # 处理单一帧
            self.processed_masks[frame_idx] = filter_by_area_single(self.processed_masks[frame_idx])
            return self.processed_masks[frame_idx]
        else:
            # 处理所有帧
            for i in range(len(self.processed_masks)):
                self.processed_masks[i] = filter_by_area_single(self.processed_masks[i])
            return self.processed_masks[0] if self.processed_masks else None
    
    def exclude_edge_objects(self, enable: bool) -> None:
        """
        设置是否剔除边缘对象，并立即应用到所有帧的掩膜
        
        Args:
            enable: 是否启用剔除边缘对象
        """
        # 记录设置状态变化
        self.edge_exclusion = enable
        
        # 如果启用剔除但没有掩膜数据，则直接返回
        if not self.processed_masks:
            return
            
        # 如果启用剔除，立即应用到所有帧
        if enable:
            for frame_idx in range(len(self.processed_masks)):
                mask = self.processed_masks[frame_idx]
                h, w = mask.shape
                
                # 创建边缘掩膜
                edge_mask = np.zeros((h, w), dtype=np.uint8)
                edge_mask[0, :] = edge_mask[h-1, :] = edge_mask[:, 0] = edge_mask[:, w-1] = 1
                edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=1)
                
                # 获取非零像素的唯一值
                unique_values = np.unique(mask)
                unique_values = unique_values[unique_values > 0]  # 排除背景值(0)
                
                # 处理每个唯一的非零值
                for val in unique_values:
                    # 创建二值掩膜
                    binary_mask = (mask == val).astype(np.uint8)
                    
                    # 检查对象是否与边缘相交
                    if np.any(binary_mask & edge_mask):
                        # 如果对象与边缘相交，将其从掩膜中移除
                        mask[binary_mask == 1] = 0
                
                # 更新处理后的掩膜
                self.processed_masks[frame_idx] = mask
            
            # 重置分析状态，因为掩膜已修改
            self.is_analyzed = False
        
    def reset_masks(self, frame_idx: int = -1) -> np.ndarray:
        """
        重置掩膜为原始掩膜
        
        Args:
            frame_idx: 帧索引，-1表示重置所有帧
            
        Returns:
            重置后的掩膜
        """
        if not self.mask_images:
            return None
            
        if frame_idx >= 0 and frame_idx < len(self.mask_images):
            # 重置单一帧
            self.processed_masks[frame_idx] = self.mask_images[frame_idx].copy()
            return self.processed_masks[frame_idx]
        else:
            # 重置所有帧
            for i in range(len(self.mask_images)):
                self.processed_masks[i] = self.mask_images[i].copy()
            return self.processed_masks[0] if self.processed_masks else None
    
    def get_overlay_image(self, frame_idx: int, show_edges: bool = False, 
                         show_centers: bool = False, show_axes: bool = False,
                         show_excluded_marks: bool = True) -> np.ndarray:
        """
        获取叠加了掩膜的原始图像
        
        Args:
            frame_idx: 帧索引
            show_edges: 是否只显示边缘轮廓
            show_centers: 是否显示对象中心点
            show_axes: 是否显示对象长短轴
            show_excluded_marks: 是否显示手动剔除的标记
            
        Returns:
            叠加了掩膜的原始图像
        """
        if (frame_idx < 0 or frame_idx >= len(self.original_images) or 
            frame_idx >= len(self.processed_masks)):
            return None
            
        original = self.get_original_image(frame_idx)
        if original is None:
            return None
            
        original = original.copy()
        mask = self.processed_masks[frame_idx]
        
        # 创建彩色图像
        if len(original.shape) == 2:  # 灰度图转为彩色图
            original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        elif original.shape[2] == 3:  # 已经是彩色图
            original_rgb = original
        else:  # 可能是RGBA或其他格式，转换为BGR
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGRA2BGR)
            
        # 调整尺寸以确保原始图和掩膜尺寸一致
        if original_rgb.shape[:2] != mask.shape[:2]:
            original_rgb = cv2.resize(original_rgb, (mask.shape[1], mask.shape[0]))
            
        # 不同的对象使用不同的颜色显示
        overlay = original_rgb.copy()
        
        # 如果已分析并且需要显示高级特征(轮廓、中心点、轴线)
        if self.is_analyzed and (show_edges or show_centers or show_axes):
            # 获取分析好的对象数据
            objects_data = self.analyzed_data[frame_idx] if frame_idx < len(self.analyzed_data) else []
            
            # 批量处理轮廓绘制优化
            if show_edges and objects_data:
                # 预先准备所有轮廓和颜色
                all_contours = []
                all_colors = []
                
                for obj_data in objects_data:
                    label_value = obj_data["label_value"]
                    color_index = label_value % len(self._cached_base_colors)
                    color = self._cached_base_colors[color_index]
                    
                    contour = obj_data["contour"]
                    all_contours.append(contour)
                    all_colors.append(color)
                
                # 批量绘制所有轮廓
                if all_contours:
                    cv2.drawContours(overlay, all_contours, -1, (0,0,0), 1)  # 先绘制黑色边框增强可见性
                    for i, (contour, color) in enumerate(zip(all_contours, all_colors)):
                        cv2.drawContours(overlay, [contour], -1, color, 1)
            
            # 批量处理中心点绘制优化
            if show_centers and objects_data:
                centers = []
                for obj_data in objects_data:
                    center_x = int(obj_data["center_x"])
                    center_y = int(obj_data["center_y"])
                    centers.append((center_x, center_y))
                
                # 绘制中心点 - 红色十字标记（优化绘制方法）
                cross_size = 4  # 十字大小
                
                # 批量创建点数组以加快绘制
                if centers:
                    for cx, cy in centers:
                        # 绘制水平线
                        cv2.line(overlay, 
                              (cx - cross_size, cy),
                              (cx + cross_size, cy),
                              (0, 0, 255), 1)
                        # 绘制垂直线
                        cv2.line(overlay,
                              (cx, cy - cross_size),
                              (cx, cy + cross_size),
                              (0, 0, 255), 1)
            
            # 批量处理轴线绘制
            if show_axes and objects_data:
                major_lines = []
                minor_lines = []
                
                for obj_data in objects_data:
                    # 检查是否有轴线坐标信息
                    has_axis_coordinates = all(key in obj_data for key in [
                        "major_axis_start_x", "major_axis_start_y", 
                        "major_axis_end_x", "major_axis_end_y",
                        "minor_axis_start_x", "minor_axis_start_y",
                        "minor_axis_end_x", "minor_axis_end_y"
                    ])
                    
                    if has_axis_coordinates:
                        # 准备长轴数据
                        major_start = (int(obj_data["major_axis_start_x"]), int(obj_data["major_axis_start_y"]))
                        major_end = (int(obj_data["major_axis_end_x"]), int(obj_data["major_axis_end_y"]))
                        major_lines.append((major_start, major_end))
                        
                        # 准备短轴数据
                        minor_start = (int(obj_data["minor_axis_start_x"]), int(obj_data["minor_axis_start_y"]))
                        minor_end = (int(obj_data["minor_axis_end_x"]), int(obj_data["minor_axis_end_y"]))
                        minor_lines.append((minor_start, minor_end))
                    else:
                        # 如果没有轴线坐标，则尝试计算
                        if all(key in obj_data for key in ["center_x", "center_y", "major_axis_pixels", "minor_axis_pixels"]):
                            try:
                                # 获取必要信息
                                cx, cy = obj_data["center_x"], obj_data["center_y"]
                                major_len = obj_data["major_axis_pixels"] / 2
                                minor_len = obj_data["minor_axis_pixels"] / 2
                                angle = obj_data.get("angle_degrees", 0)
                                
                                # 转换角度为弧度
                                angle_rad = np.radians(angle)
                                
                                # 计算长轴的起点和终点
                                major_dx = major_len * np.cos(angle_rad)
                                major_dy = major_len * np.sin(angle_rad)
                                
                                major_start = (int(cx - major_dx), int(cy - major_dy))
                                major_end = (int(cx + major_dx), int(cy + major_dy))
                                major_lines.append((major_start, major_end))
                                
                                # 计算短轴的起点和终点
                                minor_angle_rad = angle_rad + np.pi/2  # 加90度
                                minor_dx = minor_len * np.cos(minor_angle_rad)
                                minor_dy = minor_len * np.sin(minor_angle_rad)
                                
                                minor_start = (int(cx - minor_dx), int(cy - minor_dy))
                                minor_end = (int(cx + minor_dx), int(cy + minor_dy))
                                minor_lines.append((minor_start, minor_end))
                            except Exception as e:
                                logger.error(f"计算轴线时出错: {e}")
                
                # 批量绘制所有轴线
                for start, end in major_lines:
                    cv2.line(overlay, start, end, (0, 0, 255), 1)
                
                for start, end in minor_lines:
                    cv2.line(overlay, start, end, (0, 255, 0), 1)
        else:
            # 简单模式：未分析或不显示高级特征，只根据掩膜的灰度值添加伪彩
            # 优化版：使用更高效的向量化操作和缓存的颜色映射
            
            # 如果掩膜中没有对象，直接返回原始图像
            if np.all(mask == 0):
                return overlay
                
            # 获取非零像素的唯一值
            unique_values = np.unique(mask)
            unique_values = unique_values[unique_values > 0]  # 排除背景值(0)
            
            if len(unique_values) == 0:
                return overlay
            
            # 缓存颜色映射
            max_label = int(np.max(unique_values))
            # 检查缓存是否存在此大小的颜色映射
            if max_label not in self._color_cache:
                # 创建新的颜色映射并缓存
                colormap = np.zeros((max_label + 1, 3), dtype=np.uint8)
                # 为每个标签值分配颜色
                for val in range(1, max_label + 1):
                    color_index = val % len(self._cached_base_colors)
                    colormap[val] = self._cached_base_colors[color_index]
                # 缓存结果
                self._color_cache[max_label] = colormap
            else:
                # 使用缓存的颜色映射
                colormap = self._color_cache[max_label]
            
            # 使用LUT加速颜色映射 - 避免创建中间数组
            # 创建预设透明度的彩色掩膜图像
            h, w = mask.shape
            alpha = 0.5
            
            # 使用mask作为索引直接查找颜色
            # 方法1: 先创建彩色掩膜，再与原图混合
            # mask_3d = np.zeros((h, w, 3), dtype=np.uint8)
            # non_zero_mask = mask > 0
            # mask_3d[non_zero_mask] = colormap[mask[non_zero_mask]]
            
            # overlay[non_zero_mask] = cv2.addWeighted(
            #    overlay[non_zero_mask], alpha, 
            #    mask_3d[non_zero_mask], 1-alpha, 
            #    0, dtype=cv2.CV_8U
            # )
            
            # 方法2: 使用NumPy的高级索引和广播功能，减少内存使用
            non_zero_mask = mask > 0
            if np.any(non_zero_mask):
                # 获取原始图像和对应掩膜颜色
                src = overlay[non_zero_mask].astype(np.float32)
                color_indices = mask[non_zero_mask]
                color_pixels = colormap[color_indices].astype(np.float32)
                
                # 直接计算混合结果
                blended = (src * alpha + color_pixels * (1 - alpha)).astype(np.uint8)
                
                # 赋值回原图
                overlay[non_zero_mask] = blended
        
        # 绘制手动剔除对象的标记（白色X标记）
        if show_excluded_marks and frame_idx in self.excluded_objects_marked:
            excluded_marks = self.excluded_objects_marked[frame_idx]
            for _, x, y in excluded_marks:
                # 使用cv2.drawMarker绘制白色X标记（带抗锯齿）
                cv2.drawMarker(overlay, (x, y), (255, 255, 255), 
                               cv2.MARKER_TILTED_CROSS, 8, 1, cv2.LINE_AA)
        
        return overlay
    
    def analyze_objects_all(self) -> None:
        """
        分析所有帧的对象并存储结果
        """
        # 清空之前的分析数据
        self.analyzed_data = []
        
        # 分析每一帧
        for i in range(len(self.processed_masks)):
            frame_objects = self.analyze_objects(i)
            self.analyzed_data.append(frame_objects)
        
        # 标记为已分析
        self.is_analyzed = True
        
    def analyze_objects(self, frame_idx: int) -> List[Dict[str, Any]]:
        """
        分析指定帧的对象并返回结果
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            对象信息列表
        """
        if (not self.processed_masks or 
            frame_idx < 0 or 
            frame_idx >= len(self.processed_masks)):
            return []
            
        # 获取当前掩膜
        mask = self.processed_masks[frame_idx]
        
        # 存储对象信息的列表
        objects = []
        
        # 创建边缘掩膜用于检测边缘对象
        h, w = mask.shape
        edge_mask = np.zeros((h, w), dtype=np.uint8)
        edge_mask[0, :] = edge_mask[h-1, :] = edge_mask[:, 0] = edge_mask[:, w-1] = 1
        
        # 找到所有不同的对象标签 - 使用更高效的预处理
        unique_values = np.unique(mask)
        unique_values = unique_values[unique_values > 0]  # 排除背景值0
        
        # 如果没有对象，直接返回空列表
        if len(unique_values) == 0:
            return []
        
        # 计算物理系数的平方，避免在循环中重复计算
        scale_factor_squared = self.scale_factor ** 2
        
        # 使用更高效的向量化处理
        # 创建标签到二值掩膜的映射，避免在循环中重复创建
        binary_masks = {}
        edge_intersection = {}
        
        # 获取当前帧的手动剔除对象列表
        manually_excluded = self.get_manually_excluded_objects(frame_idx)
        
        # 预先处理所有标签的二值掩膜和边缘检测
        for val in unique_values:
            # 如果对象被手动剔除，则跳过预处理
            if val in manually_excluded:
                continue
                
            binary_mask = (mask == val).astype(np.uint8)
            binary_masks[val] = binary_mask
            # 如果启用边缘剔除，检查对象是否与边缘相交
            edge_intersection[val] = np.any(binary_mask & edge_mask) if self.edge_exclusion else False
        
        # 批处理所有对象
        for val in unique_values:
            # 如果是边缘对象且启用了边缘剔除，则跳过
            if val in edge_intersection and edge_intersection[val]:
                continue
                
            # 如果对象被手动剔除，则跳过
            if val in manually_excluded:
                continue
            
            # 检查对象ID是否在手动剔除列表中
            obj_id = f"{frame_idx}-{val}"
            if obj_id in self.manually_excluded_objects:
                continue
                
            # 如果值不在binary_masks中（可能因为之前的预处理跳过了），则创建binary_mask
            if val not in binary_masks:
                binary_mask = (mask == val).astype(np.uint8)
            else:
                binary_mask = binary_masks[val]
            
            # 使用连通组件分析来确保正确识别每个独立对象 - 一次性分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            # 批处理所有连通区域
            for i in range(1, num_labels):  # 跳过背景(0)
                # 提取当前连通区域
                component = (labels == i).astype(np.uint8)
                
                # 找到轮廓 - 只需查找外部轮廓以提高速度
                contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 确保找到轮廓
                if not contours:
                    continue
                    
                # 获取最大轮廓
                contour = max(contours, key=cv2.contourArea)
                
                # 使用stats直接获取面积，避免再次计算轮廓面积
                area_pixels = stats[i, cv2.CC_STAT_AREA]
                
                # 跳过面积为0的对象
                if area_pixels == 0:
                    continue
                
                # 面积（物理单位 - μm²）- 使用预计算的scale_factor_squared
                area_um2 = area_pixels * scale_factor_squared
                
                # 中心点坐标 - 直接使用centroids结果
                center_x, center_y = centroids[i]
                
                # 使用轮廓计算最小外接矩形
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                angle = rect[2]
                
                # 确保width总是长边（长轴）
                if width < height:
                    width, height = height, width
                    # 旋转角度需要调整90度
                    angle = angle - 90
                
                # 计算周长 - 使用更简化的方式
                perimeter_pixels = cv2.arcLength(contour, True)
                
                # 一次性计算所有物理量，减少乘法运算
                # 转换为物理单位
                center_x_um = center_x * self.scale_factor
                center_y_um = center_y * self.scale_factor
                major_axis_um = width * self.scale_factor
                minor_axis_um = height * self.scale_factor
                perimeter_um = perimeter_pixels * self.scale_factor
                
                # 计算纵横比（长轴/短轴）- 防止除以0
                aspect_ratio = width / height if height > 0 else 0
                
                # 计算圆度 = 4π * 面积 / 周长²
                # 优化计算，减少除法操作
                circularity = 0
                if perimeter_pixels > 0:
                    circularity = (4 * np.pi * area_pixels) / (perimeter_pixels * perimeter_pixels)
                
                # 计算等效圆直径 = 2 * sqrt(面积/π)
                # 优化计算，使用np.sqrt一次计算
                equivalent_diameter_pixels = 2 * np.sqrt(area_pixels / np.pi)
                equivalent_diameter_um = equivalent_diameter_pixels * self.scale_factor
                
                # 优化轴线计算
                # 将角度转换为弧度 - 只计算一次
                angle_rad = np.radians(angle)
                
                # 计算半轴长度 - 只计算一次
                half_major = width / 2
                half_minor = height / 2
                
                # 计算长轴方向的正弦和余弦 - 只计算一次
                cos_angle = np.cos(angle_rad)
                sin_angle = np.sin(angle_rad)
                
                # 计算短轴方向的正弦和余弦 - 使用加法而不是再次计算
                minor_angle_rad = angle_rad + np.pi/2  # 加90度
                cos_minor = np.cos(minor_angle_rad)  # 或直接用 -sin_angle
                sin_minor = np.sin(minor_angle_rad)  # 或直接用 cos_angle
                
                # 更高效地计算轴线端点
                major_dx = half_major * cos_angle
                major_dy = half_major * sin_angle
                
                major_axis_start_x = center_x - major_dx
                major_axis_start_y = center_y - major_dy
                major_axis_end_x = center_x + major_dx
                major_axis_end_y = center_y + major_dy
                
                minor_dx = half_minor * cos_minor
                minor_dy = half_minor * sin_minor
                
                minor_axis_start_x = center_x - minor_dx
                minor_axis_start_y = center_y - minor_dy
                minor_axis_end_x = center_x + minor_dx
                minor_axis_end_y = center_y + minor_dy
                
                # 创建对象信息字典
                obj_info = {
                    "id": f"{frame_idx+1}-{val}",  # 新的ID格式：帧序号-对象掩膜灰度值
                    "label_value": val,
                    "area_pixels": area_pixels,
                    "area_um2": area_um2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "center_x_um": center_x_um,
                    "center_y_um": center_y_um,
                    "major_axis_pixels": width,
                    "minor_axis_pixels": height,
                    "major_axis_um": major_axis_um,
                    "minor_axis_um": minor_axis_um,
                    "aspect_ratio": aspect_ratio,
                    "angle_degrees": angle,
                    "perimeter_pixels": perimeter_pixels,
                    "perimeter_um": perimeter_um,
                    "circularity": circularity,
                    "equivalent_diameter_pixels": equivalent_diameter_pixels,
                    "equivalent_diameter_um": equivalent_diameter_um,
                    "contour": contour,  # 保存轮廓数据，用于后续绘制
                    # 添加长轴和短轴的起点和终点坐标
                    "major_axis_start_x": major_axis_start_x,
                    "major_axis_start_y": major_axis_start_y,
                    "major_axis_end_x": major_axis_end_x,
                    "major_axis_end_y": major_axis_end_y,
                    "minor_axis_start_x": minor_axis_start_x,
                    "minor_axis_start_y": minor_axis_start_y,
                    "minor_axis_end_x": minor_axis_end_x,
                    "minor_axis_end_y": minor_axis_end_y,
                    # 记录当前的缩放系数，用于后续更新
                    "_scale_factor": self.scale_factor,
                    # 存储帧序号供后续使用
                    "frame": frame_idx
                }
                
                objects.append(obj_info)
                
        return objects
    
    def get_frame_stats(self, frame_idx: int) -> Dict[str, Any]:
        """
        获取单一帧的统计数据
        
        Args:
            frame_idx: 帧索引
            
        Returns:
            包含统计数据的字典
        """
        # 检查analyzed_data是否正确初始化，及元素数目是否匹配
        if len(self.analyzed_data) != len(self.processed_masks):
            # 如果analyzed_data尚未初始化或大小不匹配，调整大小
            self.analyzed_data = [None] * len(self.processed_masks)
            
        # 如果已分析，使用保存的分析结果
        objects = None
        if frame_idx < len(self.analyzed_data):
            objects = self.analyzed_data[frame_idx]
            
        # 如果没有分析数据，返回空结果（不再自动分析）
        if not objects:
            return {
                "frame": frame_idx,
                "object_count": 0,
                "total_area_pixels": 0,
                "total_area_um2": 0,
                "area_fraction": 0,
                "objects": []
            }
            
        # 计算总面积
        total_area_pixels = sum(obj["area_pixels"] for obj in objects)
        total_area_um2 = sum(obj["area_um2"] for obj in objects)
        
        # 计算面积分数 - 使用处理后的掩膜来反映所有掩膜处理操作的影响
        if len(self.original_images) > frame_idx:
            img_height, img_width = self.original_images[frame_idx].shape[:2]
            total_image_area = img_height * img_width
            
            # 使用处理后的掩膜计算面积分数，以反映膨胀、腐蚀等操作的影响
            processed_mask = self.processed_masks[frame_idx]
            processed_mask_area = np.count_nonzero(processed_mask)
            area_fraction = processed_mask_area / total_image_area
        else:
            area_fraction = 0
            
        return {
            "frame": frame_idx,
            "object_count": len(objects),
            "total_area_pixels": total_area_pixels,
            "total_area_um2": total_area_um2,
            "area_fraction": area_fraction,
            "objects": objects
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """
        获取所有帧的全局统计数据
        
        Returns:
            包含全局统计数据的字典
        """
        # 初始化基本全局统计数据
        total_frames = len(self.processed_masks)
        time_points = [i / self.frame_rate for i in range(total_frames)]
        object_counts = [0] * total_frames
        area_fractions = [0] * total_frames
        all_objects = []
        
        # 如果分析状态标记为未分析，但analyzed_data中有局部数据，
        # 使用局部数据生成统计结果，不再需要空结果
        has_partial_data = False
        if not self.is_analyzed and len(self.analyzed_data) > 0:
            for frame_data in self.analyzed_data:
                if frame_data is not None and len(frame_data) > 0:
                    has_partial_data = True
                    break
        
        if has_partial_data or self.is_analyzed:
            # 如果已分析或有部分分析数据，使用这些数据生成统计信息
            all_frame_stats = []
            
            # 直接从处理后的掩膜计算面积分数，以反映膨胀、腐蚀等操作的影响
            for i in range(len(self.processed_masks)):
                # 计算处理后掩膜面积占比
                if len(self.original_images) > i:
                    # 使用处理后的掩膜来计算面积分数，反映所有掩膜处理的效果
                    mask = self.processed_masks[i]
                    img_height, img_width = self.original_images[i].shape[:2]
                    total_image_area = img_height * img_width
                    
                    # 计算非零像素的数量
                    mask_area = np.count_nonzero(mask)
                    
                    # 计算面积分数
                    area_fractions[i] = mask_area / total_image_area
            
            for i in range(len(self.processed_masks)):
                # 确保i在analyzed_data范围内
                if i < len(self.analyzed_data) and self.analyzed_data[i] is not None:
                    frame_stats = self.get_frame_stats(i)
                    
                    # 确保使用一致的面积分数
                    frame_stats["area_fraction"] = area_fractions[i]
                    
                    all_frame_stats.append(frame_stats)
                    all_objects.extend(frame_stats["objects"])
                    
                    # 更新对象数量
                    if "object_count" in frame_stats:
                        object_counts[i] = frame_stats["object_count"]
            
            # 获取对象面积、长轴、短轴和纵横比数据
            areas = [obj["area_um2"] for obj in all_objects if "area_um2" in obj]
            major_axes = [obj["major_axis_um"] for obj in all_objects if "major_axis_um" in obj]
            minor_axes = [obj["minor_axis_um"] for obj in all_objects if "minor_axis_um" in obj]
            aspect_ratios = [obj["aspect_ratio"] for obj in all_objects if "aspect_ratio" in obj]
            
            return {
                "frame_stats": all_frame_stats,
                "time_series": {
                    "frames": list(range(total_frames)),
                    "time_points": time_points,
                    "object_counts": object_counts,
                    "area_fractions": area_fractions
                },
                "distribution_data": {
                    "areas": areas,
                    "major_axes": major_axes,
                    "minor_axes": minor_axes,
                    "aspect_ratios": aspect_ratios
                },
                "total_frames": total_frames,
                "total_objects": len(all_objects),
                "scale_factor": self.scale_factor,
                "frame_rate": self.frame_rate
            }
        else:
            # 未分析时返回基本信息
            return {
                "frame_stats": [],
                "time_series": {
                    "frames": list(range(total_frames)),
                    "time_points": time_points,
                    "object_counts": object_counts,
                    "area_fractions": area_fractions
                },
                "distribution_data": {
                    "areas": [],
                    "major_axes": [],
                    "minor_axes": [],
                    "aspect_ratios": []
                },
                "total_frames": total_frames,
                "total_objects": 0,
                "scale_factor": self.scale_factor,
                "frame_rate": self.frame_rate
            } 
    
    def toggle_manually_exclude_object(self, frame_idx: int, obj_value: int, center_x: int, center_y: int) -> bool:
        """
        切换对象的手动剔除状态
        
        Args:
            frame_idx: 帧索引
            obj_value: 对象标签值
            center_x: 对象中心点x坐标
            center_y: 对象中心点y坐标
            
        Returns:
            是否已剔除 (True表示已剔除，False表示未剔除)
        """
        # 生成对象ID
        obj_id = f"{frame_idx}-{obj_value}"
        
        # 切换剔除状态
        if obj_id in self.manually_excluded_objects:
            # 如果对象已剔除，则取消剔除
            self.manually_excluded_objects.remove(obj_id)
            
            # 更新标记点列表
            if frame_idx in self.excluded_objects_marked:
                self.excluded_objects_marked[frame_idx] = [
                    (val, x, y) for val, x, y in self.excluded_objects_marked[frame_idx] 
                    if val != obj_value
                ]
                # 如果该帧没有标记点了，删除该帧记录
                if not self.excluded_objects_marked[frame_idx]:
                    del self.excluded_objects_marked[frame_idx]
            
            return False
        else:
            # 如果对象未剔除，则添加到剔除列表
            self.manually_excluded_objects.add(obj_id)
            
            # 添加标记点
            if frame_idx not in self.excluded_objects_marked:
                self.excluded_objects_marked[frame_idx] = []
            
            self.excluded_objects_marked[frame_idx].append((obj_value, center_x, center_y))
            
            return True
    
    def get_manually_excluded_objects(self, frame_idx: int = None) -> list:
        """
        获取手动剔除的对象列表
        
        Args:
            frame_idx: 指定帧索引，如果为None则返回所有帧的剔除对象
            
        Returns:
            剔除对象列表：如果frame_idx不为None，返回该帧的对象值列表
            否则返回所有剔除的对象ID列表
        """
        if frame_idx is not None:
            # 返回指定帧的剔除对象值列表
            excluded_values = []
            for obj_id in self.manually_excluded_objects:
                parts = obj_id.split('-')
                if len(parts) == 2 and int(parts[0]) == frame_idx:
                    excluded_values.append(int(parts[1]))
            return excluded_values
        else:
            # 返回所有剔除对象ID
            return list(self.manually_excluded_objects) 