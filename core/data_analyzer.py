#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据分析模块：提供数据分析和可视化功能
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from typing import List, Dict, Any, Tuple, Optional
import platform
import os

from core.logger import get_logger

# 获取模块日志记录器
logger = get_logger("data_analyzer")


class ChartGenerator:
    def __init__(self):
        """初始化图表生成器"""
        self.style_config = {
            'figure.figsize': (5, 4),
            'figure.dpi': 100,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': 'gray',
            'axes.grid': True,
            'grid.color': 'lightgray',
            'font.size': 9,
        }
        plt.rcParams.update(self.style_config)
        
        # 设置图表字体大小配置
        self.title_fontsize = 14      # 标题字体大小
        self.axis_label_fontsize = 12 # 坐标轴标签字体大小
        self.tick_fontsize = 10       # 刻度标签字体大小
        
        # 配置中文字体
        self.configure_chinese_font()
        
    def configure_chinese_font(self):
        """配置中文字体支持"""
        system = platform.system()
        font_found = False
        
        if system == 'Windows':
            # 优先尝试微软雅黑，它支持更多Unicode字符
            font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑
            if os.path.exists(font_path):
                matplotlib.font_manager.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = ['Microsoft YaHei']
                font_found = True
            else:
                # 尝试其他常见中文字体
                for font in ['simhei.ttf', 'simkai.ttf', 'simsun.ttc']:
                    font_path = f'C:/Windows/Fonts/{font}'
                    if os.path.exists(font_path):
                        matplotlib.font_manager.fontManager.addfont(font_path)
                        if 'simhei' in font:
                            plt.rcParams['font.family'] = ['SimHei']
                        elif 'simsun' in font:
                            plt.rcParams['font.family'] = ['SimSun']
                        elif 'simkai' in font:
                            plt.rcParams['font.family'] = ['KaiTi']
                        font_found = True
                        break
        elif system == 'Linux':
            # Linux系统字体
            linux_fonts = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'AR PL UMing CN']
            plt.rcParams['font.family'] = linux_fonts
            font_found = True  # 假设Linux中有这些字体之一
        elif system == 'Darwin':  # macOS
            # macOS系统字体
            mac_fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC']
            plt.rcParams['font.family'] = mac_fonts
            font_found = True  # 假设macOS中有这些字体之一
        
        # 如果没有找到合适的中文字体，尝试使用系统默认字体
        if not font_found:
            try:
                # 使用matplotlib自带的字体
                plt.rcParams['font.family'] = ['sans-serif']
                # 设置回退字体，可能支持部分中文字符
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'Arial', 'sans-serif']
                logger.warning("未找到中文字体，图表中的中文可能无法正确显示")
            except Exception as e:
                logger.error(f"设置字体出错: {e}")
        
        # 通用配置
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        
    def create_time_series_chart(self, 
                                time_points: List[float], 
                                values: List[float],
                                x_label: str = "时间 (s)",
                                y_label: str = "",
                                title: str = "",
                                color: str = 'blue') -> Tuple[Figure, FigureCanvas]:
        """
        创建时间序列图表
        
        Args:
            time_points: 时间点列表
            values: 值列表
            x_label: x轴标签
            y_label: y轴标签
            title: 图表标题
            color: 线条颜色
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        fig = Figure(figsize=self.style_config['figure.figsize'], dpi=self.style_config['figure.dpi'])
        canvas = FigureCanvas(fig)
        
        ax = fig.add_subplot(111)
        
        # 检查数据是否为空或全零
        has_valid_data = len(time_points) > 0 and len(values) > 0 and any(v != 0 for v in values)
        
        if not has_valid_data:
            # 创建空白图表但显示有意义的信息
            ax.set_xlabel(x_label, fontsize=self.axis_label_fontsize)
            ax.set_ylabel(y_label, fontsize=self.axis_label_fontsize)
            ax.set_title(title, fontsize=self.title_fontsize, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
            
            # 添加无数据提示文本
            ax.text(0.5, 0.5, '无数据可显示',
                   transform=ax.transAxes, verticalalignment='center', horizontalalignment='center',
                   fontsize=self.title_fontsize, fontweight='bold', color='gray')
            
            # 确保坐标轴有合理的范围
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            # 设置网格线在底层
            ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
            
            fig.tight_layout()
            return fig, canvas
        
        # 正常绘制时间序列图
        ax.plot(time_points, values, '-', color=color, linewidth=2)
        
        # 设置坐标轴标签和字体大小
        ax.set_xlabel(x_label, fontsize=self.axis_label_fontsize)
        ax.set_ylabel(y_label, fontsize=self.axis_label_fontsize)
        
        # 设置标题和加粗
        ax.set_title(title, fontsize=self.title_fontsize, fontweight='bold')
        
        # 设置刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置轴范围
        if len(time_points) > 0:
            ax.set_xlim([min(time_points), max(time_points) + 0.1])
        if len(values) > 0:
            y_min = min(0, min(values))
            y_max = max(values) * 1.1 if max(values) > 0 else 1
            ax.set_ylim([y_min, y_max])
        
        # 添加统计信息框 - 修改为放置在右下角
        if len(values) > 0:
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)
            max_val = np.max(values)
            min_val = np.min(values)
            
            stats_text = f"均值: {mean_val:.2f}\n中位数: {median_val:.2f}\n标准差: {std_val:.2f}\n最大值: {max_val:.2f}\n最小值: {min_val:.2f}\n样本数: {len(values)}"
            ax.text(0.95, 0.05, stats_text,
                   transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=self.tick_fontsize, zorder=3)  # 设置统计信息字体大小和层级
            
        fig.tight_layout()
        return fig, canvas
    
    def create_histogram(self, 
                         data: List[float], 
                         x_label: str = "",
                         y_label: str = "频率",
                         title: str = "",
                         color: str = 'blue',
                         bins: int = 30) -> Tuple[Figure, FigureCanvas]:
        """
        创建归一化直方图，确保所有柱子高度总和等于1
        
        Args:
            data: 数据列表
            x_label: x轴标签
            y_label: y轴标签
            title: 图表标题
            color: 柱状图颜色
            bins: 直方图的箱数
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        fig = Figure(figsize=self.style_config['figure.figsize'], dpi=self.style_config['figure.dpi'])
        canvas = FigureCanvas(fig)
        
        # 检查数据列表是否为空或None
        if data is None or len(data) == 0:
            # 创建空白图表但显示有意义的信息
            ax = fig.add_subplot(111)
            ax.set_xlabel(x_label, fontsize=self.axis_label_fontsize)
            ax.set_ylabel(y_label, fontsize=self.axis_label_fontsize)
            ax.set_title(title, fontsize=self.title_fontsize, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
            
            # 添加无数据提示文本
            ax.text(0.5, 0.5, '无数据可显示',
                   transform=ax.transAxes, verticalalignment='center', horizontalalignment='center',
                   fontsize=self.title_fontsize, fontweight='bold', color='gray')
            
            # 确保坐标轴有合理的范围
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            
            # 设置网格线在底层
            ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
            
            fig.tight_layout()
            return fig, canvas
        
        ax = fig.add_subplot(111)
        
        # 先设置网格线 - 将网格线绘制在底层
        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        
        # 计算合适的直方图箱数
        if len(data) < bins:
            bins = max(5, len(data) // 2)
        
        # 计算直方图中的频率
        hist, bin_edges = np.histogram(data, bins=bins)
        
        # 归一化处理，确保柱子高度和为1（而不是面积和为1）
        if np.sum(hist) > 0:
            weights = np.ones_like(data) / len(data)
        else:
            weights = None
            
        # 绘制直方图 - 使用weights参数确保柱子高度和为1
        n, bins_arr, patches = ax.hist(data, bins=bins, color=color, alpha=0.7, 
                                      edgecolor='black', zorder=2, weights=weights)
        
        # 设置坐标轴标签和字体大小
        ax.set_xlabel(x_label, fontsize=self.axis_label_fontsize)
        ax.set_ylabel(y_label, fontsize=self.axis_label_fontsize)
        
        # 设置标题和加粗
        ax.set_title(title, fontsize=self.title_fontsize, fontweight='bold')
        
        # 设置刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=self.tick_fontsize)
        
        # 添加统计信息
        if len(data) > 0:
            mean_val = np.mean(data)
            median_val = np.median(data)
            std_val = np.std(data)
            
            stats_text = f"均值: {mean_val:.2f}\n中位数: {median_val:.2f}\n标准差: {std_val:.2f}\n样本数: {len(data)}"
            ax.text(0.95, 0.95, stats_text,
                   transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=self.tick_fontsize, zorder=3)  # 设置统计信息字体大小和层级
            
        fig.tight_layout()
        return fig, canvas
    
    def create_object_count_chart(self, global_stats: Dict[str, Any]) -> Tuple[Figure, FigureCanvas]:
        """
        创建对象数量-时间曲线图
        
        Args:
            global_stats: 全局统计数据
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        time_points = global_stats["time_series"]["time_points"]
        object_counts = global_stats["time_series"]["object_counts"]
        
        return self.create_time_series_chart(
            time_points=time_points,
            values=object_counts,
            x_label="时间 (s)",
            y_label="对象数量",
            title="对象数量-时间曲线",
            color='blue'
        )
    
    def create_area_fraction_chart(self, global_stats: Dict[str, Any]) -> Tuple[Figure, FigureCanvas]:
        """
        创建面积分数-时间曲线图
        
        Args:
            global_stats: 全局统计数据
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        time_points = global_stats["time_series"]["time_points"]
        area_fractions = global_stats["time_series"]["area_fractions"]
        
        # 将面积分数转换为百分比
        area_fractions_percent = [af * 100 for af in area_fractions]
        
        return self.create_time_series_chart(
            time_points=time_points,
            values=area_fractions_percent,
            x_label="时间 (s)",
            y_label="面积分数 (%)",
            title="面积分数-时间曲线",
            color='green'
        )
    
    def create_area_histogram(self, global_stats: Dict[str, Any]) -> Tuple[Figure, FigureCanvas]:
        """
        创建对象面积分布直方图
        
        Args:
            global_stats: 全局统计数据
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        areas = global_stats["distribution_data"]["areas"]
        
        return self.create_histogram(
            data=areas,
            x_label="面积 (μm²)",
            y_label="频率",
            title="对象面积分布",
            color='red',
            bins=30
        )
    
    def create_major_axis_histogram(self, global_stats: Dict[str, Any]) -> Tuple[Figure, FigureCanvas]:
        """
        创建对象长轴分布直方图
        
        Args:
            global_stats: 全局统计数据
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        major_axes = global_stats["distribution_data"]["major_axes"]
        
        return self.create_histogram(
            data=major_axes,
            x_label="长轴 (μm)",
            y_label="频率",
            title="对象长轴分布",
            color='purple',
            bins=30
        )
    
    def create_minor_axis_histogram(self, global_stats: Dict[str, Any]) -> Tuple[Figure, FigureCanvas]:
        """
        创建对象短轴分布直方图
        
        Args:
            global_stats: 全局统计数据
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        minor_axes = global_stats["distribution_data"]["minor_axes"]
        
        return self.create_histogram(
            data=minor_axes,
            x_label="短轴 (μm)",
            y_label="频率",
            title="对象短轴分布",
            color='teal',
            bins=30
        )
    
    def create_aspect_ratio_histogram(self, global_stats: Dict[str, Any]) -> Tuple[Figure, FigureCanvas]:
        """
        创建对象纵横比分布直方图
        
        Args:
            global_stats: 全局统计数据
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        aspect_ratios = global_stats["distribution_data"]["aspect_ratios"]
        
        return self.create_histogram(
            data=aspect_ratios,
            x_label="纵横比 (长轴/短轴)",
            y_label="频率",
            title="对象纵横比分布",
            color='orange',
            bins=30
        )
    
    # 以下是针对当前帧的直方图创建方法
    def create_frame_area_histogram(self, frame_stats: Dict[str, Any]) -> Tuple[Figure, FigureCanvas]:
        """
        创建当前帧的对象面积分布直方图
        
        Args:
            frame_stats: 当前帧的统计数据
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        # 从当前帧数据中提取面积信息
        areas = [obj["area_um2"] for obj in frame_stats["objects"]]
        
        return self.create_histogram(
            data=areas,
            x_label="面积 (μm²)",
            y_label="频率",
            title=f"对象面积分布 (当前帧: {frame_stats['frame']})",
            color='red',
            bins=30
        )
    
    def create_frame_major_axis_histogram(self, frame_stats: Dict[str, Any]) -> Tuple[Figure, FigureCanvas]:
        """
        创建当前帧的对象长轴分布直方图
        
        Args:
            frame_stats: 当前帧的统计数据
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        # 从当前帧数据中提取长轴信息
        major_axes = [obj["major_axis_um"] for obj in frame_stats["objects"]]
        
        return self.create_histogram(
            data=major_axes,
            x_label="长轴 (μm)",
            y_label="频率",
            title=f"对象长轴分布 (当前帧: {frame_stats['frame']})",
            color='purple',
            bins=30
        )
    
    def create_frame_minor_axis_histogram(self, frame_stats: Dict[str, Any]) -> Tuple[Figure, FigureCanvas]:
        """
        创建当前帧的对象短轴分布直方图
        
        Args:
            frame_stats: 当前帧的统计数据
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        # 从当前帧数据中提取短轴信息
        minor_axes = [obj["minor_axis_um"] for obj in frame_stats["objects"]]
        
        return self.create_histogram(
            data=minor_axes,
            x_label="短轴 (μm)",
            y_label="频率",
            title=f"对象短轴分布 (当前帧: {frame_stats['frame']})",
            color='teal',
            bins=30
        )
    
    def create_frame_aspect_ratio_histogram(self, frame_stats: Dict[str, Any]) -> Tuple[Figure, FigureCanvas]:
        """
        创建当前帧的对象纵横比分布直方图
        
        Args:
            frame_stats: 当前帧的统计数据
            
        Returns:
            matplotlib Figure和FigureCanvas对象
        """
        # 从当前帧数据中提取纵横比信息
        aspect_ratios = [obj["aspect_ratio"] for obj in frame_stats["objects"]]
        
        return self.create_histogram(
            data=aspect_ratios,
            x_label="纵横比 (长轴/短轴)",
            y_label="频率",
            title=f"对象纵横比分布 (当前帧: {frame_stats['frame']})",
            color='orange',
            bins=30
        )
    
    def save_chart(self, fig: Figure, filepath: str, dpi: int = 300) -> bool:
        """
        保存图表为图像文件
        
        Args:
            fig: matplotlib Figure对象
            filepath: 保存路径
            dpi: 输出图像的DPI
            
        Returns:
            是否保存成功
        """
        try:
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            return True
        except Exception as e:
            logger.error(f"保存图表失败: {e}")
            return False
            
    def export_data_to_csv(self, data: Dict[str, Any], filepath: str) -> bool:
        """
        将数据导出为CSV文件
        
        Args:
            data: 要导出的数据
            filepath: 导出路径
            
        Returns:
            是否导出成功
        """
        import csv
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入标题行
                if isinstance(data, dict) and "headers" in data and "rows" in data:
                    writer.writerow(data["headers"])
                    writer.writerows(data["rows"])
                elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    # 如果是字典列表，使用第一个字典的键作为标题
                    if data:
                        headers = list(data[0].keys())
                        writer.writerow(headers)
                        for item in data:
                            writer.writerow([item.get(h, "") for h in headers])
                else:
                    # 简单列表或其他类型
                    if isinstance(data, list):
                        for item in data:
                            writer.writerow([item])
                    else:
                        writer.writerow([data])
                
            return True
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")
            return False
    
    def prepare_export_data(self, global_stats: Dict[str, Any]) -> Dict[str, Dict]:
        """
        准备导出数据
        
        Args:
            global_stats: 全局统计数据
            
        Returns:
            准备好的导出数据字典
        """
        # 时间序列数据
        time_series_data = {
            "headers": ["帧序号", "时间点(s)", "对象数量", "面积分数(%)"],
            "rows": []
        }
        
        frames = global_stats["time_series"]["frames"]
        time_points = global_stats["time_series"]["time_points"]
        object_counts = global_stats["time_series"]["object_counts"]
        area_fractions = global_stats["time_series"]["area_fractions"]
        
        for i in range(len(frames)):
            time_series_data["rows"].append([
                frames[i],
                time_points[i],
                object_counts[i],
                area_fractions[i] * 100  # 转为百分比
            ])
        
        # 对象详细数据
        objects_data = {
            "headers": [
                "对象ID", "帧序号", "面积(像素)", "面积(μm²)", 
                "中心点X", "中心点Y", "长轴(像素)", "短轴(像素)", 
                "长轴(μm)", "短轴(μm)", "纵横比", "角度(度)",
                "长轴起点X", "长轴起点Y", "长轴终点X", "长轴终点Y",
                "短轴起点X", "短轴起点Y", "短轴终点X", "短轴终点Y"
            ],
            "rows": []
        }
        
        all_objects = []
        for frame_stats in global_stats["frame_stats"]:
            all_objects.extend(frame_stats["objects"])
            
        for obj in all_objects:
            # 直接使用对象的ID，无需再从label_value构建
            obj_id = obj.get("id", f"{obj.get('frame', 0)+1}-{obj.get('label_value', 0)}")
            
            # 为每个对象构建一行数据，使用get方法安全获取可能缺失的键，提供默认值
            row = [
                obj_id,  # 使用新格式的对象ID: "帧序号-对象掩膜灰度值"
                obj.get("frame", 0),
                obj.get("area_pixels", 0),
                obj.get("area_um2", 0),
                obj.get("center_x", 0),
                obj.get("center_y", 0),
                obj.get("major_axis_pixels", obj.get("major_axis_um", 0) / obj.get("_scale_factor", 1)),  # 如果没有像素值，从物理值转换
                obj.get("minor_axis_pixels", obj.get("minor_axis_um", 0) / obj.get("_scale_factor", 1)),
                obj.get("major_axis_um", 0),
                obj.get("minor_axis_um", 0),
                obj.get("aspect_ratio", 1),
                obj.get("angle_degrees", 0),
                obj.get("major_axis_start_x", 0),
                obj.get("major_axis_start_y", 0),
                obj.get("major_axis_end_x", 0),
                obj.get("major_axis_end_y", 0),
                obj.get("minor_axis_start_x", 0),
                obj.get("minor_axis_start_y", 0),
                obj.get("minor_axis_end_x", 0),
                obj.get("minor_axis_end_y", 0)
            ]
            objects_data["rows"].append(row)
            
        # 分布数据
        area_dist_data = {
            "headers": ["面积(μm²)"],
            "rows": [[area] for area in global_stats["distribution_data"]["areas"]]
        }
        
        major_axis_dist_data = {
            "headers": ["长轴(μm)"],
            "rows": [[axis] for axis in global_stats["distribution_data"]["major_axes"]]
        }
        
        minor_axis_dist_data = {
            "headers": ["短轴(μm)"],
            "rows": [[axis] for axis in global_stats["distribution_data"]["minor_axes"]]
        }
        
        aspect_ratio_dist_data = {
            "headers": ["纵横比"],
            "rows": [[ratio] for ratio in global_stats["distribution_data"]["aspect_ratios"]]
        }
        
        # 汇总所有数据
        export_data = {
            "time_series": time_series_data,
            "objects": objects_data,
            "area_distribution": area_dist_data,
            "major_axis_distribution": major_axis_dist_data,
            "minor_axis_distribution": minor_axis_dist_data,
            "aspect_ratio_distribution": aspect_ratio_dist_data
        }
        
        return export_data 