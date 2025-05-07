#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
帮助对话框：显示程序的使用帮助信息
"""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QPushButton, 
    QDialogButtonBox, QTabWidget, QWidget, QLabel,
    QScrollArea
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont


class HelpDialog(QDialog):
    """帮助对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.init_ui()
    
    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("帮助")
        self.resize(600, 500)
        
        main_layout = QVBoxLayout(self)
        
        # 创建选项卡组件
        tab_widget = QTabWidget()
        
        # 功能介绍选项卡
        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)
        
        features_text = QTextEdit()
        features_text.setReadOnly(True)
        features_text.setHtml(self.get_features_html())
        
        features_layout.addWidget(features_text)
        tab_widget.addTab(features_tab, "功能介绍")
        
        # 使用方法选项卡
        usage_tab = QWidget()
        usage_layout = QVBoxLayout(usage_tab)
        
        usage_text = QTextEdit()
        usage_text.setReadOnly(True)
        usage_text.setHtml(self.get_usage_html())
        
        usage_layout.addWidget(usage_text)
        tab_widget.addTab(usage_tab, "使用方法")
        
        # 关于选项卡
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setHtml(self.get_about_html())
        
        about_layout.addWidget(about_text)
        tab_widget.addTab(about_tab, "关于")
        
        main_layout.addWidget(tab_widget)
        
        # 添加关闭按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
    
    def get_features_html(self):
        """获取功能介绍HTML内容"""
        return """
        <h2>掩膜数据分析工具功能介绍</h2>
        
        <h3>一、原始数据处理</h3>
        <p>本工具可处理原始图片序列与相应的掩膜图片序列，掩膜图片必须为灰度图。
        工具会自动将不同灰度值识别为不同对象进行分析处理。</p>
        
        <h3>二、图像增强优化</h3>
        <ul>
          <li>支持高位深图像（16位深度或浮点型）的自动增强</li>
          <li>提供基于直方图调整的图像对比度增强，支持以下参数控制：
            <ul>
              <li>下限/上限百分比：通过直方图截断增强图像对比度</li>
              <li>亮度/对比度：精细调整图像显示效果</li>
            </ul>
          </li>
          <li>基于实时预览的交互式参数调整，直观展示效果变化</li>
          <li>直方图可视化展示，帮助理解图像数据分布</li>
        </ul>
        
        <h3>三、数据加载</h3>
        <ul>
          <li>可浏览选择原始图片序列文件夹与掩膜图片序列文件夹</li>
          <li>可设定需要处理分析的图片序号范围</li>
          <li>可设定图片序列的拍摄帧率和μm/pixel换算系数</li>
        </ul>
        
        <h3>四、掩膜处理</h3>
        <ul>
          <li>支持膨胀、腐蚀、开运算、闭运算等多种形态学操作</li>
          <li>支持填补空洞功能</li>
          <li>支持面积阈值噪点过滤</li>
          <li>支持剔除被视野边缘截断的掩膜数据</li>
          <li>支持手动剔除特定对象，适用于处理质量控制</li>
          <li>可选择对单帧或所有帧应用上述操作</li>
        </ul>
        
        <h3>五、实时预览</h3>
        <ul>
          <li>可实时预览原始图片与掩膜叠加显示</li>
          <li>可播放、暂停和逐帧浏览图片序列</li>
          <li>可显示/隐藏对象的边缘轮廓、中心点和长短轴</li>
          <li>可通过鼠标左键点击查看对象详细信息</li>
          <li>可通过鼠标右键点击标记并剔除特定对象</li>
        </ul>
        
        <h3>六、数据分析</h3>
        <ul>
          <li>可分析每个对象的面积、长短轴、纵横比、中心点坐标等数据</li>
          <li>可生成对象数量-时间曲线图、面积分数-时间曲线图</li>
          <li>可生成面积分布、长轴分布、短轴分布和纵横比分布直方图</li>
          <li>在分析计算中会自动跳过被手动剔除的对象</li>
        </ul>
        
        <h3>七、数据导出</h3>
        <ul>
          <li>可保存导出修改后的掩膜图片序列</li>
          <li>可导出各种统计图表和原始数据</li>
          <li>可导出每帧中的对象详细数据</li>
        </ul>
        """
    
    def get_usage_html(self):
        """获取使用方法HTML内容"""
        return """
        <h2>使用方法</h2>
        
        <h3>1. 数据加载</h3>
        <ol>
          <li>点击"浏览..."按钮选择原始图片目录和掩膜图片目录</li>
          <li>设置需要处理的图片序号范围（默认为全部图片）</li>
          <li>设置μm/pixel换算系数和拍摄帧率</li>
          <li>点击"加载图像"按钮加载数据</li>
        </ol>
        
        <h3>2. 数据分析</h3>
        <ol>
          <li>加载图像后，点击"分析计算"按钮开始分析</li>
          <li>分析完成后，右侧面板将显示各种统计图表</li>
        </ol>
        
        <h3>3. 掩膜处理</h3>
        <ol>
          <li>在左侧"掩膜处理"面板中选择需要的处理操作</li>
          <li>设置各种处理操作的参数</li>
          <li>选择是否应用于所有帧</li>
          <li>点击"应用处理"按钮执行处理</li>
          <li>如需撤销操作，可使用"撤销最近处理"按钮</li>
          <li>如需还原至原始状态，可使用"还原初始掩膜"按钮</li>
        </ol>
        
        <h3>4. 图像对比度调整</h3>
        <ol>
          <li>点击菜单栏中的"视图" → "图像增强" → "调整图像对比度"</li>
          <li>在弹出的对话框中可以看到当前帧的预览效果和直方图</li>
          <li>使用以下滑块调整参数：</li>
          <ul>
            <li><b>下限百分比</b>：调整直方图截断的下限（较低值可保留更多暗部细节）</li>
            <li><b>上限百分比</b>：调整直方图截断的上限（较低值可增强亮部细节）</li>
            <li><b>亮度</b>：调整整体图像亮度（大于1增加亮度，小于1降低亮度）</li>
            <li><b>对比度</b>：调整图像对比度（大于1增加对比度，小于1降低对比度）</li>
          </ul>
          <li>参数调整时，预览区域会实时显示效果变化</li>
          <li>直方图上的红线和绿线分别表示下限和上限阈值位置</li>
          <li>点击"重置参数"可恢复默认值</li>
          <li>完成调整后点击"应用"按钮保存设置并关闭对话框</li>
        </ol>
        
        <h3>5. 手动剔除对象</h3>
        <ol>
          <li>使用鼠标右键点击需要剔除的对象</li>
          <li>对象中心会出现白色十字标记，表示已选中</li>
          <li>右侧"手动剔除对象"列表中会显示被剔除对象的ID</li>
          <li>再次右键点击已标记对象可取消剔除</li>
          <li>点击"清空当前帧剔除标记"按钮可移除所有标记</li>
          <li>点击"应用处理"按钮后，被剔除的对象将在分析中被忽略</li>
        </ol>
        
        <h3>6. 图像预览</h3>
        <ol>
          <li>使用播放按钮开始/停止图片序列的播放</li>
          <li>使用上一帧/下一帧按钮手动浏览图片</li>
          <li>使用进度条跳转到特定帧</li>
          <li>使用"显示/隐藏边缘轮廓"等按钮切换显示模式</li>
          <li>鼠标左键点击对象可查看详细信息</li>
        </ol>
        
        <h3>7. 数据导出</h3>
        <ol>
          <li>点击菜单栏中的"文件" → "导出分析结果"</li>
          <li>在弹出的对话框中选择要导出的内容：</li>
          <ul>
            <li>修改后的掩膜图片序列</li>
            <li>分析结果图表（包含所有全局分析图表和每帧分析图表）</li>
            <li>详细数据表格（每帧中所有对象的信息）</li>
          </ul>
          <li>选择导出目录</li>
          <li>点击"确定"按钮完成导出</li>
        </ol>
        
        <h3>8. 自动增强高位深图像</h3>
        <ol>
          <li>点击菜单栏中的"视图" → "图像增强" → "自动增强高位深图像"</li>
          <li>选中此选项后，系统会自动对高位深图像进行直方图拉伸处理，以增强对比度</li>
          <li>您可以通过"调整图像对比度"功能进一步调整参数</li>
        </ol>
        
        <h3>快捷键</h3>
        <ul>
          <li>空格键: 播放/暂停</li>
          <li>左箭头: 上一帧</li>
          <li>右箭头: 下一帧</li>
        </ul>
        """
    
    def get_about_html(self):
        """获取关于HTML内容"""
        return """
        <h2>关于掩膜数据分析工具</h2>
        
        <p><b>版本:</b> 1.1.0</p>
        <p><b>作者:</b> Lucien</p>
        <p><b>联系方式:</b> lucien-6@qq.com</p>
        
        <h3>简介</h3>
        <p>掩膜数据分析工具是一款专为生物学研究中微生物图像分析设计的软件。它能够对原始图像序列和掩膜图像序列进行处理、分析和可视化，帮助研究人员更好地理解微生物的数量、分布和形态特征。</p>
        
        <h3>最新功能</h3>
        <ul>
          <li><b>增强的图像对比度调整功能</b>：采用直方图调整方法，提供更直观的交互式界面，使用户能实时预览调整效果</li>
          <li><b>高位深图像自动增强</b>：针对16位深度或浮点型图像，采用百分比阈值截断的直方图拉伸方法，提高图像对比度</li>
          <li><b>交互式参数调整</b>：支持通过滑块直观调整下限和上限百分比、亮度和对比度参数</li>
          <li><b>实时预览</b>：参数调整时实时显示效果变化，帮助用户获得最佳显示效果</li>
          <li>新增手动剔除对象功能，支持通过鼠标右键标记和清除特定对象</li>
          <li>改进UI布局，优化各组件的视觉效果和交互体验</li>
          <li>增强分析性能，改进多线程处理效率</li>
          <li>优化图表显示，提供更详细的数据可视化</li>
        </ul>
        
        <h3>技术特点</h3>
        <ul>
          <li><b>智能图像处理</b>：自动检测高位深图像并应用适当的增强算法</li>
          <li><b>高效运算</b>：使用多线程技术加速大规模图像序列处理</li>
          <li><b>内存优化</b>：通过批处理和缓存机制减少内存占用</li>
          <li><b>精准分析</b>：支持像素到物理单位的精确转换</li>
        </ul>
        
        <h3>致谢</h3>
        <p>感谢以下开源项目的支持：</p>
        <ul>
          <li>Python</li>
          <li>OpenCV</li>
          <li>NumPy</li>
          <li>Matplotlib</li>
          <li>PyQt5</li>
          <li>Pandas</li>
          <li>openpyxl</li>
        </ul>
        """ 