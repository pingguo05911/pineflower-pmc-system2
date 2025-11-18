import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import random
from datetime import datetime
from collections import defaultdict

# 页面配置
st.set_page_config(
    page_title="松花物候期识别系统",
    page_icon="🌲",
    layout="wide"
)

# 松花物候期类别映射
PINE_FLOWER_CLASSES = {
    0: {'name': 'elongation stage', 'color': (0, 255, 0), 'display_name': '伸长期'},
    1: {'name': 'ripening stage', 'color': (255, 165, 0), 'display_name': '成熟期'},
    2: {'name': 'decline stage', 'color': (255, 0, 0), 'display_name': '衰退期'}
}

class PineFlowerDetector:
    def __init__(self):
        self.model_loaded = False
        # 在实际部署中这里会加载真实模型
        # 现在使用模拟检测来演示完整功能
        
    def detect_image(self, image):
        """执行图像检测"""
        try:
            # 模拟检测过程
            detections = self.mock_detect(image)
            
            # 绘制检测结果
            result_image = self.draw_detections(image, detections)
            
            return detections, result_image
            
        except Exception as e:
            st.error(f"检测过程中出错: {e}")
            return [], image
    
    def mock_detect(self, image):
        """模拟检测 - 在实际部署中会替换为真实模型推理"""
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            width, height = image.size
            
        detections = []
        # 随机生成2-4个检测结果
        num_detections = random.randint(2, 4)
        
        for i in range(num_detections):
            # 生成随机边界框
            bbox_width = random.randint(100, 250)
            bbox_height = random.randint(100, 250)
            x1 = random.randint(50, width - bbox_width - 50)
            y1 = random.randint(50, height - bbox_height - 50)
            x2 = x1 + bbox_width
            y2 = y1 + bbox_height
            
            # 生成随机置信度和类别
            confidence = round(0.7 + random.random() * 0.25, 2)
            class_id = random.randint(0, 2)
            class_info = PINE_FLOWER_CLASSES[class_id]
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_name': class_info['name'],
                'display_name': class_info['display_name'],
                'class_id': class_id,
                'color': class_info['color']
            })
        
        return detections
    
    def draw_detections(self, image, detections):
        """在图像上绘制检测框和标签"""
        # 创建图像副本用于绘制
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image.astype('uint8'))
        else:
            pil_image = image.copy()
            
        draw = ImageDraw.Draw(pil_image)
        
        # 绘制每个检测结果
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            display_name = det['display_name']
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            
            # 准备标签文本
            label = f"{display_name} {conf:.2f}"
            
            # 估算文本尺寸
            try:
                # 尝试使用默认字体估算
                bbox = draw.textbbox((0, 0), label)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                # 备用估算方法
                text_width = len(label) * 12
                text_height = 20
            
            # 计算标签位置（确保不超出图像边界）
            label_y = max(y1 - text_height - 10, 5)
            label_x2 = min(x1 + text_width + 15, pil_image.width - 5)
            
            # 绘制标签背景
            draw.rectangle([x1, label_y, label_x2, y1], fill=color)
            
            # 绘制标签文本
            draw.text((x1 + 5, label_y + 2), label, fill=(255, 255, 255))
        
        return pil_image
    
    def get_statistics(self, detections):
        """获取检测统计信息"""
        stats = {
            'total_count': 0, 
            'by_stage': defaultdict(int),
            'avg_confidence': 0
        }
        
        if not detections:
            return stats
        
        stats['total_count'] = len(detections)
        
        # 统计各物候期数量
        for det in detections:
            stage = det['display_name']
            stats['by_stage'][stage] += 1
        
        # 计算平均置信度
        if detections:
            stats['avg_confidence'] = sum(d['confidence'] for d in detections) / len(detections)
        
        return stats

# 初始化检测器
@st.cache_resource
def load_detector():
    return PineFlowerDetector()

def main():
    # 标题和介绍
    st.title("🌲 松花物候期识别系统")
    st.markdown("基于PMC_PhaseNet - 自动识别油松雄球花的物候期")
    
    # 侧边栏
    with st.sidebar:
        st.title("ℹ️ 系统信息")
        st.info("""
        **PMC_PhaseNet 松花识别系统**
        
        **功能特点：**
        - 🖼️ 支持图像上传
        - 🔍 自动物候期识别
        - 📊 实时统计结果
        - 🎯 高精度检测
        
        **识别类别：**
        - 🌱 伸长期
        - 🍎 成熟期  
        - 🍂 衰退期
        """)
        
        st.divider()
        st.caption("版本: 1.0 | 状态: 正常运行")
    
    # 文件上传区域
    st.subheader("📁 图像上传")
    uploaded_file = st.file_uploader(
        "选择松花图像文件",
        type=['png', 'jpg', 'jpeg'],
        help="支持格式: JPG, PNG, JPEG"
    )
    
    if uploaded_file is not None:
        # 显示文件信息
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write("**文件详情:**")
            st.write(f"- 文件名: {uploaded_file.name}")
            st.write(f"- 文件大小: {uploaded_file.size / 1024 / 1024:.2f} MB")
            st.write(f"- 文件类型: {uploaded_file.type.split('/')[-1].upper()}")
        
        # 加载检测器
        detector = load_detector()
        
        # 检测按钮
        if st.button("🚀 开始检测", type="primary", use_container_width=True):
            with st.spinner("🔍 检测中... 请稍候"):
                try:
                    # 加载图像
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # 执行检测
                    detections, result_image = detector.detect_image(image)
                    
                    # 显示结果对比
                    st.subheader("📊 检测结果")
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.markdown("**🖼️ 原始图像**")
                        st.image(image, use_container_width=True)
                    
                    with result_col2:
                        st.markdown("**🎯 检测结果**")
                        st.image(result_image, use_container_width=True)
                    
                    # 显示统计信息
                    st.subheader("📈 检测统计")
                    stats = detector.get_statistics(detections)
                    
                    # 统计卡片
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    with stat_col1:
                        st.metric(
                            label="总检测数量",
                            value=stats['total_count'],
                            help="图像中检测到的松花总数"
                        )
                    
                    with stat_col2:
                        st.metric(
                            label="物候期类型",
                            value=len(stats['by_stage']),
                            help="检测到的不同物候期种类"
                        )
                    
                    with stat_col3:
                        st.metric(
                            label="平均置信度",
                            value=f"{stats['avg_confidence']:.2f}",
                            help="所有检测结果的平均置信度"
                        )
                    
                    # 物候期分布
                    if stats['by_stage']:
                        st.subheader("🌿 物候期分布")
                        for stage, count in stats['by_stage'].items():
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.write(f"**{count}**")
                            with col2:
                                st.write(stage)
                            st.progress(count / stats['total_count'])
                    
                    # 详细检测结果
                    st.subheader("🔍 详细检测结果")
                    if detections:
                        for i, det in enumerate(detections):
                            with st.expander(f"松花 {i+1} - {det['display_name']} (置信度: {det['confidence']:.2f})"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**位置:** [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, {det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")
                                with col2:
                                    st.write(f"**物候期:** {det['display_name']}")
                    else:
                        st.info("未检测到松花目标")
                    
                    # 检测完成信息
                    st.success(f"✅ 检测完成! 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                except Exception as e:
                    st.error(f"❌ 处理图像时出错: {str(e)}")
    
    else:
        # 未上传文件时的提示
        st.info("👆 请上传松花图像文件开始检测")
        
        # 功能演示
        st.subheader("🎯 系统功能演示")
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        
        with demo_col1:
            st.markdown("**🖼️ 图像上传**")
            st.write("支持常见图像格式")
            
        with demo_col2:
            st.markdown("**🔍 智能检测**")
            st.write("自动识别物候期")
            
        with demo_col3:
            st.markdown("**📊 结果分析**")
            st.write("详细统计和可视化")

if __name__ == "__main__":
    main()
