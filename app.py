import streamlit as st
from PIL import Image
import numpy as np
import random
from datetime import datetime

st.set_page_config(
    page_title="松花物候期识别系统", 
    page_icon="🌲",
    layout="wide"
)

st.title("🌲 松花物候期识别系统")
st.markdown("基于PMC_PhaseNet - 检测伸长期、成熟期、衰退期")

# 物候期类别
CLASSES = {
    0: "伸长期 (Elongation Stage)",
    1: "成熟期 (Ripening Stage)", 
    2: "衰退期 (Decline Stage)"
}

def simulate_detection(image):
    """模拟检测函数"""
    width, height = image.size
    detections = []
    
    # 随机生成1-3个检测结果
    for i in range(random.randint(1, 3)):
        x1 = random.randint(50, width-150)
        y1 = random.randint(50, height-150) 
        x2 = x1 + random.randint(100, 200)
        y2 = y1 + random.randint(100, 200)
        confidence = round(0.7 + random.random() * 0.25, 2)
        class_id = random.randint(0, 2)
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence,
            'class_name': CLASSES[class_id]
        })
    
    return detections

def draw_detections(image, detections):
    """绘制检测框"""
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    colors = [(0, 255, 0), (255, 165, 0), (255, 0, 0)]  # 绿, 橙, 红
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = colors[list(CLASSES.keys())[list(CLASSES.values()).index(det['class_name'])]]
        
        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 画标签
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.rectangle([x1, y1-25, x1+len(label)*8, y1], fill=color)
        draw.text((x1+5, y1-20), label, fill=(255,255,255))
    
    return draw_image

# 文件上传
uploaded_file = st.file_uploader("选择松花图像文件", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # 显示文件信息
    st.write(f"**文件信息**: {uploaded_file.name} ({uploaded_file.size/1024/1024:.2f} MB)")
    
    # 处理图像
    image = Image.open(uploaded_file)
    
    # 显示结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始图像")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("检测结果")
        # 模拟检测
        detections = simulate_detection(image)
        result_image = draw_detections(image, detections)
        st.image(result_image, use_container_width=True)
    
    # 显示统计
    st.subheader("📊 检测统计")
    if detections:
        total = len(detections)
        stages = {}
        for det in detections:
            stages[det['class_name']] = stages.get(det['class_name'], 0) + 1
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总检测数", total)
        with col2:
            st.metric("物候期类型", len(stages))
        with col3:
            avg_conf = sum(d['confidence'] for d in detections) / total
            st.metric("平均置信度", f"{avg_conf:.2f}")
        
        # 详细结果
        st.subheader("🔍 检测详情")
        for i, det in enumerate(detections):
            st.write(f"**目标 {i+1}**: {det['class_name']} (置信度: {det['confidence']:.2f})")
    
    st.success(f"检测完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.sidebar.title("系统信息")
st.sidebar.info("""
**PMC_PhaseNet 松花识别系统**

- 版本: 1.0 (演示版)
- 状态: 正常运行
- 框架: Streamlit + PIL
- 功能: 物候期识别演示
""")
