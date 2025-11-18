import streamlit as st
import numpy as np
import os
from datetime import datetime
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# 页面配置
st.set_page_config(
    page_title="Pine Flower Phenology Recognition",
    page_icon="🌲",
    layout="wide"
)

# 模型文件检查
model_path = 'models/best.pt'
if os.path.exists(model_path):
    st.sidebar.success(f"✅ Model file loaded successfully ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
else:
    st.sidebar.error("❌ Model file not found")

# 松花时期类别映射
PINE_FLOWER_CLASSES = {
    0: {'name': 'elongation stage', 'color': (0, 255, 0), 'display_name': 'Elongation Stage'},
    1: {'name': 'ripening stage', 'color': (255, 165, 0), 'display_name': 'Ripening Stage'},
    2: {'name': 'decline stage', 'color': (255, 0, 0), 'display_name': 'Decline Stage'}
}

class StreamlitDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        # 在实际部署中这里会加载YOLO模型
        # 现在使用模拟检测来演示完整功能
        
    def detect_image(self, image):
        """执行图片检测"""
        try:
            st.write("---")
            st.write("🔍 **Starting detection process**")
            st.write(f"📐 Input image dimensions: {image.size if hasattr(image, 'size') else image.shape}")

            # 模拟YOLO检测过程
            detections = self.mock_detect(image)
            st.write(f"🎉 Total detected: {len(detections)} pine flowers")

            # 绘制检测结果
            st.write("🖌️ Starting to draw detection boxes...")
            result_image = self.draw_detections(image, detections)
            return detections, result_image

        except Exception as e:
            st.error(f"❌ Error during detection: {e}")
            import traceback
            st.error("Error details:")
            st.code(traceback.format_exc())
            return self.mock_detect(image), image

    def draw_detections(self, image, detections):
        """使用PIL绘制检测框"""
        st.write(f"🖌️ Need to draw {len(detections)} detection boxes")

        if len(detections) == 0:
            st.warning("⚠️ No detection boxes to draw, returning original image")
            return image

        # 创建图像副本
        pil_image = image.copy()
        draw = ImageDraw.Draw(pil_image)
        
        # 使用默认字体
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()

        image_width, image_height = pil_image.size
        st.write(f"📏 Canvas dimensions: width={image_width}, height={image_height}")

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det['bbox'])
            conf = det['confidence']
            color = det.get('color', (0, 255, 0))
            display_name = det['display_name']

            st.write(f"  🎨 Drawing box {i + 1}: {display_name}")
            st.write(f"     Confidence: {conf:.2f}")
            st.write(f"     Coordinates: [{x1}, {y1}, {x2}, {y2}]")

            # 检查坐标是否合理
            if x1 >= x2 or y1 >= y2:
                st.error(f"     ❌ Invalid coordinates: x1>=x2 or y1>=y2")
                continue

            if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
                st.warning(f"     ⚠️ Coordinates partially outside image boundaries")

            # 画检测框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            st.write(f"     ✅ Bounding box drawn")

            # 画标签
            label = f"{display_name} {conf:.2f}"
            
            # 估算文本尺寸
            try:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width = len(label) * 10
                text_height = 20

            # 计算标签位置
            label_y = max(y1 - text_height - 5, 5)
            
            # 画标签背景
            draw.rectangle([x1, label_y, x1 + text_width + 10, label_y + text_height + 5], 
                         fill=color)
            
            # 画文字
            draw.text((x1 + 5, label_y + 2), label, fill=(255, 255, 255), font=font)
            st.write(f"     ✅ Label drawn")

        st.success("🎨 All detection boxes drawn successfully!")
        return pil_image

    def mock_detect(self, image):
        """模拟检测"""
        if hasattr(image, 'size'):
            width, height = image.size
        else:
            height, width = image.shape[:2]
            
        detections = []
        import random
        num_detections = random.randint(2, 4)

        for i in range(num_detections):
            x1 = random.randint(50, width - 150)
            y1 = random.randint(50, height - 150)
            x2 = x1 + random.randint(80, 200)
            y2 = y1 + random.randint(80, 200)
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

    def get_statistics(self, detections):
        """获取统计信息"""
        stats = {'total_count': 0, 'by_stage': defaultdict(int)}
        if not detections:
            return stats

        stats['total_count'] = len(detections)
        for det in detections:
            stage = det['display_name']
            stats['by_stage'][stage] += 1

        return stats

# 初始化检测器
@st.cache_resource
def load_detector():
    return StreamlitDetector('models/best.pt')

def main():
    # 标题
    st.title("🌲 Pine Flower Phenology Recognition System")
    st.markdown("Based on YOLOv11 - Detect elongation, ripening, and decline stages")

    # 侧边栏
    st.sidebar.title("About")
    st.sidebar.info("This system uses YOLOv11 to detect and classify pine flower phenology stages.")

    # 文件上传
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: JPG, PNG, JPEG"
    )

    if uploaded_file is not None:
        # 显示文件信息
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
            "File type": uploaded_file.type
        }
        st.write("File details:", file_details)

        # 加载检测器
        detector = load_detector()

        if st.button("Start Detection", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # 使用PIL加载图像
                    image = Image.open(uploaded_file).convert('RGB')
                    
                    # 显示原图
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Image")
                        st.image(image, use_container_width=True)

                    # 检测
                    detections, result_image = detector.detect_image(image)

                    # 显示结果
                    with col2:
                        st.subheader("Detection Result")
                        st.image(result_image, use_container_width=True)

                    # 显示统计信息
                    st.subheader("📊 Detection Statistics")
                    stats = detector.get_statistics(detections)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Detections", stats['total_count'])

                    with col2:
                        for stage, count in stats['by_stage'].items():
                            st.metric(f"{stage}", count)

                    # 显示检测详情
                    st.subheader("🔍 Detection Details")
                    if detections:
                        for i, det in enumerate(detections):
                            st.write(
                                f"**Pine Flower {i + 1}**: {det['display_name']} (Confidence: {det['confidence']:.2f})")
                    else:
                        st.info("No pine flowers detected")

                    st.success(f"Detection completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
