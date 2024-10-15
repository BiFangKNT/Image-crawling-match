import os
import cv2
from PIL import Image, ImageOps
import numpy as np

# 从原始脚本中复制的函数
def resize_with_padding(image, target_size=(224, 224)):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_pil = ImageOps.pad(img_pil, target_size, color=(0, 0, 0))
    return np.array(img_pil)

def extract_areas(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None, None

    h, w = img.shape[:2]
    
    # 输出原图坐标大小
    print(f"原图大小: 宽度 = {w}, 高度 = {h}")

    # 提取第一个对比区域（无需检测，假设左上角区域）
    area1_x2, area1_y2 = min(600, w), min(520, h)
    area1 = img[50:area1_y2, 50:area1_x2]

    # 提取第二个对比区域并检查是否超出图片大小
    area2_x1, area2_y1 = min(0, w), min(1200, h)
    area2_x2, area2_y2 = min(area2_x1 + 240, w), min(area2_y1 + 280, h)
    
    if area2_x2 <= area2_x1 or area2_y2 <= area2_y1:
        print(f"第二个区域超出图片范围或无效: {image_path}")
        area2 = None
    else:
        area2 = img[area2_y1:area2_y2, area2_x1:area2_x2]

    # 返回两个区域
    return (resize_with_padding(area1) if area1.size > 0 else None, 
            resize_with_padding(area2) if area2 is not None and area2.size > 0 else None)

# 测试脚本主体
def test_second_area_extraction():
    # 设置输入和输出文件夹
    input_folder = "UrlDownload"
    output_folder = "SecondAreaTest"

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 指定要测试的图片
    test_image = "13.jpg"
    input_path = os.path.join(input_folder, test_image)

    if not os.path.exists(input_path):
        print(f"指定的图片不存在: {input_path}")
        return

    # 提取区域
    _, second_area = extract_areas(input_path)

    if second_area is not None:
        # 保存第二区域图片
        output_path = os.path.join(output_folder, f"second_area_{test_image}")
        cv2.imwrite(output_path, cv2.cvtColor(second_area, cv2.COLOR_RGB2BGR))
        print(f"第二区域已保存到: {output_path}")
    else:
        print(f"无法提取第二区域从图片: {test_image}")

if __name__ == "__main__":
    test_second_area_extraction()
