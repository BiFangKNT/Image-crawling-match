import os
import shutil
import cv2
import numpy as np
import requests
from lxml import html
from PIL import Image, ImageOps
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import sys

# 创建一个自定义的写入器类，同时写入到文件和控制台
class TeeWriter:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, message):
        self.file.write(message)
        self.stdout.write(message)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

# 设置图片下载保存的文件夹
output_dir = "UrlDownload"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 检查文件夹是否为空
def is_folder_empty(folder_path):
    if os.path.exists(folder_path) and os.listdir(folder_path):
        print(f"文件夹 {folder_path} 不为空，跳过下载步骤。")
        return False
    return True

# 下载图片并按顺序重命名
def download_image(url, img_num):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        img_name = f"{img_num}.jpg"
        img_path = os.path.join(output_dir, img_name)

        with open(img_path, 'wb') as f:
            f.write(response.content)

        print(f"图片已保存为: {img_path}")
        return img_path

    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return None

# 抓取网页图片并下载
def fetch_images_from_page():
    page_url = "https://mp.weixin.qq.com/s/rrVy1yt15JdpxDmy6SSsSA"
    try:
        # 发送GET请求获取网页内容
        response = requests.get(page_url)
        response.raise_for_status()

        # 解析网页
        tree = html.fromstring(response.content)

        # 循环遍历每个图片位置的 p[x]，尝试获取 data-src
        base_xpath = "/html/body/div[2]/div[2]/div[2]/div/div[1]/div[2]/section[2]/span/span"
        img_xpath_template = "/p[x]/img"
        p_counter = 1
        img_num = 1  # 用于图片的递增命名

        while True:
            img_xpath = img_xpath_template.replace("[x]", f"[{p_counter}]")
            full_xpath = f"{base_xpath}{img_xpath}"

            # 尝试提取图片的 data-src 属性
            img_elements = tree.xpath(f"{full_xpath}/@data-src")
            if not img_elements:
                print(f"未找到更多图片，停止在 p[{p_counter}]。")
                break

            img_url = img_elements[0]
            print(f"找到图片: {img_url}")

            # 下载图片并按顺序重命名
            download_image(img_url, img_num)
            img_num += 1  # 增加图片计数器

            # 增加计数器，继续寻找下一张图片
            p_counter += 1

    except requests.exceptions.RequestException as e:
        print(f"无法获取网页: {e}")
    except Exception as e:
        print(f"发生错误: {e}")


# 保持纵横比的缩放，并使用填充补齐
def resize_with_padding(image, target_size=(224, 224)):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    img_pil = ImageOps.pad(img_pil, target_size, color=(0, 0, 0))
    return np.array(img_pil)


# 提取图像中的两个对比区域
def extract_areas(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None, None

    h, w = img.shape[:2]
    
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


# 加载预训练的VGG16模型，使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
model.eval()  # 设置为评估模式

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# # 计算两张图片的相似度 (基于VGG16特征)
# def calculate_similarity(img1, img2):
#     img1_tensor = preprocess(Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
#     img2_tensor = preprocess(Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
#
#     with torch.no_grad():
#         features1 = model(img1_tensor)
#         features2 = model(img2_tensor)
#
#     # 使用余弦相似度计算特征相似度
#     similarity = F.cosine_similarity(features1.flatten(), features2.flatten(), dim=0).item()
#     return similarity

# 使用SSIM（结构相似度）计算两张图片的相似度
def calculate_similarity(img1, img2):
    # 转换图像为灰度图
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算SSIM相似度
    similarity, _ = ssim(img1_gray, img2_gray, full=True)
    return similarity


# 生成对比图，支持田字格布局
def generate_comparison_image(img1_path, img2_path, output_path):
    img1_area1, img1_area2 = extract_areas(img1_path)
    img2_area1, img2_area2 = extract_areas(img2_path)

    if img1_area1 is None or img2_area1 is None or img1_area2 is None or img2_area2 is None:
        print(f"无法生成对比图，区域提取失败: {img1_path}, {img2_path}")
        return

    # 转换为PIL图像
    img1_area1 = Image.fromarray(cv2.cvtColor(img1_area1, cv2.COLOR_BGR2RGB))
    img2_area1 = Image.fromarray(cv2.cvtColor(img2_area1, cv2.COLOR_BGR2RGB))
    img1_area2 = Image.fromarray(cv2.cvtColor(img1_area2, cv2.COLOR_BGR2RGB))
    img2_area2 = Image.fromarray(cv2.cvtColor(img2_area2, cv2.COLOR_BGR2RGB))

    # 使用田字格布局
    max_width = max(img1_area1.width, img2_area1.width, img1_area2.width, img2_area2.width)
    total_height = img1_area1.height + img1_area2.height

    comparison_img = Image.new('RGB', (2 * max_width, total_height))

    # 拼接第一个区域（上方）
    comparison_img.paste(img1_area1, (0, 0))
    comparison_img.paste(img2_area1, (max_width, 0))

    # 拼接第二个区域（下方）
    comparison_img.paste(img1_area2, (0, img1_area1.height))
    comparison_img.paste(img2_area2, (max_width, img1_area1.height))

    # 保存生成的对比图
    comparison_img.save(output_path)
    print(f"对比图已生成并保存: {output_path}")


# 主函数，进行图片匹配，增加匹配轮数限制
def match_images(folder1, folder2, output_txt, comparison_folder, rounds=None):
    if not os.path.exists(comparison_folder):
        os.makedirs(comparison_folder)

    with open(output_txt, 'w', encoding='utf-8') as f:
        tee = TeeWriter(f)
        old_stdout = sys.stdout
        sys.stdout = tee

        try:
            folder1_images = sorted([img for img in os.listdir(folder1) if img.lower().endswith('.jpg')])
            folder2_images = sorted([img for img in os.listdir(folder2) if img.lower().endswith('.jpg')])

            total_rounds = len(folder1_images)
            if rounds is not None and rounds > 0:
                total_rounds = min(rounds, total_rounds)

            for i, img1_name in enumerate(folder1_images[:total_rounds]):
                img1_path = os.path.join(folder1, img1_name)
                img1_area1, img1_area2 = extract_areas(img1_path)

                if img1_area1 is None or img1_area2 is None:
                    print(f"跳过无法处理的图片: {img1_name}")
                    continue

                print(f"开始匹配: {img1_name}")

                # 第一步：根据第一区域选出相似度最高的前四个
                top_4_matches = []
                for img2_name in folder2_images:
                    img2_path = os.path.join(folder2, img2_name)
                    img2_area1, img2_area2 = extract_areas(img2_path)

                    if img2_area1 is None or img2_area2 is None:
                        print(f"跳过无法处理的图片: {img2_name}")
                        continue

                    similarity_area1 = calculate_similarity(img1_area1, img2_area1)
                    top_4_matches.append((img2_name, img2_path, similarity_area1, img2_area1, img2_area2))

                top_4_matches.sort(key=lambda x: x[2], reverse=True)
                top_4_matches = top_4_matches[:4]

                print("第一区域匹配结果（前4名）:")
                for rank, (name, _, sim, _, _) in enumerate(top_4_matches, 1):
                    print(f"  {rank}. {name}, 相似度: {sim:.4f}")

                # 第二步：在前四个中根据第二区域选出相似度最高的
                best_match = max(top_4_matches, key=lambda x: calculate_similarity(img1_area2, x[4]))
                best_img2_name, best_match_path, _, best_area1, best_area2 = best_match

                final_similarity = calculate_similarity(img1_area2, best_area2)
                print(f"最终最佳匹配: {img1_name} -> {best_img2_name}，第二区域相似度: {final_similarity:.4f}")

                # 生成对比图
                comparison_img_name = f"{img1_name}_vs_{best_img2_name}.jpg"
                comparison_img_path = os.path.join(comparison_folder, comparison_img_name)
                generate_comparison_image(img1_path, best_match_path, comparison_img_path)

                # 重命名文件
                new_img1_path = os.path.join(folder1, f"{best_img2_name}")
                os.rename(img1_path, new_img1_path)
                print(f"已将 {img1_name} 重命名为 {new_img1_path}")

        finally:
            sys.stdout = old_stdout


# 主程序
if __name__ == "__main__":
    # 文件夹路径
    folder1 = "UrlDownload"  # 下载图片的文件夹（参考图片）
    folder2 = "ElysianRealm-Data"  # 需要匹配的图片
    output_txt = "matching_results.txt"
    comparison_folder = "output_comparison"

    # 检查文件夹1是否为空，如果为空，则开始下载图片
    if is_folder_empty(folder1):
        fetch_images_from_page()

    # 设置匹配轮数（为None时执行全部匹配，设置为1时只执行一轮）
    rounds_to_run = None  # 改为 None 来执行所有轮次

    # 进行图片匹配
    match_images(folder1, folder2, output_txt, comparison_folder, rounds=rounds_to_run)