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
from torch.nn import functional as F
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

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


# 提取图像中的头像部分（左上角区域）
def extract_avatar(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        return None
    # 假设头像区域在图片的左上角, 手动定义裁剪区域 (根据图片具体情况调整)
    cropped_img = img[50:520, 50:600]  # 示例范围，需根据实际调整
    resized_avatar = resize_with_padding(cropped_img)  # 保持比例的缩放并填充
    return resized_avatar


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


# 生成左右对比图 (对比头像区域)
def generate_comparison_image(img1_path, img2_path, output_path):
    # 提取图像中的头像部分
    img1_avatar = extract_avatar(img1_path)
    img2_avatar = extract_avatar(img2_path)

    if img1_avatar is None or img2_avatar is None:
        print(f"无法生成对比图，头像区域提取失败: {img1_path}, {img2_path}")
        return

    # 将两个头像区域拼接为左右对比图
    img1_avatar = Image.fromarray(cv2.cvtColor(img1_avatar, cv2.COLOR_BGR2RGB))
    img2_avatar = Image.fromarray(cv2.cvtColor(img2_avatar, cv2.COLOR_BGR2RGB))

    total_width = img1_avatar.width + img2_avatar.width
    max_height = max(img1_avatar.height, img2_avatar.height)

    comparison_img = Image.new('RGB', (total_width, max_height))
    comparison_img.paste(img1_avatar, (0, 0))
    comparison_img.paste(img2_avatar, (img1_avatar.width, 0))

    # 保存生成的对比图
    comparison_img.save(output_path)
    print(f"对比图已生成并保存: {output_path}")


# 主函数，进行图片匹配，增加匹配轮数限制
def match_images(folder1, folder2, output_txt, comparison_folder, rounds=None):
    if not os.path.exists(comparison_folder):
        os.makedirs(comparison_folder)

    # 使用 utf-8 编码打开文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        folder1_images = sorted([img for img in os.listdir(folder1) if img.lower().endswith('.jpg')])
        folder2_images = sorted([img for img in os.listdir(folder2) if img.lower().endswith('.jpg')])

        # 如果有匹配轮数限制，只执行部分匹配
        total_rounds = len(folder1_images)
        if rounds is not None and rounds > 0:
            total_rounds = min(rounds, total_rounds)  # 限制执行的轮数

        # 遍历文件夹1中的每一张图片
        for i, img1_name in enumerate(folder1_images[:total_rounds]):
            img1_path = os.path.join(folder1, img1_name)
            img1_avatar = extract_avatar(img1_path)

            if img1_avatar is None:
                continue  # 跳过无法读取的图片

            # 记录三次匹配中的最佳匹配信息
            overall_best_match = None
            overall_best_similarity = -1
            overall_best_img2_name = ""

            # 重复若干次匹配
            for match_round in range(1):
                print(f"开始第 {match_round + 1} 轮匹配: {img1_name}")

                best_match = None
                best_similarity = -1
                best_img2_name = ""

                # 对文件夹2中的每张图片进行匹配
                for img2_name in folder2_images:
                    img2_path = os.path.join(folder2, img2_name)
                    img2_avatar = extract_avatar(img2_path)

                    if img2_avatar is None:
                        continue  # 跳过无法读取的图片

                    # 计算相似度
                    similarity = calculate_similarity(img1_avatar, img2_avatar)
                    print(f"匹配 {img1_name} -> {img2_name}, 相似度: {similarity:.4f}")

                    # 更新该轮次的最佳匹配
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = img2_path
                        best_img2_name = img2_name

                # 打印当前轮次最佳匹配结果
                print(f"第 {match_round + 1} 轮最佳匹配: {img1_name} -> {best_img2_name}，相似度: {best_similarity:.4f}")

                # 记录每轮的最佳匹配结果，并与之前轮次的最佳结果比较
                if best_similarity > overall_best_similarity:
                    overall_best_similarity = best_similarity
                    overall_best_match = best_match
                    overall_best_img2_name = best_img2_name

                # 生成对比图
                comparison_img_name = f"{img1_name}_vs_{best_img2_name}_round{match_round + 1}.jpg"
                comparison_img_path = os.path.join(comparison_folder, comparison_img_name)
                generate_comparison_image(img1_path, best_match, comparison_img_path)

                # 记录当前轮次的最佳相似度的匹配结果
                f.write(f"第 {match_round + 1} 轮: {img1_name} -> {best_img2_name}, 相似度: {best_similarity:.4f}\n")

            # 打印并记录当前图片的最终匹配结果
            print(f"{img1_name} 的最终最佳匹配是: {img1_name} -> {overall_best_img2_name}，相似度: {overall_best_similarity:.4f}")
            f.write(f"{img1_name} 最终最佳匹配 -> {overall_best_img2_name}, 相似度: {overall_best_similarity:.4f}\n")

            # 覆盖文件名，将最佳匹配的文件名替换原始图片的文件名
            new_img1_path = os.path.join(folder1, f"{overall_best_img2_name}")
            os.rename(img1_path, new_img1_path)
            print(f"已将 {img1_name} 重命名为 {new_img1_path}")


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
    rounds_to_run = 20  # 改为 None 来执行所有轮次

    # 进行图片匹配
    match_images(folder1, folder2, output_txt, comparison_folder, rounds=rounds_to_run)
