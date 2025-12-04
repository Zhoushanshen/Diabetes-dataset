import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
import numpy as np


def merge_specific_figure_pairs():
    """
    将指定的图片对合并为符合期刊要求的复合图
    符合要求：300 DPI，TIFF格式，添加(a)(b)标签，单文件输出
    """

    # 设置图片目录路径
    image_dir = r"D:\scientific data\上传图"

    # 设置字体为Times New Roman，与之前代码保持一致
    plt.rcParams['font.family'] = 'Times New Roman'

    # 定义要合并的图片对
    figure_pairs = [
        {
            "input_files": ["Figure.1(a).tif", "Figure.1(b).tif"],
            "output_file": "Figure_1.tiff"
        },
        {
            "input_files": ["BMI.tif", "PG2h.tif"],
            "output_file": "Figure_2.tiff"
        },
        {
            "input_files": ["Figure.6(a).tif", "Figure.6(b).tif"],
            "output_file": "Figure_6.tiff"
        }
    ]

    # 处理每个图片对
    for pair in figure_pairs:
        input_files = pair["input_files"]
        output_file = pair["output_file"]

        # 构建完整路径
        input_paths = [os.path.join(image_dir, f) for f in input_files]
        output_path = os.path.join(image_dir, output_file)

        print(f"\n处理 {input_files[0]} 和 {input_files[1]} -> {output_file}")

        # 检查输入文件是否存在
        missing_files = [f for f in input_paths if not os.path.exists(f)]
        if missing_files:
            print(f"❌ 找不到文件: {[os.path.basename(f) for f in missing_files]}")
            continue

        # 读取图片并获取尺寸信息
        images = []
        img_sizes = []
        for img_path in input_paths:
            try:
                img = Image.open(img_path)
                images.append(img)
                img_sizes.append(img.size)  # (width, height)
            except Exception as e:
                print(f"  ❌ 错误: 无法加载图片 {img_path}: {e}")
                continue

        if len(images) != 2:
            print(f"  ❌ 错误: 需要两张图片，但只成功加载了 {len(images)} 张")
            continue

        # 计算合适的画布尺寸，确保两张图片高度一致
        max_height = max(img_sizes[0][1], img_sizes[1][1])
        total_width = img_sizes[0][0] + img_sizes[1][0]

        # 根据图片比例调整画布大小
        fig_width = 10  # 基础宽度
        fig_height = fig_width * max_height / total_width * 1.2  # 保持比例，留出标签空间

        # 创建1行2列的复合图
        fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=300)

        # 处理每张图片
        for i, (ax, img, img_path) in enumerate(zip(axes, images, input_paths)):
            try:
                # 显示图片，保持原始宽高比
                ax.imshow(img)
                ax.axis('off')  # 隐藏坐标轴

                # 添加子图标签 (a), (b)... 无方框，字体大小比之前代码小两号
                # 之前代码中字体大小为14，小两号即为12
                ax.text(0.02, 0.98, f'{chr(97 + i)}',
                        transform=ax.transAxes,
                        fontsize=12,  # 从14改为12
                        fontweight='bold',
                        va='top',
                        color='black')  # 移除bbox参数，去掉方框

                print(f"  ✅ 已处理: {os.path.basename(img_path)} -> ({chr(97 + i)})")

            except Exception as e:
                print(f"  ❌ 错误: 处理图片 {img_path} 时出错: {e}")
                ax.text(0.5, 0.5, f'Error\n{os.path.basename(img_path)}',
                        ha='center', va='center',
                        transform=ax.transAxes,
                        color='red',
                        fontsize=10)  # 错误信息也使用较小字体
                ax.axis('off')

        # 调整子图间距，确保图片对齐
        plt.subplots_adjust(wspace=0.05, hspace=0)  # 减少水平间距，确保无垂直间距

        # 保存为高质量TIFF文件
        plt.savefig(output_path,
                    dpi=300,  # 300 DPI 分辨率
                    format='tiff',  # TIFF 格式
                    bbox_inches='tight',  # 紧凑布局
                    facecolor='white',  # 白色背景
                    edgecolor='none',  # 无边框
                    pil_kwargs={'compression': 'tiff_lzw'}  # 无损压缩
                    )

        print(f"  ✅ 复合图已保存: {output_file}")

        # 关闭图形，释放内存
        plt.close(fig)

        # 关闭图片文件
        for img in images:
            img.close()

    print("\n" + "=" * 50)
    print("✅ 所有指定的图片对已处理完成!")
    print("📊 输出文件:")
    for pair in figure_pairs:
        output_path = os.path.join(image_dir, pair["output_file"])
        if os.path.exists(output_path):
            print(f"   - {pair['output_file']} (已创建)")
        else:
            print(f"   - {pair['output_file']} (创建失败)")
    print("=" * 50)


def find_available_figures(image_dir):
    """
    查找目录中可用的图片文件，帮助调试
    """
    print("扫描目录中的图片文件...")
    image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
    all_images = []

    for extension in image_extensions:
        all_images.extend(glob.glob(os.path.join(image_dir, extension)))

    if all_images:
        print("找到以下图片文件:")
        for img in sorted(all_images):
            print(f"  - {os.path.basename(img)}")
    else:
        print("未找到任何图片文件")


# 运行主程序
if __name__ == "__main__":
    image_dir = r"D:\scientific data\上传图"

    # 检查目录是否存在
    if not os.path.exists(image_dir):
        print(f"❌ 目录不存在: {image_dir}")
        print("请检查路径是否正确")
    else:
        # 可选：显示目录中的图片文件，用于调试
        find_available_figures(image_dir)

        # 合并指定的图片对
        merge_specific_figure_pairs()