# from PIL import Image, ImageDraw

# def draw_bounding_box(image_path, normalized_bounding_box, output_path):
#     # 打开图像
#     image = Image.open(image_path)
    
#     # 获取图像的宽度和高度
#     image_width, image_height = image.size
    
#     # 提取归一化的边界框坐标
#     x_min, y_min, x_max, y_max = normalized_bounding_box
    
#     # 将归一化的坐标转换为实际像素坐标
#     left = int(x_min * image_width)
#     top = int(y_min * image_height)
#     right = int(x_max * image_width)
#     bottom = int(y_max * image_height)
    
#     # 创建一个可以在图像上绘图的对象
#     draw = ImageDraw.Draw(image)
    
#     # 使用红色绘制边界框
#     draw.rectangle([left, top, right, bottom], outline="red", width=3)
    
#     # 保存带有边界框的图像
#     image.save(output_path)

# # 示例使用
# image_path = "images/sculpture.jpg"  # 替换为你的图片路径
# #normalized_bounding_box = [0.684, 0.556, 0.744, 0.644] # 归一化的边界框坐标
# normalized_bounding_box = [0.63, 0.56,0.69, 0.68]
# output_path = "images_box/sculpture_box_qwen.jpg"  # 保存的图片路径

# draw_bounding_box(image_path, normalized_bounding_box, output_path)

from PIL import Image, ImageDraw, ImageFont

def draw_bounding_boxes(image_path, bounding_boxes, output_path):
    # 打开图像
    image = Image.open(image_path)
    
    # 获取图像的宽度和高度
    image_width, image_height = image.size
    
    # 创建一个可以在图像上绘图的对象
    draw = ImageDraw.Draw(image)
    
    # 加载字体（确保系统中存在该字体文件）
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # 遍历每个边界框
    for bbox in bounding_boxes:
        # 提取归一化的边界框坐标和标签信息
        x_min, y_min, x_max, y_max, color = bbox
        
        # 将归一化的坐标转换为实际像素坐标
        left = int(x_min * image_width)
        top = int(y_min * image_height)
        right = int(x_max * image_width)
        bottom = int(y_max * image_height)
        
        # 绘制边界框
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        
        # 在边界框旁边添加文本标注
        # draw.text((right + 5, top), label, fill=color, font=font)
    
    # 保存带有边界框的图像
    image.save(output_path)

# 示例使用
image_path = "images/sculpture.jpg"  # 替换为你的图片路径

# 定义两个边界框，每个边界框包含归一化坐标、颜色和标签
bounding_boxes = [
    [0.705, 0.528, 0.745, 0.626, "red"]
]

output_path = "images_exp/sculpture_sft_2w.jpg"  # 保存的图片路径

draw_bounding_boxes(image_path, bounding_boxes, output_path)