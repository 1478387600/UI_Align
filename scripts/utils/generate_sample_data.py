"""
生成示例数据的脚本，用于快速测试项目功能（迁移至 scripts/utils/）
"""
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_sample_image(text, size=(224, 224), bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    img = Image.new('RGB', size, bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    draw.text((x, y), text, fill=text_color, font=font)
    draw.rectangle([10, 10, size[0]-10, 50], outline=(200, 200, 200), width=2)
    draw.rectangle([10, size[1]-50, size[0]-10, size[1]-10], outline=(200, 200, 200), width=2)
    return img


def generate_rico_sample_data():
    print("生成RICO示例数据...")
    images_dir = Path("data/rico_screen2words/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        ("rico_001.jpg", "移动应用主页界面，包含导航菜单和内容列表"),
        ("rico_002.jpg", "设置页面，显示用户配置选项和开关按钮"),
        ("rico_003.jpg", "登录页面，包含用户名密码输入框"),
        ("rico_004.jpg", "商品列表页面，展示多个商品项目"),
        ("rico_005.jpg", "个人资料页面，显示用户信息和头像"),
        ("rico_006.jpg", "搜索页面，包含搜索框和筛选选项"),
        ("rico_007.jpg", "消息页面，显示聊天记录和输入框"),
        ("rico_008.jpg", "地图页面，显示位置信息和导航按钮"),
        ("rico_009.jpg", "相册页面，展示图片缩略图网格"),
        ("rico_010.jpg", "播放器页面，包含音乐控制按钮")
    ]
    captions = []
    for filename, caption in samples:
        img = create_sample_image(filename.split('.')[0], bg_color=(240, 248, 255))
        img.save(images_dir / filename)
        captions.append({"image": filename, "caption": caption})
    with open("data/rico_screen2words/captions.jsonl", 'w', encoding='utf-8') as f:
        for item in captions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"生成了{len(samples)}个RICO示例")


def generate_custom_sample_data():
    print("生成自定义示例数据...")
    apps = {
        'mcdonalds': {
            'pages': ['home', 'order', 'cart', 'payment', 'profile'],
            'color': (255, 196, 37),
            'captions': {
                'home': '麦当劳主页，展示推荐套餐和优惠活动',
                'order': '麦当劳点单页，展示套餐和加购按钮',
                'cart': '麦当劳购物车，显示已选商品和总价',
                'payment': '麦当劳支付页面，包含支付方式选择',
                'profile': '麦当劳个人中心，显示用户信息和订单历史'
            }
        },
        'luckin': {
            'pages': ['home', 'order', 'cart', 'payment', 'profile'],
            'color': (0, 112, 192),
            'captions': {
                'home': '瑞幸咖啡主页，展示新品推荐和门店信息',
                'order': '瑞幸下单页，列出热卖咖啡并可加入购物车',
                'cart': '瑞幸购物车，显示选择的咖啡和配送信息',
                'payment': '瑞幸支付页面，包含优惠券和支付选项',
                'profile': '瑞幸个人中心，显示会员等级和积分信息'
            }
        },
        'ctrip': {
            'pages': ['home', 'flight_search', 'hotel_search', 'order_detail', 'profile'],
            'color': (22, 119, 255),
            'captions': {
                'home': '携程主页，提供机票酒店等旅游服务入口',
                'flight_search': '航旅纵横机票搜索页，包含日期与城市选择',
                'hotel_search': '携程酒店搜索页，显示酒店列表和筛选条件',
                'order_detail': '携程订单详情页，显示预订信息和状态',
                'profile': '携程个人中心，包含会员信息和订单管理'
            }
        }
    }
    captions = []
    labels = []
    label_map = {}
    label_id = 0
    for app_name, app_data in apps.items():
        app_dir = Path(f"data/custom_app/images/{app_name}")
        app_dir.mkdir(parents=True, exist_ok=True)
        for page in app_data['pages']:
            label_name = f"{app_name}_{page}"
            label_map[label_name] = label_id
            label_id += 1
            for i in range(3):
                filename = f"{app_name}_{page}_{i+1:03d}.jpg"
                text = f"{app_name.upper()}\n{page.upper()}\n#{i+1}"
                img = create_sample_image(text, bg_color=app_data['color'])
                img.save(app_dir / filename)
                relative_path = f"{app_name}/{filename}"
                captions.append({"image": relative_path, "caption": app_data['captions'][page]})
                labels.append({"image": relative_path, "label": label_name})
    with open("data/custom_app/captions.jsonl", 'w', encoding='utf-8') as f:
        for item in captions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open("data/custom_app/labels.jsonl", 'w', encoding='utf-8') as f:
        for item in labels:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open("data/custom_app/label_map.json", 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"生成了{len(captions)}个自定义样本，{len(label_map)}个类别")
    print("类别映射:")
    for name, id in label_map.items():
        print(f"  {name}: {id}")


def main():
    print("=" * 50)
    print("生成示例数据")
    print("=" * 50)
    generate_rico_sample_data()
    print()
    generate_custom_sample_data()
    print()
    print("=" * 50)
    print("✅ 示例数据生成完成!")
    print("=" * 50)
    print("\n生成的文件:")
    print("- data/rico_screen2words/images/ (10张示例图片)")
    print("- data/rico_screen2words/captions.jsonl")
    print("- data/custom_app/images/ (45张示例图片)")
    print("- data/custom_app/captions.jsonl")
    print("- data/custom_app/labels.jsonl")
    print("- data/custom_app/label_map.json")
    print("\n现在可以运行训练脚本进行测试:")
    print("- Linux: bash scripts/linux/stage1.sh 及 bash scripts/linux/stage2.sh")
    print("- Windows: scripts\\windows\\stage1.bat 及 scripts\\windows\\stage2.bat")


if __name__ == "__main__":
    main()