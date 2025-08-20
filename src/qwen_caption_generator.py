"""
使用Qwen-VL生成图像描述的脚本
"""
import os
import json
import argparse
import glob
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# 注意：这里只是示例代码框架，具体的Qwen-VL模型加载和推理需要根据实际使用的版本调整


def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions using Qwen-VL")
    
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help="Qwen-VL模型路径")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="图像目录路径")
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出captions文件路径")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批次大小")
    parser.add_argument("--max_length", type=int, default=30,
                        help="描述最大长度")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备")
    
    return parser.parse_args()


def load_qwen_vl_model(model_path, device):
    """加载Qwen-VL模型"""
    try:
        # 这里需要根据实际使用的Qwen-VL版本调整
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype='auto',
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading Qwen-VL model: {e}")
        print("Please install the required dependencies and adjust the model loading code.")
        return None, None


def generate_caption(model, tokenizer, image, prompt):
    """生成单张图像的描述"""
    try:
        # 这里的API需要根据具体的Qwen-VL版本调整
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image"}
                ]
            }
        ]
        
        # 调用模型生成描述（具体API需要调整）
        # response = model.chat(tokenizer=tokenizer, messages=messages, images=[image])
        # caption = response.strip().split('\n')[0][:40]
        
        # 临时返回示例描述
        caption = "移动应用界面截图"
        
        return caption
    
    except Exception as e:
        print(f"Error generating caption: {e}")
        return "应用界面截图"


def main():
    args = parse_args()
    
    # 加载模型
    print("Loading Qwen-VL model...")
    model, tokenizer = load_qwen_vl_model(args.model_path, args.device)
    
    if model is None:
        print("Failed to load model. Using placeholder captions.")
        model, tokenizer = None, None
    
    # 准备提示词
    prompt = """你是一名移动应用界面理解助手。请用一句中文简洁总结这张手机App截图的页面类型和主要目的。
要求：不超过30个字，避免无关修饰词；尽量包含品牌/功能关键词。
输出：仅一句话。"""
    
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        pattern = os.path.join(args.images_dir, '**', ext)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(image_files)} images")
    
    # 生成描述
    captions = []
    
    for image_path in tqdm(image_files, desc="Generating captions"):
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 生成描述
            if model is not None:
                caption = generate_caption(model, tokenizer, image, prompt)
            else:
                # 使用基于文件名的简单描述作为占位符
                filename = Path(image_path).stem
                if 'mcd' in filename.lower() or 'mcdonald' in filename.lower():
                    caption = "麦当劳应用界面截图"
                elif 'luckin' in filename.lower():
                    caption = "瑞幸咖啡应用界面截图"
                elif 'ctrip' in filename.lower() or 'trip' in filename.lower():
                    caption = "航旅纵横应用界面截图"
                else:
                    caption = "移动应用界面截图"
            
            # 保存结果
            relative_path = os.path.relpath(image_path, args.images_dir)
            captions.append({
                "image": relative_path,
                "caption": caption
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    # 保存到文件
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for item in captions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Generated {len(captions)} captions and saved to {args.output_file}")


if __name__ == "__main__":
    main()