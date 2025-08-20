"""
项目安装和验证脚本（迁移至 scripts/utils/setup.py）
"""
import os
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    if version.major != 3 or version.minor < 10:
        print(f"❌ Python版本不符合要求: {version.major}.{version.minor}")
        print("   需要Python 3.10+")
        return False
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True


def check_gpu():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"✅ GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("❌ 未检测到可用GPU")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def install_dependencies():
    """安装依赖包"""
    print("正在安装依赖包...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False


def verify_installation():
    """验证关键包的安装"""
    packages = [
        'torch', 'transformers', 'accelerate', 'peft', 'bitsandbytes',
        'datasets', 'pillow', 'scikit-learn', 'matplotlib', 'seaborn'
    ]
    
    failed = []
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed.append(package)
    
    return len(failed) == 0


def check_directory_structure():
    """检查目录结构"""
    required_dirs = [
        'data/rico_screen2words/images',
        'data/custom_app/images/mcdonalds',
        'data/custom_app/images/luckin', 
        'data/custom_app/images/ctrip',
        'src', 'configs', 'outputs'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
        else:
            print(f"✅ {dir_path}")
    
    if missing:
        print("\n缺少的目录:")
        for dir_path in missing:
            print(f"❌ {dir_path}")
        return False
    
    return True


def check_config_files():
    """检查配置文件"""
    config_files = [
        'configs/accelerate_config.yaml',
        'configs/stage1.yaml',
        'configs/stage2.yaml',
        'data/custom_app/label_map.json'
    ]
    
    missing = []
    for file_path in config_files:
        if not Path(file_path).exists():
            missing.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing:
        print("\n缺少的配置文件:")
        for file_path in missing:
            print(f"❌ {file_path}")
        return False
    
    return True


def main():
    """主函数"""
    print("=" * 60)
    print("UI_Align 项目初始化和验证")
    print("=" * 60)
    
    print("\n1. 检查Python版本...")
    if not check_python_version():
        sys.exit(1)
    
    print("\n2. 检查目录结构...")
    if not check_directory_structure():
        print("请确保所有必需的目录都已创建")
    
    print("\n3. 检查配置文件...")
    if not check_config_files():
        print("请确保所有配置文件都已创建")
    
    print("\n4. 安装依赖包...")
    if not install_dependencies():
        sys.exit(1)
    
    print("\n5. 验证包安装...")
    if not verify_installation():
        print("请手动安装缺失的包")
        sys.exit(1)
    
    print("\n6. 提示：Accelerate 配置")
    print("如需要：accelerate config --config_file configs/accelerate_config.yaml")
    
    print("\n" + "=" * 60)
    print("✅ 项目初始化完成!")
    print("=" * 60)
    
    print("\n下一步:")
    print("- 生成示例数据：python scripts/utils/generate_sample_data.py")
    print("- 训练（Linux）：bash scripts/linux/stage1.sh && bash scripts/linux/stage2.sh")
    print("- 训练（Windows）：scripts\\windows\\stage1.bat && scripts\\windows\\stage2.bat")
    print("- 评估：bash scripts/linux/eval.sh 或 scripts\\windows\\eval.bat")


if __name__ == "__main__":
    main()