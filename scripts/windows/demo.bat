@echo off
REM 推理演示（Windows）
set PROJECT_ROOT=%~dp0\..\..
cd /d %PROJECT_ROOT%

set MODEL_PATH=outputs\stage2\best\pytorch_model.bin
set LABEL_MAP=data\custom_app\label_map.json
set IMAGE_PATH=%1

if "%IMAGE_PATH%"=="" (
  echo Usage: %0 ^<image_path^>
  exit /b 1
)

if not exist "%MODEL_PATH%" (
  echo [Error] Model not found at %MODEL_PATH%
  exit /b 1
)
if not exist "%LABEL_MAP%" (
  echo [Error] Label map not found at %LABEL_MAP%
  exit /b 1
)
if not exist "%IMAGE_PATH%" (
  echo [Error] Image not found at %IMAGE_PATH%
  exit /b 1
)

python src\infer_demo.py --model_path %MODEL_PATH% --label_map %LABEL_MAP% --image %IMAGE_PATH% --save_visualization --output_dir demo_results
pause

