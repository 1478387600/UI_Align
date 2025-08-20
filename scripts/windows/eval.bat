@echo off
REM 评估（Windows）
set PROJECT_ROOT=%~dp0\..\..
cd /d %PROJECT_ROOT%

set MODEL_PATH=outputs\stage2\best\pytorch_model.bin
set TEST_IMAGES=data\custom_app\images
set TEST_CAPTIONS=data\custom_app\captions.jsonl
set TEST_LABELS=data\custom_app\labels.jsonl
set LABEL_MAP=data\custom_app\label_map.json

if not exist "%MODEL_PATH%" (
  echo [Error] Model not found at %MODEL_PATH%
  pause
  exit /b 1
)

python src\eval_retrieval.py --model_path %MODEL_PATH% --test_images %TEST_IMAGES% --test_captions %TEST_CAPTIONS% --test_labels %TEST_LABELS% --label_map %LABEL_MAP% --output_dir eval_results\retrieval --save_examples
python src\eval_cls.py --model_path %MODEL_PATH% --test_images %TEST_IMAGES% --test_labels %TEST_LABELS% --label_map %LABEL_MAP% --test_captions %TEST_CAPTIONS% --output_dir eval_results\classification
pause

