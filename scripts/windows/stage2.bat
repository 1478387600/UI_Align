@echo off
REM Stage-2 训练（Windows）
set PROJECT_ROOT=%~dp0\..\..
cd /d %PROJECT_ROOT%

if not exist outputs\stage1 (
  echo [Error] outputs\stage1 not found. Run stage1 first.
  pause
  exit /b 1
)

accelerate launch --config_file configs/accelerate_config.yaml src/train_stage2_align_cls.py --config configs/stage2.yaml --use_wandb
pause

