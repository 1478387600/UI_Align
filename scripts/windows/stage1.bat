@echo off
REM Stage-1 训练（Windows）
set PROJECT_ROOT=%~dp0\..\..
cd /d %PROJECT_ROOT%

accelerate launch --config_file configs/accelerate_config.yaml src/train_stage1_align.py --config configs/stage1.yaml --use_wandb
pause

