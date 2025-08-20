# Makefile for Linux/macOS

.PHONY: setup stage1 stage2 eval demo clean

setup:
	python -m pip install -r requirements.txt

stage1:
	bash scripts/linux/stage1.sh

stage2:
	bash scripts/linux/stage2.sh

eval:
	bash scripts/linux/eval.sh

demo:
	bash scripts/linux/demo.sh $(img)

clean:
	rm -rf outputs/* eval_results/* demo_results/*

