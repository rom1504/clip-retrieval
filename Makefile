install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-dev: ## [Local development] Install test requirements
	python -m pip install -r requirements-test.txt

lint: ## [Local development] Run mypy, pylint and black
	python -m mypy clip_retrieval
	python -m pylint clip_retrieval
	python -m black --check -l 120 clip_retrieval

black: ## [Local development] Auto-format python code using black
	python -m black -l 120 .

build-pex:
	python3 -m venv .pexing
	. .pexing/bin/activate && python -m pip install -U pip && python -m pip install pex
	. .pexing/bin/activate && python -m pex --layout packed  -f https://download.pytorch.org/whl/cu113/torch_stable.html setuptools gcsfs==2022.1.0 s3fs==2022.1.0 pyspark==3.2.0 torch==1.10.2+cu113 torchvision==0.11.3+cu113 . -o clip_retrieval.pex -v
	rm -rf .pexing
	tar czf clip_retrieval_torch.tgz clip_retrieval.pex/.deps/torch-1.10.2+cu113-cp38-cp38-linux_x86_64.whl
	tar czf clip_retrieval.tgz --exclude clip_retrieval.pex/.deps/torch-1.10.2+cu113-cp38-cp38-linux_x86_64.whl clip_retrieval.pex

venv-lint-test: ## [Continuous integration]
	python3 -m venv .env && . .env/bin/activate && make install install-dev lint test && rm -rf .env

test: ## [Local development] Run unit tests
	rm -rf tests/test_folder/
	python -m pytest -x -s -v tests

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
