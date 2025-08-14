install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

install-dev: ## [Local development] Install test requirements
	python -m pip install -r requirements-test.txt

lint: ## [Local development] Run mypy, pylint and black
	python -m mypy clip_retrieval
	python -m pylint clip_retrieval
	python -m black --check -l 120 .

black: ## [Local development] Auto-format python code using black
	python -m black -l 120 .

build-pex:
	python3.12 -m venv .pexing
	. .pexing/bin/activate && python3.12 -m pip install -U pip && python3.12 -m pip install pex
	. .pexing/bin/activate && python3.12 -m pex --layout packed setuptools gcsfs charset-normalizer s3fs pyspark torch torchvision "numpy==1.26.4" "opencv-python-headless==4.11.0.86" . -o clip_retrieval.pex -v
	rm -rf .pexing
	tar czf clip_retrieval_torch.tgz clip_retrieval.pex/.deps/torch-*
	tar czf clip_retrieval.tgz --exclude clip_retrieval.pex/.deps/torch-* clip_retrieval.pex

venv-lint-test: ## [Continuous integration]
	python3 -m venv .env && . .env/bin/activate && make install install-dev lint test && rm -rf .env

test: ## [Local development] Run unit tests
	rm -rf tests/test_folder/
	python -m pytest -x -s -v tests

.PHONY: help

help: # Run `make help` to get help on the make commands
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
