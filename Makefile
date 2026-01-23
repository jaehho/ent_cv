.SILENT:

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ent_cv
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Help
.PHONY: help
help: ## Show this help message
	echo "Available targets:"
	echo "=================="
	grep -E '(^[a-zA-Z_-]+:.*?## .*$$|^## )' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; \
		     /^## / {gsub("^## ", ""); print "\n\033[1;35m" $$0 "\033[0m"}; \
		     /^[a-zA-Z_-]+:/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## Setup
.PHONY: requirements
requirements: ## Install Python dependencies
	uv sync

## Code Quality
.PHONY: clean
clean: ## Delete all compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

.PHONY: lint
lint: ## Lint using ruff (use `make format` to do formatting)
	ruff format --check
	ruff check

.PHONY: format
format: ## Format source code with ruff
	ruff check --fix
	ruff format

## Data Operations
.PHONY: scrape
scrape: ## Download videos from URLs listed in scripts/scrape/urls.txt
	bash scripts/scrape/run_downloads.sh

.PHONY: sync_data_down
sync_data_down: ## Download data from storage system
	gsutil -m rsync -r gs://ent-cv.jaehho.com/data/ data/

.PHONY: sync_data_up
sync_data_up: ## Upload data to storage system
	gsutil -m rsync -r data/ gs://ent-cv.jaehho.com/data/
