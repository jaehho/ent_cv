SHELL := /bin/bash
.SILENT:
.IGNORE:
.DEFAULT_GOAL := help

REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

## General
help: ## Show this help message
	echo "Available targets:"
	echo "=================="
	grep -E '(^[a-zA-Z_-]+:.*?## .*$$|^## )' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; \
		     /^## / {gsub("^## ", ""); print "\n\033[1;35m" $$0 "\033[0m"}; \
		     /^[a-zA-Z_-]+:/ {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

## Modeling
# Common paths — override on the command line: make train DATA=…
WEIGHTS ?= /mnt/data/ent_cv/models/v1/weights/best.pt
DATA    ?= /mnt/data/ent_cv/datasets/combined_new/data_with_val.yaml
SOURCE  ?= /mnt/data/ent_cv/raw/test
PREDICTIONS ?= /mnt/data/ent_cv/predictions/test

.PHONY: train predict val compare postprocess batch
train: ## Train a YOLO model  (override DATA=, EPOCHS=200, …)
	uv run ent-cv train --data $(DATA) --model $(or $(MODEL),yolo11x.pt)

predict: ## Run prediction  (set SOURCE=)
	uv run ent-cv predict --source $(SOURCE) --weights $(WEIGHTS) --verbose

val: ## Validate a model
	uv run ent-cv val --weights $(WEIGHTS) --data $(DATA)

compare: ## Compare trained models in MODELS_DIR
	uv run ent-cv compare --models-dir /mnt/data/ent_cv/models --verbose

postprocess: ## Run post-processing on predictions  (set PREDICTIONS=)
	uv run ent-cv postprocess --raw-json $(PREDICTIONS)/detections.json

batch: ## Run batch ops from YAML config
	uv run ent-cv batch /home/jaeho/ent_cv/ent_cv/modeling/configs/batch.yaml

## Data Management
.PHONY: disk-usage clean-frames clean-all clean-dry-run
disk-usage: ## Show disk usage for /mnt/data/ent_cv/
	du -sh /mnt/data/ent_cv/*/

clean-frames: ## Delete extracted prediction frames
	uv run ent-cv clean --frames

clean-all: ## Delete all prediction artifacts (frames, videos, labels)
	uv run ent-cv clean --all

clean-dry-run: ## Show what would be deleted (no changes)
	uv run ent-cv clean --all --dry-run

## CVAT
CVAT_HOST := cvat.jaehho.com
COMPOSE_FILES := \
	-f cvat/docker-compose.yml \
	-f cvat/components/serverless/docker-compose.serverless.yml \
	-f cvat/docker-compose.override.yml

export CVAT_HOST

.PHONY: cvat-up cvat-down cvat-build cvat-superuser
cvat-up: ## Start CVAT services
	docker compose --project-directory cvat $(COMPOSE_FILES) up -d

cvat-down: ## Stop CVAT services
	docker compose --project-directory cvat $(COMPOSE_FILES) down

cvat-build: ## Build CVAT services
	docker compose --project-directory cvat $(COMPOSE_FILES) -f cvat/docker-compose.dev.yml build --pull

cvat-superuser: ## Create a CVAT superuser
	docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

## Web App – Development
WEB_PORT    := 8050
DJANGO_PORT := 8787
WEB_SESSION := ent-cv-web
VENV_PYTHON := $(REPO_ROOT)/.venv/bin/python

.PHONY: web-django web-vite web-dev web-stop web-attach web-db web-db-down
web-django: ## Start Django dev server on :$(DJANGO_PORT) (foreground)
	cd $(REPO_ROOT)/web/backend && set -a && source $(REPO_ROOT)/.env && set +a && $(VENV_PYTHON) manage.py runserver $(DJANGO_PORT)

web-vite: ## Start Vite dev server on :$(WEB_PORT) (foreground)
	cd $(REPO_ROOT)/web/frontend && DJANGO_PORT=$(DJANGO_PORT) npx vite --port $(WEB_PORT) --host

web-dev: ## Start Django + Vite in a tmux session (two panes)
	tmux new-session -d -s $(WEB_SESSION) -c $(REPO_ROOT) \
		'set -a && source $(REPO_ROOT)/.env && set +a && cd web/backend && $(VENV_PYTHON) manage.py runserver $(DJANGO_PORT)'
	tmux split-window -t $(WEB_SESSION) -c $(REPO_ROOT) \
		'cd web/frontend && DJANGO_PORT=$(DJANGO_PORT) npx vite --port $(WEB_PORT) --host'
	echo "Web servers started in tmux session '$(WEB_SESSION)'"
	echo "  Django → :$(DJANGO_PORT)   Vite → :$(WEB_PORT)"
	echo "  attach: make web-attach"

web-stop: ## Stop the web dev tmux session
	tmux kill-session -t $(WEB_SESSION)

web-attach: ## Attach to the web dev tmux session
	tmux attach -t $(WEB_SESSION)

web-db: ## Start only the Postgres container (for dev — shares prod DB)
	docker compose --env-file $(REPO_ROOT)/.env -f $(REPO_ROOT)/web/docker-compose.yml up -d db

web-db-down: ## Stop the Postgres container
	docker compose --env-file $(REPO_ROOT)/.env -f $(REPO_ROOT)/web/docker-compose.yml stop db

## Web App – Production
.PHONY: web-build web-prod web-prod-down web-prod-logs web-migrate web-rollback web-createsuperuser
web-build: ## Build frontend for production
	cd $(REPO_ROOT)/web/frontend && npx vite build

web-prod: ## Start full production stack (Docker Compose)
	docker compose --env-file $(REPO_ROOT)/.env -f $(REPO_ROOT)/web/docker-compose.yml up -d --build

web-prod-down: ## Stop full production stack
	docker compose --env-file $(REPO_ROOT)/.env -f $(REPO_ROOT)/web/docker-compose.yml down

web-prod-logs: ## Tail production stack logs
	docker compose --env-file $(REPO_ROOT)/.env -f $(REPO_ROOT)/web/docker-compose.yml logs -f --tail=50

web-migrate: ## Run Django migrations  (dev by default; set WEB_PROD=1 for prod)
	if [ "$(WEB_PROD)" = "1" ]; then \
		docker compose --env-file $(REPO_ROOT)/.env -f $(REPO_ROOT)/web/docker-compose.yml exec backend python manage.py migrate --noinput; \
	else \
		cd $(REPO_ROOT)/web/backend && set -a && source $(REPO_ROOT)/.env && set +a && $(VENV_PYTHON) manage.py migrate --noinput; \
	fi

web-rollback: ## Rollback production to the previous image
	echo "Rolling back to previous images..."
	docker compose --env-file $(REPO_ROOT)/.env -f $(REPO_ROOT)/web/docker-compose.yml down
	docker compose --env-file $(REPO_ROOT)/.env -f $(REPO_ROOT)/web/docker-compose.yml up -d
	echo "Rollback complete (using cached images). Run 'make web-migrate' if needed."

web-createsuperuser: ## Create a Django superuser  (dev by default; set WEB_PROD=1 for prod)
	if [ "$(WEB_PROD)" = "1" ]; then \
		docker compose --env-file $(REPO_ROOT)/.env -f $(REPO_ROOT)/web/docker-compose.yml exec backend python manage.py createsuperuser; \
	else \
		cd $(REPO_ROOT)/web/backend && set -a && source $(REPO_ROOT)/.env && set +a && $(VENV_PYTHON) manage.py createsuperuser; \
	fi

## Cloudflared Tunnel (systemd)
CLOUDFLARED_SERVICE ?= cloudflared

.PHONY: cloudflared-status cloudflared-start cloudflared-stop cloudflared-restart cloudflared-info
cloudflared-status: ## Show cloudflared tunnel service status
	sudo systemctl status $(CLOUDFLARED_SERVICE) --no-pager

cloudflared-start: ## Start cloudflared tunnel service
	sudo systemctl start $(CLOUDFLARED_SERVICE)

cloudflared-stop: ## Stop cloudflared tunnel service
	sudo systemctl stop $(CLOUDFLARED_SERVICE)

cloudflared-restart: ## Restart cloudflared tunnel service
	sudo systemctl restart $(CLOUDFLARED_SERVICE)

cloudflared-info: ## Show cloudflared tunnel info and config
	echo "Tunnel information for 'mililab':"
	echo "-------------------------------"
	cloudflared tunnel info mililab
	echo "Tunnel configuration:"
	echo "---------------------"
	cat /etc/cloudflared/config.yml

## Testing & Linting
lint: ## Run all linters (Ruff + ESLint)
	uv run ruff check .
	uv run ruff format --check .
	cd $(REPO_ROOT)/web/frontend && npx eslint .

test: ## Run all tests (pytest + vitest)
	uv run pytest web/backend/ --tb=short -q
	cd $(REPO_ROOT)/web/frontend && npx vitest run
