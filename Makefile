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

## Web app
WEB_PORT    := 8050
WEB_SESSION := ent-cv-web

.PHONY: web-dev web-bg web-stop web-attach web-route
web-dev: ## Start the web dev server on :$(WEB_PORT) (foreground)
	cd web && npm run dev -- --port $(WEB_PORT) --host

web-bg: ## Start the web dev server on :$(WEB_PORT) in a tmux session
	tmux new-session -d -s $(WEB_SESSION) -c $(REPO_ROOT)/web 'npm run dev -- --port $(WEB_PORT) --host'
	echo "Web server started in tmux session '$(WEB_SESSION)' on :$(WEB_PORT)"

web-stop: ## Stop the web dev server tmux session
	tmux kill-session -t $(WEB_SESSION)

web-attach: ## Attach to the web dev server tmux session
	tmux attach -t $(WEB_SESSION)

web-route: ## Show cloudflared ingress rule for entcv.jaehho.com → :$(WEB_PORT)
	grep -A1 'hostname: entcv.jaehho.com' /etc/cloudflared/config.yml

## Cloudflared tunnel (systemd)
CLOUDFLARED_SERVICE ?= cloudflared

.PHONY: cloudflared-status cloudflared-start cloudflared-stop cloudflared-restart cloudflared-info cloudflared-edit cloudflared-route
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

cloudflared-edit: ## Edit cloudflared tunnel configuration
	sudo nvim /etc/cloudflared/config.yml

cloudflared-route: ## Show cloudflared tunnel route
	echo "Tunnel route information for 'mililab':"
	echo "-------------------------------------"
	cloudflared tunnel route ip mililab
