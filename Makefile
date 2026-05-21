SHELL := /bin/bash
.SILENT:
.DEFAULT_GOAL := help

REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Most things have their own CLI — use those directly:
#   modeling/data:  uv run ent-cv --help
#   CVAT compose:   see infra/cvat/README.md
#   web prod:       docker compose --env-file .env -f web/docker-compose.yml ...
#   cloudflared:    systemctl {status,start,stop,restart} cloudflared
#
# This Makefile only wraps things that are genuinely awkward to type.

.PHONY: help web-dev lint test

help: ## Show this help message
	awk 'BEGIN{FS=":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

web-dev: ## Start Django + Vite dev servers in a tmux session (attach: tmux a -t ent-cv-web)
	set -e; \
	if tmux has-session -t ent-cv-web 2>/dev/null; then \
		echo "tmux session 'ent-cv-web' already exists."; \
		echo "  Attach:  tmux a -t ent-cv-web"; \
		echo "  Restart: tmux kill-session -t ent-cv-web && make web-dev"; \
		exit 0; \
	fi; \
	for port in 8000 5173; do \
		if ss -ltn "sport = :$$port" 2>/dev/null | grep -q LISTEN; then \
			echo "Error: port $$port already in use."; \
			sessions=$$(tmux ls 2>/dev/null | awk -F: '{print $$1}' | paste -sd, -); \
			[ -n "$$sessions" ] && echo "  Existing tmux sessions: $$sessions"; \
			echo "  A prior dev server may already be running — try http://localhost:5173"; \
			echo "  Note: prod compose uses :8050 and :8787; dev intentionally uses different ports."; \
			echo "  To find the process: ss -ltnp 'sport = :$$port'"; \
			exit 1; \
		fi; \
	done; \
	tmux new-session -d -s ent-cv-web -c $(REPO_ROOT) \
		'set -a && source $(REPO_ROOT)/.env && set +a && cd web/backend && $(REPO_ROOT)/.venv/bin/python manage.py runserver 8000'; \
	tmux split-window -t ent-cv-web -c $(REPO_ROOT) \
		'cd web/frontend && DJANGO_PORT=8000 npx vite --port 5173 --host'; \
	echo "Started in tmux session 'ent-cv-web'  (Django :8000, Vite :5173)"; \
	echo "Attach: tmux a -t ent-cv-web    Kill: tmux kill-session -t ent-cv-web"

lint: ## Ruff (Python) + ESLint (frontend)
	uv run ruff check .
	uv run ruff format --check .
	cd $(REPO_ROOT)/web/frontend && npx eslint .

test: ## pytest (backend) + vitest (frontend)
	uv run pytest web/backend/ --tb=short -q
	cd $(REPO_ROOT)/web/frontend && npx vitest run
