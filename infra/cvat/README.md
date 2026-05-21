# CVAT compose layout

CVAT runs from the upstream `cvat-ai/cvat` submodule (pinned via `CVAT_VERSION`)
plus our overrides in `infra/cvat/docker-compose.override.yml`.

The compose stack spans three files. Export `CVAT_HOST` and `CVAT_VERSION` and
pass all three to every `docker compose` call:

```bash
export CVAT_HOST=cvat.jaehho.com CVAT_VERSION=v2.64.0

docker compose --project-directory cvat \
  -f cvat/docker-compose.yml \
  -f cvat/components/serverless/docker-compose.serverless.yml \
  -f infra/cvat/docker-compose.override.yml \
  up -d        # or: down, pull, config, logs, etc.
```

Other useful incantations:

```bash
# create a superuser
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

# print merged compose config (debugging)
docker compose --project-directory cvat -f cvat/docker-compose.yml \
  -f cvat/components/serverless/docker-compose.serverless.yml \
  -f infra/cvat/docker-compose.override.yml config
```
