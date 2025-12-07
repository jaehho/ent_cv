#!/usr/bin/env bash
set -e

# Stop all containers
docker stop $(docker ps -aq) 2>/dev/null || true

# Remove all containers
docker rm -f $(docker ps -aq) 2>/dev/null || true

# Remove all images
docker rmi -f $(docker images -aq) 2>/dev/null || true

# Remove all volumes
docker volume rm $(docker volume ls -q) 2>/dev/null || true

# Remove all networks except defaults
for net in $(docker network ls -q); do
    case "$net" in
        bridge|host|none) ;; 
        *) docker network rm "$net" 2>/dev/null || true ;;
    esac
done

# Final prune
docker system prune -a --volumes -f
