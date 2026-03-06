FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir gunicorn

WORKDIR /app

# Install Python project (ent_cv + Django + ML deps)
COPY pyproject.toml /app/
COPY README.md /app/
COPY ent_cv/ /app/ent_cv/
RUN pip install --no-cache-dir .[db]

# Copy Django project
COPY web/backend/ /app/web/backend/

WORKDIR /app/web/backend

ENV DJANGO_SETTINGS_MODULE=config.settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN python manage.py collectstatic --noinput

EXPOSE 8787

CMD ["gunicorn", "config.wsgi:application", \
     "--bind", "0.0.0.0:8787", \
     "--workers", "2", \
     "--timeout", "120"]
