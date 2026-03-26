FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN python -c "\
import tomllib;\
d=tomllib.load(open('pyproject.toml','rb'));\
open('_deps.txt','w').write('\n'.join(d['project']['dependencies']))" && \
    pip install --no-cache-dir -r _deps.txt && rm _deps.txt

COPY app/ ./app/

EXPOSE 8200

CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8200", "--timeout", "120", "app.main:app"]
