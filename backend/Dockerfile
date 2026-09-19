# Built with the repo root as build context (see docker-compose.yml's
# `context: .`), because backend/app/main.py imports root-level modules
# (pipeline_core.py, layer1.py..layer5.py, trade_glue.py, portfolio_glue.py)
# as plain top-level imports — the same "run from repo root" convention
# run_pipeline.py/run_portfolio.py already use, just inside a container
# instead of a local shell.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
