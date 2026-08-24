FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY apps/webapp/requirements.txt /app/apps/webapp/requirements.txt
RUN pip install --no-cache-dir -r /app/apps/webapp/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY apps/webapp/backend /app/apps/webapp/backend
COPY apps/webapp/frontend /app/apps/webapp/frontend
COPY src/sentence_splitter.py /app/src/sentence_splitter.py

RUN touch /app/apps/__init__.py /app/apps/webapp/__init__.py \
    /app/apps/webapp/backend/__init__.py /app/src/__init__.py

ENV DOC_MODEL_ID=sandeep1103/lilly-fashion-muse-doc-gated
ENV ABSA_MODEL_ID=sandeep1103/lilly-fashion-muse-absa
ENV HF_HOME=/tmp/hf_cache

EXPOSE 7860
CMD ["uvicorn", "apps.webapp.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]

