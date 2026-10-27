# Local research workstation; independent field validation is not bundled.
FROM python:3.12-slim-bookworm AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt opencv-python-headless traci sumolib "eclipse-sumo==1.28.0" && \
    SUMO_DIR="$(python -c 'import sumo; print(sumo.SUMO_HOME)')" && \
    ln -s "$SUMO_DIR/bin/sumo" /usr/local/bin/sumo && \
    ln -s "$SUMO_DIR/bin/netconvert" /usr/local/bin/netconvert && \
    sumo --version && netconvert --version
COPY . /app
RUN useradd -u 10001 -m -s /usr/sbin/nologin streetlab && \
    mkdir -p /app/.streetlab-m5 /app/artifacts && chown -R 10001:10001 /app/.streetlab-m5 /app/artifacts
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=7s --start-period=45s --retries=3 CMD ["python","scripts/container_healthcheck.py"]
FROM base AS worker
USER root
RUN python -m pip install --no-cache-dir "sahi[ultralytics]" "trackers==2.6.0" "openvino"
USER 10001:10001
CMD ["python","-m","streetlab_integration.worker","--workdir","/app/.streetlab-m5","--model-dir","/frozen/model","--frozen-provenance","/frozen/provenance.json"]
FROM base AS web
CMD ["python","-m","uvicorn","streetlab_phase2.api:app","--host","0.0.0.0","--port","8000","--no-access-log"]
