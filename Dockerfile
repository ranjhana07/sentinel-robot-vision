FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libasound2 alsa-utils espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
ARG EXTRAS=tts
RUN pip install ".[${EXTRAS}]"

COPY configs ./configs

VOLUME ["/app/models", "/app/logs"]

ENTRYPOINT ["sentinel"]
CMD ["run", "--config", "configs/default.yaml"]
