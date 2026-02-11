FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# System deps (keep small); add more if build fails
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install python deps
# Copy requirements (build context is ./backend when building)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application (includes modelos/ and frontend/ as you requested)
COPY . /app/

    # ensure entrypoint is executable
RUN if [ -f /app/entrypoint.sh ]; then chmod +x /app/entrypoint.sh; fi

# Create unprivileged user
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

ENV PORT=8080
EXPOSE 8080

# Entrypoint will start uvicorn using $PORT
CMD ["/app/entrypoint.sh"]
