
# ── Base image ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Logs (print statements in node.py, etc.) flush immediately to docker logs
ENV PYTHONUNBUFFERED=1

# Install system dependencies (curl needed for HEALTHCHECK below)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ───────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ─────────────────────────────────────────────────────
# .dockerignore keeps .env, .git, __pycache__ etc out of the image
COPY . .

# ── Run as non-root ────────────────────────────────────────────────────────
RUN useradd --create-home appuser
USER appuser

# ── Expose Streamlit port ──────────────────────────────────────────────────
EXPOSE 8501

# ── Health check (Streamlit's built-in endpoint) ───────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Run the app ────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
