# Multi-stage build for optimized Streamlit + TensorFlow image
# Optimisé pour maximiser l'utilisation du cache Docker

# Stage 1: Builder avec toutes les dépendances de build
FROM python:3.11-slim as builder

# Install system dependencies needed for TensorFlow and build tools
# Cette couche sera mise en cache si les dépendances système ne changent pas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip first (cached layer)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements FIRST - cette couche sera mise en cache si requirements.txt ne change pas
# C'est la clé pour accélérer les builds : les dépendances ne seront réinstallées 
# que si requirements.txt change
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
# Le cache Docker standard fonctionne très bien ici - pas besoin de BuildKit
# Les dépendances seront mises en cache si requirements.txt ne change pas
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Final stage - minimal runtime image
FROM python:3.11-slim

# Install only runtime dependencies (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code LAST - cette couche change souvent mais ne nécessite pas
# de réinstaller les dépendances si seul le code change
COPY src/ ./src/
COPY templates/ ./templates/
COPY PrédictionTempératuresWeb.py ./

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && \
    chown -R streamlit:streamlit /app

USER streamlit

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import socket; s = socket.socket(); s.connect(('localhost', 8501)); s.close()" || exit 1

# Run Streamlit
CMD ["streamlit", "run", "PrédictionTempératuresWeb.py", "--server.port=8501", "--server.address=0.0.0.0"]

