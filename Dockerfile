# ============================================
# Legal Document OCR - Production Dockerfile
# Multi-stage build for optimized image size
# ============================================

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=Asia/Seoul \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # PDF processing (poppler for pdf2image)
    poppler-utils \
    # Fonts for Korean
    fonts-nanum \
    fonts-nanum-coding \
    fonts-noto-cjk \
    # Utilities
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ============================================
# Dependencies Stage
# ============================================
FROM base AS dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ============================================
# Production Stage
# ============================================
FROM dependencies AS production

# Create non-root user for security
RUN useradd -m -u 1000 ocruser && \
    mkdir -p /app/uploads /app/outputs /app/logs /app/weights && \
    chown -R ocruser:ocruser /app

# Copy application code
COPY --chown=ocruser:ocruser . .

# Create necessary directories
RUN mkdir -p /app/temp_uploads /app/static /app/templates

# Switch to non-root user
USER ocruser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default port
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
