FROM python:3.11-slim

# System deps for Playwright
RUN apt-get update && apt-get install -y \
    wget gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers
RUN playwright install chromium --with-deps

# App code
COPY src/ /app/src/
WORKDIR /app

# Cloud Run uses PORT env var (default 8080)
EXPOSE 8080

CMD ["uvicorn", "src.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--timeout-keep-alive", "300"]
