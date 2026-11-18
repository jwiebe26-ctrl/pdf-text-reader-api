# PDF Text Reader API — server-side OCR (Render free tier, Docker)
FROM python:3.12-slim

# Tesseract binary. tesseract-ocr-eng is kept as a fallback; the
# fast model below is what the API actually uses (TESSDATA_PREFIX).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng curl \
    && rm -rf /var/lib/apt/lists/*

# tessdata_fast English model (small, fast). The 'best' model measured
# worse than 'fast' on Jake's drawings, so fast is the one we ship.
RUN mkdir -p /app/tessdata \
    && curl -fsSL -o /app/tessdata/eng.traineddata \
        https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata

ENV TESSDATA_PREFIX=/app/tessdata
# Single-threaded tesseract: faster on shared-CPU hosts and avoids
# oversubscription (the desktop pool only behaved with it).
ENV OMP_THREAD_LIMIT=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY ocr_core.py ocr_enhanced.py app.py ./

EXPOSE 5000
# One worker, two threads: OCR is sequential by design (512 MB free
# tier), the extra thread keeps /health answering during a long read.
# Timeout 600s so a big multi-page read is not cut off mid-file.
CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 600"]
