FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-fetch the heavy MediaPipe pose model at build time so cold requests
# do not pay a runtime download penalty.
RUN python -c "import mediapipe as mp; pose = mp.solutions.pose.Pose(model_complexity=2); pose.close()"

COPY app ./app
COPY migrations ./migrations
COPY actionlab_setup.sh ./actionlab_setup.sh

ENV PORT=8080

CMD ["sh", "-c", "uvicorn app.orchestrator.orchestrator:app --host 0.0.0.0 --port ${PORT}"]
