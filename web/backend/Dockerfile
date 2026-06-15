# --- Stage 1: build SPA ---
FROM node:20-slim AS fe
ARG VITE_OAUTH_CLIENT_ID
WORKDIR /fe
COPY web/frontend/package.json web/frontend/package-lock.json* ./
RUN npm install
COPY web/frontend/ ./
RUN VITE_OAUTH_CLIENT_ID=${VITE_OAUTH_CLIENT_ID} npm run build

# --- Stage 2: python backend serving the SPA ---
FROM python:3.12-slim
WORKDIR /app
COPY web/backend/requirements.txt ./req.txt
RUN pip install --no-cache-dir -r req.txt
COPY web/__init__.py ./web/__init__.py
COPY web/backend/ ./web/backend/
COPY --from=fe /fe/dist ./web/backend/static
ENV PYTHONUNBUFFERED=1
CMD ["sh", "-c", "uvicorn web.backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
