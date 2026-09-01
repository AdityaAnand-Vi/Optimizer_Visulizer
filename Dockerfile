# Stage 1: Build the Vite frontend
FROM node:20-alpine AS build-frontend

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Build the FastAPI backend and serve everything
FROM python:3.11-slim

WORKDIR /app

# Copy the requirements file and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ /app/

# Copy the built frontend static files from Stage 1 into the backend's dist folder
COPY --from=build-frontend /app/frontend/dist /app/dist

# Expose the port that Cloud Run expects (8080 by default, though we can configure it)
EXPOSE 8080

# Start the application - use $PORT env var injected by Cloud Run (defaults to 8080)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
