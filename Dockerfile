FROM python:3.11.8-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY medical_assistant/ ./medical_assistant/

# Set working directory to medical_assistant for the app
WORKDIR /app/medical_assistant

# Expose port (Railway will set the PORT env var)
EXPOSE 5000

# Run the application
# Railway sets PORT env var automatically
CMD gunicorn webhook:app --bind 0.0.0.0:$PORT --timeout 120 --workers 2 --threads 2

