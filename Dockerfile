FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Railway için doğru port kullanımı:
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:${PORT}", "ai_server:app"]
