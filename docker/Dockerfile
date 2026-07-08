FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY knowledge_base/ knowledge_base/

EXPOSE 8000

CMD ["uvicorn", "src.retinocare.api.main:app", "--host", "0.0.0.0", "--port", "8000"]