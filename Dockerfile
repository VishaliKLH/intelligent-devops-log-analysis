FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the full pipeline once so output/ files exist inside the image,
# then launch the dashboard.
RUN python src/preprocess.py && \
    python src/train_model.py && \
    python src/evaluate.py

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
