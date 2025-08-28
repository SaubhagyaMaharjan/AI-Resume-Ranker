# Step 1: Start with a stable, well-supported Python base image (Debian 12 "Bookworm")
FROM python:3.11-slim-bookworm

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system-level dependencies (This will now work)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     openjdk-17-jdk-headless \
#     tesseract-ocr \
#     libgl1 \
#     && apt-get clean && rm -rf /var/lib/apt/lists/*

# Step 4: Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your application code into the container
COPY . .

# Step 6: Pre-download the AI models during the build
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"

# Step 7: Expose the port your application runs on
EXPOSE 8000

# Step 8: The command to run your application when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]