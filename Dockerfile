# -------------------------------------------------------
# 1. Base Image (Python 3.13)
# -------------------------------------------------------
FROM python:3.13-slim

# -------------------------------------------------------
# 2. System Dependencies
# -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    libpng-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------
# 3. Set Work Directory
# -------------------------------------------------------
WORKDIR /app

# -------------------------------------------------------
# 4. Copy Project Files
# -------------------------------------------------------
COPY . /app

# -------------------------------------------------------
# 5. Install Python Dependencies
# -------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other project dependencies
RUN pip install -r requirements.txt

# -------------------------------------------------------
# 6. Expose Flask Port
# -------------------------------------------------------
EXPOSE 5000

# -------------------------------------------------------
# 7. Run Flask App
# -------------------------------------------------------
ENV FLASK_APP=web_app/app.py
CMD ["python", "web_app/app.py"]
