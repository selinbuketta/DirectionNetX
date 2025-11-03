# ==========================
# DirectionNet (TF2-compatible)
# ==========================
FROM tensorflow/tensorflow:2.11.0-gpu

LABEL maintainer="DirectionNet Docker"
LABEL description="Docker image for DirectionNet (CVPR 2020) running on TensorFlow 2.11 in compat.v1 mode"

# ---- Basic setup ----
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    git wget unzip libgl1-mesa-glx libglib2.0-0 python3-tk \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python dependencies ----
RUN pip install --upgrade pip && \
    pip install setuptools==65.5.1 wheel && \
    pip install \
    tensorflow-addons==0.19.0 \
    tensorflow-graphics==2021.12.3 \
    tensorflow-probability==0.19.0 \
    tf_slim==1.1.0 \
    absl-py==2.1.0 \
    numpy==1.24.3 \
    matplotlib==3.7.5 \
    scikit-image==0.20.0 \
    opencv-python==4.8.1.78 \
    Pillow==9.5.0

# ---- Create workspace ----
WORKDIR /workspace/DirectionNet

# Copy your project files into container
COPY . /workspace/DirectionNet

# ---- Optional: set PYTHONPATH for pano_utils if it's local ----
ENV PYTHONPATH="/workspace/DirectionNet/pano_utils:${PYTHONPATH}"

# ---- Set default command ----
CMD ["bash"]
