# Base image: official TensorFlow 2.15 with GPU support (CUDA 12.2, cuDNN 8.9)
FROM tensorflow/tensorflow:2.15.0-gpu

# Optional: silence TensorFlow logs
ENV TF_CPP_MIN_LOG_LEVEL=2

# Fix cuSolverDN issue on RTX 40 GPUs
ENV TF_USE_CUSOLVER=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# System dependencies
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git wget vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install your Python dependencies
RUN python3 -m pip install \
    "tensorflow==2.15.0" \
    "keras==2.15.0" \
    "tensorflow-probability==0.23.0" \
    "tf-slim==1.1.0" \
    "tensorflow-graphics" \
    tensorboard \
    numpy matplotlib opencv-python absl-py

# Optional (if on Apple Silicon; ignored on Linux)
RUN python3 -m pip install tensorflow-metal || true

# Set workspace directory
WORKDIR /workspace/DirectionNetGPU

# Expose TensorBoard port
EXPOSE 6006

# Default command: open bash shell
CMD ["/bin/bash"]

