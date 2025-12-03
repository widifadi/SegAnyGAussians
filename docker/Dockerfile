# ============================================
# Stage 1: Builder stage for compilation
# ============================================
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV DISTUTILS_USE_SDK=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Install build dependencies (will be discarded in final stage)
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    ninja-build \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CMake 3.22+ (required for building)
RUN wget -O /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v3.27.7/cmake-3.27.7-linux-x86_64.sh && \
    sh /tmp/cmake.sh --prefix=/usr/local --skip-license && \
    rm /tmp/cmake.sh

# Clone SegAnyGaussians repository with submodules (with retry and buffer increase)
WORKDIR /workspace
RUN git config --global http.postBuffer 524288000 && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 999999 && \
    git config --global url."https://github.com/".insteadOf git@github.com: && \
    git config --global url."https://".insteadOf git:// && \
    git clone https://github.com/Jumpat/SegAnyGAussians.git && \
    cd SegAnyGAussians && \
    git submodule update --init --recursive

# Create Python virtual environment
WORKDIR /workspace/SegAnyGAussians
RUN python3 -m venv gaussian_env && \
    gaussian_env/bin/pip install --upgrade pip

# Install build dependencies first
RUN gaussian_env/bin/pip install --no-cache-dir \
    wheel \
    setuptools \
    ninja && \
    rm -rf /root/.cache/pip

# Install PyTorch with CUDA 11.6 support (matching environment.yml requirements)
RUN for i in 1 2 3; do \
        gaussian_env/bin/pip install --no-cache-dir \
        torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 \
        --index-url https://download.pytorch.org/whl/cu116 && break || \
        (echo "PyTorch install failed, retry $i/3..." && sleep 10); \
    done && \
    rm -rf /root/.cache/pip

# Install other Python dependencies from environment.yml
RUN gaussian_env/bin/pip install --no-cache-dir \
    plyfile==0.8.1 \
    tqdm \
    opencv-python \
    pillow \
    hdbscan \
    matplotlib \
    dearpygui \
    open-clip-torch \
    joblib==1.1.0 && \
    rm -rf /root/.cache/pip

# Install fvcore and iopath (required for pytorch3d)
RUN gaussian_env/bin/pip install --no-cache-dir fvcore iopath && \
    rm -rf /root/.cache/pip

# Install pytorch3d from source (no compatible prebuilt wheel for CUDA 11.6 + PyTorch 1.12.1)
RUN FORCE_CUDA=1 MAX_JOBS=4 gaussian_env/bin/pip install --no-cache-dir \
    'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2' && \
    rm -rf /root/.cache/pip

# Install submodules for diff-gaussian-rasterization (REQUIRED - do not continue on failure)
RUN if [ -d "submodules/diff-gaussian-rasterization" ]; then \
        cd submodules/diff-gaussian-rasterization && \
        TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" ../../gaussian_env/bin/pip install --no-build-isolation .; \
    else \
        echo "ERROR: diff-gaussian-rasterization directory not found!" && exit 1; \
    fi && \
    rm -rf /root/.cache/pip

# Install submodules for diff-gaussian-rasterization_contrastive_f (REQUIRED - do not continue on failure)
RUN if [ -d "submodules/diff-gaussian-rasterization_contrastive_f" ]; then \
        cd submodules/diff-gaussian-rasterization_contrastive_f && \
        TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" ../../gaussian_env/bin/pip install --no-build-isolation .; \
    else \
        echo "ERROR: diff-gaussian-rasterization_contrastive_f directory not found!" && exit 1; \
    fi && \
    rm -rf /root/.cache/pip

# Install simple-knn submodule (REQUIRED - do not continue on failure)
RUN if [ -d "submodules/simple-knn" ]; then \
        cd submodules/simple-knn && \
        TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" ../../gaussian_env/bin/pip install --no-build-isolation .; \
    else \
        echo "ERROR: simple-knn directory not found!" && exit 1; \
    fi && \
    rm -rf /root/.cache/pip

# Install diff-gaussian-rasterization-depth submodule (REQUIRED - do not continue on failure)
RUN if [ -d "submodules/diff-gaussian-rasterization-depth" ]; then \
        cd submodules/diff-gaussian-rasterization-depth && \
        TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" ../../gaussian_env/bin/pip install --no-build-isolation .; \
    else \
        echo "ERROR: diff-gaussian-rasterization-depth directory not found!" && exit 1; \
    fi && \
    rm -rf /root/.cache/pip

# Install segment-anything from third_party (REQUIRED - do not continue on failure)
RUN if [ -d "third_party/segment-anything" ]; then \
        gaussian_env/bin/pip install --no-cache-dir -e third_party/segment-anything; \
    else \
        echo "ERROR: segment-anything directory not found!" && exit 1; \
    fi && \
    rm -rf /root/.cache/pip

# Install kmeans_pytorch from third_party (REQUIRED - do not continue on failure)
RUN if [ -d "third_party/kmeans_pytorch" ]; then \
        gaussian_env/bin/pip install --no-cache-dir -e third_party/kmeans_pytorch; \
    else \
        echo "ERROR: kmeans_pytorch directory not found!" && exit 1; \
    fi && \
    rm -rf /root/.cache/pip

# Download SAM model checkpoint
RUN mkdir -p third_party/segment-anything/sam_ckpt && \
    wget -O third_party/segment-anything/sam_ckpt/sam_vit_h_4b8939.pth \
    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth || \
    (mkdir -p sam_ckpt && \
     wget -O sam_ckpt/sam_vit_h_4b8939.pth \
     https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

# Clean up unnecessary files from repository to reduce size
RUN cd /workspace/SegAnyGAussians && \
    rm -rf .git && \
    find . -type d -name ".git" -exec rm -rf {} + 2>/dev/null || true && \
    find . -type f -name "*.pyc" -delete && \
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# ============================================
# Stage 2: Runtime stage (minimal image)
# ============================================
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Install only runtime dependencies (minimal set)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    python3 \
    python3-venv \
    libglew2.1 \
    libassimp5 \
    libboost-filesystem1.71.0 \
    libboost-system1.71.0 \
    libboost-thread1.71.0 \
    libgtk-3-0 \
    libopencv-core4.2 \
    libopencv-imgproc4.2 \
    libopencv-highgui4.2 \
    libglfw3 \
    libavdevice58 \
    libavcodec58 \
    libeigen3-dev \
    libxxf86vm1 \
    libembree3-3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy built application and virtual environment from builder stage
COPY --from=builder /workspace/SegAnyGAussians /workspace/SegAnyGAussians

# Set library path for PyTorch and CUDA extensions
ENV LD_LIBRARY_PATH=/workspace/SegAnyGAussians/gaussian_env/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH}

# Create proper entrypoint script that activates environment
RUN echo '#!/bin/bash' > /entrypoint.sh \
    && echo 'cd /workspace/SegAnyGAussians' >> /entrypoint.sh \
    && echo 'source gaussian_env/bin/activate' >> /entrypoint.sh \
    && echo 'export LD_LIBRARY_PATH=/workspace/SegAnyGAussians/gaussian_env/lib/python3.8/site-packages/torch/lib:${LD_LIBRARY_PATH}' >> /entrypoint.sh \
    && echo 'exec "$@"' >> /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Set working directory
WORKDIR /workspace/SegAnyGAussians

# Default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
