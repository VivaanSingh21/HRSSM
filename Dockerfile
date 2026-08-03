FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    unzip \
    nano \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libosmesa6-dev \
    libglfw3 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

COPY setup/conda_env.yml /tmp/conda_env.yml
ENV PIP_NO_BUILD_ISOLATION=1
RUN conda env create -f /tmp/conda_env.yml

# Legacy MuJoCo 2.0 binary, needed by this dm_control fork's build
RUN mkdir -p /root/.mujoco \
    && cd /root/.mujoco \
    && wget -q https://www.roboti.us/download/mujoco200_linux.zip \
    && unzip -q mujoco200_linux.zip \
    && mv mujoco200_linux mujoco200_linux_tmp \
    && mv mujoco200_linux_tmp mujoco200_linux \
    && rm mujoco200_linux.zip

# Rendering env vars, baked in permanently
ENV PYOPENGL_PLATFORM=osmesa
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
ENV MUJOCO_GL=osmesa

WORKDIR /workspace
