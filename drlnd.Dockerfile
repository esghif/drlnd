ARG cuda_version=9.2
ARG ubuntu_version=16.04
FROM nvidia/cuda:$cuda_version-cudnn7-devel-ubuntu$ubuntu_version

ENV http_proxy=http://172.17.0.1:3128
ENV https_proxy=http://172.17.0.1:3128
ENV HTTP_PROXY=http://172.17.0.1:3128
ENV HTTPS_PROXY=http://172.17.0.1:3128

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    apt-utils \
    libjpeg-dev \
    libpng-dev \
    cmake \
    build-essential \
    libx11-6 \
    libgtk2.0-0 \
    libcanberra-gtk-module

# OpenAI gym requirements
RUN apt-get install -y python-pyglet python3-opengl zlib1g-dev libjpeg-dev patchelf \
        cmake swig libboost-all-dev libsdl2-dev libosmesa6-dev xvfb ffmpeg    

RUN echo "proxy_servers:" >> $HOME/.condarc
RUN echo "    https: $https_proxy" >> $HOME/.condarc
RUN echo "    http: $http_proxy" >> $HOME/.condarc
RUN echo "ssl_verify: False" >> $HOME/.condarc

# Install Miniconda
RUN curl --proxy http://172.17.0.1:3128 -o ~/miniconda.sh -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

# Ensure conda version is at least 4.4.11
# (because of this issue: https://github.com/conda/conda/issues/6811)
ENV CONDA_AUTO_UPDATE_CONDA=true
RUN conda install -y "conda>=4.4.11" python=3.6 && conda clean -ya

# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests && conda clean -ya

# Update pip
RUN pip install --upgrade pip

# Install python requirements
COPY drlnd.requirements /tmp/
RUN pip install -r /tmp/drlnd.requirements

# Install OpenAI Gym
RUN pip install cython
RUN pip install gym
RUN pip install gym[atari]

# Install OpenCV3 Python bindings
RUN conda install -c conda-forge opencv

# Create an IPython kernel for the drlnd environment
RUN python -m ipykernel install --user --name drlnd --display-name "drlnd"

WORKDIR /data
RUN chmod -R a+w /data
WORKDIR /workspace
RUN chmod -R a+w /workspace

